#!/usr/bin/env python3
# TinyLLaMA-ES Instruction Tuned
# Base optimizada + máscara por roles <|user|> / <|assistant|>

import os, argparse, json, shutil
from pathlib import Path
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm
from safetensors.torch import save_file, load_file

# ============================================================
# CONFIG
# ============================================================

class Config:
    vocab_size = 1401 #8192
    hidden_size = 512
    intermediate_size = 2048
    num_hidden_layers = 6
    num_attention_heads = 8
    max_position_embeddings = 512 # 256
    rope_theta = 10000.0
    rms_norm_eps = 1e-6
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 3

# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# ============================================================
# RoPE
# ============================================================

def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)

def apply_rope(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

# ============================================================
# ATTENTION
# ============================================================

class LlamaAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_size
        self.n_heads = cfg.num_attention_heads
        self.head_dim = h // cfg.num_attention_heads

        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))

# ============================================================
# MLP
# ============================================================

class LlamaMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# ============================================================
# DECODER LAYER
# ============================================================

class LlamaDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = LlamaAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = LlamaMLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

# ============================================================
# MODEL
# ============================================================

class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = nn.Module()        
        self.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.model.layers = nn.ModuleList(
            [LlamaDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.model.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)

    def build_rope(self, T, device):
        dim = self.cfg.hidden_size // self.cfg.num_attention_heads
        inv = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(T, device=device)
        freqs = torch.einsum("i,j->ij", t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos()[None, None, :, :]
        self.sin = emb.sin()[None, None, :, :]

    def forward(self, x):
        if self.cos is None or self.cos.size(2) < x.size(1):
            self.build_rope(x.size(1), x.device)

        h = self.model.embed_tokens(x)

        for layer in self.model.layers:
            h = layer(h, self.cos[:, :, :x.size(1)], self.sin[:, :, :x.size(1)])

        return self.lm_head(self.model.norm(h))

# ============================================================
# DATASET CON MÁSCARA POR ROLES
# ============================================================
class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, max_len, stride, assistant_id):
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)

        self.tokens = tokens.long()
        self.max_len = max_len
        self.stride = stride
        self.assistant_id = assistant_id

        total = len(self.tokens) - self.max_len - 1
        self.length = max(0, total // self.stride)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_len

        x = self.tokens[start:end]
        y = self.tokens[start+1:end+1].clone()

        assistant_pos = (x == self.assistant_id).nonzero(as_tuple=True)[0]

        if len(assistant_pos) > 0:
            last = assistant_pos[-1].item()
            y[:last+1] = -100
        else:
            y[:] = -100

        return x, y

# ============================================================
# TRAIN
# ============================================================
def train(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.bfloat16

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ---------- TOKENIZER ----------
    sp = spm.SentencePieceProcessor(model_file=args.sp_model)

    assistant_id = sp.piece_to_id("<|assistant|>")
    if assistant_id == -1:
        raise ValueError("Tokenizer no contiene <|assistant|>")

    # ---------- PRETOKEN ----------
    token_path = Path(args.tokens)

    if not token_path.exists():
        print("[*] Pretokenizando corpus (una sola vez)...")
        tokens = []

        with open(args.corpus, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = sp.encode(line)
                tokens.extend(ids)
                tokens.append(sp.eos_id())

        tokens = torch.tensor(tokens, dtype=torch.long)
        torch.save(tokens, token_path)
        print(f"[OK] tokens guardados en {token_path}")
    else:
        tokens = torch.load(token_path, map_location="cpu", weights_only=True)

    tokens = tokens.contiguous()

    # ---------- MODEL ----------
    model = LlamaForCausalLM(Config).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05
    )

    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if scaler and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1

    max_tokens = 12_000_000

    # ---------- TRAIN LOOP ----------
    for epoch in range(start_epoch, args.epochs):

        # --- SUBMUESTREO POR EPOCH (CORRECTO) ---
        if len(tokens) > max_tokens:
            start = torch.randint(0, len(tokens) - max_tokens, (1,)).item()
            tokens_epoch = tokens[start:start + max_tokens]
        else:
            tokens_epoch = tokens

        ds = TokenDataset(
            tokens_epoch,
            Config.max_position_embeddings,
            args.stride,
            assistant_id
        )

        num_workers = min(12, os.cpu_count())

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=8,
            drop_last=True
        )

        model.train()
        total_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")):

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=dtype):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-100
                ) / args.accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # ---- CORRECTO: step SOLO cuando toca ----
            if (step + 1) % args.accum_steps == 0:
                if scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                opt.zero_grad(set_to_none=True)

            total_loss += loss.item() * args.accum_steps

        # ---- FLUSH FINAL SI QUEDÓ RESTO ----
        if (step + 1) % args.accum_steps != 0:
            if scaler:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            opt.zero_grad(set_to_none=True)

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dl):.4f}")

        os.makedirs(args.out, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "epoch": epoch,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            },
            Path(args.out) / f"checkpoint{epoch+1}.pt"
        )

        torch.save(
            model.state_dict(),
            Path(args.out) / f"model_epoch{epoch+1}.pt"
        )


# ============================================================
# SAVE HF
# ============================================================
def save_hf(model, sp_model, out_dir, epoch):
    path = Path(out_dir) / f"hf_epoch_{epoch}"
    path.mkdir(parents=True, exist_ok=True)

    # ---- config HF REAL ----
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": Config.vocab_size,
        "hidden_size": Config.hidden_size,
        "intermediate_size": Config.intermediate_size,
        "num_hidden_layers": Config.num_hidden_layers,
        "num_attention_heads": Config.num_attention_heads,
        "max_position_embeddings": Config.max_position_embeddings,
        "rms_norm_eps": Config.rms_norm_eps,
        "rope_theta": Config.rope_theta,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "use_cache": True,
        "bos_token_id": Config.bos_token_id,
        "eos_token_id": Config.eos_token_id,
        "pad_token_id": Config.pad_token_id,
        "torch_dtype": "float16"
    }

    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    shutil.copy(sp_model, path / "tokenizer.model")   
    sd = model.state_dict().copy()  # .copy() evita modificar el original
    sd.pop('lm_head.weight', None)  # Elimina si existe, sin error si no está
    
    save_file(sd, path / "model.safetensors")
    print(f"[OK] HF export correcto: {path}")
    
@torch.no_grad()
def generate(
    model,
    sp,
    prompt,
    max_new_tokens=128,
    temperature=0.9,
    top_k=40,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    ids = [sp.bos_id()] + sp.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    max_ctx = model.cfg.max_position_embeddings

    for _ in range(max_new_tokens):
        if x.size(1) > max_ctx:
            x = x[:, -max_ctx:]

        logits = model(x)[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)

        x = torch.cat([x, next_id], dim=1)

        if next_id.item() == sp.eos_id():
            break

    return sp.decode(x[0].tolist())


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--sp_model", required=True)
    ap.add_argument("--tokens", default="tokens.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=24) # 8 o 96
    ap.add_argument("--accum_steps", type=int, default=2) # 4 o 8
    ap.add_argument("--stride", type=int, default=256) # 128 o 896
    ap.add_argument("--resume", help="checkpoint .pt para continuar entrenamiento")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--lr", type=float, default=1.5e-4) # learning rate 3e-4

    # ---- modo inference ----
    ap.add_argument("--checkpoint")
    ap.add_argument("--prompt", default="Érase una vez,")

    args = ap.parse_args()

    # ================= INFERENCE =================
    if args.checkpoint:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        sp = spm.SentencePieceProcessor(model_file=args.sp_model)
        model = LlamaForCausalLM(Config)

        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.suffix == ".safetensors":
            state = load_file(str(checkpoint_path))
            model.load_state_dict(state, strict=False) # True
        elif checkpoint_path.suffix == ".pt":
            ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=True)
            else:
                model.load_state_dict(ckpt, strict=True)  # fallback si es state_dict directo
        else:
            raise ValueError(f"Formato de checkpoint no soportado: {checkpoint_path.suffix}")
        
        out = generate(model, sp, args.prompt, device=device)

        print("\n=== GENERACIÓN LOCAL ===\n")
        print(out)
        exit(0)

    # ================= TRAIN =================
    if not args.corpus:
        raise SystemExit("Falta --corpus para entrenar")

    train(args)
