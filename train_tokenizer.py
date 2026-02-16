import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    #input="chat_corpus_es.txt",
    input="chat_corpus_es_v3.txt",
    model_prefix="sp_es",
    model_type="bpe",
    vocab_size=1401,#8192,
    character_coverage=0.9995,
    
    # IDs
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    
    # CHAT TOKENS (ya bien)
    user_defined_symbols=["<|user|>", "<|assistant|>"],
    
    # NORMALIZACIÓN (correcto)
    normalization_rule_name="nfkc",
    add_dummy_prefix=False,
    escape_whitespaces=True,
    remove_extra_whitespaces=True,
    split_by_whitespace=True,
    split_by_number=True,
    split_by_unicode_script=True,
    allow_whitespace_only_pieces=False,
    byte_fallback=False,
    hard_vocab_limit=True,
    num_threads=16,
    
    # ¡Cambios clave para corpus grandes!
    input_sentence_size=1000000,          # Muestrea ~1 millón de líneas (suficiente para vocab 768)
    shuffle_input_sentence=True,          # Mezcla aleatorio para mejor representatividad
    # Opcional: si quieres aún más rápido y menos RAM
    # seed_sentencepiece_size=1000000,    # Tamaño inicial de muestreo interno (default ok)
)
