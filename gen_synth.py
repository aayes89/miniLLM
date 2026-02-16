#!/usr/bin/env python3
import random

OUTPUT = "chat_corpus_es.txt"
TARGET_MB = 400          # Recomendado: 300–600 MB para tu modelo
ANTI_EOS_RATIO = 0.25    # 25% de conversaciones sin cierre definitivo
MULTI_TURN_PROB = 0.60   # Probabilidad de conversaciones con 3+ turnos

# Temas ampliados (copia y pega más aquí para llegar a 50–100)
temas = [
    "una variable en programación",
    "listas y tuplas en Python",
    "funciones y parámetros",
    "funciones lambda",
    "clases y objetos",
    "herencia en programación orientada a objetos",
    "manejo de excepciones",
    "módulos y paquetes en Python",
    "programación asincrónica con asyncio",
    "bases de datos relacionales",
    "SQLite para principiantes",
    "SQL vs NoSQL",
    "creación de APIs REST con FastAPI",
    "Docker y contenedores",
    "Git y control de versiones",
    "GitHub para colaboración",
    "inteligencia artificial conceptos básicos",
    "qué es una red neuronal",
    "entrenamiento de modelos de machine learning",
    "PyTorch para principiantes",
    "fine-tuning de modelos grandes",
    "qué es un modelo de lenguaje grande (LLM)",
    "prompt engineering técnicas",
    "herramientas locales como Ollama o LM Studio",
    "formato GGUF y cuantización",
    "Q4_K_M vs f16 en modelos",
    "pandas para análisis de datos",
    "numpy arrays y operaciones",
    "visualización con matplotlib",
    "web scraping con BeautifulSoup",
    "ética en inteligencia artificial",
    "sesgos en modelos de IA",
    "privacidad y datos en IA",
    "juegos simples con pygame",
    "ciberseguridad conceptos básicos",
    "autenticación y JWT",
    "despliegue de aplicaciones web",
    "cloud computing AWS o Google Cloud",
    "virtualización vs contenedores",
    # Añade aquí fácilmente más temas:
    # "algoritmos de búsqueda", "recursión", "árboles binarios", "grafos",
    # "machine learning supervisado", "clasificación vs regresión", etc.
]

# Preguntas de usuario neutras y naturales
preguntas_usuario = [
    "¿Qué es {tema} y para qué sirve?",
    "Explícame {tema} de forma sencilla.",
    "Dame un ejemplo práctico de {tema}.",
    "¿Cuál es la diferencia entre {tema} y {tema2}?",
    "¿Cómo se usa {tema} en proyectos reales?",
    "Enséñame {tema} paso a paso para principiantes.",
    "¿Qué recursos recomiendas para aprender {tema}?",
    "¿Sigue siendo relevante {tema} en 2026?",
    "¿Puedes explicarme {tema} con un ejemplo de código?",
]

# Preguntas de seguimiento (para multi-turno)
seguimientos = [
    "Sí, explícame más sobre esa parte.",
    "¿Cómo lo implementaría en código?",
    "Dame otro ejemplo, por favor.",
    "¿Y si quiero hacerlo de forma más avanzada?",
    "¿Funciona igual en versiones recientes?",
    "Gracias, ¿qué más debo saber al respecto?",
]

def generar_conversacion():
    tema = random.choice(temas)
    tema2 = random.choice(temas) if "{tema2}" in random.choice(preguntas_usuario) else tema
    
    conv = []
    num_turnos = random.choices([2, 3, 4, 5, 6], weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]
    
    # Primer mensaje del usuario
    preg = random.choice(preguntas_usuario).format(tema=tema, tema2=tema2)
    conv.append(f"<|user|> {preg}")
    
    for i in range(num_turnos):
        # Respuesta del assistant (placeholder – idealmente generar con LLM grande)
        base_resp = f"{tema.capitalize()} es un concepto fundamental en programación. "
        if i < num_turnos - 1 or random.random() >= ANTI_EOS_RATIO:
            base_resp += "Por ejemplo, en la práctica se utiliza de esta forma: ... ¿Te gustaría ver más detalles o un código de muestra?"
        else:
            base_resp += "¿Quieres que profundicemos en algún aspecto concreto?"
        
        conv.append(f"<|assistant|> {base_resp}")
        
        # Siguiente turno del usuario (si aplica)
        if i < num_turnos - 1:
            conv.append(f"<|user|> {random.choice(seguimientos)}")
    
    return "\n".join(conv) + "\n"

# Generación principal
with open(OUTPUT, "w", encoding="utf-8") as f:
    size = 0
    target_bytes = TARGET_MB * 1024 * 1024
    from tqdm import tqdm
    pbar = tqdm(total=target_bytes, unit="B", unit_scale=True, desc="Generando corpus")
    
    while size < target_bytes:
        texto = generar_conversacion() + "\n"
        f.write(texto)
        added = len(texto.encode("utf-8"))
        size += added
        pbar.update(added)
    
    pbar.close()

print(f"¡Listo! Corpus generado: {OUTPUT} (~{size / 1024**2:.1f} MB)")
