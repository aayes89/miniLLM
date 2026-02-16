# miniLLM

Este script permite entrenar LLM desde cero funcional (~25–30M parámetros)

---
## Funcionalidades

* RoPE real
* RMSNorm correcto
* Instruction masking por <|assistant|>
* AMP
* Resume
* Export HF
* Generación local
---
# Como usar

* Generar el corpus.txt inicial ejecutando <code>python gen_synth.py</code>
* Ejecutar <code>python train_tokenizer.py</code> para generar **sp_es.model** y **sp_es.vocab**. (ajustar script para que coincida con el corpus)
* Ejecutar <code>python minLLM.py</code> y esperar a que termine.
---
# Comandos
* **--corpus** indicar el texto para entrenamiento
* **--sp_model** parámetro obligatorio, se obtiene de **train_tokenizer.py**
* **--tokens** se obtiene automáticamente al ejecutar **minLLM.py**
* **--epochs** fijado a 1 época por defecto, valor variable según necesidades
* **--batch** fijado a 24 pero ajustable según capacidad de tu equipo
* **--accum_steps** fijado a 2 pero ajustable según necesidades
* **--stride** fijado a 256 pero ajustable según capacidad del equipo
* **--resume** ruta del checkpoint **.pt** para continuar entrenamiento
* **--out** directorio donde almacena cada checkpoint, fijado a **ckpt**
* **--lr** tasa de aprendizaje, valor tipo float, fijado a 1.5e-4, cambiar a 3e-4 o según necesidades

---
# Posibles usos

* Chat técnico privado
* Asistente para código
* Dataset propio especializado
* Modelo para hardware limitado (6–8GB VRAM)

### Prototipar datasets

* Validar formato de conversaciones
* Ajustar máscara por roles
* Medir si el modelo realmente aprende patrón diálogo
* Probar estrategias de stride/contexto

Resumen:  Es un laboratorio.

### Modelo embebible

* Permite correr en CPU moderno
* Permite correr en GPU pequeña
* Integración en sistemas edge
---
# Ayuda

Se necesita de un generador de texto sintético más eficiente para el entrenamiento de los modelos, cualquier sugerencia en función de mejora y progreso es bienvenida.
