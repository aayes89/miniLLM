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
# Usos

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
