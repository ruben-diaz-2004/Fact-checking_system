# Hito 2: sistema RAG con búsqueda híbrida y LLM de pesos abiertos

Fact-checking de mitología griega, romana y nórdica. Detalle en el [README](README.md) (secciones 2 y 11 a 13) y en [`notebooks/hito2.ipynb`](notebooks/hito2.ipynb) (ejecutado entero).

**1. Recuperación densa.** Los mismos 19.792 fragmentos del Hito 1, con el título delante, se representan con `intfloat/multilingual-e5-base` (768 dim., multilingüe, porque las afirmaciones pueden llegar en español) y se guardan en **Chroma** (HNSW, coseno) con los datos para citar ([`src/indexa_vectores.py`](src/indexa_vectores.py)). Se compara con `BAAI/bge-small-en-v1.5` (solo inglés). Misma evaluación que en el Hito 1 (84 queries, de fragmentos a documentos, IC 95 % por *bootstrap*). La búsqueda aproximada da las mismas métricas que la exacta.

**2. Búsqueda híbrida.** e5 y BM25 fallan en queries distintas (9 aciertos en el top 10 solo de e5, sobre todo paráfrasis de los clásicos; 3 solo de BM25, palabras raras como *opuntii*). Se fusionan los fragmentos con *Reciprocal Rank Fusion* (c = 60, sin ajustar).

| Métrica | BM25 | e5 | bge | **Híbrida** | Híbrida − BM25 |
|---|---|---|---|---|---|
| Recall@10 | 0,845 | 0,917 | 0,869 | **0,917** | +0,071 [0,012; 0,131] |
| Recall@100 | 0,976 | 0,964 | 0,952 | **0,976** | 0,000 |
| MRR@10 | 0,645 | 0,651 | 0,624 | **0,729** | +0,085 [0,041; 0,132] |

**Decisión: búsqueda híbrida BM25 + e5** (unos 35 ms por consulta).

**3. Generación.** Para cada afirmación: 5 fragmentos de la búsqueda híbrida (máx. 2 por documento), numerados, al LLM del servidor de la UC3M con temperatura 0 y salida JSON. El prompt pide **copiar primero la frase de las pruebas** sobre el dato de la afirmación y después dar el veredicto (*verdadero*, *falso* o *sin información*), una explicación en el idioma de la afirmación (detectado con `langdetect`) y los fragmentos citados. Una regla en el código convierte en *sin información* todo veredicto sin frase de prueba, y en ese caso el usuario recibe una frase fija, no lo que "sabe" el LLM. La respuesta muestra veredicto, explicación, prueba y fuentes (título, URL, segundo del vídeo y fragmento).

**4. Evaluación.** **250 afirmaciones** con veredicto conocido ([`data/afirmaciones.jsonl`](data/afirmaciones.jsonl)): 102 verdaderas y 102 falsas (un dato cambiado), sacadas del fragmento de origen de cada query; 46 sin información (otras mitologías y detalles inventados); 50 traducidas al español. Las generó `llama3.1:8b` y se revisaron todas a mano: de 208 candidatas, 126 se mantienen, 74 se corrigen y 8 se descartan. Métricas: **calidad** (acierto, F1 por clase, rechazo correcto, cita correcta, prueba literal, idioma), **cobertura** (Recall@5 del documento de origen) y **tiempo**.

**5. Resultados** (IC 95 %):

| Métrica | `llama3.1:8b` | `qwen3:8b` |
|---|---|---|
| Acierto global | 0,672 | **0,824** (+0,152 [0,096; 0,208]) |
| F1 macro | 0,664 | **0,812** |
| Rechazo correcto (sin información) | **0,891** | 0,783 |
| Cita el documento correcto | 0,564 | **0,711** |
| Idioma de la explicación | 1,000 | 1,000 |
| Recall@5 (cobertura) | 0,824 | 0,824 |
| Tiempo por afirmación (media) | **2,7 s** | 11,4 s |

**Decisión: `qwen3:8b`.** llama dice *falso* sin prueba en el 28 % de las respuestas, y su rechazo alto viene de la regla del código; con qwen la regla solo actúa 2 veces. Cuando la búsqueda trae el documento correcto, qwen acierta el 91 %; cuando no, el 47 %. En español, el Recall@5 baja de 0,85 a 0,65 y el acierto de 0,86 a 0,70.

**6. Problemas y decisiones.** El primer prompt hacía que el LLM respondiera *falso* a afirmaciones sin información, aunque su explicación dijera que no aparecían en las fuentes: de ahí pedir la prueba antes del veredicto y la regla. La respuesta en JSON se rompía en pocos casos (1 por modelo), que cuentan como fallo. Los errores de qwen son sobre todo de dos tipos: da por verdaderas falsas cuyo dato cambiado es secundario (14), y responde *sin información* cuando la búsqueda no trae el documento (14, la mayoría en español). Varios errores son del propio conjunto: versiones contradictorias en el corpus (la muerte de Androgeo) y afirmaciones "sin información" que el corpus menciona de pasada. Limitaciones: las afirmaciones parafrasean un solo fragmento (más fáciles que las reales), las generó uno de los modelos evaluados, el prompt se ajustó con 5 de ellas y los IC son anchos.

**7. Siguiente paso.** Mejorar la búsqueda en español (traducir la afirmación al inglés para buscar o buscar con las dos versiones), hacer que el LLM compruebe cada dato de la afirmación, añadir el veredicto *depende de la fuente*, probar un *reranker*, pasar el verificador a un módulo de `src/` y fijar las versiones de `requirements.txt`. Para el Hito 3, modelado de tópicos o un ajuste con LoRA/QLoRA.

**8. Uso de IA generativa.** (a) **En el sistema**: `qwen3:8b` es el LLM del verificador y `llama3.1:8b` se evaluó como alternativa; `llama3.1:8b` generó las afirmaciones candidatas y sus traducciones, todas revisadas a mano. (b) **Como asistente**: Claude (Anthropic) para diseñar el verificador y la evaluación, escribir y depurar el código y el notebook, proponer decisiones en la revisión de las afirmaciones y las traducciones (comprobadas y confirmadas por el equipo), analizar los resultados y redactar la documentación. Todo lo revisó y ejecutó el equipo, que asume su autoría.
