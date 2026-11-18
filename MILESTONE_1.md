# Hito 1: colección, fragmentación y baseline de recuperación

Fact-checking de mitología griega, romana y nórdica. Detalle en el [README](README.md) y en [`notebooks/hito1.ipynb`](notebooks/hito1.ipynb) (ejecutado entero).

**1. Colección.** Corpus en inglés de **5.035 documentos** (≈ 4,26 M palabras, 19.792 fragmentos): Wikipedia (4.231, enciclopédica), textos clásicos de Gutenberg y Perseus (365, primaria), World History Encyclopedia (206, divulgativa) y transcripciones de 233 vídeos de YouTube revisados a mano (divulgativa). Wikipedia se obtuvo del árbol de categorías (6.839 artículos) y se limpió con un clasificador LLM, una regla de rescate y etiquetas manuales (96 % de acierto en 100 artículos de validación; 692 excluidos, en [`data/wikipedia_excluidos.csv`](data/wikipedia_excluidos.csv)). Los datos no se suben; se reconstruyen con `src/` (README, sección 5).

**2. Unidad de indexación: el fragmento** ([`src/troceo.py`](src/troceo.py)). ≈ 280 palabras con 15 % de solape, sin cortar frases y empezando fragmento nuevo en cada sección; máximo 1,4× el objetivo. En YouTube las frases sin puntuación se parten cada 42 palabras y se guarda el segundo de inicio. Cada fragmento lleva `doc_id`, `chunk_id`, fuente, tipo, título y sección.

**3. Consultas.** [`data/queries.jsonl`](data/queries.jsonl), **84 queries** (28 clásicos, 23 Wikipedia, 21 YouTube, 12 WHE). Un LLM propuso 135 candidatas a partir de fragmentos al azar, pidiéndole que no copiara el texto; todas se revisaron a mano: 64 se mantienen, 20 se reescriben y 51 se descartan (motivos en `data/revision_queries_t*.csv`).

**4. Evaluación.** A nivel de documento: se recuperan 20·*k* fragmentos (mín. 200), cada documento puntúa por su mejor fragmento y se toman los *K* primeros. IC del 95 % por *bootstrap* (2.000 remuestreos). Sistemas: TF-IDF + coseno (Gensim) y BM25 (`bm25s`, k1 = 1,5, b = 0,75), sobre lemas de spaCy con el título delante. N-gramas, filtros, normalización nórdica, *stemming* y otros k1/b no mejoraron el MRR.

**5. Resultados** (entre corchetes, IC 95 %):

| Métrica | TF-IDF | BM25 | BM25 − TF-IDF |
|---|---|---|---|
| Recall@5 | 0,774 | **0,810** | +0,036 [−0,024; 0,095] |
| Recall@10 | 0,821 | **0,845** | +0,024 [−0,024; 0,071] |
| Recall@50 | 0,917 | **0,952** | +0,036 [0,000; 0,083] |
| Recall@100 | 0,952 | **0,976** | +0,024 [0,000; 0,060] |
| MRR@10 | 0,558 | **0,645** | +0,087 [0,027; 0,152] |

BM25 solo es claramente mejor en MRR (coloca más arriba el documento) y será el baseline del Hito 2. Por fuente (Recall@10, BM25): clásicos 0,71, YouTube 0,81, Wikipedia 0,96, WHE 1,00. Los fallos suelen deberse a otro documento del mismo tema o a vocabulario distinto (*islands of the blessed* / *Fortunate Isles*).

*Comprobación con 87 queries.* Tres queries se descartaron por genéricas al ver que fallaban con los dos métodos (q024, q030, q117). Al reincorporarlas, BM25 sigue por encima en todas las métricas y su ventaja en MRR se mantiene (+0,084 [0,023; 0,145]); Recall@50 pasa también a ser diferencia clara (+0,046 [0,011; 0,092]), por una sola query (q024). El Recall@10 de BM25 baja 0,029 (a 0,816) y su MRR 0,022 (a 0,623). La elección de BM25 no cambia.

**Limitaciones**: con 84 queries los IC son anchos (±0,08 en Recall@10); las variantes se eligieron con las mismas queries; quitar las tres queries genéricas hace los resultados algo optimistas (ver arriba); la evaluación *known-item* penaliza otros documentos que también responden, y el solape léxico query–fragmento favorece a los métodos léxicos.

**6. Uso de IA generativa.** (a) **En el sistema**: `llama3.1:8b` (servidor de la UC3M) para clasificar artículos de Wikipedia fuera de tema y para proponer las queries candidatas (temperatura 0,3); en ambos casos la salida se validó a mano (etiquetas manuales y revisión de las 135 queries). (b) **Como asistente**: Claude Code (Anthropic) para ayudar a escribir y depurar scripts y el notebook, y a redactar la documentación. Todo el código y los resultados se revisaron y ejecutaron por el equipo, que asume su autoría y corrección.

**Reproducción**: Python 3.10, `pip install -r requirements.txt`, `python -m spacy download en_core_web_sm`; construir el corpus (README, sección 5) y ejecutar el notebook entero.
