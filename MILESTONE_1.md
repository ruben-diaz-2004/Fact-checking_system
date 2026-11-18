# Hito 1: colección, fragmentación y baseline de recuperación

Fact-checking de mitología griega, romana y nórdica. Este fichero resume lo entregado en el Hito 1; el detalle está en el [README](README.md) y en el notebook [`notebooks/hito1.ipynb`](notebooks/hito1.ipynb), que está ejecutado entero y es lo que hay que revisar.

## 1. Colección

Corpus en **inglés** de **5.035 documentos** (≈ 4,26 M palabras), de cuatro fuentes. Cada documento es una unidad con sentido propio: un artículo, un libro de la *Ilíada*, una fábula de Ovidio, un poema de la Edda, un vídeo.

| Fuente | Tipo de fuente | Documentos | Palabras | Fragmentos |
|---|---|---|---|---|
| Wikipedia | enciclopédica | 4.231 | 2,67 M | 12.723 |
| Textos clásicos (Homero, Hesíodo, Ovidio, Eddas, Apolodoro, Bulfinch) | primaria | 365 | 733 k | 3.383 |
| YouTube (transcripciones de 12 canales) | divulgativa (experta o popular) | 233 | 535 k | 2.258 |
| World History Encyclopedia | divulgativa | 206 | 319 k | 1.428 |
| **Total** | | **5.035** | **≈ 4,26 M** | **19.792** |

- **Wikipedia**: árbol de categorías de las tres mitologías hasta profundidad 3 (6.839 artículos) y limpieza de los artículos fuera de tema con un clasificador LLM (`llama3.1:8b` de la UC3M), una regla de rescate y 255 etiquetas manuales. La decisión final acierta en el 96 % de 100 artículos de validación etiquetados a mano. Quedan fuera 692 artículos, listados con su motivo en [`data/wikipedia_excluidos.csv`](data/wikipedia_excluidos.csv). Después se aplican filtros de secciones de relleno, listas, tema del primer párrafo y longitud mínima.
- **Clásicos**: Project Gutenberg y Perseus, sin notas ni introducciones de los traductores.
- **WHE**: definiciones y artículos del índice del sitio que encajan con nuestras mitologías.
- **YouTube**: 250 vídeos seleccionados y revisados a mano ([`data/youtube_seleccion.csv`](data/youtube_seleccion.csv)); se descargaron los subtítulos en inglés de 233.

Los datos descargados no se suben (pesan y tienen derechos de autor); se reconstruyen con los scripts de `src/` siguiendo la sección 5 del README.

## 2. Unidad de indexación

La unidad que se indexa y se recupera es el **fragmento**, y se mantendrá en todo el proyecto. Se trocea porque los documentos son muy desiguales (de decenas a decenas de miles de palabras) y porque la prueba citada tiene que ser corta. Implementación: [`src/troceo.py`](src/troceo.py).

- Unas **280 palabras** por fragmento, con un **15 % de solape** entre fragmentos consecutivos.
- No se cortan frases (solo las de más de 280 palabras, como versos sin puntuación). Al cambiar de sección se empieza fragmento nuevo, sin solape.
- Ningún fragmento pasa de 1,4 veces el objetivo; si el último queda muy corto, se une al anterior.
- En YouTube, los bloques de 30 segundos de los subtítulos no cuentan como secciones: solo sirven para guardar el segundo en que empieza cada fragmento. Como los subtítulos automáticos no tienen puntuación, las frases de más de 42 palabras se parten en trozos de ese tamaño para que el solape funcione.
- Cada fragmento guarda el identificador de su documento (`doc_id`, p. ej. `wiki_78881`) y uno propio (`chunk_id`, p. ej. `wiki_78881_c11`), además de la fuente, el tipo de fuente, el título, la sección y, en YouTube, el segundo en que empieza.

Resultado: 19.792 fragmentos, de 238 palabras de media. Clásicos, WHE y YouTube se acercan al objetivo (264, 252 y 266 palabras de media); en Wikipedia hay muchos fragmentos cortos por los artículos y secciones cortas.

## 3. Consultas de prueba

Fichero: [`data/queries.jsonl`](data/queries.jsonl), **84 queries**, con el formato pedido más la fuente:

```json
{"qid": "q003", "query": "What does Plutarch's account of Pan's death have to do with Jesus Christ and early Christianity?", "relevant_ids": ["wiki_78881"], "source_chunk": "wiki_78881_c11", "fuente": "wikipedia"}
```

- **Generación** ([`src/genera_queries.py`](src/genera_queries.py)): fragmentos al azar (semilla fija, al menos 150 palabras, un número fijo por fuente y nunca dos del mismo documento). Al LLM (`llama3.1:8b`, temperatura 0,3) se le pide la consulta que escribiría un usuario para encontrar esa información, **con sus propias palabras y sin copiar el texto**. Se hicieron dos tandas (45 + 90 candidatas).
- **Revisión manual de las 135 candidatas**, cada una leída junto a su fragmento, comprobando entre otras cosas que no copiara el texto: 64 se mantienen, 20 se reescriben y 51 se descartan (sobre todo por ambiguas o porque su documento quedó fuera al limpiar Wikipedia). Seis de los descartes salen de una segunda revisión al terminar el hito: tres copiaban el texto y tres eran demasiado genéricas. Decisiones y motivos en [`data/revision_queries_t1.csv`](data/revision_queries_t1.csv) y [`data/revision_queries_t2.csv`](data/revision_queries_t2.csv).
- **Reparto final**: 28 de clásicos, 23 de Wikipedia, 21 de YouTube y 12 de WHE.

## 4. Evaluación

Evaluación a nivel de **documento**, como se pide:

1. Se recuperan fragmentos: para pedir *k* documentos se recuperan 20·*k* fragmentos (como mínimo 200).
2. Cada documento recibe la puntuación de **su mejor fragmento**.
3. Se eliminan los documentos repetidos y se toman los *K* primeros.
4. Se calculan **Recall@5, @10, @50, @100 y MRR@10** sobre esa lista, con **intervalos de confianza del 95 % por *bootstrap*** (2.000 remuestreos de las queries) y comparaciones emparejadas entre sistemas.

Las estadísticas de TF-IDF y BM25 (IDF y longitud media) se calculan **sobre los fragmentos**.

**Sistemas**:
- **TF-IDF + coseno**: Gensim (`TfidfModel` por defecto y `SparseMatrixSimilarity`).
- **BM25**: `bm25s`, variante Lucene, k1 = 1,5, b = 0,75.
- **Preprocesado** ([`src/preprocesado.py`](src/preprocesado.py)): lemas de spaCy (`en_core_web_sm`) con el título delante. Se compararon 12 variantes (N-gramas × filtros por frecuencia documental), la normalización de letras nórdicas, el *stemming* y un barrido de k1 y b. Ninguna mejora clara del MRR sobre la configuración simple, así que se queda sin N-gramas, sin filtros y sin normalización.

## 5. Resultados

84 queries; entre corchetes, el intervalo del 95 %.

| Métrica | TF-IDF | BM25 | BM25 − TF-IDF |
|---|---|---|---|
| Recall@5 | 0,774 [0,679; 0,857] | **0,810** [0,726; 0,893] | +0,036 [−0,024; 0,095] |
| Recall@10 | 0,821 [0,738; 0,893] | **0,845** [0,762; 0,917] | +0,024 [−0,024; 0,071] |
| Recall@50 | 0,917 [0,857; 0,976] | **0,952** [0,905; 0,988] | +0,036 [0,000; 0,083] |
| Recall@100 | 0,952 [0,905; 0,988] | **0,976** [0,940; 1,000] | +0,024 [0,000; 0,060] |
| MRR@10 | 0,558 [0,473; 0,644] | **0,645** [0,561; 0,733] | +0,087 [0,027; 0,152] |

- **BM25 es claramente mejor solo en MRR**: los dos encuentran el documento en un número parecido de queries, pero BM25 lo coloca más arriba. Con BM25 el documento relevante está entre los 100 primeros en el 98 % de las queries. Es el baseline con el que se comparará el RAG del Hito 2.
- **Por fuente** (Recall@10 con BM25): clásicos 0,71, YouTube 0,81, Wikipedia 0,96 y WHE 1,00. Los clásicos son la fuente más difícil.
- **Errores** (12 queries que fallan los dos métodos en el top 10, 8 de ellas de los clásicos): sobre todo gana otro documento del mismo tema (p. ej., el artículo de Wikipedia frente al libro de Ovidio del que sale la query); también hay queries con otras palabras que el texto (*islands of the blessed* frente a *Fortunate Isles*).

## 6. Limitaciones

- Con 84 queries los intervalos son anchos (unos ±0,08 en Recall@10), y las variantes se eligieron con las mismas queries con las que se evalúa.
- Las tres queries demasiado genéricas se detectaron porque fallaban con los dos métodos, así que quitarlas hace los resultados algo optimistas.
- La evaluación *known-item* cuenta como fallo otro documento que también responde, y las queries comparten bastantes palabras con su fragmento, lo que favorece a los métodos léxicos.
- No se ha medido la evaluación opcional a nivel de fragmento (si el `source_chunk` aparece entre los *K* primeros).

## 7. Cómo reproducirlo

Instalación en la sección 3 del README (Python 3.10, `pip install -r requirements.txt` y `python -m spacy download en_core_web_sm`). La construcción del corpus, paso a paso, está en la sección 5. Con `data/processed/` generado, se abre `notebooks/hito1.ipynb` y se ejecuta entero (*Restart + Run All*, unos minutos).
