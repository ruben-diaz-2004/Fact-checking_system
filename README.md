# Fact-checking de mitología griega, romana y nórdica

Proyecto final de **Procesamiento del Lenguaje Natural** (Máster en Inteligencia Artificial Aplicada, UC3M).

Sistema de **verificación de hechos** basado en RAG (*Retrieval-Augmented Generation*) sobre mitología griega, romana (las versiones romanas de los mitos griegos, como las de Ovidio) y nórdica. Dada una afirmación ("Zeus se tragó a sus hijos", "Thor es hijo de Loki"), el sistema busca en un corpus descargado e indexado previamente y responde con un veredicto citando sus fuentes:

- **Verdadero** / **Falso**
- **Depende de la fuente**: distintos autores cuentan versiones distintas del mito
- **Información insuficiente**: el corpus no permite responder

Dos rasgos propios del tema:

- **Equivalencias de nombres** entre mitologías: Zeus = Júpiter = Jove, Odiseo = Ulises, Heracles = Hércules…
- **Tipos de fuente**: cada fragmento sabe si viene de una fuente **primaria** (textos clásicos), **enciclopédica** (Wikipedia) o **divulgativa** (artículos y vídeos), y los vídeos de YouTube además si son de nivel **experto** o **popular**. Esto permite citar con precisión y estudiar cómo afecta el tipo de fuente a la cobertura y a los errores.

---

## Índice

1. [Estado y calendario](#1-estado-y-calendario)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Instalación](#3-instalación)
4. [El corpus](#4-el-corpus)
5. [Cómo se construye todo, paso a paso](#5-cómo-se-construye-todo-paso-a-paso)
6. [Formato de los datos](#6-formato-de-los-datos)
7. [Preprocesado](#7-preprocesado)
8. [Queries de evaluación](#8-queries-de-evaluación)
9. [Hito 1: baseline de recuperación](#9-hito-1-baseline-de-recuperación)
10. [Limitaciones conocidas y mejoras pendientes](#10-limitaciones-conocidas-y-mejoras-pendientes)
11. [Licencias de las fuentes](#11-licencias-de-las-fuentes)
12. [Uso de herramientas de IA generativa](#12-uso-de-herramientas-de-ia-generativa)

---

## 1. Estado y calendario

| Fecha | Entrega | Estado |
|---|---|---|
| Lun 16 nov, 23:59 | Grupo, aplicación y repositorio en Aula Global | Pendiente de confirmar |
| Jue 19 nov, 23:59 | **Hito 1**: corpus, preprocesado y baselines TF-IDF y BM25 | Hecho en la rama `santi`; falta `MILESTONE_1.md` y pasarlo a `main` |
| Vie 4 dic, 23:59 | **Hito 2**: sistema RAG con LLM de pesos abiertos y base de datos vectorial | — |
| Vie 18 dic, 23:59 | **Hito 3**: extensión (modelado de tópicos o LoRA/QLoRA) | — |
| Enero | Entrega final (informe y código) y presentación | — |

Los hitos se corrigen sobre la rama **`main`** en el commit indicado en Aula Global. El trabajo se desarrolla en ramas propias y se lleva a `main` con un *pull request*.

---

## 2. Estructura del repositorio

```
Fact-checking_system/
├── README.md
├── requirements.txt
├── .gitignore
├── .env                      # NO se sube: clave de la API del LLM de la UC3M
├── src/                      # todos los scripts del pipeline
│   ├── lista_wikipedia.py
│   ├── descarga_wikipedia.py
│   ├── procesa_wikipedia.py
│   ├── descarga_clasicos.py
│   ├── procesa_clasicos.py
│   ├── descarga_whe.py
│   ├── lista_youtube.py
│   ├── descarga_youtube.py
│   ├── troceo.py
│   ├── construye_corpus.py
│   ├── preprocesado.py
│   ├── genera_queries.py
│   ├── exporta_revision.py
│   └── crea_queries.py
├── data/
│   ├── youtube_seleccion.csv     # los 250 vídeos elegidos tras la revisión manual
│   ├── queries.jsonl             # las 95 queries de evaluación
│   ├── revision_queries_t1.csv   # revisión manual de la tanda 1 de queries
│   ├── revision_queries_t2.csv   # revisión manual de la tanda 2 de queries
│   ├── raw/                      # NO se sube: descargas tal cual
│   └── processed/                # NO se sube: corpus, fragmentos y tokens
├── notebooks/
│   └── hito1.ipynb               # baselines TF-IDF y BM25, ejecutado
└── referencia/                   # notebooks de la asignatura
    ├── Text_Vectorization_I.ipynb      # completado; base del Hito 1
    ├── Text_Vectorization_II.ipynb
    ├── Using_spaCy_preprocessing.ipynb
    └── imdb_lemmas_clean.txt
```

**El texto descargado no está en el repositorio** (`data/raw/` y `data/processed/` están en el `.gitignore`), por derechos de autor y por tamaño. Se regenera con los scripts (sección 5). Sí se suben los ficheros pequeños que son fruto de decisiones manuales y que hacen el trabajo reproducible: la selección de vídeos, las queries y las tablas de revisión.

---

## 3. Instalación

Desarrollado en **Ubuntu (WSL2)** con **Python 3.10** y VS Code.

```bash
git clone https://github.com/ruben-diaz-2004/Fact-checking_system.git
cd Fact-checking_system

python3 -m venv .venv
source .venv/bin/activate          # hay que activarlo en cada terminal nueva

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Librerías principales: `requests`, `beautifulsoup4`, `lxml` y `yt-dlp` (descargas); `spacy` y `contractions` (preprocesado); `gensim` y `bm25s` (recuperación); `numpy`, `pandas`, `matplotlib`, `seaborn` y `tqdm`; `ipykernel` (para usar el entorno en Jupyter/VS Code).

**Clave del LLM de la UC3M.** Solo hace falta para generar queries. Crea un fichero `.env` en la raíz con:

```
UC3M_API_KEY=sk-...
```

La clave es la del enunciado del proyecto (sección 7). El `.env` está en el `.gitignore`: **nunca se sube**, porque el repositorio es público.

**Para usar el entorno en los notebooks**: abrir la carpeta desde WSL (`code .`), instalar las extensiones Python y Jupyter en WSL y elegir el kernel `.venv` con *Select Kernel → Python Environments*.

---

## 4. El corpus

Corpus en **inglés**, de cuatro fuentes. Cada documento es una unidad con sentido propio: un artículo, un libro de la *Ilíada*, una fábula de Ovidio, un poema de la Edda, un capítulo, un vídeo.

| Fuente | Tipo de fuente | Documentos | Palabras | Fragmentos | % fragmentos |
|---|---|---|---|---|---|
| Wikipedia | enciclopédica | 4.923 | 3,49 M | 16.470 | 68,6 % |
| Textos clásicos | primaria (Bulfinch: divulgativa) | 365 | 733 k | 3.383 | 14,1 % |
| YouTube (transcripciones) | divulgativa (experta o popular) | 233 | 535 k | 2.740 | 11,4 % |
| World History Encyclopedia | divulgativa | 206 | 319 k | 1.428 | 5,9 % |
| **Total** | | **5.727** | **≈ 5,1 M** | **24.021** | |

Fragmentos por tipo de fuente: 16.470 enciclopédicos, 4.700 divulgativos y 2.851 primarios. Por mitología (un fragmento puede tener varias): griega 17.264, romana 8.035 y nórdica 4.622.

### 4.1. Wikipedia

- **Selección**: recorrido en anchura del árbol de categorías desde `Category:Greek mythology`, `Category:Roman mythology` y `Category:Norse mythology`, hasta **profundidad 3**, con la API de MediaWiki (`list=categorymembers`).
- Se excluyen las subcategorías de cultura popular, arte, geografía, arqueología, "por región", etc., y los artículos cuyo título indica una obra moderna (cómic, película, cuadro…). Resultado: **6.701 artículos**, todos descargados (5,84 M de palabras).
- **Descarga**: texto plano completo (`prop=extracts`, `explaintext`) con las secciones marcadas como `== Sección ==`, más la URL y el número de revisión (`revid`), para saber exactamente qué versión se usó.
- **Limpieza y filtrado** (`procesa_wikipedia.py`):
  - se quitan las secciones de relleno (*References*, *Notes*, *See also*, *External links*, *Bibliography*…) y las de cultura popular, con sus subsecciones;
  - se descartan las listas y las desambiguaciones que no son de mitología, los artículos cuyo primer párrafo no habla de mitos, dioses, héroes, leyendas o tragedias, los de otras mitologías que no mencionan las nuestras y los de menos de 50 palabras.
  - Resultado: **4.923 artículos**. Lo descartado y el motivo quedan en `data/raw/wikipedia/descartados_procesado.csv`.

### 4.2. Textos clásicos (fuentes primarias)

| Obra | Traducción | Origen | Documento = |
|---|---|---|---|
| Homero, *Ilíada* | Butler (prosa) | Gutenberg 2199 | un libro |
| Homero, *Odisea* | Butler (prosa) | Gutenberg 1727 | un libro |
| Hesíodo e himnos homéricos | Evelyn-White | Gutenberg 348 | una obra o un himno |
| Ovidio, *Metamorfosis* I–VII y VIII–XV | Riley | Gutenberg 21765 y 26073 | una fábula |
| *Edda poética* | Bellows | Gutenberg 73533 | un poema |
| *Edda en prosa* | — | Gutenberg 18947 | un capítulo o relato |
| Apolodoro, *Biblioteca* y *Epítome* | Frazer | Perseus (XML TEI) | un capítulo |
| Bulfinch, *The Age of Fable* (caps. I–XXVI y XXXI–XXXIII) | — | Gutenberg 3327 | un capítulo (**divulgativa**) |

`procesa_clasicos.py` corta cada libro en documentos y quita el material de los traductores: notas al pie, introducciones, epílogos y notas de transcripción. Las lagunas del texto se marcan como `[...]`. De Hesíodo se excluyen los fragmentos menores (*Epigramas*, *Margites*, *Batracomiomaquia*…).

**Ojo:** las traducciones de Butler, Riley y Bulfinch usan los **nombres romanos** (*Jove*, *Juno*, *Minerva*) incluso para los mitos griegos.

### 4.3. World History Encyclopedia (WHE)

- WHE no tiene categorías utilizables, así que se parte del índice del sitio (`sitemap.txt`):
  - **Definiciones** (páginas tipo `/Zeus/`): entran si su nombre coincide con el título de un artículo de nuestra lista de Wikipedia, salvo una lista de excepciones revisada a mano.
  - **Artículos** (`/article/…`): entran si su título contiene palabras de nuestras mitologías (comparando palabras completas) y ninguna de la lista de descarte. Las listas se ajustaron revisando a mano todos los candidatos.
- Se descarga la versión en Markdown de cada página (añadiendo `.md` a la URL), a **una petición por segundo** como pide el sitio.
- Se cortan la revisión editorial, la bibliografía y los bloques de anuncios. Resultado: **206 documentos**.

### 4.4. YouTube (transcripciones)

- **12 canales**, cada uno con su nivel:
  - **Experta**: Jackson Crawford, Religion for Breakfast, Crash Course (*World Mythology*), TED-Ed, World History Encyclopedia, Gresham College, History Hit, Arith Härger.
  - **Popular**: Overly Sarcastic Productions, Mythology & Fiction Explained, Tiny Epics, Extra History (*Extra Mythology*).
- `lista_youtube.py` lista todos los vídeos de cada canal con `yt-dlp`, sin descargarlos, y filtra por título (vocabulario de cada mitología; las palabras ambiguas como *Mars* o *Rome* exigen además una palabra general de mito), por duración (3–60 min) y por un máximo de 30 por canal.
- **Revisión manual** de los candidatos: se quitaron 56 vídeos y se añadieron 30. El resultado son **250 vídeos** (151 expertos y 99 populares), en `data/youtube_seleccion.csv`, que sí se sube.
- `descarga_youtube.py` descarga **solo los subtítulos** en inglés: los manuales si existen y, si no, los automáticos. Se pudieron descargar **233** (137 manuales y 96 automáticos); 140 de nivel experto (369 k palabras) y 93 popular (166 k palabras).
- El texto se divide en bloques de 30 segundos marcados como `== t=125 ==`, para poder citar el minuto exacto.
- **Hay que ejecutarlo en un ordenador personal**: YouTube bloquea las IP de servidores en la nube.

### 4.5. Troceo en fragmentos (`troceo.py`)

El **fragmento** es la unidad que se indexa y se recupera:

- Unas **280 palabras** por fragmento, con un **15 % de solape** entre fragmentos consecutivos.
- Se respetan las frases (nunca se corta una a la mitad) y las secciones: se intenta empezar fragmento nuevo al cambiar de sección.
- Ningún fragmento pasa de 1,4 veces el objetivo; si el último queda muy corto, se une al anterior.
- En YouTube, cada fragmento guarda el segundo en que empieza (`segundo`).

Resultado: 24.021 fragmentos de 230 palabras de media (mediana 251, máximo 392).

---

## 5. Cómo se construye todo, paso a paso

Todos los comandos se ejecutan **desde la carpeta raíz del repositorio y con el entorno activado**. Las descargas se pueden interrumpir con Ctrl+C y retomar: lo ya descargado no se repite.

```bash
# 1. Wikipedia
python src/lista_wikipedia.py --contacto tu_correo@ejemplo.com     # lista de artículos (--profundidad 3 por defecto)
python src/descarga_wikipedia.py --contacto tu_correo@ejemplo.com  # texto de los 6.701 artículos
python src/procesa_wikipedia.py                                    # limpieza y filtros -> 4.923

# 2. Textos clásicos
python src/descarga_clasicos.py
python src/procesa_clasicos.py

# 3. World History Encyclopedia
python src/descarga_whe.py --solo-lista    # opcional: ver solo los candidatos
python src/descarga_whe.py

# 4. YouTube (en un ordenador personal)
python src/descarga_youtube.py             # usa data/youtube_seleccion.csv (~6 s por vídeo)
# lista_youtube.py solo hace falta para rehacer la selección de vídeos desde cero

# 5. Corpus y fragmentos
python src/construye_corpus.py

# 6. Preprocesado con spaCy (~3-4 min con 2 procesos)
python src/preprocesado.py --procesos 2
# prueba rápida con 500 fragmentos al azar: python src/preprocesado.py --muestra 500

# 7. Hito 1
# abrir notebooks/hito1.ipynb y ejecutarlo entero (Restart + Run All, ~4 min)
```

`--contacto` es obligatorio para Wikipedia: va en el *User-Agent* de las peticiones, como pide su política de uso.

**Queries de evaluación** (ya están en el repositorio; solo hace falta para rehacerlas):

```bash
python src/genera_queries.py --tanda 1           # candidatas con el LLM (requiere .env)
python src/exporta_revision.py --tanda 1         # cada query junto a su fragmento, para revisar
# ... revisión manual -> data/revision_queries_t1.csv ...
python src/crea_queries.py                       # junta las tablas de revisión -> data/queries.jsonl
```

### Qué produce cada script

| Script | Entrada | Salida |
|---|---|---|
| `lista_wikipedia.py` | API de Wikipedia | `data/raw/wikipedia/lista_articulos.csv`, `descartados.csv` |
| `descarga_wikipedia.py` | `lista_articulos.csv` | `data/raw/wikipedia/articulos.jsonl` |
| `procesa_wikipedia.py` | `articulos.jsonl` | `data/processed/wikipedia.jsonl`, `data/raw/wikipedia/descartados_procesado.csv` |
| `descarga_clasicos.py` | Gutenberg y Perseus | `data/raw/clasicos/` |
| `procesa_clasicos.py` | `data/raw/clasicos/` | `data/processed/clasicos.jsonl` |
| `descarga_whe.py` | sitemap de WHE | `data/raw/whe/`, `data/processed/whe.jsonl` |
| `lista_youtube.py` | canales (yt-dlp) | `data/raw/youtube/candidatos.csv`, `todos.csv` |
| `descarga_youtube.py` | `data/youtube_seleccion.csv` | `data/raw/youtube/subs/`, `data/processed/youtube.jsonl` |
| `construye_corpus.py` | los 4 `.jsonl` de las fuentes | `data/processed/corpus.jsonl`, `chunks.jsonl` |
| `preprocesado.py` | `chunks.jsonl` | `data/processed/chunks_tokens.jsonl` |
| `genera_queries.py` | `chunks.jsonl` + LLM | `data/processed/queries_candidatas_t{N}.jsonl` |
| `exporta_revision.py` | candidatas + `chunks.jsonl` | `data/processed/revision_queries_t{N}.md` |
| `crea_queries.py` | `data/revision_queries_t*.csv` | `data/queries.jsonl` |

---

## 6. Formato de los datos

**Documento** (`data/processed/corpus.jsonl`, uno por línea):

| Campo | Descripción |
|---|---|
| `doc_id` | Identificador único: `wiki_{pageid}`, `yt_{video_id}`, `whe_{nnn}`, `iliada_06`, `ovidio2_032`… |
| `fuente` | `wikipedia`, `clasicos`, `whe` o `youtube` |
| `titulo`, `obra`, `autor`, `url` | Para mostrar y citar |
| `tipo_fuente` | `primaria`, `enciclopedica` o `divulgativa` |
| `nivel` | En YouTube: `experta` o `popular` |
| `mitologia` | `griega`, `romana` y/o `nordica` |
| `subtitulos` | En YouTube: `manuales` o `automaticos` |
| `revid` | En Wikipedia: revisión exacta del artículo |
| `texto` | Texto limpio, con las secciones marcadas `== Sección ==` |

**Fragmento** (`data/processed/chunks.jsonl`): `chunk_id` (`{doc_id}_cNN`), `doc_id`, `seccion`, `texto`, `n_palabras`, los campos del documento necesarios para filtrar y citar (`fuente`, `titulo`, `autor`, `tipo_fuente`, `nivel`, `mitologia`, `url`) y, en YouTube, `segundo`.

**Tokens** (`data/processed/chunks_tokens.jsonl`): `chunk_id`, `doc_id` y `tokens`, la lista de términos tras el preprocesado.

**Query** (`data/queries.jsonl`), con el formato del enunciado más dos campos:

```json
{"qid": "q002", "query": "What is the origin of the name \"Opuntii\" in ancient Greek mythology?",
 "relevant_ids": ["wiki_24959971"], "source_chunk": "wiki_24959971_c01", "fuente": "wikipedia"}
```

---

## 7. Preprocesado

`src/preprocesado.py` sigue la función `prepare_data` del notebook *Text Vectorization I*:

1. Se quitan las URLs y se expanden las contracciones (*I'll* → *I will*).
2. spaCy `en_core_web_sm` con `nlp.pipe`, sin parser ni NER.
3. Se quedan los **lemas en minúsculas** de los tokens alfabéticos de categoría **NOUN, VERB, ADJ o PROPN** que no son stopwords: las de spaCy y unas pocas del dominio (*video*, *channel*, *subscribe*, *patreon*, *chapter*, *book*, *footnote*, *translation*…).

Dos cambios pensados para los nombres propios:

- Se quitan **tildes y diéresis** (*Jörmungandr* → *jormungandr*), porque cada fuente lo escribe de una manera.
- Al texto de cada fragmento se le pone delante el **título de su documento**, porque muchos fragmentos de un artículo no repiten el nombre del personaje.

Las queries pasan por **la misma función** (`from preprocesado import preparar`), sin el título.

El resto del preprocesado (N-gramas con `Phrases`, diccionario y filtrado por frecuencia documental con `filter_extremes`) se hace en el notebook, porque es rápido y se eligió comparando variantes (sección 9).

---

## 8. Queries de evaluación

**95 queries de tipo *known-item***: cada una describe un fragmento concreto, y el documento de ese fragmento es el único relevante.

**Generación** (`genera_queries.py`):
- LLM de pesos abiertos de la UC3M (**`llama3.1:8b`**, servidor `yiyuan.tsc.uc3m.es`), con *temperature* 0,3.
- Fragmentos al azar (semilla fija), de al menos 150 palabras, con **un número fijo por fuente** para que no salgan casi todos de Wikipedia, y nunca dos del mismo documento ni de documentos de tandas anteriores.
- Se pide la query que escribiría alguien que buscara ese fragmento (8–20 palabras), con sus propias palabras y como mucho un nombre propio.
- Dos tandas:
  - **Tanda 1**: 45 candidatas (15 de Wikipedia, 12 de clásicos, 10 de YouTube y 8 de WHE).
  - **Tanda 2**: 90 candidatas (30, 24, 20 y 16), con el prompt ajustado tras revisar la primera: pedir lo concreto del fragmento y no inventar nada. Además, el LLM indica en JSON si el fragmento es de mitología.

**Revisión manual de las 135 candidatas**, cada una leída junto a su fragmento. Las decisiones y sus motivos están en `data/revision_queries_t1.csv` y `data/revision_queries_t2.csv`:

| | Mantener | Reescribir | Descartar | Total |
|---|---|---|---|---|
| Tanda 1 | 25 | 10 | 10 | 45 |
| Tanda 2 | 49 | 11 | 30 | 90 |
| **Total** | **74** | **21** | **40** | **135** |

Motivos de descarte: documento fuera de tema (14), query ambigua (el mismo mito está en muchos documentos), trivial (copia el texto o el título) o con errores de contenido. Motivos de reescritura: copiaba el texto, describía mal el fragmento, era demasiado genérica o inventaba algo.

**Resultado**: 95 queries (30 de clásicos, 28 de Wikipedia, 24 de YouTube y 13 de WHE), 21 de ellas reescritas.

**Hallazgos de la revisión**:
- **Hay ruido en Wikipedia**: aparecieron artículos que no son de nuestras mitologías (listas de personas, santos, una raza de oca, baladas…).
- **La marca de "no es mitología" del LLM no es fiable**: acertó 4 de los 6 fragmentos que marcó y se le escaparon 8.
- **El prompt de la tanda 2** dio queries más concretas, pero que copian más el texto: el solape medio con el fragmento pasó de 0,46 a 0,64. Esto favorece a los métodos léxicos.

---

## 9. Hito 1: baseline de recuperación

Todo está en `notebooks/hito1.ipynb`.

**Recuperación**:
- Se indexan **fragmentos** y se evalúa por **documento**: para cada query se recuperan los 200 fragmentos más parecidos, cada documento puntúa con **su mejor fragmento** y aparece una sola vez.
- El **IDF se calcula sobre los fragmentos**.
- **TF-IDF con similitud coseno**: Gensim, `TfidfModel` con sus valores por defecto (frecuencia sin transformar, IDF log₂(N/df) y vectores normalizados) y `SparseMatrixSimilarity`.
- **BM25**: librería **`bm25s`**, variante **Lucene**, **k1 = 1,5** y **b = 0,75**.
- Los dos métodos usan el mismo diccionario, así que ven exactamente los mismos términos.

**Elección del preprocesado**, comparando 12 variantes (N-gramas sí/no × `no_below` 1/2/4 × `no_above` 0,5/0,1):
- **Sin N-gramas**: empeoran siempre. Al sustituir dos palabras por el N-grama se pierden coincidencias con la query.
- **Sin filtro por frecuencia documental**: se aplicó y se evaluó, y en búsqueda *known-item* perjudica. Quita justo las palabras raras que mejor identifican un documento: con `no_below=2` desaparece *opuntii*, que solo está en el fragmento relevante de su query. `no_above=0.1` (80 términos como *god*, *greek* o *king*) tampoco ayuda.
- **k1 y b de BM25**: el barrido (k1 ∈ {0,5; 1,2; 1,5; 2}, b ∈ {0; 0,5; 0,75; 1}) solo da diferencias pequeñas, y ajustadas con las mismas queries, así que se mantienen los valores por defecto.

**Resultados** (95 queries):

| Método | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|
| TF-IDF | 0,705 | 0,789 | 0,546 |
| **BM25** | **0,779** | **0,811** | **0,630** |

Por fuente (Recall@10 con BM25): clásicos 0,67, WHE 1,00, Wikipedia 0,93 y YouTube 0,75. Las queries reescritas a mano son más difíciles que las del LLM (con BM25, MRR de 0,50 frente a 0,67).

**Análisis de errores**: los dos métodos fallan en 17 queries, 10 de ellas de los clásicos. Las causas:
1. **Gana otro documento del mismo tema** (la más frecuente). Por ejemplo, el artículo *Freyr* o el de *Lethe*. Muchas veces lo devuelto también serviría, pero la evaluación *known-item* solo cuenta el documento de origen.
2. **La query usa otras palabras que el texto**: "islands of the blessed" frente a *Fortunate Isles*.
3. **Libros de una misma obra muy parecidos**: los libros de la *Ilíada*.
4. **Un término clave que no coincide**: una errata del LLM.

**Configuración final del baseline**: lemas de spaCy con el título delante, sin N-gramas, sin filtro por frecuencia documental, y BM25 (`bm25s`, Lucene, k1 = 1,5, b = 0,75).

---

## 10. Limitaciones conocidas y mejoras pendientes

- **Ruido en Wikipedia.** En la muestra revisada, entre un 20 % y un 30 % de los documentos de Wikipedia no son de nuestras mitologías. Vienen de la profundidad 3 del recorrido de categorías, de artículos tipo "X in popular culture" o "X in fiction" y de secciones de cultura popular con encabezados no previstos. Se puede validar un filtro mejor con los 45 documentos ya etiquetados en la revisión, o clasificarlos con el LLM.
- **Tokens que no coinciden:**
  - letras nórdicas sin normalizar (*Óðinn* → `oðinn` ≠ `odin`);
  - singular y plural (`trojans` frente a `trojan`, `gods` frente a `god`) cuando spaCy etiqueta mal frases cortas o en minúscula;
  - probables errores en los subtítulos automáticos, que no tienen mayúsculas ni puntuación.

  Una forma de probarlo: transliterar las letras nórdicas y aplicar *stemming* (Snowball) como variante.
- **Nombres romanos en los clásicos** (*Jove*, *Venus*): un método léxico no los relaciona con los griegos. Encaja con el Hito 2 (equivalencias de nombres y *embeddings*).
- **Evaluación**: con 95 queries, una diferencia de 0,01 es una query. Faltan intervalos de confianza (*bootstrap*) y comparaciones por parejas. Las variantes se eligieron con las mismas queries con las que se evalúa, así que los resultados pueden ser algo optimistas.
- **Queries**: q086 tiene una errata (*Automatedon* por *Automedon*) que se escapó en la revisión.
- 17 de los 250 vídeos seleccionados no tenían subtítulos en inglés descargables.
- Pendiente para el Hito 2: decidir qué nivel (experto o popular) se asigna a WHE y a Bulfinch para el experimento de cobertura.

---

## 11. Licencias de las fuentes

- **Wikipedia**: CC BY-SA. Se guarda la URL y la revisión de cada artículo.
- **Project Gutenberg**: dominio público en EE. UU.
- **Perseus Digital Library** (Apolodoro): según la licencia del repositorio `canonical-greekLit`.
- **World History Encyclopedia**: CC BY-NC-SA 4.0 (uso no comercial, con atribución).
- **YouTube**: los subtítulos pertenecen a sus autores. Solo se usan para este trabajo académico y **no se redistribuyen**.

Por todo ello, el texto descargado no se sube al repositorio.

---

## 12. Uso de herramientas de IA generativa

- **LLM de la UC3M (`llama3.1:8b`)**: generación de las queries candidatas de evaluación, todas revisadas a mano (sección 8).
- **Claude (Anthropic)**: asistencia en el diseño del pipeline, la escritura de los scripts y del notebook, la propuesta de decisiones en la revisión de queries (confirmadas por el grupo) y la redacción de la documentación. Todo el código se ejecutó y comprobó en local.