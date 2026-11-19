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
| Lun 16 nov, 23:59 | Grupo, aplicación y repositorio en Aula Global | Hecho |
| Jue 19 nov, 23:59 | **Hito 1**: corpus, preprocesado y baselines TF-IDF y BM25 | Entregado: rama `main`, resumen en `MILESTONE_1.md` (fijar las versiones de `requirements.txt` queda para el Hito 2) |
| Vie 4 dic, 23:59 | **Hito 2**: sistema RAG con LLM de pesos abiertos y base de datos vectorial | — |
| Vie 18 dic, 23:59 | **Hito 3**: extensión (modelado de tópicos o LoRA/QLoRA) | — |
| Enero | Entrega final (informe y código) y presentación | — |

Los hitos se corrigen sobre la rama **`main`** en el commit indicado en Aula Global. El trabajo se hace en la rama **`desarrollo`** y se lleva a `main` con un *pull request* en cada entrega.

---

## 2. Estructura del repositorio

```
Fact-checking_system/
├── README.md
├── MILESTONE_1.md            # resumen de la entrega del Hito 1
├── requirements.txt
├── .gitignore
├── .env                      # NO se sube: clave de la API del LLM de la UC3M
├── src/                      # todos los scripts del pipeline
│   ├── lista_wikipedia.py            # Wikipedia: lista de artículos por categorías
│   ├── descarga_wikipedia.py         #   descarga de los textos
│   ├── procesa_wikipedia.py          #   limpieza y filtros
│   ├── exporta_validacion_wikipedia.py  # limpieza de Wikipedia: conjunto de validación
│   ├── clasifica_wikipedia.py        #   clasificador con el LLM
│   ├── revisa_clasificacion.py       #   revisión de lo que quita el clasificador
│   ├── rescata_wikipedia.py          #   rescate con la regla y decisión final
│   ├── descarga_clasicos.py          # clásicos: descarga
│   ├── procesa_clasicos.py           #   división en documentos y limpieza
│   ├── descarga_whe.py               # World History Encyclopedia
│   ├── lista_youtube.py              # YouTube: vídeos candidatos
│   ├── descarga_youtube.py           #   subtítulos
│   ├── troceo.py                     # troceo en fragmentos
│   ├── actualiza_source_chunk.py     # reasigna el fragmento de origen de las queries si cambia el troceo
│   ├── construye_corpus.py           # junta las fuentes y trocea
│   ├── preprocesado.py               # preprocesado con spaCy
│   ├── diagnostico.py                # mediciones del corpus y del preprocesado (no cambia nada)
│   ├── genera_queries.py             # queries candidatas con el LLM
│   ├── exporta_revision.py           #   fichero para revisarlas
│   └── crea_queries.py               #   conjunto final de queries
├── data/
│   ├── youtube_seleccion.csv     # los 250 vídeos elegidos tras la revisión manual
│   ├── etiquetas_wikipedia.csv   # 255 artículos de Wikipedia etiquetados a mano (dentro/fuera)
│   ├── wikipedia_excluidos.csv   # los 692 artículos que quedan fuera, con el motivo
│   ├── queries.jsonl             # las 84 queries de evaluación
│   ├── revision_queries_t1.csv   # revisión manual de la tanda 1 de queries
│   ├── revision_queries_t2.csv   # revisión manual de la tanda 2 de queries
│   ├── raw/                      # NO se sube: descargas tal cual
│   └── processed/                # NO se sube: corpus, fragmentos y tokens
├── notebooks/
│   └── hito1.ipynb               # Hito 1: corpus, queries y baselines TF-IDF y BM25, ejecutado
└── referencia/                   # notebooks de la asignatura
    ├── Text_Vectorization_I.ipynb      # completado; base del Hito 1
    ├── Text_Vectorization_II.ipynb
    ├── Using_spaCy_preprocessing.ipynb
    └── imdb_lemmas_clean.txt
```

**El texto descargado no está en el repositorio** (`data/raw/` y `data/processed/` están en el `.gitignore`), por derechos de autor y por tamaño. Se regenera con los scripts (sección 5). Sí se suben los ficheros pequeños que son fruto de decisiones manuales y que hacen el trabajo reproducible: la selección de vídeos, las etiquetas y la lista de excluidos de Wikipedia, las queries y las tablas de revisión.

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

Librerías principales: `requests`, `beautifulsoup4`, `lxml` y `yt-dlp` (descargas); `spacy`, `contractions` y `nltk` (preprocesado; `nltk` solo para el *stemming* que se prueba en el notebook); `gensim` y `bm25s` (recuperación); `numpy`, `pandas`, `matplotlib`, `seaborn` y `tqdm`; `ipykernel` (para usar el entorno en Jupyter/VS Code).

**Clave del LLM de la UC3M.** Solo hace falta para generar queries y para clasificar los artículos de Wikipedia. Crea un fichero `.env` en la raíz con:

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
| Wikipedia | enciclopédica | 4.231 | 2,67 M | 12.723 | 64,3 % |
| Textos clásicos | primaria (Bulfinch: divulgativa) | 365 | 733 k | 3.383 | 17,1 % |
| YouTube (transcripciones) | divulgativa (experta o popular) | 233 | 535 k | 2.258 | 11,4 % |
| World History Encyclopedia | divulgativa | 206 | 319 k | 1.428 | 7,2 % |
| **Total** | | **5.035** | **≈ 4,26 M** | **19.792** | |

Fragmentos por tipo de fuente: 12.723 enciclopédicos, 4.218 divulgativos y 2.851 primarios. Las mitologías no están repartidas por igual: la griega domina en las fuentes escritas, la nórdica tiene poca presencia en los clásicos (solo las dos *Eddas*) y en Wikipedia, y en YouTube es la mayoritaria. El detalle, con gráficas, está en la sección 1 del notebook.

### 4.1. Wikipedia

- **Selección**: recorrido en anchura del árbol de categorías desde `Category:Greek mythology`, `Category:Roman mythology` y `Category:Norse mythology`, hasta **profundidad 3**, con la API de MediaWiki (`list=categorymembers`).
- Se excluyen las subcategorías de cultura popular, arte, geografía, arqueología, "por región", etc., y los artículos cuyo título indica una obra moderna (cómic, película, cuadro…). De 6.839 artículos encontrados quedan **6.701**, todos descargados.
- **Descarga**: texto plano completo (`prop=extracts`, `explaintext`) con las secciones marcadas como `== Sección ==`, más la URL y el número de revisión (`revid`), para saber exactamente qué versión se usó.
- **Limpieza de los artículos fuera de tema.** Por las categorías entra mucho ruido: películas, óperas, cuadros, asteroides, barcos y personas históricas que solo llevan un nombre mitológico. El criterio que se fijó es estricto: **dentro quedan los mitos** (personajes, historias, lugares y culto) **y los textos antiguos**; fuera, las obras modernas, lo que solo lleva un nombre mitológico, las personas históricas, el folklore posterior y otras mitologías. Se aplica en tres pasos:
  1. **Clasificador con un LLM** (`clasifica_wikipedia.py`, `llama3.1:8b` de la UC3M), que lee el título y la introducción de cada artículo.
  2. **Rescate con una regla** (`rescata_wikipedia.py`): vuelven a entrar los artículos que el LLM quitó si su primera frase dice *Greek / Roman / Norse… mythology / religion / legend*. Se probó también un rescate con una segunda pregunta al LLM, pero devolvía más de la mitad del ruido y **se descartó** (resultado negativo, documentado en el script y en el notebook).
  3. **Revisión manual**: 255 artículos etiquetados a mano (`data/etiquetas_wikipedia.csv`), que mandan sobre lo anterior. Se etiquetaron en cuatro conjuntos: *validación* (100 al azar, para medir), *largos* y *azar* (artículos que quitaba el clasificador) y *regla* (los que rescata la regla).

  Sobre los 100 artículos de validación, la decisión final acierta en el **96 %**: quita 22 de los 24 artículos de ruido y solo pierde 2 de los 76 buenos. Quedan fuera **692 artículos** (579 por el clasificador y 113 por la revisión manual), listados con su motivo en `data/wikipedia_excluidos.csv`.
- **Filtros finales** (`procesa_wikipedia.py`, que lee `wikipedia_excluidos.csv`):
  - se quitan las secciones de relleno (*References*, *Notes*, *See also*, *External links*, *Bibliography*…) y las de cultura popular (*In popular culture*, *Film*, *Comics*, *Television*…), con sus subsecciones;
  - se descartan los artículos excluidos, los de título de cultura popular, las listas y las desambiguaciones que no son de mitología, los artículos cuyo primer párrafo no habla de mitos, dioses, héroes, leyendas o tragedias, los de otras mitologías que no mencionan las nuestras y los de menos de 50 palabras.

  | Paso | Se quitan | Quedan |
  |---|---|---|
  | Encontrados en las categorías | | 6.839 |
  | Título de obra moderna (al listar) | 138 | 6.701 |
  | Fuera de tema (LLM + regla + revisión) | 692 | 6.009 |
  | Título de cultura popular | 1 | 6.008 |
  | Listas y desambiguaciones | 49 | 5.959 |
  | El primer párrafo no habla de mitos | 1.345 | 4.614 |
  | Otra mitología | 44 | 4.570 |
  | Menos de 50 palabras | 339 | **4.231** |

  Lo descartado por `procesa_wikipedia.py` y el motivo quedan en `data/raw/wikipedia/descartados_procesado.csv`.

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
- Dos canales de mitología nórdica, Jackson Crawford y Arith Härger, aportan casi la mitad de los fragmentos de YouTube: el tope de 30 vídeos por canal limita el número de vídeos, no su duración.
- El texto se divide en bloques de 30 segundos marcados como `== t=125 ==`, para poder citar el minuto exacto.
- **Hay que ejecutarlo en un ordenador personal**: YouTube bloquea las IP de servidores en la nube.

### 4.5. Troceo en fragmentos (`troceo.py`)

El **fragmento** es la unidad que se indexa y se recupera. Se trocea porque los documentos son muy desiguales (de unas decenas de palabras a decenas de miles), porque la prueba que se cite tiene que ser corta y porque los modelos de *embeddings* del Hito 2 tienen un límite de longitud.

- Unas **280 palabras** por fragmento, con un **15 % de solape** entre fragmentos consecutivos.
- Se respetan las frases (solo se parten las de más de 280 palabras, como versos sin puntuación) y las secciones: se intenta empezar fragmento nuevo al cambiar de sección, y al cambiar de sección no hay solape.
- Ningún fragmento pasa de 1,4 veces el objetivo; si el último queda muy corto, se une al anterior.
- En YouTube, las marcas de tiempo de los bloques de 30 segundos **no son secciones**: no cortan el fragmento ni quitan el solape, y solo sirven para que cada fragmento guarde el segundo en que empieza (`segundo`). Como los subtítulos automáticos no tienen puntuación, en los vídeos las frases de más de 42 palabras se parten en trozos de ese tamaño para que el solape funcione.

Resultado: 19.792 fragmentos, de unas 238 palabras de media. Clásicos, WHE y YouTube se acercan al objetivo (264, 252 y 266 palabras de media), y solapan el 95 %, el 66 % y el 100 % de sus pares de fragmentos seguidos (en WHE menos, porque se empieza fragmento nuevo en cada sección); en Wikipedia hay muchos fragmentos cortos (artículos y secciones cortas). En una primera versión, los bloques de 30 segundos de YouTube contaban como secciones y los fragmentos de vídeo salían de 140 a 220 palabras y casi sin solape (2.740 fragmentos); se corrigió antes de entregar el Hito 1 para fijar la unidad de indexación, sin cambiar ningún fragmento de las otras fuentes. Si se vuelve a cambiar el troceo, `actualiza_source_chunk.py` reasigna el `source_chunk` de las queries (sección 5).

---

## 5. Cómo se construye todo, paso a paso

Todos los comandos se ejecutan **desde la carpeta raíz del repositorio y con el entorno activado**. Las descargas se pueden interrumpir con Ctrl+C y retomar: lo ya descargado no se repite.

```bash
# 1. Wikipedia
python src/lista_wikipedia.py --contacto tu_correo@ejemplo.com     # lista de artículos (--profundidad 3 por defecto)
python src/descarga_wikipedia.py --contacto tu_correo@ejemplo.com  # texto de los 6.701 artículos
python src/procesa_wikipedia.py                                    # limpieza y filtros -> 4.231
# (usa data/wikipedia_excluidos.csv, que ya está en el repositorio)

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
# (solo si se cambia el troceo y ya había queries: antes de construir, copiar el chunks.jsonl
#  anterior a data/processed/chunks_antes.jsonl, y después ejecutar
#  python src/actualiza_source_chunk.py y python src/crea_queries.py)

# 6. Preprocesado con spaCy (~3-4 min con 2 procesos)
python src/preprocesado.py --procesos 2
# prueba rápida con 500 fragmentos al azar: python src/preprocesado.py --muestra 500

# 7. Hito 1
# abrir notebooks/hito1.ipynb y ejecutarlo entero (Restart + Run All, unos minutos)
```

`--contacto` es obligatorio para Wikipedia: va en el *User-Agent* de las peticiones, como pide su política de uso.

**Limpieza de Wikipedia** (ya hecha: su resultado es `data/wikipedia_excluidos.csv`; solo hace falta para rehacerla, y requiere el `.env`). Se parte de un `wikipedia.jsonl` procesado **sin** la lista de excluidos:

```bash
python src/exporta_validacion_wikipedia.py       # 100 artículos para etiquetar a mano -> data/etiquetas_wikipedia.csv
python src/clasifica_wikipedia.py --validacion   # mide el clasificador con esos 100
python src/clasifica_wikipedia.py --hilos 4      # clasifica todos los artículos (unas 4.900 llamadas)
python src/revisa_clasificacion.py               # qué quita, para revisarlo a mano
python src/rescata_wikipedia.py                  # regla + etiquetas manuales -> data/wikipedia_excluidos.csv
python src/procesa_wikipedia.py                  # vuelve a filtrar, ya con los excluidos
```

`rescata_wikipedia.py` escribe además `data/processed/clasificacion_final_wikipedia.csv`, que el notebook usa para medir el clasificador (sección 1.5).

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
| `procesa_wikipedia.py` | `articulos.jsonl`, `data/wikipedia_excluidos.csv` | `data/processed/wikipedia.jsonl`, `data/raw/wikipedia/descartados_procesado.csv` |
| `exporta_validacion_wikipedia.py` | `corpus.jsonl`, `lista_articulos.csv` | `data/processed/validacion_wikipedia.md` |
| `clasifica_wikipedia.py` | `wikipedia.jsonl` + LLM | `data/processed/clasificacion_wikipedia.csv` |
| `revisa_clasificacion.py` | clasificación + corpus + queries | `data/processed/revision_clasificacion.md` |
| `rescata_wikipedia.py` | clasificación, `articulos.jsonl`, `etiquetas_wikipedia.csv` | `data/wikipedia_excluidos.csv`, `data/processed/clasificacion_final_wikipedia.csv`, `revision_regla.md` |
| `descarga_clasicos.py` | Gutenberg y Perseus | `data/raw/clasicos/` |
| `procesa_clasicos.py` | `data/raw/clasicos/` | `data/processed/clasicos.jsonl` |
| `descarga_whe.py` | sitemap de WHE | `data/raw/whe/`, `data/processed/whe.jsonl` |
| `lista_youtube.py` | canales (yt-dlp) | `data/raw/youtube/candidatos.csv`, `todos.csv` |
| `descarga_youtube.py` | `data/youtube_seleccion.csv` | `data/raw/youtube/subs/`, `data/processed/youtube.jsonl` |
| `construye_corpus.py` | los 4 `.jsonl` de las fuentes | `data/processed/corpus.jsonl`, `chunks.jsonl` |
| `preprocesado.py` | `chunks.jsonl` | `data/processed/chunks_tokens.jsonl` |
| `diagnostico.py` | corpus, fragmentos y tokens | resumen por pantalla (no cambia nada) |
| `genera_queries.py` | `chunks.jsonl` + LLM | `data/processed/queries_candidatas_t{N}.jsonl` |
| `exporta_revision.py` | candidatas + `chunks.jsonl` | `data/processed/revision_queries_t{N}.md` |
| `crea_queries.py` | `data/revision_queries_t*.csv` | `data/queries.jsonl` |
| `actualiza_source_chunk.py` | `chunks_antes.jsonl`, `chunks.jsonl`, `data/revision_queries_t*.csv` | `data/revision_queries_t*.csv` (columna `source_chunk`) |

**Qué necesita el notebook**: `data/processed/` (`corpus.jsonl`, `chunks.jsonl`, `chunks_tokens.jsonl`, `clasificacion_final_wikipedia.csv`), los tres CSV de `data/raw/wikipedia/` y los ficheros de `data/` que están en el repositorio.

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

**Query** (`data/queries.jsonl`), con el formato del enunciado más un campo (`fuente`):

```json
{"qid": "q002", "query": "What is the origin of the name \"Opuntii\" in ancient Greek mythology?",
 "relevant_ids": ["wiki_24959971"], "source_chunk": "wiki_24959971_c01", "fuente": "wikipedia"}
```

**Etiquetas de Wikipedia** (`data/etiquetas_wikipedia.csv`): una fila por artículo etiquetado a mano, con `doc_id`, `titulo`, profundidad y rama de la categoría, `etiqueta` (`dentro` o `fuera`), si es dudoso, `motivo` y `conjunto` (`validacion`, `largos`, `azar` o `regla`).

**Excluidos de Wikipedia** (`data/wikipedia_excluidos.csv`): `doc_id`, `titulo`, `decidido_por` (`clasificador (LLM)` o `revisión manual`), `motivo` y `palabras`.

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

El resto del preprocesado (N-gramas con `Phrases`, diccionario, filtrado por frecuencia documental con `filter_extremes` y normalización de las palabras) se hace en el notebook, porque es rápido y se eligió comparando variantes (sección 9).

---

## 8. Queries de evaluación

**Evaluación *known-item***: cada query describe un fragmento concreto, y el documento de ese fragmento es el único relevante. Es barata y automática, pero tiene tres limitaciones: si el sistema devuelve otro documento que también responde, cuenta como fallo (los resultados se quedan cortos); la query sale del texto y comparte sus palabras (favorece a los métodos léxicos); y las queries describen pasajes, no son afirmaciones como las que verificará el sistema.

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
| Tanda 1 | 21 | 9 | 15 | 45 |
| Tanda 2 | 43 | 11 | 36 | 90 |
| **Total** | **64** | **20** | **51** | **135** |

Motivos de descarte: query ambigua, porque el mismo episodio se cuenta en muchos documentos (22); documento fuera de tema o excluido al limpiar Wikipedia (19); copia el texto (3), demasiado genérica (3), error de contenido (2) y trivial (2). Motivos de reescritura: ambigua o mal enfocada (7), describía mal el fragmento o inventaba algo (7), copiaba el texto (3) y demasiado genérica (3). Cinco de los descartes son posteriores: su documento quedó fuera al limpiar Wikipedia. Otros seis salen de una segunda revisión al terminar el hito: q091, q109 y q131 copiaban el texto (solape 1,0) y q024, q030 y q117 eran demasiado genéricas. Las tres genéricas se detectaron porque fallaban con los dos métodos, así que quitarlas hace los resultados algo optimistas.

**Resultado**: 84 queries (28 de clásicos, 23 de Wikipedia, 21 de YouTube y 12 de WHE), 20 de ellas reescritas.

**Hallazgos de la revisión**:
- **Había ruido en Wikipedia**: aparecieron artículos que no son de nuestras mitologías (listas de personas, santos, una raza de oca, baladas…). Fue lo que llevó a la limpieza de la sección 4.1.
- **La marca de "no es mitología" del LLM no es fiable**: acertó 4 de los 6 fragmentos que marcó y se le escaparon 8.
- **El prompt de la tanda 2** dio queries más concretas, pero que copian más el texto: el solape medio con el fragmento pasó de 0,46 a 0,64. Esto favorece a los métodos léxicos.
- **Reescribir apenas bajó el solape** (0,52 antes y 0,51 después): la mayoría de las ediciones corregían errores o ambigüedades, no copias. Las reescritas son más largas (casi todas de 13 a 17 palabras, frente a una mediana de 10).

---

## 9. Hito 1: baseline de recuperación

Todo está en `notebooks/hito1.ipynb`, que además explica en detalle cómo se construyeron el corpus (sección 1) y las queries (sección 2).

**Recuperación**:
- Se indexan **fragmentos** y se evalúa por **documento**: para pedir *k* documentos se recuperan 20·*k* fragmentos (como mínimo 200), cada documento puntúa con **su mejor fragmento** y aparece una sola vez.
- El **IDF se calcula sobre los fragmentos**.
- **TF-IDF con similitud coseno**: Gensim, `TfidfModel` con sus valores por defecto (frecuencia sin transformar, IDF log₂(N/df) y vectores normalizados) y `SparseMatrixSimilarity`.
- **BM25**: librería **`bm25s`**, variante **Lucene**, **k1 = 1,5** y **b = 0,75**.
- Los dos métodos usan el mismo diccionario, así que ven exactamente los mismos términos.

**Métricas**: Recall@5, @10, @50 y @100 y MRR@10, con **intervalos de confianza del 95 % por *bootstrap*** (2.000 remuestreos de las queries) y **comparaciones emparejadas** entre sistemas. Una diferencia se considera clara si el intervalo de la diferencia no incluye el 0.

**Elección del preprocesado**:
- **12 variantes** (N-gramas sí/no × `no_below` 1/2/4 × `no_above` 0,5/0,1). Ninguna mejora a la más simple. Los **N-gramas** empeoran con BM25 en las tres métricas, y claramente en las tres combinaciones con más filtros. El **filtro por frecuencia documental** quita justo las palabras raras que mejor identifican un documento: con `no_below=2` desaparece *opuntii*, que solo está en el fragmento relevante de su query. **Se queda sin N-gramas y sin filtro.**
- **Normalización**: se probó transcribir las letras nórdicas (*ð*, *þ*, *æ*…) y aplicar *stemming* (Snowball, NLTK). Las letras nórdicas no cambian nada (casi todos esos términos son ligaduras latinas de las traducciones antiguas, como *æther*). El *stemming* **sube Recall@10** (+0,024 con BM25 y +0,036 con TF-IDF, con intervalos que empiezan justo en 0), **pero no el MRR**. La regla fijada antes de ejecutar exigía una mejora clara del MRR de BM25, así que **no se adopta**; queda para volver a probarlo en el Hito 2.
- **k1 y b de BM25**: en el barrido (k1 ∈ {0,5; 1,2; 1,5; 2}, b ∈ {0; 0,5; 0,75; 1}) ninguna combinación se separa claramente de la de por defecto, así que se mantiene.

**Resultados** (84 queries; entre corchetes, el intervalo del 95 %):

| Métrica | TF-IDF | BM25 | BM25 − TF-IDF |
|---|---|---|---|
| Recall@5 | 0,774 [0,679; 0,857] | **0,810** [0,726; 0,893] | +0,036 [−0,024; 0,095] |
| Recall@10 | 0,821 [0,738; 0,893] | **0,845** [0,762; 0,917] | +0,024 [−0,024; 0,071] |
| Recall@50 | 0,917 [0,857; 0,976] | **0,952** [0,905; 0,988] | +0,036 [0,000; 0,083] |
| Recall@100 | 0,952 [0,905; 0,988] | **0,976** [0,940; 1,000] | +0,024 [0,000; 0,060] |
| MRR@10 | 0,558 [0,473; 0,644] | **0,645** [0,561; 0,733] | +0,087 [0,027; 0,152] |

BM25 es **claramente mejor solo en MRR**: los dos encuentran el documento en un número parecido de queries, pero BM25 lo coloca más arriba. Con BM25, el documento relevante está entre los 100 primeros en el 98 % de las queries, lo que es buena base para una etapa de reordenación (*reranker*, Hito 2).

**Comprobación con 87 queries** (las 84 más q024, q030 y q117, las tres descartadas por genéricas al ver que fallaban): BM25 mantiene la ventaja clara en MRR (+0,084 [0,023; 0,145]) y la gana también en Recall@50 (+0,046 [0,011; 0,092]) por una sola query, q024. Su Recall@10 baja 0,029 (a 0,816) y su MRR 0,022 (a 0,623). La elección de BM25 no cambia.

Por fuente (Recall@10 con BM25): clásicos 0,71, WHE 1,00, Wikipedia 0,96 y YouTube 0,81. Los clásicos son la única fuente claramente más difícil. Las queries reescritas a mano también son más difíciles que las del LLM (con BM25, MRR de 0,46 frente a 0,70), y 12 de las 20 son de los clásicos.

**Análisis de errores**: los dos métodos fallan en 12 de las 84 queries, 8 de ellas de los clásicos, pero solo 2 quedan fuera de los 100 primeros con los dos. Las causas:
1. **Gana otro documento del mismo tema** (la más frecuente). Por ejemplo, los artículos de Wikipedia de *Phaethon*, *Lethe* o *Diomedes* ganan al libro de Ovidio o de la *Ilíada* del que sale la query. Muchas veces lo devuelto también serviría, pero la evaluación *known-item* solo cuenta el documento de origen.
2. **La query usa otras palabras que el texto**: "islands of the blessed" frente a *Fortunate Isles*, "the love goddess" frente a *Aphrodite*. Son justo las tres que quedan más lejos (dos fuera de los 100 primeros y una en el puesto 99).
3. **Queries demasiado generales**, que encajan con muchos documentos ("Norse mythology vs nature worship"). Las tres que había se descartaron en la segunda revisión (sección 8).
4. **El lematizador no une las formas**: en una query, spaCy deja *gods* y *giants* en plural.

**Configuración final del baseline**: lemas de spaCy con el título delante, sin N-gramas, sin filtro por frecuencia documental, sin normalización, y BM25 (`bm25s`, Lucene, k1 = 1,5, b = 0,75).

---

## 10. Limitaciones conocidas y mejoras pendientes

- **Filtro de tema de Wikipedia.** La expresión regular de `procesa_wikipedia.py` que exige que el primer párrafo hable de mitos es el paso que más quita (1.345 artículos) y también se lleva artículos buenos (*Muspelheim*, *Ludi Romani*, *Temple of Quirinus*). El clasificador también pierde algunos artículos buenos (templos y tragedias antiguas).
- **Tokens que no coinciden:**
  - nombres propios terminados en *-s* que spaCy lematiza distinto según el contexto (*Kronos* → *krono*, *Pyrrhus* → *pyrrhu*, *Ares* → *are*);
  - singular y plural (*trojans* frente a *trojan*, *gods* frente a *god*): el *stemming* lo corrige y sube Recall@10, pero no se adoptó (sección 9);
  - errores en los subtítulos automáticos (41 % de los vídeos), que no tienen mayúsculas ni puntuación.
- **Nombres romanos en los clásicos** (*Jove*, *Venus*): un método léxico no los relaciona con los griegos. Encaja con el Hito 2 (equivalencias de nombres y *embeddings*).
- **Evaluación**: con 84 queries los intervalos son anchos (unos ±0,08 en Recall@10). Las variantes se eligieron con las mismas queries con las que se evalúa, así que los resultados pueden ser algo optimistas. Las queries comparten muchas palabras con su fragmento, lo que favorece a los métodos léxicos. Quitar las tres queries genéricas sube el Recall@10 de BM25 en 0,029 y su MRR en 0,022 (sección 9).
- **Reparto de mitologías**: la nórdica pesa poco en clásicos y Wikipedia, y la romana casi no aparece en YouTube.
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

- **LLM de la UC3M (`llama3.1:8b`)**:
  - generación de las queries candidatas de evaluación, todas revisadas a mano (sección 8);
  - clasificación de los artículos de Wikipedia como dentro o fuera de tema, validada con 100 artículos etiquetados a mano y corregida con 255 etiquetas manuales (sección 4.1). Un segundo uso como rescate se probó y se descartó por sus malos resultados.
- **Claude Code (Anthropic)**: asistencia en el diseño del pipeline, la escritura de los scripts y del notebook, la propuesta de decisiones en la revisión de queries y de artículos de Wikipedia (confirmadas por el grupo) y la redacción de la documentación. Todo el código se ejecutó y comprobó en local.
