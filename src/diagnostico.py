"""
Diagnóstico del corpus y del preprocesado (no cambia nada, solo mide).

Comprueba con datos los problemas detectados al revisar el Hito 1:

  1. Letras nórdicas (ð, þ, æ, ø) que quitar_tildes no normaliza
     (Óðinn -> "oðinn", que no coincide con "odin").
  2. Singulares y plurales que no se igualan (trojans / trojan, gods / god).
  3. Nombres propios terminados en -s que se pierden o se "singularizan"
     (hermes -> herme, ares -> are, que además es stopword), por fuente y
     separando los subtítulos manuales de los automáticos de YouTube.
  4. Ruido en Wikipedia:
       a) artículos y secciones de cultura popular que se han colado;
       b) de dónde vienen los artículos (profundidad y rama de categorías) y
          cómo se reparten los que marcamos como fuera de tema en la revisión.

Entrada:  data/processed/corpus.jsonl, chunks.jsonl, chunks_tokens.jsonl
          data/raw/wikipedia/lista_articulos.csv
          data/revision_queries_t*.csv
Salida:   solo por pantalla. Para guardarla y pasármela:
    python src/diagnostico.py | tee data/processed/diagnostico.txt
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PROCESSED = RAIZ / "data" / "processed"

NORDICAS = {"ð": "d", "þ": "th", "æ": "ae", "ø": "o", "œ": "oe"}

# Nombres terminados en -s que el lematizador podría estropear
NOMBRES_S = ["zeus", "hermes", "ares", "achilles", "odysseus", "heracles", "hercules",
             "perseus", "theseus", "dionysus", "remus", "romulus", "cronus", "kronos",
             "uranus", "tantalus", "sisyphus", "orpheus", "icarus", "daedalus",
             "prometheus", "atlas", "eros", "helios", "hades", "paris", "ulysses", "venus"]

# Palabras que suelen indicar cultura popular u obras modernas
CULTURA_POP = re.compile(r"popular culture|in fiction|in film|in art\b|in literature|in music|"
                         r"media|film|television|video game|games|comic|adaptation|"
                         r"modern (culture|depiction|reception|usage)|legacy|in culture|"
                         r"cultural (depiction|influence|references)", re.I)


def leer_jsonl(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f]


def titulo(texto):
    print("\n" + "=" * 78 + f"\n{texto}\n" + "=" * 78)


# --- Carga ---------------------------------------------------------------------

docs = {d["doc_id"]: d for d in leer_jsonl(PROCESSED / "corpus.jsonl")}
chunks = leer_jsonl(PROCESSED / "chunks.jsonl")
tokens = [t["tokens"] for t in leer_jsonl(PROCESSED / "chunks_tokens.jsonl")]
assert len(chunks) == len(tokens)


def grupo(chunk):
    # La fuente, separando YouTube según el tipo de subtítulos
    if chunk["fuente"] == "youtube":
        return f"youtube_{docs[chunk['doc_id']].get('subtitulos') or '?'}"
    return chunk["fuente"]


grupos = [grupo(c) for c in chunks]
frec = Counter(w for doc in tokens for w in doc)
frec_grupo = defaultdict(Counter)
for g, doc in zip(grupos, tokens):
    frec_grupo[g].update(doc)
ORDEN = ["clasicos", "whe", "wikipedia", "youtube_manuales", "youtube_automaticos"]
print(f"{len(docs)} documentos, {len(chunks)} fragmentos, {len(frec)} términos distintos")
print("Fragmentos por grupo:", dict(Counter(grupos)))


# --- 1. Letras nórdicas ----------------------------------------------------------

titulo("1. LETRAS NÓRDICAS SIN NORMALIZAR (ð, þ, æ, ø)")


def transliterar(w):
    for letra, sustituta in NORDICAS.items():
        w = w.replace(letra, sustituta)
    return w


nordicos = {w: n for w, n in frec.items() if any(l in w for l in NORDICAS)}
print(f"Términos distintos con alguna de esas letras: {len(nordicos)}")
print(f"Apariciones en total: {sum(nordicos.values())}")
print("Apariciones por grupo:",
      {g: sum(n for w, n in frec_grupo[g].items() if w in nordicos) for g in ORDEN})
print("\nLos 25 más frecuentes -> transliterado (y cuántas veces aparece ya esa forma):")
for w, n in Counter(nordicos).most_common(25):
    t = transliterar(w)
    print(f"  {w:22s} {n:6d}  ->  {t:20s} {frec.get(t, 0):6d}")


# --- 2. Singulares y plurales ----------------------------------------------------

titulo("2. SINGULARES Y PLURALES QUE NO SE IGUALAN")
pares = [(w, w[:-1]) for w in frec if len(w) > 3 and w.endswith("s") and w[:-1] in frec]
print(f"Pares 'xs' / 'x' en el vocabulario: {len(pares)}")
print("\nLos 30 pares en que el plural es más frecuente (plural, singular):")
for pl, sg in sorted(pares, key=lambda p: -frec[p[0]])[:30]:
    print(f"  {pl:16s} {frec[pl]:6d}   {sg:16s} {frec[sg]:6d}")

print("\nPor grupo (apariciones):")
ejemplos = ["gods", "god", "giants", "giant", "trojans", "trojan",
            "greeks", "greek", "titans", "titan", "romans", "roman"]
tabla = pd.DataFrame({g: [frec_grupo[g][w] for w in ejemplos] for g in ORDEN}, index=ejemplos)
print(tabla.to_string())


# --- 3. Nombres terminados en -s -------------------------------------------------

titulo("3. NOMBRES TERMINADOS EN -S")
# Cuántas veces aparece cada nombre en el texto original (título + texto, como al preprocesar)
crudo = defaultdict(Counter)
for c, g in zip(chunks, grupos):
    palabras = re.findall(r"[a-z]+", (c["titulo"] + ". " + c["texto"]).lower())
    crudo[g].update(p for p in palabras if p in NOMBRES_S)

filas = []
for nombre in NOMBRES_S:
    for g in ORDEN:
        texto = crudo[g][nombre]
        if texto == 0:
            continue
        filas.append({"nombre": nombre, "grupo": g, "en el texto": texto,
                      "token igual": frec_grupo[g][nombre],
                      "token sin -s": frec_grupo[g][nombre[:-1]]})
nombres = pd.DataFrame(filas)
nombres["% conservado"] = (100 * nombres["token igual"] / nombres["en el texto"]).round(0)
print("'en el texto' = apariciones de la palabra en el texto original;")
print("'token igual' = cuántas veces sobrevive tal cual tras el preprocesado;")
print("'token sin -s' = cuántas veces aparece el token sin la -s final (posible 'singularización').")
print("(Los nombres pueden perderse también si spaCy no los etiqueta como NOUN/VERB/ADJ/PROPN.)\n")
print(nombres.to_string(index=False))

print("\nResumen por grupo (% de apariciones de estos nombres que sobreviven tal cual):")
resumen = nombres.groupby("grupo")[["en el texto", "token igual", "token sin -s"]].sum()
resumen["% conservado"] = (100 * resumen["token igual"] / resumen["en el texto"]).round(1)
print(resumen.reindex(ORDEN).to_string())


# --- 4a. Cultura popular en Wikipedia ---------------------------------------------

titulo("4a. CULTURA POPULAR EN WIKIPEDIA")
wiki = [d for d in docs.values() if d["fuente"] == "wikipedia"]
titulos_pop = sorted(d["titulo"] for d in wiki if CULTURA_POP.search(d["titulo"]))
print(f"Artículos cuyo título sugiere cultura popular u obras modernas: {len(titulos_pop)}")
for t in titulos_pop:
    print(f"  - {t}")

secciones = Counter(c["seccion"] for c in chunks if c["fuente"] == "wikipedia")
sospechosas = {s: n for s, n in secciones.items() if s and CULTURA_POP.search(s)}
print(f"\nSecciones con nombre sospechoso: {len(sospechosas)} distintas, "
      f"{sum(sospechosas.values())} fragmentos")
for s, n in Counter(sospechosas).most_common(40):
    print(f"  {n:5d}  {s}")
print("\nLas 40 secciones más frecuentes de Wikipedia (por si hay otras a vigilar):")
for s, n in secciones.most_common(40):
    print(f"  {n:5d}  {s or '(introducción)'}")


# --- 4b. Procedencia del ruido ----------------------------------------------------

titulo("4b. PROCEDENCIA DE LOS ARTÍCULOS DE WIKIPEDIA")
lista = pd.read_csv(RAIZ / "data" / "raw" / "wikipedia" / "lista_articulos.csv")
lista["doc_id"] = "wiki_" + lista["pageid"].astype(str)
en_corpus = lista[lista["doc_id"].isin(docs)]
print(f"Artículos del corpus con su procedencia: {len(en_corpus)} de {len(wiki)}")
print("\nPor profundidad:")
print(en_corpus["profundidad"].value_counts().sort_index().to_string())
print("\nLas 30 ramas (subcategoría de primer nivel) que más artículos aportan:")
print(en_corpus["rama"].value_counts().head(30).to_string())

# Los documentos de Wikipedia que revisamos al hacer las queries
revision = pd.concat([pd.read_csv(p) for p in sorted((RAIZ / "data").glob("revision_queries_t*.csv"))])
rev = revision[revision["fuente"] == "wikipedia"].copy()
rev["fuera_de_tema"] = rev["motivo"].fillna("").str.startswith("Documento fuera de tema")
rev = rev.merge(lista[["doc_id", "profundidad", "rama", "categoria"]],
                left_on="relevant_id", right_on="doc_id", how="left")
print(f"\nDocumentos de Wikipedia revisados: {len(rev)}, "
      f"marcados fuera de tema: {rev['fuera_de_tema'].sum()}")
print("\nFuera de tema por profundidad (fuera / total):")
print(rev.groupby("profundidad")["fuera_de_tema"].agg(["sum", "count"]).to_string())
print("\nLos marcados fuera de tema, con su procedencia:")
for r in rev[rev["fuera_de_tema"]].itertuples():
    print(f"  {r.titulo[:40]:40s} prof={r.profundidad}  rama={r.rama}  categoría={r.categoria}")
print("\nLos que están dentro de tema, con su procedencia:")
for r in rev[~rev["fuera_de_tema"]].itertuples():
    print(f"  {r.titulo[:40]:40s} prof={r.profundidad}  rama={r.rama}  categoría={r.categoria}")