"""
Crea el conjunto final de queries a partir de las tablas de revisión manual.

Cada tabla (una por tanda) tiene, para cada query candidata, la decisión
tomada en la revisión (mantener, editar o descartar), el motivo y la query
final. Se quedan las que no se han descartado.

Formato de salida (el del enunciado, más source_chunk y fuente):
    {"qid": "q002", "query": "...", "relevant_ids": ["wiki_24959971"],
     "source_chunk": "wiki_24959971_c01", "fuente": "wikipedia"}

Entrada:  data/revision_queries_t*.csv
Salida:   data/queries.jsonl

Uso:
    python src/crea_queries.py
"""

import json
from pathlib import Path

import pandas as pd

CARPETA = Path("data")

# 1. Juntar las tablas de revisión de todas las tandas
tablas = sorted(CARPETA.glob("revision_queries_t*.csv"))
revision = pd.concat([pd.read_csv(t) for t in tablas], ignore_index=True)
print(f"{len(revision)} queries candidatas en {len(tablas)} tandas\n")

# 2. Resumen de la revisión
print("Decisiones por tanda:")
print(pd.crosstab(revision["tanda"], revision["decision"], margins=True, margins_name="total"), "\n")
print("Solape medio con el fragmento (query original) por tanda:")
print(revision.groupby("tanda")["solape"].mean().round(2), "\n")

# 3. Quedarse con las no descartadas
finales = revision[revision["decision"] != "descartar"]
assert finales["query_final"].notna().all(), "Hay queries sin query_final"
assert finales["qid"].is_unique, "Hay qid repetidos"

# Comprobar que los documentos relevantes existen en el corpus
ruta_corpus = CARPETA / "processed" / "corpus.jsonl"
if ruta_corpus.exists():
    with open(ruta_corpus, encoding="utf-8") as f:
        doc_ids = {json.loads(linea)["doc_id"] for linea in f}
    faltan = set(finales["relevant_id"]) - doc_ids
    print(f"Documentos relevantes que no están en el corpus: {len(faltan)} {sorted(faltan)}\n")

with open(CARPETA / "queries.jsonl", "w", encoding="utf-8") as f:
    for q in finales.itertuples():
        f.write(json.dumps({"qid": q.qid, "query": q.query_final,
                            "relevant_ids": [q.relevant_id],
                            "source_chunk": q.source_chunk,
                            "fuente": q.fuente}, ensure_ascii=False) + "\n")

print(f"Guardadas {len(finales)} queries en {CARPETA / 'queries.jsonl'}\n")
print("Queries finales por fuente:")
print(finales["fuente"].value_counts().to_string())