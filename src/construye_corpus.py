"""
Junta las cuatro fuentes en un único corpus y lo trocea en fragmentos.

Entrada (generadas por los scripts de cada fuente):
    data/processed/clasicos.jsonl    (procesa_clasicos.py)
    data/processed/whe.jsonl         (descarga_whe.py)
    data/processed/wikipedia.jsonl   (procesa_wikipedia.py)
    data/processed/youtube.jsonl     (descarga_youtube.py)

Salida:
    data/processed/corpus.jsonl  -> un documento por línea, todos con los mismos campos
    data/processed/chunks.jsonl  -> un fragmento por línea: lo que se indexa en la búsqueda

Uso (desde la carpeta principal del repositorio, con el entorno activado):
    python src/construye_corpus.py
"""

import json
from pathlib import Path

import pandas as pd

from troceo import trocear_documento

CARPETA = Path(__file__).resolve().parents[1] / "data" / "processed"
FUENTES = ["clasicos", "whe", "wikipedia", "youtube"]
RUTA_CORPUS = CARPETA / "corpus.jsonl"
RUTA_CHUNKS = CARPETA / "chunks.jsonl"

# Campos comunes de cada documento (los que una fuente no tenga quedan vacíos)
CAMPOS = ["doc_id", "fuente", "titulo", "obra", "autor", "tipo_fuente", "nivel",
          "mitologia", "url", "subtitulos", "revid", "texto"]
# Campos del documento que se copian a cada fragmento, para poder filtrar y citar sin
# tener que volver al documento
CAMPOS_EN_CHUNK = ["fuente", "titulo", "autor", "tipo_fuente", "nivel", "mitologia", "url"]


def guardar_jsonl(registros, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    # 1. Cargar y unificar las fuentes
    partes = []
    for fuente in FUENTES:
        ruta = CARPETA / f"{fuente}.jsonl"
        if not ruta.exists():
            print(f"AVISO: falta {ruta.name}; el corpus se construye sin esa fuente.")
            continue
        df = pd.read_json(ruta, lines=True, dtype=False)
        df["fuente"] = fuente
        partes.append(df.reindex(columns=CAMPOS))
    corpus = pd.concat(partes, ignore_index=True)
    corpus = corpus.astype(object).where(corpus.notna(), None)   # NaN -> None (null en JSON)

    repetidos = corpus["doc_id"][corpus["doc_id"].duplicated()]
    if len(repetidos):
        raise ValueError(f"Hay doc_id repetidos: {list(repetidos[:10])}")

    # 2. Trocear cada documento
    chunks = []
    for doc in corpus.to_dict("records"):
        for frag in trocear_documento(doc["doc_id"], doc["texto"]):
            for campo in CAMPOS_EN_CHUNK:
                frag[campo] = doc[campo]
            # En YouTube la sección es "t=125": se guarda también el segundo como número
            if doc["fuente"] == "youtube" and str(frag["seccion"]).startswith("t="):
                frag["segundo"] = int(frag["seccion"][2:])
            chunks.append(frag)

    # 3. Guardar
    guardar_jsonl(corpus.to_dict("records"), RUTA_CORPUS)
    guardar_jsonl(chunks, RUTA_CHUNKS)

    # --- Resumen ---
    corpus["palabras"] = corpus["texto"].str.split().str.len()
    ch = pd.DataFrame(chunks)
    resumen = corpus.groupby("fuente").agg(documentos=("doc_id", "count"),
                                           palabras=("palabras", "sum"))
    resumen["fragmentos"] = ch.groupby("fuente").size()
    resumen["% fragmentos"] = (100 * resumen["fragmentos"] / len(ch)).round(1)
    resumen = resumen.reindex([f for f in FUENTES if f in resumen.index])

    print("=" * 64)
    print(f"Corpus: {len(corpus)} documentos  ->  {RUTA_CORPUS}")
    print(f"Fragmentos: {len(ch)}  ->  {RUTA_CHUNKS}")
    print("=" * 64)
    print("\nPor fuente:")
    print(resumen.to_string())
    print("\nFragmentos por tipo de fuente:")
    print(ch["tipo_fuente"].value_counts().to_string())
    print("\nFragmentos por mitología (uno puede tener varias):")
    print(ch["mitologia"].fillna("(sin dato)").str.split(";").explode().value_counts().to_string())
    print("\nPalabras por fragmento:")
    print(ch["n_palabras"].describe(percentiles=[0.05, 0.5, 0.95]).round(0).to_string())


if __name__ == "__main__":
    main()