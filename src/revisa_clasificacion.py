"""
Revisa la clasificación de Wikipedia hecha por clasifica_wikipedia.py antes de aplicarla.

Muestra:
  1. Cuántos artículos y cuántos fragmentos quitaría el filtro, y por profundidad.
  2. Las queries de evaluación cuyo documento relevante quedaría fuera.
  3. Los 40 artículos más largos que quita (los que más fragmentos se llevan),
     para rescatar a mano los que sean importantes.
  4. Una muestra al azar de 40 artículos que quita y que NO están en la validación,
     para comprobar el acierto en artículos que no hemos visto nunca.

Entrada:  data/processed/clasificacion_wikipedia.csv, wikipedia.jsonl, chunks.jsonl
          data/raw/wikipedia/lista_articulos.csv
          data/etiquetas_wikipedia.csv, data/queries.jsonl
Salida:   resumen por pantalla y data/processed/revision_clasificacion.md (para pasármelo)

Uso:
    python src/revisa_clasificacion.py
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PROCESSED = RAIZ / "data" / "processed"
SEMILLA = 42
PALABRAS_INTRO = 250


def leer_jsonl(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f]


def principio(texto, n=PALABRAS_INTRO):
    intro = texto.split("\n== ")[0]
    if len(intro.split()) < 60:
        intro = texto
    palabras = intro.split()
    return " ".join(palabras[:n]) + (" [...]" if len(palabras) > n else "")


clas = pd.read_csv(PROCESSED / "clasificacion_wikipedia.csv")
textos = {d["doc_id"]: d["texto"] for d in leer_jsonl(PROCESSED / "wikipedia.jsonl")}
fragmentos = Counter(c["doc_id"] for c in leer_jsonl(PROCESSED / "chunks.jsonl")
                     if c["fuente"] == "wikipedia")
lista = pd.read_csv(RAIZ / "data" / "raw" / "wikipedia" / "lista_articulos.csv")
lista["doc_id"] = "wiki_" + lista["pageid"].astype(str)

clas["fragmentos"] = clas["doc_id"].map(fragmentos).fillna(0).astype(int)
clas = clas.merge(lista[["doc_id", "profundidad"]], on="doc_id", how="left")
fuera = clas[clas["etiqueta"] == "fuera"]

# 1. Cuánto quita
print(f"Artículos: {len(clas)}, fuera: {len(fuera)} ({100 * len(fuera) / len(clas):.1f} %)")
print(f"Fragmentos de Wikipedia: {clas['fragmentos'].sum()}, de artículos fuera: "
      f"{fuera['fragmentos'].sum()} ({100 * fuera['fragmentos'].sum() / clas['fragmentos'].sum():.1f} %)")
print("\nPor profundidad (artículos fuera / total, y % de fragmentos fuera):")
tabla = clas.groupby("profundidad").apply(lambda g: pd.Series({
    "fuera": (g["etiqueta"] == "fuera").sum(), "total": len(g),
    "% fragmentos fuera": round(100 * g.loc[g["etiqueta"] == "fuera", "fragmentos"].sum()
                                / g["fragmentos"].sum(), 1)}))
print(tabla.to_string())

# 2. Queries afectadas
queries = leer_jsonl(RAIZ / "data" / "queries.jsonl")
ids_fuera = set(fuera["doc_id"])
afectadas = [q for q in queries if q["relevant_ids"][0] in ids_fuera]
print(f"\nQueries cuyo documento relevante quedaría fuera: {len(afectadas)}")
for q in afectadas:
    titulo = clas.loc[clas["doc_id"] == q["relevant_ids"][0], "titulo"].iloc[0]
    print(f"  {q['qid']}  {titulo}  <-  {q['query']}")

# 3 y 4. Artículos para revisar a mano
validacion = set(pd.read_csv(RAIZ / "data" / "etiquetas_wikipedia.csv")["doc_id"])
largos = fuera.sort_values("fragmentos", ascending=False).head(40)
muestra = fuera[~fuera["doc_id"].isin(validacion) & ~fuera["doc_id"].isin(largos["doc_id"])]
muestra = muestra.sample(40, random_state=SEMILLA)

with open(PROCESSED / "revision_clasificacion.md", "w", encoding="utf-8") as f:
    for nombre, grupo in [("LARGOS", largos), ("MUESTRA AL AZAR", muestra)]:
        f.write(f"# {nombre}\n\n")
        for r in grupo.itertuples():
            f.write(f"## {r.doc_id} | {r.titulo} | {r.fragmentos} fragmentos | prof. {r.profundidad}\n\n")
            f.write(f"*Motivo del LLM: {r.motivo}*\n\n{principio(textos[r.doc_id])}\n\n---\n\n")

print(f"\n40 largos y 40 al azar escritos en {PROCESSED / 'revision_clasificacion.md'}")
print("\nLos 40 más largos que quita:")
for r in largos.itertuples():
    print(f"  {r.fragmentos:4d}  {r.titulo[:45]:45s} ({r.motivo})")