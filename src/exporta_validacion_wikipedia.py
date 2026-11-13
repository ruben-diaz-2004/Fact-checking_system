"""
Prepara el conjunto de validación para el filtro de Wikipedia.

Junta 100 artículos de Wikipedia del corpus para etiquetarlos a mano como
"dentro" o "fuera" de tema con el criterio A:

    DENTRO: trata de los mitos en sí (personajes, relatos, lugares, culto,
            textos antiguos y los estudios sobre ellos).
    FUERA:  obras modernas inspiradas en mitos (cuadros, óperas, novelas,
            teatro, cómics...), cosas que solo llevan el nombre de un dios,
            personas, folklore posterior y otras mitologías.

  - Los 45 que ya revisamos al hacer las queries (origen = "queries").
  - 55 nuevos al azar, repartidos por profundidad en el árbol de categorías,
    para que haya bastantes de cada nivel (origen = "nuevo"):
        10 de profundidad 0-1, 20 de profundidad 2 y 25 de profundidad 3.

Para cada artículo se muestra el título, su procedencia y el principio del
texto (la introducción, hasta unas 250 palabras), que es lo mismo que verá
después el clasificador.

Entrada:  data/processed/corpus.jsonl
          data/raw/wikipedia/lista_articulos.csv
          data/revision_queries_t*.csv
Salida:   data/processed/validacion_wikipedia.md   (para leerlo y pasármelo)

Uso:
    python src/exporta_validacion_wikipedia.py
"""

import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PROCESSED = RAIZ / "data" / "processed"
SEMILLA = 42
NUEVOS = {(0, 1): 10, (2,): 20, (3,): 25}   # profundidades -> cuántos artículos nuevos
PALABRAS_INTRO = 250


def principio(texto, n=PALABRAS_INTRO):
    # Introducción (hasta la primera sección) y, si es muy corta, lo siguiente
    intro = texto.split("\n== ")[0]
    if len(intro.split()) < 60:
        intro = texto
    palabras = intro.split()
    return " ".join(palabras[:n]) + (" [...]" if len(palabras) > n else "")


# Artículos de Wikipedia del corpus con su procedencia
with open(PROCESSED / "corpus.jsonl", encoding="utf-8") as f:
    wiki = pd.DataFrame([d for d in map(json.loads, f) if d["fuente"] == "wikipedia"])
lista = pd.read_csv(RAIZ / "data" / "raw" / "wikipedia" / "lista_articulos.csv")
lista["doc_id"] = "wiki_" + lista["pageid"].astype(str)
wiki = wiki.merge(lista[["doc_id", "profundidad", "rama", "categoria"]], on="doc_id", how="left")

# 1. Los 45 que ya revisamos
revision = pd.concat([pd.read_csv(p) for p in sorted((RAIZ / "data").glob("revision_queries_t*.csv"))])
revisados = set(revision.loc[revision["fuente"] == "wikipedia", "relevant_id"])
elegidos = wiki[wiki["doc_id"].isin(revisados)].assign(origen="queries")

# 2. Los nuevos, al azar por profundidad
resto = wiki[~wiki["doc_id"].isin(revisados)]
for profundidades, n in NUEVOS.items():
    muestra = resto[resto["profundidad"].isin(profundidades)].sample(n, random_state=SEMILLA)
    elegidos = pd.concat([elegidos, muestra.assign(origen="nuevo")])

elegidos = elegidos.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)   # mezclados
elegidos["vid"] = [f"v{i + 1:03d}" for i in range(len(elegidos))]

with open(PROCESSED / "validacion_wikipedia.md", "w", encoding="utf-8") as f:
    for r in elegidos.itertuples():
        f.write(f"## {r.vid} | {r.doc_id} | {r.titulo}\n\n")
        f.write(f"*Profundidad {r.profundidad} · {r.rama} · {r.categoria} · origen: {r.origen}*\n\n")
        f.write(f"{principio(r.texto)}\n\n---\n\n")

print(f"{len(elegidos)} artículos escritos en {PROCESSED / 'validacion_wikipedia.md'}")
print(elegidos.groupby(["profundidad", "origen"]).size().unstack(fill_value=0).to_string())