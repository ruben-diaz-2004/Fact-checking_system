"""
Actualiza el source_chunk de las queries después de cambiar el troceo.

Al corregir el troceo de YouTube (las marcas de tiempo ya no cortan los fragmentos)
cambian los límites y la numeración de los fragmentos de los vídeos, así que el
source_chunk que guardan las tablas de revisión deja de apuntar al texto del que
salió cada query. Este script busca, para cada query, el fragmento nuevo del mismo
documento que más palabras comparte con el fragmento antiguo, y reescribe las tablas.

El documento relevante (relevant_id) no cambia: la evaluación por documento es la misma.
En las fuentes que no son YouTube el troceo no ha cambiado y el fragmento nuevo es
idéntico al antiguo.

Pasos (desde la carpeta raíz del repositorio):
    cp data/processed/chunks.jsonl data/processed/chunks_antes.jsonl   # copia del troceo antiguo
    python src/construye_corpus.py                                     # troceo nuevo
    python src/actualiza_source_chunk.py                               # este script
    python src/crea_queries.py                                         # regenera data/queries.jsonl
    python src/preprocesado.py --procesos 2                            # tokens del troceo nuevo

Solo se puede ejecutar una vez: al terminar, chunks_antes.jsonl se renombra a
chunks_antes_aplicado.jsonl, porque las tablas ya apuntan al troceo nuevo.

Entrada:  data/processed/chunks_antes.jsonl, data/processed/chunks.jsonl,
          data/revision_queries_t*.csv
Salida:   data/revision_queries_t*.csv (misma tabla, con la columna source_chunk al día)
"""

import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

DATOS = Path("data")
RUTA_ANTES = DATOS / "processed" / "chunks_antes.jsonl"
RUTA_AHORA = DATOS / "processed" / "chunks.jsonl"


def leer_chunks(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f]


def comunes(a, b):
    # Palabras compartidas contando repeticiones
    return sum((Counter(a) & Counter(b)).values())


def main():
    if not RUTA_ANTES.exists():
        raise SystemExit(f"No encuentro {RUTA_ANTES}. Si ya se aplicó el cambio "
                         "(existe chunks_antes_aplicado.jsonl), no hay que repetirlo.")
    antes = {c["chunk_id"]: c for c in leer_chunks(RUTA_ANTES)}
    ahora_por_doc = defaultdict(list)
    for c in leer_chunks(RUTA_AHORA):
        ahora_por_doc[c["doc_id"]].append(c)

    cambiados, sin_documento = 0, []
    for tabla in sorted(DATOS.glob("revision_queries_t*.csv")):
        with open(tabla, encoding="utf-8", newline="") as f:
            contenido = f.read()
        salto = "\r\n" if "\r\n" in contenido else "\n"
        lector = csv.DictReader(io.StringIO(contenido, newline=""))
        filas = list(lector)
        columnas = lector.fieldnames
        if not filas:
            continue

        for fila in filas:
            viejo = antes.get(fila["source_chunk"])
            candidatos = ahora_por_doc.get(fila["relevant_id"], [])
            if viejo is None or not candidatos:
                # Queries ya descartadas porque su documento salió del corpus: se dejan igual
                sin_documento.append(fila["qid"])
                continue
            palabras = viejo["texto"].split()
            # A igualdad de palabras compartidas, se prefiere el fragmento con el mismo id
            mejor = max(candidatos, key=lambda c: (comunes(palabras, c["texto"].split()),
                                                   c["chunk_id"] == fila["source_chunk"]))
            cubierto = comunes(palabras, mejor["texto"].split()) / max(len(palabras), 1)
            if cubierto < 0.5:
                print(f"AVISO {fila['qid']}: el mejor fragmento nuevo solo contiene el "
                      f"{100 * cubierto:.0f} % del antiguo; revisar a mano")
            if mejor["texto"] != viejo["texto"] or mejor["chunk_id"] != fila["source_chunk"]:
                cambiados += 1
                print(f"{fila['qid']} [{fila['fuente']}, {fila['decision']}] "
                      f"{fila['source_chunk']} -> {mejor['chunk_id']}  "
                      f"(contiene el {100 * cubierto:.0f} % del fragmento antiguo)")
            fila["source_chunk"] = mejor["chunk_id"]

        with open(tabla, "w", encoding="utf-8", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=columnas, lineterminator=salto)
            escritor.writeheader()
            escritor.writerows(filas)

    RUTA_ANTES.rename(RUTA_ANTES.with_name("chunks_antes_aplicado.jsonl"))
    print(f"\nFragmentos de origen que cambian: {cambiados}")
    print(f"Queries cuyo fragmento o documento ya no está en el corpus (no se tocan): "
          f"{len(sin_documento)} {sin_documento}")
    print("Ahora ejecuta: python src/crea_queries.py")


if __name__ == "__main__":
    main()
