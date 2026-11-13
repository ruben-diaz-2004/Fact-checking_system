"""
Prepara la revisión manual de una tanda de queries candidatas.

Junta cada query con el texto del fragmento del que salió, para poder juzgar
si la query describe bien ese fragmento (y solo ese) sin copiar sus frases.

Entrada:  data/processed/queries_candidatas_t{N}.jsonl
          data/processed/chunks.jsonl
Salida:   data/processed/revision_queries_t{N}.md   (para leer y pasármelo)

Uso:
    python src/exporta_revision.py --tanda 2
"""

import argparse
import json
from pathlib import Path

CARPETA = Path("data/processed")

parser = argparse.ArgumentParser()
parser.add_argument("--tanda", type=int, required=True)
args = parser.parse_args()

with open(CARPETA / f"queries_candidatas_t{args.tanda}.jsonl", encoding="utf-8") as f:
    queries = [json.loads(linea) for linea in f]

# Solo hace falta el texto de los fragmentos de origen
buscados = {q["source_chunk"] for q in queries}
textos = {}
with open(CARPETA / "chunks.jsonl", encoding="utf-8") as f:
    for linea in f:
        fr = json.loads(linea)
        if fr["chunk_id"] in buscados:
            textos[fr["chunk_id"]] = fr["texto"]

salida = CARPETA / f"revision_queries_t{args.tanda}.md"
with open(salida, "w", encoding="utf-8") as f:
    for q in queries:
        aviso = "" if q.get("on_topic", True) else "  **[EL LLM DICE: NO ES MITOLOGÍA]**"
        f.write(f"## {q['qid']} [{q['fuente']}] {q['titulo']}{aviso}\n\n")
        f.write(f"**Query:** {q['query']}  (solape {q['solape']:.2f})\n\n")
        f.write(f"**Fragmento {q['source_chunk']}:**\n\n{textos[q['source_chunk']]}\n\n---\n\n")

print(f"{len(queries)} queries escritas en {salida}")