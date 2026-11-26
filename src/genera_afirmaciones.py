"""
Genera las afirmaciones candidatas para evaluar la generación del sistema RAG.

Las 84 queries del Hito 1 sirven para evaluar la recuperación, pero no los veredictos:
son descripciones de un pasaje, no afirmaciones con una respuesta correcta. Aquí se
crean afirmaciones con su veredicto correcto conocido, de tres tipos:

  - verdadero: un hecho que afirma el fragmento de origen de una query del Hito 1.
  - falso:     el mismo hecho con UN dato cambiado (un nombre, una relación, un lugar...),
               de modo que el fragmento lo desmiente.
  - sin_informacion: afirmaciones que el corpus no puede confirmar ni desmentir:
        * hechos de otras mitologías (egipcia, azteca, hindú...), que no están en el corpus;
        * detalles inventados sobre personajes conocidos de nuestras mitologías (un objeto,
          lugar o compañero con un nombre inventado), que ningún texto menciona.

Se parte de los fragmentos de origen de las queries porque ya están revisados a mano y
son de mitología, y así cada afirmación verdadera o falsa tiene su prueba conocida
(el fragmento y su documento).

Todas las candidatas hay que revisarlas a mano antes de usarlas, como las queries.

La clave de la API se lee del archivo .env (que no se sube a GitHub).

Entrada:  data/queries.jsonl, data/processed/chunks.jsonl
Salida:   data/processed/afirmaciones_candidatas.jsonl   (todas las candidatas)
          data/processed/revision_afirmaciones.md        (cada afirmación junto a su fragmento, para leer)
          data/revision_afirmaciones.csv                 (tabla para anotar la revisión; NO se
                                                          sobrescribe si ya existe)

Uso (desde la carpeta principal del repositorio, con el entorno activado):
    python src/genera_afirmaciones.py
    python src/genera_afirmaciones.py --modelo qwen3:8b
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
import urllib3

from clasifica_wikipedia import URL, leer_clave

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RAIZ = Path(__file__).resolve().parents[1]
RUTA_QUERIES = RAIZ / "data" / "queries.jsonl"
RUTA_CHUNKS = RAIZ / "data" / "processed" / "chunks.jsonl"
RUTA_CANDIDATAS = RAIZ / "data" / "processed" / "afirmaciones_candidatas.jsonl"
RUTA_MD = RAIZ / "data" / "processed" / "revision_afirmaciones.md"
RUTA_CSV = RAIZ / "data" / "revision_afirmaciones.csv"

# Para las afirmaciones sin información
OTRAS_MITOLOGIAS = ["Egyptian", "Mesopotamian", "Hindu", "Aztec", "Maya",
                    "Chinese", "Japanese", "Celtic", "Slavic", "Yoruba"]          # 2 afirmaciones de cada una
PERSONAJES = ["Zeus", "Hera", "Athena", "Apollo", "Artemis", "Hermes", "Heracles", "Perseus",
              "Odysseus", "Achilles", "Jupiter", "Venus", "Romulus", "Aeneas",
              "Odin", "Thor", "Loki", "Freyja", "Baldr", "Tyr"]                    # 1 detalle inventado de cada uno

PROMPT_FRAGMENTO = """You will receive a passage from a collection about Greek, Roman and Norse mythology.
Write two short factual claims (8-25 words each) to test a fact-checking system:

1. "true_claim": one specific fact that the passage clearly states. Choose a fact that is central to the passage, not a minor aside.
2. "false_claim": the same claim with exactly ONE detail changed (a name, a relationship, a place, a number or an action), so that the passage clearly contradicts it. Keep it plausible: replace the detail with another one of the same kind from the same mythology.
3. "change": what you changed, in a few words.

Rules for both claims:
- Self-contained: name the characters, places or works explicitly. No pronouns without a clear referent, and never "the passage" or "this text".
- Use your own words: do not copy sentences from the passage.

Answer only with JSON: {{"true_claim": "...", "false_claim": "...", "change": "..."}}

Passage (from "{titulo}"):
{texto}"""

PROMPT_OTRA = """Write two short factual claims (8-25 words each) about {mitologia} mythology.
Each claim must state a specific, well-known fact about a god, hero, creature or myth of {mitologia} mythology, naming the characters explicitly.
The two claims must be about different characters or myths. Do not mention Greek, Roman or Norse mythology.

Answer only with JSON: {{"claims": ["...", "..."]}}"""

PROMPT_INVENTADA = """We are building a test set for a fact-checking system about Greek, Roman and Norse mythology.
We need claims that NO ancient or modern source can confirm or refute.

Write one short claim (8-25 words) about {personaje} that adds an invented detail: a made-up object, place, companion or event with an invented proper name.
It must sound plausible, but it must not change or contradict anything known about {personaje} (parents, spouse, children, deeds, symbols).

Answer only with JSON: {{"claim": "...", "invented": "the invented element, in a few words"}}"""


def preguntar(prompt, modelo, clave, temperatura, intentos=3):
    datos = {"model": modelo, "prompt": prompt, "stream": False, "format": "json",
             "options": {"temperature": temperatura}}
    if modelo.startswith("qwen3"):
        datos["think"] = False          # qwen3 "piensa" en voz alta si no se le dice
    for intento in range(intentos):
        try:
            r = requests.post(URL, json=datos, headers={"X-API-KEY": clave}, verify=False, timeout=120)
            r.raise_for_status()
            return json.loads(r.json()["response"])
        except Exception as e:
            if intento == intentos - 1:
                print(f"  ERROR: {e}")
                return None
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="llama3.1:8b")
    args = parser.parse_args()
    clave = leer_clave()

    with open(RUTA_QUERIES, encoding="utf-8") as f:
        queries = [json.loads(linea) for linea in f]
    buscados = {q["source_chunk"] for q in queries}
    fragmentos = {}
    with open(RUTA_CHUNKS, encoding="utf-8") as f:
        for linea in f:
            fr = json.loads(linea)
            if fr["chunk_id"] in buscados:
                fragmentos[fr["chunk_id"]] = fr
    faltan = buscados - set(fragmentos)
    assert not faltan, f"Fragmentos de origen que no están en chunks.jsonl: {sorted(faltan)[:5]}"

    candidatas = []

    def anadir(etiqueta, afirmacion, origen, nota="", q=None):
        candidatas.append({
            "aid": f"a{len(candidatas) + 1:03d}", "etiqueta": etiqueta, "afirmacion": afirmacion.strip(),
            "idioma": "en", "origen": origen, "nota": nota,
            "qid": q["qid"] if q else "", "relevant_id": q["relevant_ids"][0] if q else "",
            "source_chunk": q["source_chunk"] if q else "", "fuente": q["fuente"] if q else "",
            "titulo": fragmentos[q["source_chunk"]]["titulo"] if q else ""})

    # 1. Verdaderas y falsas, a partir del fragmento de origen de cada query
    print(f"Modelo: {args.modelo}\n\n1. Verdaderas y falsas ({len(queries)} fragmentos)")
    for i, q in enumerate(queries, 1):
        fr = fragmentos[q["source_chunk"]]
        r = preguntar(PROMPT_FRAGMENTO.format(titulo=fr["titulo"], texto=fr["texto"]), args.modelo, clave, 0.3)
        if not r or not r.get("true_claim") or not r.get("false_claim"):
            print(f"  {q['qid']}: respuesta no válida, se salta")
            continue
        anadir("verdadero", r["true_claim"], "fragmento", q=q)
        anadir("falso", r["false_claim"], "fragmento", nota=str(r.get("change", "")), q=q)
        if i % 10 == 0 or i == len(queries):
            print(f"  {i}/{len(queries)}")

    # 2. Sin información: otras mitologías
    print(f"\n2. Otras mitologías ({len(OTRAS_MITOLOGIAS)} x 2)")
    for mitologia in OTRAS_MITOLOGIAS:
        r = preguntar(PROMPT_OTRA.format(mitologia=mitologia), args.modelo, clave, 0.7)
        for afirmacion in (r or {}).get("claims", [])[:2]:
            anadir("sin_informacion", afirmacion, "otra_mitologia", nota=mitologia)

    # 3. Sin información: detalles inventados sobre personajes conocidos
    print(f"\n3. Detalles inventados ({len(PERSONAJES)})")
    for personaje in PERSONAJES:
        r = preguntar(PROMPT_INVENTADA.format(personaje=personaje), args.modelo, clave, 0.9)
        if r and r.get("claim"):
            anadir("sin_informacion", r["claim"], "inventada", nota=f"{personaje}: {r.get('invented', '')}")

    # 4. Guardar las candidatas, el fichero para leer y la tabla de revisión
    with open(RUTA_CANDIDATAS, "w", encoding="utf-8") as f:
        for c in candidatas:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(RUTA_MD, "w", encoding="utf-8") as f:
        f.write("# Revisión de las afirmaciones candidatas\n\n")
        for qid in dict.fromkeys(c["qid"] for c in candidatas if c["qid"]):
            par = [c for c in candidatas if c["qid"] == qid]
            fr = fragmentos[par[0]["source_chunk"]]
            f.write(f"## {qid} [{fr['fuente']}] {fr['titulo']}\n\n")
            for c in par:
                extra = f"  *(cambio: {c['nota']})*" if c["nota"] else ""
                f.write(f"- **{c['aid']} {c['etiqueta'].upper()}:** {c['afirmacion']}{extra}\n")
            f.write(f"\n**Fragmento {fr['chunk_id']}:**\n\n{fr['texto']}\n\n---\n\n")
        f.write("## Sin información\n\n")
        for c in candidatas:
            if c["etiqueta"] == "sin_informacion":
                f.write(f"- **{c['aid']}** [{c['origen']}: {c['nota']}] {c['afirmacion']}\n")

    if RUTA_CSV.exists():
        print(f"\n{RUTA_CSV} ya existe: no se sobrescribe (puede tener la revisión hecha)")
    else:
        tabla = pd.DataFrame(candidatas)[["aid", "etiqueta", "origen", "qid", "relevant_id", "source_chunk",
                                         "fuente", "titulo", "nota", "afirmacion"]]
        tabla.insert(len(tabla.columns), "decision", "")
        tabla.insert(len(tabla.columns), "motivo", "")
        tabla.insert(len(tabla.columns), "afirmacion_final", tabla["afirmacion"])
        tabla.to_csv(RUTA_CSV, index=False)

    resumen = pd.DataFrame(candidatas).groupby(["etiqueta", "origen"]).size()
    print(f"\nCandidatas: {len(candidatas)}")
    print(resumen.to_string())
    print(f"\n-> {RUTA_CANDIDATAS}\n-> {RUTA_MD}\n-> {RUTA_CSV}")


if __name__ == "__main__":
    main()
