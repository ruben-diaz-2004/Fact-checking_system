"""
Genera queries candidatas con el LLM de la UC3M, por tandas.

Pasos:
    1. Elige fragmentos al azar, un número fijo por fuente, para que no salgan
       casi todos de Wikipedia (que es el 69 % de los fragmentos). No repite
       documentos que ya salieron en tandas anteriores.
    2. Para cada fragmento pide al LLM la query que escribiría alguien que
       buscara ese fragmento, sin copiar sus frases (como en el notebook).
       El LLM además dice si el fragmento trata de mitología (para descartar
       el ruido del corpus sin tener que leerlo).
    3. Guarda las candidatas para revisarlas a mano. El documento de origen
       es el relevante: relevant_ids = [doc_id].

Prompt de la tanda 2 ajustado con lo visto al revisar la tanda 1:
pedir lo concreto del fragmento (no el tema general del personaje) y no
inventar nada que no esté en el texto.

La clave de la API se lee del archivo .env (que no se sube a GitHub):
    UC3M_API_KEY=sk-...

Entrada:  data/processed/chunks.jsonl
Salida:   data/processed/queries_candidatas_t{N}.jsonl

Uso:
    python src/genera_queries.py --tanda 2
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import requests
import urllib3

# El servidor usa un certificado propio (por eso el -k en el curl del enunciado)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://yiyuan.tsc.uc3m.es/api/generate"
CARPETA = Path("data/processed")
RUTA_CHUNKS = CARPETA / "chunks.jsonl"

# Fragmentos a elegir de cada fuente (90 en total)
POR_FUENTE = {"wikipedia": 30, "clasicos": 24, "youtube": 20, "whe": 16}
MIN_PALABRAS = 150    # fragmentos muy cortos dan queries muy pobres

PROMPT = """You will receive a passage from a collection about Greek, Roman and Norse mythology.
Write a short search query (8-20 words) that a user looking for THIS passage could type.

Rules:
- Describe what is specific to this passage (a particular episode, detail or claim),
  not the general topic of the character, so that the query fits this passage better than others.
- Use your own words: do not copy phrases from the passage.
- Use at most one proper name (a god, hero, place or work), only if it is essential.
- Do not add anything that is not in the passage.

Also say whether the passage is about mythology, legend, ancient religion or their
reception (art, literature, scholarship). A list of people, a modern fairy tale or an
unrelated topic is NOT.

Answer only with JSON: {{"on_topic": true or false, "query": "..."}}

Passage:
{texto}"""


def leer_clave():
    for linea in Path(".env").read_text(encoding="utf-8").splitlines():
        if linea.startswith("UC3M_API_KEY="):
            return linea.split("=", 1)[1].strip()
    raise SystemExit("No encuentro UC3M_API_KEY en el archivo .env")


def preguntar_llm(prompt, modelo, clave):
    datos = {"model": modelo, "prompt": prompt, "stream": False, "format": "json",
             "options": {"temperature": 0.3}}
    if modelo.startswith("qwen3"):
        datos["think"] = False          # qwen3 "piensa" en voz alta si no se le dice
    r = requests.post(URL, json=datos, headers={"X-API-KEY": clave},
                      verify=False, timeout=120)
    r.raise_for_status()
    return r.json()["response"]


def leer_respuesta(texto):
    # El LLM debe devolver JSON; si falla, se guarda lo que haya para revisarlo
    try:
        datos = json.loads(texto)
        return bool(datos.get("on_topic", True)), str(datos.get("query", "")).strip(" \"'")
    except json.JSONDecodeError:
        return True, texto.strip().splitlines()[0] if texto.strip() else ""


def solape(query, texto):
    # Proporción de palabras de la query (de más de 3 letras) que están en el fragmento.
    # Sirve para detectar en la revisión las queries que copian el texto.
    palabras_q = {p for p in re.findall(r"[a-z]+", query.lower()) if len(p) > 3}
    palabras_t = set(re.findall(r"[a-z]+", texto.lower()))
    return len(palabras_q & palabras_t) / max(len(palabras_q), 1)


def tandas_anteriores(tanda):
    # Documentos y número de queries ya usados en las tandas anteriores
    docs, n = set(), 0
    for t in range(1, tanda):
        with open(CARPETA / f"queries_candidatas_t{t}.jsonl", encoding="utf-8") as f:
            for linea in f:
                q = json.loads(linea)
                docs.update(q["relevant_ids"])
                n += 1
    return docs, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tanda", type=int, required=True)
    parser.add_argument("--modelo", default="llama3.1:8b")
    args = parser.parse_args()
    clave = leer_clave()

    with open(RUTA_CHUNKS, encoding="utf-8") as f:
        fragmentos = [json.loads(linea) for linea in f]
    usados, n_previas = tandas_anteriores(args.tanda)
    print(f"Tandas anteriores: {n_previas} queries de {len(usados)} documentos (no se repiten)")

    # 1. Muestra al azar por fuente (como mucho un fragmento por documento)
    random.seed(100 + args.tanda)
    elegidos = []
    for fuente, n in POR_FUENTE.items():
        candidatos = [fr for fr in fragmentos
                      if fr["fuente"] == fuente and fr["n_palabras"] >= MIN_PALABRAS
                      and fr["doc_id"] not in usados]
        random.shuffle(candidatos)
        vistos = set()
        for fr in candidatos:
            if fr["doc_id"] not in vistos:
                elegidos.append(fr)
                vistos.add(fr["doc_id"])
            if len(vistos) == n:
                break
    print(f"{len(elegidos)} fragmentos elegidos. Modelo: {args.modelo}\n")

    # 2. Una query por fragmento (los qid siguen la numeración de las tandas anteriores)
    queries = []
    for i, fr in enumerate(elegidos, n_previas + 1):
        inicio = time.time()
        on_topic, query = leer_respuesta(
            preguntar_llm(PROMPT.format(texto=fr["texto"]), args.modelo, clave))
        queries.append({
            "qid": f"q{i:03d}",
            "query": query,
            "relevant_ids": [fr["doc_id"]],
            "source_chunk": fr["chunk_id"],
            "fuente": fr["fuente"],
            "titulo": fr["titulo"],
            "solape": round(solape(query, fr["texto"]), 2),
            "on_topic": on_topic,
        })
        aviso = "" if on_topic else "   <-- NO ES MITOLOGÍA"
        print(f"q{i:03d} [{fr['fuente']}] {fr['titulo'][:50]}  ({time.time() - inicio:.1f} s){aviso}")
        print(f"     -> {query}   (solape {queries[-1]['solape']:.2f})")

    # 3. Guardar las candidatas
    salida = CARPETA / f"queries_candidatas_t{args.tanda}.jsonl"
    with open(salida, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\nGuardadas {len(queries)} queries en {salida}")
    print(f"Marcadas como no mitología: {sum(not q['on_topic'] for q in queries)}")
    print(f"Solape medio con el fragmento: {sum(q['solape'] for q in queries) / len(queries):.2f}")


if __name__ == "__main__":
    main()