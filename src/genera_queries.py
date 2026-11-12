"""
Genera las queries candidatas del Hito 1 con el LLM de la UC3M.

Pasos:
    1. Elige fragmentos al azar, un número fijo por fuente, para que no salgan
       casi todos de Wikipedia (que es el 69 % de los fragmentos).
    2. Para cada fragmento pide al LLM la query que escribiría alguien que
       buscara ese fragmento, sin copiar sus frases (como en el notebook).
    3. Guarda las candidatas para revisarlas a mano. El documento de origen
       es el relevante: relevant_ids = [doc_id].

Se generan más de 30 (45) porque en la revisión se descartarán algunas.

La clave de la API se lee del archivo .env (que no se sube a GitHub):
    UC3M_API_KEY=sk-...

Entrada:  data/processed/chunks.jsonl
Salida:   data/processed/queries_candidatas.jsonl

Uso:
    python src/genera_queries.py
    python src/genera_queries.py --modelo qwen3:8b
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
RUTA_CHUNKS = Path("data/processed/chunks.jsonl")
RUTA_SALIDA = Path("data/processed/queries_candidatas.jsonl")

# Fragmentos a elegir de cada fuente (45 en total)
POR_FUENTE = {"wikipedia": 15, "clasicos": 12, "youtube": 10, "whe": 8}
MIN_PALABRAS = 150    # fragmentos muy cortos dan queries muy pobres
SEMILLA = 42

PROMPT = """You will receive a passage about Greek, Roman or Norse mythology.
Write a short search query (8-20 words) that a user looking for THIS passage could type.
Describe its content with your own words: do not copy phrases from the passage.
You may use at most one proper name (a god, hero, place or work), only if it is essential.
Answer with the query only: no quotes, no explanations.

Passage:
{texto}"""


def leer_clave():
    for linea in Path(".env").read_text(encoding="utf-8").splitlines():
        if linea.startswith("UC3M_API_KEY="):
            return linea.split("=", 1)[1].strip()
    raise SystemExit("No encuentro UC3M_API_KEY en el archivo .env")


def preguntar_llm(prompt, modelo, clave):
    datos = {"model": modelo, "prompt": prompt, "stream": False,
             "options": {"temperature": 0.3}}
    if modelo.startswith("qwen3"):
        datos["think"] = False          # qwen3 "piensa" en voz alta si no se le dice
    r = requests.post(URL, json=datos, headers={"X-API-KEY": clave},
                      verify=False, timeout=120)
    r.raise_for_status()
    return r.json()["response"]


def limpiar_query(texto):
    # Primera línea no vacía, sin comillas ni "Query:" delante
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.S)
    linea = next((l for l in texto.splitlines() if l.strip()), "")
    linea = re.sub(r"^(search )?query:\s*", "", linea.strip(), flags=re.I)
    return linea.strip(" \"'")


def solape(query, texto):
    # Proporción de palabras de la query (de más de 3 letras) que están en el fragmento.
    # Sirve para detectar en la revisión las queries que copian el texto.
    palabras_q = {p for p in re.findall(r"[a-z]+", query.lower()) if len(p) > 3}
    palabras_t = set(re.findall(r"[a-z]+", texto.lower()))
    return len(palabras_q & palabras_t) / max(len(palabras_q), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="llama3.1:8b")
    args = parser.parse_args()
    clave = leer_clave()

    with open(RUTA_CHUNKS, encoding="utf-8") as f:
        fragmentos = [json.loads(linea) for linea in f]

    # 1. Muestra al azar por fuente (como mucho un fragmento por documento)
    random.seed(SEMILLA)
    elegidos = []
    for fuente, n in POR_FUENTE.items():
        candidatos = [fr for fr in fragmentos
                      if fr["fuente"] == fuente and fr["n_palabras"] >= MIN_PALABRAS]
        random.shuffle(candidatos)
        vistos = set()
        for fr in candidatos:
            if fr["doc_id"] not in vistos:
                elegidos.append(fr)
                vistos.add(fr["doc_id"])
            if len(vistos) == n:
                break
    print(f"{len(elegidos)} fragmentos elegidos. Modelo: {args.modelo}\n")

    # 2. Una query por fragmento
    queries = []
    for i, fr in enumerate(elegidos, 1):
        inicio = time.time()
        respuesta = preguntar_llm(PROMPT.format(texto=fr["texto"]), args.modelo, clave)
        query = limpiar_query(respuesta)
        queries.append({
            "qid": f"q{i:02d}",
            "query": query,
            "relevant_ids": [fr["doc_id"]],
            "source_chunk": fr["chunk_id"],
            "fuente": fr["fuente"],
            "titulo": fr["titulo"],
            "solape": round(solape(query, fr["texto"]), 2),
        })
        print(f"q{i:02d} [{fr['fuente']}] {fr['titulo'][:50]}  ({time.time() - inicio:.1f} s)")
        print(f"     -> {query}   (solape {queries[-1]['solape']:.2f})")

    # 3. Guardar las candidatas
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\nGuardadas {len(queries)} queries en {RUTA_SALIDA}")
    media = sum(q["solape"] for q in queries) / len(queries)
    print(f"Solape medio con el fragmento: {media:.2f}")


if __name__ == "__main__":
    main()