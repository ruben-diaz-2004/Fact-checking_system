"""
Clasifica los artículos de Wikipedia como "dentro" o "fuera" de tema con el LLM de la UC3M.

Criterio (opción A, el mismo de data/etiquetas_wikipedia.csv):
    DENTRO: trata de los mitos en sí: personajes, relatos, lugares, culto,
            textos antiguos y los estudios sobre ellos.
    FUERA:  obras modernas inspiradas en mitos, cosas que solo llevan el nombre
            de un dios, personas históricas, folklore posterior y otras mitologías.

Al LLM se le pasa el título y el principio del artículo (la introducción, hasta
unas 250 palabras): lo mismo que se usó para etiquetar a mano.

Dos modos:
    --validacion   solo los 100 artículos etiquetados a mano, y mide cuánto acierta
    (sin opción)   todos los artículos de Wikipedia (unas 4.900 llamadas)

Las respuestas se guardan en un CSV según llegan: se puede interrumpir con Ctrl+C
y retomar, y lo ya clasificado en la validación no se vuelve a pedir.

La clave de la API se lee del archivo .env:  UC3M_API_KEY=sk-...

Entrada:  data/processed/wikipedia.jsonl
          data/etiquetas_wikipedia.csv   (para --validacion)
Salida:   data/processed/clasificacion_wikipedia.csv

Uso:
    python src/clasifica_wikipedia.py --validacion
    python src/clasifica_wikipedia.py --hilos 3
"""

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RAIZ = Path(__file__).resolve().parents[1]
URL = "https://yiyuan.tsc.uc3m.es/api/generate"
RUTA_WIKI = RAIZ / "data" / "processed" / "wikipedia.jsonl"
RUTA_ETIQUETAS = RAIZ / "data" / "etiquetas_wikipedia.csv"
RUTA_SALIDA = RAIZ / "data" / "processed" / "clasificacion_wikipedia.csv"
PALABRAS_INTRO = 250
CAMPOS = ["doc_id", "titulo", "etiqueta", "motivo"]

PROMPT = """You are filtering a document collection for a fact-checking system about Greek, Roman and Norse mythology.
Decide whether this Wikipedia article is IN or OUT of scope.

IN: the article is about the myths themselves or their ancient context:
- gods, heroes, mythical characters and creatures, myths and legends
- mythical or ancient places, cults, rituals, sanctuaries, oracles, ancient religious objects
- ancient texts and authors that tell the myths (Homer, Hesiod, Ovid, Virgil, Catullus, the Eddas, the sagas...), their characters and episodes, and scholarship about them
- Germanic heroic legend closely tied to Norse myth (Völsung cycle, Beowulf)

OUT:
- modern works inspired by myths: paintings, sculptures, fountains, operas, plays, novels, films, comics, games (even if the article retells the myth)
- things that only carry a mythological name: asteroids, buildings, scientific theories, given names, animal breeds, companies
- real historical people (emperors, saints, kings) and historical events
- later folklore and fairy tales (Grimm tales, ghosts, household spirits, medieval ballads)
- other mythologies and religions (Arabian, Celtic, Egyptian, Christian...) and general concepts shared by many religions

Answer only with JSON: {{"label": "IN" or "OUT", "reason": "a few words"}}

Title: {titulo}
Article (beginning): {texto}"""


def leer_clave():
    for linea in (RAIZ / ".env").read_text(encoding="utf-8").splitlines():
        if linea.startswith("UC3M_API_KEY="):
            return linea.split("=", 1)[1].strip()
    raise SystemExit("No encuentro UC3M_API_KEY en el archivo .env")


def principio(texto, n=PALABRAS_INTRO):
    # Igual que en exporta_validacion_wikipedia.py
    intro = texto.split("\n== ")[0]
    if len(intro.split()) < 60:
        intro = texto
    palabras = intro.split()
    return " ".join(palabras[:n]) + (" [...]" if len(palabras) > n else "")


def clasificar(doc, modelo, clave, intentos=3):
    datos = {"model": modelo, "stream": False, "format": "json",
             "prompt": PROMPT.format(titulo=doc["titulo"], texto=principio(doc["texto"])),
             "options": {"temperature": 0}}
    if modelo.startswith("qwen3"):
        datos["think"] = False
    for intento in range(intentos):
        try:
            r = requests.post(URL, json=datos, headers={"X-API-KEY": clave},
                              verify=False, timeout=120)
            r.raise_for_status()
            respuesta = json.loads(r.json()["response"])
            etiqueta = "fuera" if str(respuesta.get("label", "")).upper() == "OUT" else "dentro"
            return {"doc_id": doc["doc_id"], "titulo": doc["titulo"],
                    "etiqueta": etiqueta, "motivo": str(respuesta.get("reason", ""))}
        except Exception as e:
            if intento == intentos - 1:
                print(f"  ERROR con {doc['doc_id']} ({doc['titulo']}): {e}")
                return None
            time.sleep(5)


def evaluar(clasificados):
    # Compara lo que dice el LLM con las etiquetas hechas a mano
    manual = pd.read_csv(RUTA_ETIQUETAS)
    df = manual.merge(clasificados[["doc_id", "etiqueta", "motivo"]], on="doc_id",
                      suffixes=("_manual", "_llm"))
    print(f"\nArtículos evaluados: {len(df)}")
    print("\nTabla (filas: etiqueta a mano, columnas: LLM):")
    print(pd.crosstab(df["etiqueta_manual"], df["etiqueta_llm"], margins=True,
                      margins_name="total").to_string())

    fuera = df["etiqueta_manual"] == "fuera"
    llm_fuera = df["etiqueta_llm"] == "fuera"
    print(f"\nAcierto global: {100 * (df['etiqueta_manual'] == df['etiqueta_llm']).mean():.0f} %")
    print(f"Ruido detectado (de los 'fuera', cuántos quita): "
          f"{(fuera & llm_fuera).sum()} de {fuera.sum()}")
    print(f"Contenido bueno perdido (de los 'dentro', cuántos quitaría): "
          f"{(~fuera & llm_fuera).sum()} de {(~fuera).sum()}")

    errores = df[df["etiqueta_manual"] != df["etiqueta_llm"]]
    print("\nErrores:")
    for r in errores.itertuples():
        print(f"  {r.vid} {r.titulo[:40]:40s} a mano={r.etiqueta_manual:6s} "
              f"LLM={r.etiqueta_llm:6s} ({r.motivo_llm})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validacion", action="store_true",
                        help="clasificar solo los artículos etiquetados a mano y medir el acierto")
    parser.add_argument("--modelo", default="llama3.1:8b")
    parser.add_argument("--hilos", type=int, default=1,
                        help="peticiones en paralelo (3 suele ir bien; no abusar del servidor)")
    args = parser.parse_args()
    clave = leer_clave()

    with open(RUTA_WIKI, encoding="utf-8") as f:
        docs = [json.loads(linea) for linea in f]
    if args.validacion:
        ids = set(pd.read_csv(RUTA_ETIQUETAS)["doc_id"])
        docs = [d for d in docs if d["doc_id"] in ids]

    # Lo ya clasificado no se repite
    hechos = set()
    if RUTA_SALIDA.exists():
        hechos = set(pd.read_csv(RUTA_SALIDA)["doc_id"])
    pendientes = [d for d in docs if d["doc_id"] not in hechos]
    print(f"{len(docs)} artículos, {len(docs) - len(pendientes)} ya clasificados, "
          f"{len(pendientes)} pendientes. Modelo: {args.modelo}, hilos: {args.hilos}")

    nuevo = not RUTA_SALIDA.exists()
    inicio = time.time()
    with open(RUTA_SALIDA, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS)
        if nuevo:
            escritor.writeheader()
        with ThreadPoolExecutor(max_workers=args.hilos) as ejecutor:
            futuros = [ejecutor.submit(clasificar, d, args.modelo, clave) for d in pendientes]
            for i, futuro in enumerate(as_completed(futuros), 1):
                resultado = futuro.result()
                if resultado:
                    escritor.writerow(resultado)
                    f.flush()
                if i % 50 == 0 or i == len(pendientes):
                    minutos = (time.time() - inicio) / 60
                    resto = minutos / i * (len(pendientes) - i)
                    print(f"  {i}/{len(pendientes)}  ({minutos:.1f} min, quedan ~{resto:.0f} min)")

    clasificados = pd.read_csv(RUTA_SALIDA)
    clasificados = clasificados[clasificados["doc_id"].isin({d["doc_id"] for d in docs})]
    print(f"\nClasificados: {len(clasificados)}")
    print(clasificados["etiqueta"].value_counts().to_string())
    if args.validacion:
        evaluar(clasificados)


if __name__ == "__main__":
    main()