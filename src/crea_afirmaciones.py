"""
Crea el conjunto final de afirmaciones para evaluar la generación del sistema RAG.

1. Junta las afirmaciones de data/revision_afirmaciones.csv que no se han descartado
   en la revisión manual, con su texto final.
2. Traduce al español una parte, para comprobar el requisito de responder en el idioma
   de la pregunta (y si la búsqueda funciona con afirmaciones en español sobre un
   corpus en inglés):
     - las dos afirmaciones (verdadera y falsa) de N_PARES queries elegidas al azar,
       para poder comparar la misma afirmación en los dos idiomas;
     - N_SIN_INFO afirmaciones sin información elegidas al azar.
   Las traducciones las hace el LLM y se guardan en data/traducciones_afirmaciones.csv
   para revisarlas a mano. Ese fichero NO se sobrescribe si ya existe: después de
   revisarlo, se vuelve a ejecutar el script sin --traducir y usa las traducciones revisadas.

Cada afirmación en español es una afirmación más del conjunto, con el mismo veredicto
correcto que su original en inglés (y su id con "_es").

Entrada:  data/revision_afirmaciones.csv
          data/traducciones_afirmaciones.csv  (si existe)
Salida:   data/traducciones_afirmaciones.csv  (con --traducir, si no existe)
          data/afirmaciones.jsonl             (el conjunto final)

Uso (desde la carpeta principal del repositorio, con el entorno activado):
    python src/crea_afirmaciones.py --traducir   # la primera vez: traduce con el LLM
    python src/crea_afirmaciones.py              # después de revisar las traducciones
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3

from clasifica_wikipedia import URL, leer_clave

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RAIZ = Path(__file__).resolve().parents[1]
RUTA_REVISION = RAIZ / "data" / "revision_afirmaciones.csv"
RUTA_TRADUCCIONES = RAIZ / "data" / "traducciones_afirmaciones.csv"
RUTA_SALIDA = RAIZ / "data" / "afirmaciones.jsonl"

N_PARES = 20        # queries cuyas dos afirmaciones (verdadera y falsa) se traducen
N_SIN_INFO = 10     # afirmaciones sin información que se traducen
SEMILLA = 7

PROMPT = """Translate this claim about mythology from English into Spanish.
Translate faithfully, without adding or removing anything. Use the usual Spanish form of
proper names when there is one (Odin -> Odín, Hercules -> Hércules, Jupiter -> Júpiter);
otherwise keep the name as it is.

Answer only with JSON: {{"es": "..."}}

Claim: {texto}"""


def traducir(texto, modelo, clave, intentos=3):
    datos = {"model": modelo, "prompt": PROMPT.format(texto=texto), "stream": False,
             "format": "json", "options": {"temperature": 0}}
    if modelo.startswith("qwen3"):
        datos["think"] = False
    for intento in range(intentos):
        try:
            r = requests.post(URL, json=datos, headers={"X-API-KEY": clave}, verify=False, timeout=120)
            r.raise_for_status()
            return str(json.loads(r.json()["response"]).get("es", "")).strip()
        except Exception as e:
            if intento == intentos - 1:
                print(f"  ERROR: {e}")
                return ""
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traducir", action="store_true", help="traduce con el LLM (si no hay traducciones)")
    parser.add_argument("--modelo", default="llama3.1:8b")
    args = parser.parse_args()

    # 1. Afirmaciones aprobadas en la revisión
    revision = pd.read_csv(RUTA_REVISION, dtype=str).fillna("")
    assert revision["decision"].isin(["mantener", "editar", "descartar"]).all(), "Hay decisiones sin rellenar"
    finales = revision[revision["decision"] != "descartar"].copy()
    assert (finales["afirmacion_final"] != "").all(), "Hay afirmaciones aprobadas sin afirmacion_final"
    print(f"Revisión: {len(revision)} candidatas -> {len(finales)} aprobadas")
    print(finales["etiqueta"].value_counts().to_string())

    # 2. Traducciones al español
    if RUTA_TRADUCCIONES.exists():
        traducciones = pd.read_csv(RUTA_TRADUCCIONES, dtype=str).fillna("")
        print(f"\nTraducciones: {len(traducciones)} (de {RUTA_TRADUCCIONES.name})")
    elif args.traducir:
        rng = np.random.default_rng(SEMILLA)
        # Queries con sus dos afirmaciones aprobadas
        pares = finales[finales["origen"] == "fragmento"].groupby("qid")["etiqueta"].nunique()
        qids = rng.choice(sorted(pares[pares == 2].index), N_PARES, replace=False)
        sin_info = finales[finales["etiqueta"] == "sin_informacion"]["aid"].tolist()
        elegidas = finales[finales["qid"].isin(qids) |
                           finales["aid"].isin(rng.choice(sin_info, N_SIN_INFO, replace=False))]
        clave = leer_clave()
        print(f"\nTraduciendo {len(elegidas)} afirmaciones con {args.modelo}...")
        filas = []
        for _, a in elegidas.iterrows():
            filas.append({"aid": a["aid"], "etiqueta": a["etiqueta"], "afirmacion_en": a["afirmacion_final"],
                          "afirmacion_es": traducir(a["afirmacion_final"], args.modelo, clave)})
        traducciones = pd.DataFrame(filas)
        traducciones.to_csv(RUTA_TRADUCCIONES, index=False)
        print(f"Guardadas en {RUTA_TRADUCCIONES}: revísalas antes de usarlas")
    else:
        traducciones = pd.DataFrame(columns=["aid", "afirmacion_es"])
        print("\nSin traducciones (usa --traducir la primera vez)")

    # 3. Conjunto final: las aprobadas en inglés y sus traducciones
    registros = []
    for _, a in finales.iterrows():
        base = {"etiqueta": a["etiqueta"], "origen": a["origen"], "qid": a["qid"],
                "relevant_ids": [a["relevant_id"]] if a["relevant_id"] else [],
                "source_chunk": a["source_chunk"], "fuente": a["fuente"]}
        registros.append({"aid": a["aid"], "afirmacion": a["afirmacion_final"], "idioma": "en", **base})
        traduccion = traducciones.loc[traducciones["aid"] == a["aid"], "afirmacion_es"]
        if len(traduccion) and traduccion.iloc[0]:
            registros.append({"aid": f"{a['aid']}_es", "afirmacion": traduccion.iloc[0], "idioma": "es", **base})

    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tabla = pd.DataFrame(registros)
    print(f"\n{len(tabla)} afirmaciones -> {RUTA_SALIDA}")
    print(pd.crosstab(tabla["etiqueta"], tabla["idioma"], margins=True, margins_name="total").to_string())


if __name__ == "__main__":
    main()
