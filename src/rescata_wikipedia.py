"""
Segunda pasada sobre los artículos que clasifica_wikipedia.py dejó fuera: rescata
los que en realidad son de nuestras mitologías y decide la clasificación final.

Al revisar a mano lo que quitaba la primera pasada vimos cuatro tipos de error:
lo antiguo tomado por "histórico" o "moderno" (templos, cultos, tragedias, himnos),
personajes míticos con un asteroide que lleva su nombre, leyenda nórdica tomada
por folklore, y algún personaje mítico tomado por persona histórica.

Se probaron dos rescates, solo sobre los artículos que quedaron fuera:
  1. Regla: si la primera frase dice "Greek / Roman / Norse... mythology / religion /
     legend", se rescata.
  2. Una segunda pregunta al LLM, centrada en esos cuatro tipos de error.
     RESULTADO NEGATIVO: el LLM rescata casi todo lo que menciona un mito (449 de 675),
     incluidas óperas, cuadros y asteroides, y devuelve más de la mitad del ruido de la
     validación. Se descarta: sus respuestas se guardan y se miden, pero no se usan.

Decisión final: dentro si la primera pasada lo dejó dentro o si lo rescata la regla.
Las etiquetas hechas a mano (data/etiquetas_wikipedia.csv) mandan sobre todo lo demás.

Además escribe los artículos rescatados por la regla que no están etiquetados a mano,
para revisarlos (la regla también se equivoca: rescata, por ejemplo, novelas modernas
cuya primera frase menciona la mitología griega).

Se mide el acierto de "solo primera pasada" y de "primera pasada + rescate", por
conjunto etiquetado. OJO: el rescate se diseñó viendo los errores de los conjuntos
"largos" y "azar", así que en ellos la medida es optimista; "validacion" es más
independiente y sirve sobre todo para comprobar que el rescate no devuelve el ruido.

Entrada:  data/processed/clasificacion_wikipedia.csv
          data/raw/wikipedia/articulos.jsonl  (los textos se leen de aquí y no de wikipedia.jsonl,
                                               para poder repetir el rescate después de aplicar el filtro)
          data/etiquetas_wikipedia.csv
Salida:   data/processed/rescate_wikipedia.csv            (respuestas del LLM, para retomar)
          data/processed/clasificacion_final_wikipedia.csv  (decisión final de cada artículo)
          data/processed/revision_regla.md                 (rescates de la regla, para revisar)
          data/wikipedia_excluidos.csv                     (artículos que quedan fuera, con el motivo;
                                                            se sube a GitHub y lo usa procesa_wikipedia.py)

Uso:
    python src/rescata_wikipedia.py            (usa las respuestas del LLM ya guardadas)
    python src/rescata_wikipedia.py --llm      (pide al LLM las que falten)
"""

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import urllib3

from clasifica_wikipedia import URL, leer_clave, principio

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RAIZ = Path(__file__).resolve().parents[1]
PROCESSED = RAIZ / "data" / "processed"
RUTA_ETAPA1 = PROCESSED / "clasificacion_wikipedia.csv"
RUTA_RESCATE = PROCESSED / "rescate_wikipedia.csv"
RUTA_FINAL = PROCESSED / "clasificacion_final_wikipedia.csv"
RUTA_ETIQUETAS = RAIZ / "data" / "etiquetas_wikipedia.csv"
RUTA_EXCLUIDOS = RAIZ / "data" / "wikipedia_excluidos.csv"

# "Greek mythology", "ancient Roman religion", "Norse myth", "Germanic heroic legend"...
REGLA = re.compile(r"\b(greek|roman|norse|nordic|germanic|greco-roman)\s+(heroic\s+)?"
                   r"(mythology|religion|legend|legends|myth|myths)\b", re.I)

PROMPT = """This Wikipedia article was flagged as out of scope for a fact-checking system about Greek, Roman and Norse mythology.
Check whether it should be RESCUED. Answer RESCUE if it is any of these:
1. a character, creature, place, object or episode of Greek, Roman or Norse myth or legend
   (even if the article also mentions an asteroid, a product or a modern work named after it)
2. an ancient Greek or Roman text, play, poem or hymn (Homer, Hesiod, Aeschylus, Euripides, Aristophanes, Ovid, Virgil...) or an ancient religious inscription
3. an ancient Greek or Roman temple, sanctuary, cult, festival, ritual or priesthood
4. a figure or theme of the Norse/Icelandic sagas, the Eddas or Germanic heroic legend

Answer OUT for: works made after antiquity (paintings, operas, novels, films, games...) even if they retell a myth;
things that only carry a mythological name; historical people and events; other religions and mythologies
(Egyptian, Christian, Islamic, Celtic, Arthurian, Slavic, Asian...); later folklore and fairy tales.

Answer only with JSON: {{"decision": "RESCUE" or "OUT", "reason": "a few words"}}

Title: {titulo}
Article (beginning): {texto}"""


def primera_frase(texto):
    return re.split(r"(?<=[.!?])\s", principio(texto, 60), maxsplit=1)[0]


def preguntar(doc, modelo, clave, intentos=3):
    datos = {"model": modelo, "stream": False, "format": "json", "options": {"temperature": 0},
             "prompt": PROMPT.format(titulo=doc["titulo"], texto=principio(doc["texto"]))}
    for intento in range(intentos):
        try:
            r = requests.post(URL, json=datos, headers={"X-API-KEY": clave}, verify=False, timeout=120)
            r.raise_for_status()
            resp = json.loads(r.json()["response"])
            return {"doc_id": doc["doc_id"],
                    "rescate_llm": str(resp.get("decision", "")).upper() == "RESCUE",
                    "motivo_rescate": str(resp.get("reason", ""))}
        except Exception as e:
            if intento == intentos - 1:
                print(f"  ERROR con {doc['doc_id']} ({doc['titulo']}): {e}")
                return None
            time.sleep(5)


def comparar(df, columna):
    # Aciertos de una decisión frente a la etiqueta manual, en un conjunto
    fuera_manual = df["etiqueta"] == "fuera"
    fuera_pred = df[columna] == "fuera"
    return {"artículos": len(df),
            "ruido quitado": f"{(fuera_manual & fuera_pred).sum()} de {fuera_manual.sum()}",
            "bueno perdido": f"{(~fuera_manual & fuera_pred).sum()} de {(~fuera_manual).sum()}",
            "acierto %": round(100 * (fuera_manual == fuera_pred).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="llama3.1:8b")
    parser.add_argument("--hilos", type=int, default=1)
    parser.add_argument("--llm", action="store_true", help="pedir al LLM los rescates que falten")
    args = parser.parse_args()

    docs = {}
    with open(RAIZ / "data" / "raw" / "wikipedia" / "articulos.jsonl", encoding="utf-8") as f:
        for a in map(json.loads, f):
            docs[f"wiki_{a['pageid']}"] = {"doc_id": f"wiki_{a['pageid']}", "titulo": a["titulo"],
                                           "texto": a.get("texto") or ""}
    etapa1 = pd.read_csv(RUTA_ETAPA1)[["doc_id", "titulo", "etiqueta", "motivo"]]
    etapa1 = etapa1.rename(columns={"etiqueta": "etapa1", "motivo": "motivo_etapa1"})

    # 1. Regla sobre la primera frase
    fuera1 = etapa1["etapa1"] == "fuera"
    etapa1["rescate_regla"] = False
    etapa1.loc[fuera1, "rescate_regla"] = etapa1.loc[fuera1, "doc_id"].map(
        lambda d: bool(REGLA.search(primera_frase(docs.get(d, {"texto": ""})["texto"]))))
    print(f"Fuera en la primera pasada: {fuera1.sum()}. Rescatados por la regla: {etapa1['rescate_regla'].sum()}")

    # 2. LLM sobre los que siguen fuera (se guarda según llega y se puede retomar)
    pendientes_ids = etapa1.loc[fuera1 & ~etapa1["rescate_regla"], "doc_id"].tolist()
    hechos = set(pd.read_csv(RUTA_RESCATE)["doc_id"]) if RUTA_RESCATE.exists() else set()
    pendientes = [docs[d] for d in pendientes_ids if d not in hechos and d in docs] if args.llm else []
    clave = leer_clave() if pendientes else None
    print(f"Para el LLM: {len(pendientes_ids)} ({len(pendientes)} pendientes)")
    nuevo = not RUTA_RESCATE.exists()
    with open(RUTA_RESCATE, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=["doc_id", "rescate_llm", "motivo_rescate"])
        if nuevo:
            escritor.writeheader()
        with ThreadPoolExecutor(max_workers=args.hilos) as ejecutor:
            futuros = [ejecutor.submit(preguntar, d, args.modelo, clave) for d in pendientes]
            for i, futuro in enumerate(as_completed(futuros), 1):
                if (resultado := futuro.result()):
                    escritor.writerow(resultado)
                    f.flush()
                if i % 100 == 0 or i == len(pendientes):
                    print(f"  {i}/{len(pendientes)}")

    rescate = pd.read_csv(RUTA_RESCATE)
    df = etapa1.merge(rescate, on="doc_id", how="left")
    df["rescate_llm"] = df["rescate_llm"].astype("boolean").fillna(False).astype(bool)
    df["motivo_rescate"] = df["motivo_rescate"].fillna("")
    fuera_etapa1 = df["etapa1"] == "fuera"
    df["con_regla"] = "dentro"
    df.loc[fuera_etapa1 & ~df["rescate_regla"], "con_regla"] = "fuera"
    df["con_regla_y_llm"] = "dentro"
    df.loc[fuera_etapa1 & ~df["rescate_regla"] & ~df["rescate_llm"], "con_regla_y_llm"] = "fuera"
    df["automatica"] = df["con_regla"]      # el rescate del LLM se descarta
    print(f"Rescatados por el LLM (no se usan): {df['rescate_llm'].sum()}")

    # 3. Las etiquetas manuales mandan
    manual = pd.read_csv(RUTA_ETIQUETAS)[["doc_id", "etiqueta", "conjunto", "motivo"]]
    manual = manual.rename(columns={"motivo": "motivo_manual"})
    df = df.merge(manual, on="doc_id", how="left")
    df["final"] = df["etiqueta"].fillna(df["automatica"])

    # Medida, por conjunto etiquetado
    print("\nAcierto frente a las etiquetas manuales:")
    filas = []
    for conjunto, g in df[df["etiqueta"].notna()].groupby("conjunto"):
        filas.append({"conjunto": conjunto, "decisión": "solo 1ª pasada", **comparar(g, "etapa1")})
        filas.append({"conjunto": conjunto, "decisión": "1ª + regla", **comparar(g, "con_regla")})
        filas.append({"conjunto": conjunto, "decisión": "1ª + regla + LLM", **comparar(g, "con_regla_y_llm")})
    print(pd.DataFrame(filas).to_string(index=False))

    errores = df[df["etiqueta"].notna() & (df["etiqueta"] != df["automatica"])]
    print("\nErrores que quedan con '1ª + regla' (los corrige la etiqueta manual):")
    for r in errores.itertuples():
        print(f"  [{r.conjunto}] {r.titulo[:45]:45s} manual={r.etiqueta:6s} auto={r.automatica}")

    # 4. Resultado final (tamaño en palabras, a partir de la descarga original: los fragmentos
    #    de los excluidos ya no existen si el filtro se ha aplicado antes)
    df["palabras"] = df["doc_id"].map(lambda d: len(docs.get(d, {"texto": ""})["texto"].split()))
    fuera = df[df["final"] == "fuera"].copy()
    print(f"\nFINAL: {len(fuera)} artículos fuera de {len(df)} ({100 * len(fuera) / len(df):.1f} %), "
          f"{fuera['palabras'].sum():,} palabras de {df['palabras'].sum():,} "
          f"({100 * fuera['palabras'].sum() / df['palabras'].sum():.1f} %, texto sin limpiar)")

    # Rescates de la regla que nadie ha revisado, para revisarlos a mano
    por_revisar = df[df["rescate_regla"] & df["etiqueta"].isna()].sort_values("palabras", ascending=False)
    with open(PROCESSED / "revision_regla.md", "w", encoding="utf-8") as f:
        for r in por_revisar.itertuples():
            f.write(f"## {r.doc_id} | {r.titulo} | {r.palabras} palabras\n\n"
                    f"{primera_frase(docs.get(r.doc_id, {'texto': ''})['texto'])}\n\n---\n\n")
    print(f"\nRescatados por la regla sin etiqueta manual: {len(por_revisar)} "
          f"-> {PROCESSED / 'revision_regla.md'}")

    columnas = ["doc_id", "titulo", "etapa1", "motivo_etapa1", "rescate_regla", "rescate_llm",
                "motivo_rescate", "con_regla", "con_regla_y_llm", "etiqueta", "final", "palabras"]
    df[columnas].rename(columns={"etiqueta": "manual"}).to_csv(RUTA_FINAL, index=False)
    print(f"\nGuardado en {RUTA_FINAL}")

    # Lista de excluidos (pequeña y reproducible): la usa procesa_wikipedia.py
    fuera["decidido_por"] = fuera["etiqueta"].notna().map({True: "revisión manual", False: "clasificador (LLM)"})
    manual_ok = fuera["motivo_manual"].fillna("").str.strip() != ""
    fuera["motivo"] = fuera["motivo_manual"].where(manual_ok, fuera["motivo_etapa1"]).fillna("")
    fuera[["doc_id", "titulo", "decidido_por", "motivo", "palabras"]].sort_values("titulo").to_csv(
        RUTA_EXCLUIDOS, index=False)
    print(f"Excluidos: {len(fuera)} -> {RUTA_EXCLUIDOS}")
    print(fuera["decidido_por"].value_counts().to_string())


if __name__ == "__main__":
    main()