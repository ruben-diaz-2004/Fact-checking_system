"""
Crea la base de datos vectorial del Hito 2: un embedding por fragmento, guardado en Chroma.

Se indexan los MISMOS fragmentos del Hito 1 (data/processed/chunks.jsonl), que son la
unidad de indexación de todo el proyecto. Cada vector se guarda con su texto y con los
datos necesarios para citar la fuente (doc_id, título, fuente, sección, URL...).

Modelos (se puede crear una colección por modelo para compararlos):
    e5   intfloat/multilingual-e5-base   multilingüe (las afirmaciones pueden venir en
                                         español y el corpus está en inglés), 512 tokens
    bge  BAAI/bge-small-en-v1.5          solo inglés, para comparar, 512 tokens

Los modelos E5 y BGE esperan un prefijo distinto para los documentos y para las
consultas; están en MODELOS y el notebook los reutiliza.

Al texto de cada fragmento se le pone delante el título del documento, como en el
preprocesado del Hito 1 (muchos fragmentos no repiten el nombre del personaje).

Los vectores se normalizan, así que la similitud coseno y la distancia euclídea
ordenan igual; la colección se crea con distancia coseno.

Entrada:  data/processed/chunks.jsonl
Salida:   data/processed/chroma/   (base de datos de Chroma; no se sube a GitHub)

Uso (desde la carpeta principal del repositorio, con el entorno activado):
    python src/indexa_vectores.py --modelo e5 --muestra 500   # prueba rápida
    python src/indexa_vectores.py --modelo e5                 # todos los fragmentos
    python src/indexa_vectores.py --modelo bge
"""

import argparse
import json
import time
from pathlib import Path

import chromadb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CHUNKS = RAIZ / "data" / "processed" / "chunks.jsonl"
RUTA_CHROMA = RAIZ / "data" / "processed" / "chroma"

MODELOS = {
    "e5": {"nombre": "intfloat/multilingual-e5-base",
           "prefijo_doc": "passage: ", "prefijo_query": "query: "},
    "bge": {"nombre": "BAAI/bge-small-en-v1.5",
            "prefijo_doc": "", "prefijo_query": "Represent this sentence for searching relevant passages: "},
}
LOTE_CHROMA = 5000   # Chroma no admite añadir más de ~5.400 vectores de una vez


def nombre_coleccion(clave):
    return f"fragmentos_{clave}"


def texto_a_indexar(fragmento):
    return f"{fragmento['titulo']}. {fragmento['texto']}"


def metadatos(fragmento):
    # Chroma solo admite str, int, float o bool (no None)
    campos = ["doc_id", "fuente", "titulo", "tipo_fuente", "nivel", "seccion", "url", "segundo"]
    meta = {c: fragmento.get(c) for c in campos}
    return {c: ("" if v is None else v) for c, v in meta.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", choices=MODELOS, default="e5")
    parser.add_argument("--muestra", type=int, default=None, help="solo N fragmentos al azar (prueba)")
    parser.add_argument("--lote", type=int, default=64, help="fragmentos por lote en la GPU")
    args = parser.parse_args()
    config = MODELOS[args.modelo]

    with open(RUTA_CHUNKS, encoding="utf-8") as f:
        fragmentos = [json.loads(linea) for linea in f]
    if args.muestra:
        rng = np.random.default_rng(42)
        fragmentos = [fragmentos[i] for i in rng.choice(len(fragmentos), args.muestra, replace=False)]
    print(f"Fragmentos: {len(fragmentos)}")

    # 1. Modelo de embeddings (en la GPU si la hay, y en media precisión para ir más rápido)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Modelo: {config['nombre']}  ({dispositivo})")
    modelo = SentenceTransformer(config["nombre"], device=dispositivo)
    if dispositivo == "cuda":
        modelo.half()
    modelo.max_seq_length = 512

    # Comprobación: cuántos fragmentos no caben enteros en el modelo y se cortarían
    textos = [config["prefijo_doc"] + texto_a_indexar(fr) for fr in fragmentos]
    longitudes = np.array([len(ids) for ids in modelo.tokenizer(textos)["input_ids"]])
    print(f"Tokens por fragmento: mediana {np.median(longitudes):.0f}, máximo {longitudes.max()}; "
          f"se cortan {(longitudes > modelo.max_seq_length).sum()} "
          f"({100 * (longitudes > modelo.max_seq_length).mean():.1f} %)")

    # 2. Embeddings (normalizados: el producto escalar es la similitud coseno)
    inicio = time.time()
    vectores = modelo.encode(textos, batch_size=args.lote, normalize_embeddings=True,
                             show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    print(f"Embeddings: {vectores.shape} en {time.time() - inicio:.0f} s")

    # 3. Base de datos vectorial (la colección de este modelo se rehace desde cero)
    cliente = chromadb.PersistentClient(path=str(RUTA_CHROMA))
    nombre = nombre_coleccion(args.modelo) + ("_muestra" if args.muestra else "")
    if nombre in [c.name if hasattr(c, "name") else c for c in cliente.list_collections()]:
        cliente.delete_collection(nombre)
    info = {"modelo": config["nombre"], "prefijo_query": config["prefijo_query"]}
    try:
        coleccion = cliente.create_collection(nombre, metadata=info,
                                              configuration={"hnsw": {"space": "cosine"}})
    except TypeError:   # versiones antiguas de chromadb
        coleccion = cliente.create_collection(nombre, metadata={**info, "hnsw:space": "cosine"})

    for i in range(0, len(fragmentos), LOTE_CHROMA):
        lote = fragmentos[i:i + LOTE_CHROMA]
        coleccion.add(ids=[fr["chunk_id"] for fr in lote],
                      embeddings=vectores[i:i + LOTE_CHROMA].tolist(),
                      documents=[fr["texto"] for fr in lote],
                      metadatas=[metadatos(fr) for fr in lote])
    print(f"Colección '{nombre}': {coleccion.count()} vectores  ->  {RUTA_CHROMA}")

    # 4. Prueba rápida: una búsqueda
    query = "Zeus was the son of Cronus"
    q = modelo.encode([config["prefijo_query"] + query], normalize_embeddings=True).tolist()
    res = coleccion.query(query_embeddings=q, n_results=3)
    print(f"\nPrueba: '{query}'")
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        print(f"   {1 - dist:.3f}  [{meta['fuente']}] {meta['titulo']}")


if __name__ == "__main__":
    main()
