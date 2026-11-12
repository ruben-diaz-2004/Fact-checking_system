"""
Preprocesado de los fragmentos con spaCy (Hito 1).

Sigue la misma receta que la función prepare_data del notebook
Text_Vectorization_I (sección 1.3, Ejercicio 5):

    1. wrangle_text: quitar URLs y expandir contracciones (I'll -> I will)
    2. spaCy con nlp.pipe (sin parser ni NER, que no hacen falta)
    3. Lemas en minúsculas de los tokens alfabéticos cuya categoría está en
       VALID_POS y que no son stopwords (genéricas ni del dominio)

Dos cambios respecto al notebook, pensados para los nombres propios:
    - Se quitan las tildes y diéresis (Jörmungandr -> jormungandr), porque
      cada fuente lo escribe de una manera y las queries pueden venir sin ellas.
    - Al texto de cada fragmento se le pone delante el título del documento,
      porque muchos fragmentos de un artículo no repiten el nombre del dios.

Lo que NO se hace aquí (va en el notebook, porque es rápido):
    N-gramas con Phrases, diccionario, filter_extremes, TF-IDF y BM25.

Las queries deben pasar por la misma función: en el notebook se hace
    from preprocesado import preparar

Entrada:  data/processed/chunks.jsonl
Salida:   data/processed/chunks_tokens.jsonl  -> {"chunk_id", "doc_id", "tokens"}

Uso:
    python src/preprocesado.py                 # todos los fragmentos
    python src/preprocesado.py --muestra 500   # prueba con 500 al azar
"""

import argparse
import json
import random
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import contractions
import spacy
from tqdm import tqdm

CARPETA = Path("data/processed")
RUTA_CHUNKS = CARPETA / "chunks.jsonl"
RUTA_SALIDA = CARPETA / "chunks_tokens.jsonl"

VALID_POS = {"NOUN", "VERB", "ADJ", "PROPN"}

# Stopwords del dominio: ruido de los vídeos y de la estructura de los libros
SPECIFIC_STW = {
    "video", "channel", "subscribe", "patreon", "sponsor", "link", "description",
    "chapter", "book", "page", "footnote", "translator", "translation",
}

AÑADIR_TITULO = True
BATCH_SIZE = 256

# Mismo modelo y mismos componentes desactivados que en el notebook
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def wrangle_text(text):
    # 1. Quitar URLs con una expresión regular
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # 2. Expandir las contracciones (I'll -> I will)
    text = contractions.fix(text)
    return text


def quitar_tildes(palabra):
    # ö -> o, á -> a ... (letras como þ o ð no cambian)
    descompuesta = unicodedata.normalize("NFKD", palabra)
    return "".join(c for c in descompuesta if not unicodedata.combining(c))


def preparar(textos, batch_size=BATCH_SIZE, n_process=1, progreso=False):
    """Recibe una lista de textos y devuelve una lista con los tokens limpios de cada uno."""
    # 1. Limpiamos URLs y contracciones de cada texto
    textos = [wrangle_text(t) for t in textos]

    resultado = []
    # 2. Procesamos todos los textos con nlp.pipe (más rápido que un bucle)
    docs = nlp.pipe(textos, batch_size=batch_size, n_process=n_process)
    if progreso:
        docs = tqdm(docs, total=len(textos), desc="spaCy")
    for doc in docs:
        tokens = []
        for tk in doc:
            # 3. Mismos filtros que en el notebook
            if not tk.is_alpha or tk.pos_ not in VALID_POS or tk.is_stop:
                continue
            # Lema en minúsculas y sin tildes (en los nombres propios spaCy
            # deja la palabra tal cual: Hermes -> hermes, Trojans -> trojans)
            forma = quitar_tildes(tk.lemma_.lower())
            if forma in SPECIFIC_STW or len(forma) < 2:
                continue
            tokens.append(forma)
        resultado.append(tokens)
    return resultado


def texto_a_indexar(frag):
    if AÑADIR_TITULO and frag.get("titulo"):
        return frag["titulo"] + ". " + frag["texto"]
    return frag["texto"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muestra", type=int, default=None,
                        help="procesar solo N fragmentos elegidos al azar (para probar)")
    parser.add_argument("--procesos", type=int, default=1,
                        help="procesos de spaCy en paralelo (2 suele ir bien)")
    args = parser.parse_args()

    with open(RUTA_CHUNKS, encoding="utf-8") as f:
        fragmentos = [json.loads(linea) for linea in f]
    if args.muestra:
        random.seed(42)
        fragmentos = random.sample(fragmentos, args.muestra)
    print(f"{len(fragmentos)} fragmentos leídos de {RUTA_CHUNKS}")

    inicio = time.time()
    tokens = preparar([texto_a_indexar(fr) for fr in fragmentos],
                      n_process=args.procesos, progreso=True)
    print(f"Tiempo: {(time.time() - inicio) / 60:.1f} min")

    salida = RUTA_SALIDA if not args.muestra else CARPETA / "chunks_tokens_prueba.jsonl"
    with open(salida, "w", encoding="utf-8") as f:
        for fr, tk in zip(fragmentos, tokens):
            f.write(json.dumps({"chunk_id": fr["chunk_id"], "doc_id": fr["doc_id"],
                                "tokens": tk}, ensure_ascii=False) + "\n")
    print(f"Guardado en {salida}")

    # ---- Resumen para comprobar que todo tiene sentido ----
    total = sum(len(t) for t in tokens)
    frecuencias = Counter(w for t in tokens for w in t)
    vacios = sum(1 for t in tokens if not t)
    print(f"\nTokens totales: {total:,} | media por fragmento: {total / len(tokens):.0f}")
    print(f"Vocabulario (antes de filter_extremes): {len(frecuencias):,}")
    print(f"Fragmentos que se quedan vacíos: {vacios}")
    print("\n30 tokens más frecuentes:")
    print(", ".join(f"{w} ({n})" for w, n in frecuencias.most_common(30)))

    print("\n¿Sobreviven los nombres? (apariciones en el corpus)")
    nombres = ["zeus", "jupiter", "jove", "hera", "juno", "athena", "minerva",
               "hermes", "mercury", "ares", "mars", "persephone", "proserpina",
               "odin", "thor", "loki", "jormungandr", "yggdrasil", "fenrir", "valhalla"]
    print(", ".join(f"{n}: {frecuencias.get(n, 0)}" for n in nombres))

    print("\nFragmentos por fuente:", dict(Counter(fr["fuente"] for fr in fragmentos)))

    print("\nEjemplo del primer fragmento:")
    print("  TEXTO: ", texto_a_indexar(fragmentos[0])[:300], "...")
    print("  TOKENS:", tokens[0][:40])


if __name__ == "__main__":
    main()