"""
Descarga artículos de mitología de World History Encyclopedia (WHE).

WHE publica todo su contenido con licencia CC BY-NC-SA 4.0 y ofrece cada página
en Markdown añadiendo ".md" a la URL. Pide no hacer más de una petición por segundo.

Cómo se eligen los artículos (WHE no tiene categorías que podamos usar):
  1. Se descarga el índice completo del sitio (sitemap.txt).
  2. "Definiciones" (páginas tipo /Zeus/): entran si su nombre coincide con el
     título de algún artículo de nuestra lista de mitología de Wikipedia.
  3. "Artículos" (páginas tipo /article/123/titulo/): entran si su título contiene
     palabras de mitología griega, romana o nórdica, y ninguna de la lista de descarte.
  4. Las listas de descarte se ajustaron revisando a mano todos los candidatos.

Primero conviene ver solo la lista de candidatos (no descarga nada más):
    python src/descarga_whe.py --solo-lista
Y después, descargar:
    python src/descarga_whe.py

Salida:
    data/raw/whe/candidatos.csv     -> lista de páginas elegidas (para revisarla)
    data/raw/whe/paginas/*.md       -> cada página descargada, tal cual
    data/processed/whe.jsonl        -> documentos limpios (mismo formato que clasicos.jsonl)
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_RAW = RAIZ / "data" / "raw" / "whe"
CARPETA_PAGINAS = CARPETA_RAW / "paginas"
LISTA_WIKIPEDIA = RAIZ / "data" / "raw" / "wikipedia" / "lista_articulos.csv"
RUTA_SALIDA = RAIZ / "data" / "processed" / "whe.jsonl"

BASE = "https://www.worldhistory.org"
PAUSA = 1.1   # WHE pide como máximo una petición por segundo

# Páginas generales que entran siempre
SEMILLAS = ["Greek_Mythology", "Norse_Mythology", "Roman_Mythology", "Greek_Religion",
            "Roman_Religion", "Norse_Religion", "Mythology"]

# Palabras (enteras) que delatan un artículo de nuestras mitologías, en la URL del artículo.
# Se comparan palabras completas, no trozos: así "authority" no cuenta como "thor".
PALABRAS_SI = {
    "myth", "myths", "mythology", "mythological", "god", "gods", "goddess", "goddesses",
    "titan", "titans", "hero", "heroes", "olympian", "trojan", "troy", "iliad", "odyssey",
    "aeneid", "hercules", "heracles", "zeus", "hera", "athena", "athenas", "apollo", "artemis",
    "aphrodite", "dionysus", "hades", "persephone", "demeter", "poseidon", "hermes", "medusa",
    "perseus", "theseus", "minotaur", "jason", "argonauts", "odin", "thor", "loki", "freyja",
    "frigg", "baldr", "ragnarok", "valhalla", "valkyrie", "yggdrasil", "edda", "jupiter", "juno",
    "venus", "mars", "minerva", "ovid", "hesiod", "homer", "prometheus", "pandora", "eleusinian",
    "laocoon", "norse",
}
# ... y palabras que lo descartan: otras mitologías, historia, arqueología, arte, reseñas...
# (lista ajustada tras revisar a mano los candidatos el 12-nov-2026)
PALABRAS_NO = {
    "egypt", "egyptian", "mesopotamia", "sumer", "babylon", "gilgamesh", "etana", "adapa",
    "cherokee", "kiowa", "aztec", "maya", "hindu", "chinese", "japan", "celtic",
    "temple", "statues", "phidias", "milo", "review", "interview", "diet", "pets", "america",
    "greenland", "juneteenth", "alamo", "scotland", "glastonbury", "unicorn", "discovery",
    "caesar", "desecration", "menacing", "rediscovering",
    "china", "persian", "ninkasi", "alexander", "aegina",
}

# Definiciones que coinciden con algún título de Wikipedia pero no son mitología
# (personajes y lugares históricos, otras religiones, comedias...). Revisión manual del 12-nov-2026.
EXCLUIR_DEFINICIONES = {
    "mycenae", "aegina", "julius_caesar", "mark_antony", "tiryns", "epidaurus", "thera", "nemea",
    "parthenon", "erechtheion", "sounion", "delos", "naxos", "plataea", "marathon", "sparta",
    "piraeus", "corcyra", "elagabalus", "alexander_the_great", "gortyn", "isis", "osiris",
    "peloponnesian_league", "pyrrhus", "horus", "lysander", "kos", "melqart", "tegea",
    "memphis_(ancient_egypt)", "etruscan_religion", "crete", "lucius_tarquinius_superbus", "apis",
    "beowulf", "marcus_claudius_marcellus", "the_frogs", "mount_athos", "augustus",
    "colossus_of_rhodes", "yahweh", "thesmophoriazusae", "plutus_(play)", "side", "phrygia",
    "herculaneum", "argos", "persepolis", "melusine", "serapis", "empedocles",
    "archelaus_(philosopher)", "argead_dynasty", "brunhilda_of_austrasia", "croesus", "lycurgus",
    "orichalcum", "plato", "antinous", "cyrene", "cato_the_elder", "arsinoe_iv", "the_tempest",
}


def normalizar(texto):
    """'Mars_(mythology)' / 'Mars (mythology)' / 'mars' -> 'mars'."""
    texto = re.sub(r"\(.*?\)", "", texto.lower())
    texto = re.sub(r"[_\-]", " ", texto)
    return " ".join(re.sub(r"[^\w\s]", "", texto).split())


def mitologia_por_palabras(texto):
    """Asigna mitología a un artículo según las palabras de su URL."""
    t = texto.lower()
    if any(p in t for p in ["norse", "viking", "odin", "thor", "loki", "freyja", "frigg", "baldr",
                            "ragnar", "valhalla", "valkyr", "yggdrasil", "edda"]):
        return "nordica"
    if any(p in t for p in ["roman", "jupiter", "juno", "venus", "mars-", "minerva", "aeneid", "ovid"]):
        return "griega;romana"
    return "griega"


def elegir_candidatos(sesion):
    """Descarga el sitemap y elige las páginas de mitología. Devuelve un DataFrame."""
    resp = sesion.get(f"{BASE}/sitemap.txt", timeout=60)
    resp.raise_for_status()
    urls = [u.strip() for u in resp.text.split("\n") if u.strip().startswith(BASE)]
    print(f"Páginas en el índice de WHE: {len(urls)}")

    # Títulos de Wikipedia normalizados -> mitologías
    wiki = pd.read_csv(LISTA_WIKIPEDIA)
    titulos_wiki = dict(zip(wiki["titulo"].map(normalizar), wiki["mitologias"]))

    candidatos = []
    for url in urls:
        ruta = url[len(BASE):].strip("/")
        partes = ruta.split("/")
        if len(partes) == 1 and partes[0]:                      # definición: /Zeus/
            if partes[0].lower() in EXCLUIR_DEFINICIONES:
                continue
            clave = normalizar(partes[0])
            if partes[0] in SEMILLAS:
                mito = mitologia_por_palabras(partes[0])
            elif clave in titulos_wiki:
                mito = titulos_wiki[clave]
            else:
                continue
            candidatos.append({"url": url, "tipo": "definicion", "nombre": partes[0], "mitologia": mito})
        elif len(partes) == 3 and partes[0] == "article":       # artículo: /article/123/titulo/
            slug = partes[2].lower()
            palabras = set(slug.split("-"))
            if palabras & PALABRAS_SI and not palabras & PALABRAS_NO:
                candidatos.append({"url": url, "tipo": "articulo", "nombre": partes[2],
                                   "mitologia": mitologia_por_palabras(slug)})

    df = pd.DataFrame(candidatos).drop_duplicates("url").reset_index(drop=True)
    df["doc_id"] = [f"whe_{i + 1:03d}" for i in range(len(df))]
    return df


def limpiar_markdown(md):
    """
    Convierte el Markdown de WHE en texto limpio:
    quita la cabecera YAML, las imágenes, los enlaces (deja el texto), y todo lo
    que va desde "Editorial Review" o "Bibliography" (bibliografía, autor, citas, licencia).
    Los títulos de sección se pasan al formato == Sección == que entiende troceo.py.
    Devuelve (título, autor, texto).
    """
    titulo = autor = ""
    cabecera = re.match(r"^---\n(.*?)\n---\n", md, flags=re.S)
    if cabecera:
        for linea in cabecera.group(1).split("\n"):
            if linea.startswith("title:"):
                titulo = linea[6:].strip()
            elif linea.startswith("author:"):
                autor = linea[7:].strip()
        md = md[cabecera.end():]

    lineas = []
    for linea in md.split("\n"):
        if re.match(r"^#+\s*(Editorial Review|Bibliography|About the Author)", linea):
            break                                           # a partir de aquí ya no es el artículo
        if "![" in linea or linea.startswith("_Authored by"):
            continue                                        # imágenes y firma
        m = re.match(r"^(#{2,6})\s*(.+)$", linea)
        if m:                                               # título de sección
            linea = f"== {m.group(2).strip('* ')} =="
        elif linea.startswith("# "):                        # título principal: ya lo tenemos
            continue
        linea = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", linea)   # [texto](url) -> texto
        linea = re.sub(r"[*_`>]", "", linea)                       # negritas, cursivas, citas
        lineas.append(linea.strip())
    texto = re.sub(r"\n{3,}", "\n\n", "\n".join(lineas)).strip()
    return titulo, autor, texto


def main():
    parser = argparse.ArgumentParser(description="Descarga de World History Encyclopedia")
    parser.add_argument("--solo-lista", action="store_true",
                        help="Solo elige y guarda los candidatos, sin descargar las páginas")
    args = parser.parse_args()

    sesion = requests.Session()
    sesion.headers["User-Agent"] = "FactCheckingMitologia/0.2 (proyecto UC3M)"
    CARPETA_PAGINAS.mkdir(parents=True, exist_ok=True)

    # 1. Lista de candidatos
    candidatos = elegir_candidatos(sesion)
    candidatos.to_csv(CARPETA_RAW / "candidatos.csv", index=False)
    print(f"Candidatos: {len(candidatos)}  ->  {CARPETA_RAW / 'candidatos.csv'}")
    print(candidatos.groupby(["tipo", "mitologia"]).size().to_string())
    print("\nMuestra de 25 candidatos:")
    for fila in candidatos.sample(min(25, len(candidatos)), random_state=0).itertuples():
        print(f"  [{fila.tipo:10s}] {fila.nombre}")
    if args.solo_lista:
        return

    # 2. Descarga de cada página en Markdown (se salta las que ya están)
    fallidas = []
    for fila in tqdm(candidatos.itertuples(), total=len(candidatos), desc="Descargando"):
        ruta = CARPETA_PAGINAS / f"{fila.doc_id}.md"
        if ruta.exists():
            continue
        try:
            resp = sesion.get(fila.url + ".md", timeout=60)
            resp.raise_for_status()
            ruta.write_text(resp.text, encoding="utf-8")
        except requests.RequestException as e:
            fallidas.append((fila.url, str(e)))
        time.sleep(PAUSA)

    # 3. Limpieza y documentos
    docs = []
    for fila in candidatos.itertuples():
        ruta = CARPETA_PAGINAS / f"{fila.doc_id}.md"
        if not ruta.exists():
            continue
        titulo, autor, texto = limpiar_markdown(ruta.read_text(encoding="utf-8"))
        if len(texto.split()) < 30:
            continue
        docs.append({"doc_id": fila.doc_id, "titulo": titulo or fila.nombre,
                     "obra": "World History Encyclopedia", "autor": autor,
                     "tipo_fuente": "divulgativa", "mitologia": fila.mitologia,
                     "url": fila.url, "texto": texto})

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # --- Resumen ---
    palabras = pd.Series([len(d["texto"].split()) for d in docs])
    print("\n" + "=" * 60)
    print(f"Documentos: {len(docs)}  ->  {RUTA_SALIDA}")
    print(f"Descargas fallidas: {len(fallidas)}" + (" (vuelve a ejecutar para reintentar)" if fallidas else ""))
    print(f"Palabras: total {palabras.sum():,}, mediana por documento {palabras.median():.0f}")
    print("=" * 60)
    if docs:
        d = docs[0]
        print(f"\nEjemplo [{d['doc_id']}] {d['titulo']} ({d['autor']}):\n{d['texto'][:400]}...")


if __name__ == "__main__":
    main()