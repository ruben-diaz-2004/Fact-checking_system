"""
Limpia y filtra los artículos de Wikipedia descargados por descarga_wikipedia.py.

Limpieza:
  - Se quitan las secciones de relleno (References, Notes, See also, External links,
    Bibliography...) y las de cultura popular (In popular culture, adaptaciones...),
    con todas sus subsecciones.
Filtros (en este orden):
  0. Artículos fuera de tema según la clasificación revisada (data/wikipedia_excluidos.csv,
     generado por clasifica_wikipedia.py + rescata_wikipedia.py + revisión manual) y
     artículos cuyo título es de cultura popular ("X in popular culture", "Thor in comics"...).
  1. Listas ("List of...") y desambiguaciones que no son de mitología.
     (Las de mitología, "In Greek mythology, the name X may refer to...", se quedan.)
  2. Tema: el primer párrafo debe hablar de mitos, dioses, héroes, leyendas, tragedias...
  3. Otras mitologías (acadia, egipcia, celta...) que no mencionan las nuestras.
  4. Longitud: al menos MIN_PALABRAS palabras después de limpiar.

Entrada:  data/raw/wikipedia/articulos.jsonl
          data/wikipedia_excluidos.csv   (si existe)
Salida:   data/processed/wikipedia.jsonl            (mismo formato que clasicos.jsonl)
          data/raw/wikipedia/descartados_procesado.csv (qué se descartó y por qué)

Uso (desde la carpeta principal del repositorio, con el entorno activado):
    python src/procesa_wikipedia.py
"""

import json
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_ENTRADA = RAIZ / "data" / "raw" / "wikipedia" / "articulos.jsonl"
RUTA_SALIDA = RAIZ / "data" / "processed" / "wikipedia.jsonl"
RUTA_DESCARTADOS = RAIZ / "data" / "raw" / "wikipedia" / "descartados_procesado.csv"
RUTA_EXCLUIDOS = RAIZ / "data" / "wikipedia_excluidos.csv"

MIN_PALABRAS = 50     # los personajes menores suelen tener 50-100 palabras y son datos útiles

# Secciones que se quitan enteras (se compara el título de la sección en minúsculas)
SECCIONES_RELLENO = {
    "references", "notes", "see also", "external links", "further reading", "bibliography",
    "sources", "citations", "footnotes", "works cited", "notes and references",
    "references and notes", "references and sources", "gallery", "primary sources",
    "secondary sources", "explanatory notes", "general references", "cited sources",
}
# Secciones de cultura popular: basta con que el título contenga alguna de estas expresiones
SECCIONES_CULTURA_POP = ["popular culture", "modern culture", "in fiction", "in film",
                         "in media", "video game", "modern depictions", "modern references",
                         "cultural references", "adaptations", "in music", "in television",
                         "in comics", "modern adaptations", "modern use", "modern usage",
                         # añadidas tras el diagnóstico de secciones (diagnostico.py)
                         "film", "comics", "other media", "modern media", "social media",
                         "animated adaptation", "television"]

# Títulos de cultura popular (encontrados con diagnostico.py)
TITULO_CULTURA_POP = re.compile(r"in popular culture|in fiction|in comics|cultural depictions|"
                                r"\bin culture$|\(webcomic\)", re.IGNORECASE)

# Palabras que el primer párrafo debe contener para considerar el artículo de mitología
TEMA = re.compile(
    r"\b(myths?|mytholog\w*|legends?|legendary|gods?|goddess\w*|deit(y|ies)|divine|divinit\w*|"
    r"heroe?s?|heroine|demigods?|sagas?|eddas?|eddic|homer\w*|hesiod\w*|ovid|iliad|odyssey|"
    r"aeneid|nymphs?|titans?|giants?|giantess|j[öo]tunn?|jotnar|oracles?|cults?|epithets?|"
    r"monsters?|creatures?|valkyries?|norns?|muses?|centaurs?|satyrs?|amazons?|argonauts?|"
    r"trag(edy|edies|ic)|euripides|sophocles|aeschylus|hymns?)\b",
    re.IGNORECASE)

# Otras mitologías: si la entradilla las menciona y no menciona las nuestras, fuera
OTRAS = re.compile(r"\b(akkadian|sumerian|mesopotamian|babylonian|assyrian|egyptian|hindu|vedic|"
                   r"celtic|irish|welsh|slavic|baltic|finnish|chinese|japanese|aztec|mayan|"
                   r"hittite|canaanite|phoenician|persian|zoroastrian|armenian|georgian)\b",
                   re.IGNORECASE)
NUESTRAS = re.compile(r"\b(greek|greece|roman|rome|latin|norse|germanic|scandinavian|icelandic|"
                      r"old norse|homer\w*|hesiod|ovid|virgil|edda\w*|olymp\w*)\b", re.IGNORECASE)

CABECERA = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$")


def quitar_secciones(texto):
    """Quita las secciones de relleno y de cultura popular (con sus subsecciones)."""
    lineas, saltando_nivel = [], None
    for linea in texto.split("\n"):
        m = CABECERA.match(linea.strip())
        if m:
            nivel, titulo = len(m.group(1)), m.group(2).lower().strip()
            # Una cabecera del mismo nivel o superior termina la sección que se estaba saltando
            if saltando_nivel is not None and nivel <= saltando_nivel:
                saltando_nivel = None
            if saltando_nivel is None and (titulo in SECCIONES_RELLENO or
                                           any(p in titulo for p in SECCIONES_CULTURA_POP)):
                saltando_nivel = nivel
        if saltando_nivel is None:
            lineas.append(linea)
    # Quita cabeceras que se hayan quedado sin contenido y líneas vacías repetidas
    texto = "\n".join(lineas)
    texto = re.sub(r"(^=+[^\n]*=+\s*\n)(?=\s*=+[^\n]*=+\s*$)", "", texto, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def primer_parrafo(texto, palabras=120):
    """Texto antes de la primera sección (la entradilla), limitado a unas cuantas palabras."""
    entradilla = re.split(r"^=+.*=+\s*$", texto, maxsplit=1, flags=re.M)[0]
    return " ".join(entradilla.split()[:palabras])


def motivo_descarte(titulo, texto_limpio):
    """Devuelve por qué se descarta un artículo, o None si se queda."""
    if TITULO_CULTURA_POP.search(titulo):
        return "cultura popular (título)"
    inicio = texto_limpio[:400].lower()
    entradilla = primer_parrafo(texto_limpio)
    if re.match(r"^(list|lists|index|outline) of\b", titulo.lower()):
        return "desambiguación o lista"
    # Las desambiguaciones de mitología ("In Greek mythology, the name X may refer to: ...")
    # sí se quedan: describen a varios personajes con el mismo nombre. Las demás, fuera.
    if ("may refer to" in inicio or "may also refer to" in inicio) and not TEMA.search(entradilla):
        return "desambiguación o lista"
    if not TEMA.search(entradilla):
        return "tema"
    if OTRAS.search(entradilla) and not NUESTRAS.search(entradilla):
        return "otra mitología"
    if len(texto_limpio.split()) < MIN_PALABRAS:
        return "corto"
    return None


def main():
    df = pd.read_json(RUTA_ENTRADA, lines=True)
    palabras_antes = df["texto"].fillna("").str.split().str.len().sum()

    excluidos = set()
    if RUTA_EXCLUIDOS.exists():
        excluidos = set(pd.read_csv(RUTA_EXCLUIDOS)["doc_id"])
    print(f"Artículos excluidos por la clasificación: {len(excluidos)}")

    documentos, descartados = [], []
    for art in df.itertuples():
        limpio = quitar_secciones(art.texto or "")
        if f"wiki_{art.pageid}" in excluidos:
            motivo = "fuera de tema (clasificación)"
        else:
            motivo = motivo_descarte(art.titulo, limpio)
        if motivo:
            descartados.append({"titulo": art.titulo, "motivo": motivo,
                                "palabras": len(limpio.split()),
                                "inicio": primer_parrafo(limpio, 25)})
            continue
        documentos.append({
            "doc_id": f"wiki_{art.pageid}",
            "titulo": art.titulo,
            "obra": "Wikipedia",
            "autor": "Wikipedia",
            "tipo_fuente": "enciclopedica",
            "mitologia": art.mitologias,
            "url": art.url,
            "revid": art.revid,
            "texto": limpio,
        })

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        for doc in documentos:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    desc = pd.DataFrame(descartados)
    desc.to_csv(RUTA_DESCARTADOS, index=False)

    # --- Resumen ---
    palabras = pd.Series([len(d["texto"].split()) for d in documentos])
    print("=" * 60)
    print(f"Artículos de entrada: {len(df)}")
    print(f"Se quedan: {len(documentos)}  ->  {RUTA_SALIDA}")
    print(f"Descartados: {len(desc)}  ->  {RUTA_DESCARTADOS}")
    if len(desc):
        print(desc["motivo"].value_counts().to_string())
    print(f"\nPalabras: {palabras_antes:,} antes  ->  {palabras.sum():,} después "
          f"(mediana por artículo: {palabras.median():.0f})")
    print("=" * 60)

    for motivo in ["tema", "otra mitología", "corto", "desambiguación o lista", "cultura popular (título)"]:
        muestra = desc[desc["motivo"] == motivo] if len(desc) else desc
        if len(muestra):
            print(f"\nMuestra de descartados por '{motivo}':")
            for fila in muestra.sample(min(12, len(muestra)), random_state=0).itertuples():
                print(f"  - {fila.titulo}  ({fila.palabras} pal.): {fila.inicio[:90]}")

    print("\nMuestra de artículos que SE QUEDAN:")
    for doc in pd.DataFrame(documentos).sample(min(10, len(documentos)), random_state=1).itertuples():
        print(f"  - {doc.titulo}: {primer_parrafo(doc.texto, 18)}")


if __name__ == "__main__":
    main()