"""
Lista los vídeos candidatos de YouTube para el corpus (NO descarga nada).

Para cada canal elegido, yt-dlp obtiene la lista completa de sus vídeos (título,
duración, visualizaciones) sin descargarlos. Después se filtran:
  - el título debe mencionar algo de mitología griega, romana o nórdica
    (nombres de dioses, héroes, criaturas, lugares y obras). Si solo tiene palabras
    ambiguas ("Mars", "Rome", "Viking"...), debe llevar además "myth", "god", "legend"...;
  - y no debe ser de otras mitologías ni de temas ajenos (clases de idioma, directos...);
  - duración entre 3 y 60 minutos;
  - como mucho MAX_POR_CANAL vídeos por canal (los más vistos), para que ningún canal domine.

IMPORTANTE: YouTube bloquea las IP de servidores en la nube, así que este script
hay que ejecutarlo en un ordenador personal (por ejemplo, tu portátil con WSL).

Uso (desde la carpeta principal del repositorio, con el entorno activado):
    python src/lista_youtube.py

Se puede interrumpir con Ctrl+C: los canales ya listados se guardan y no se repiten.

Salida:
    data/raw/youtube/candidatos.csv   -> vídeos elegidos (para revisarlos a mano)
    data/raw/youtube/todos.csv        -> todos los vídeos listados, para consultar
"""

import re
import time
from pathlib import Path

import pandas as pd
from yt_dlp import YoutubeDL

CARPETA_SALIDA = Path(__file__).resolve().parents[1] / "data" / "raw" / "youtube"

MAX_POR_CANAL = 30
DURACION_MIN, DURACION_MAX = 3 * 60, 60 * 60    # en segundos

# Canales elegidos. nivel: "experta" (académicos, profesores) o "popular" (divulgadores).
# "exigir": palabra que debe aparecer en el título (para canales con muchos temas).
CANALES = [
    # --- Divulgación experta ---
    {"canal": "Jackson Crawford", "nivel": "experta",
     "url": "https://www.youtube.com/channel/UCXCxNFxw6iq-Mh4uIjYvufg/videos"},
    {"canal": "Religion for Breakfast", "nivel": "experta",
     "url": "https://www.youtube.com/channel/UCct9aR7HC79Cv2g-9oDOTLw/videos"},
    {"canal": "Crash Course", "nivel": "experta", "exigir": "World Mythology",
     "url": "https://www.youtube.com/@crashcourse/videos"},
    {"canal": "TED-Ed", "nivel": "experta",
     "url": "https://www.youtube.com/@TEDEd/videos"},
    {"canal": "World History Encyclopedia", "nivel": "experta",
     "url": "https://www.youtube.com/@WorldHistoryEncyclopedia/videos"},
    {"canal": "Gresham College", "nivel": "experta",
     "url": "https://www.youtube.com/channel/UC1t6kKXoBvjdr8m9KJ2Fx7A/videos"},
    {"canal": "History Hit", "nivel": "experta",
     "url": "https://www.youtube.com/@HistoryHit/videos"},
    {"canal": "Arith Härger", "nivel": "experta",
     "url": "https://www.youtube.com/@ArithHarger/videos"},
    # --- Divulgación popular ---
    {"canal": "Overly Sarcastic Productions", "nivel": "popular",
     "url": "https://www.youtube.com/channel/UCodbH5mUeF-m_BsNueRDjcw/videos"},
    {"canal": "Mythology & Fiction Explained", "nivel": "popular",
     "url": "https://www.youtube.com/channel/UCH4Bz5Swatl5iHSuCT_Hlxg/videos"},
    {"canal": "Tiny Epics", "nivel": "popular",
     "url": "https://www.youtube.com/channel/UC87WLm0IaVihJbsyrwH01jg/videos"},
    {"canal": "Extra History (Extra Mythology)", "nivel": "popular",
     "url": "https://www.youtube.com/@ExtraHistory/videos"},
]

# --- Vocabulario de nuestras mitologías ------------------------------------------

GENERALES = ["myth", "myths", "mythology", "mythological", "legend", "legends", "god", "gods",
             "goddess", "goddesses", "hero", "heroes", "monster", "monsters", "titan", "titans",
             "olympian", "olympians", "pantheon", "underworld", "afterlife", "creation"]

GRIEGA = ["Greek", "Greeks", "Greece", "Helen of Troy", "Trojan War", "Mount Olympus", "Psyche", "Zeus", "Hera", "Poseidon", "Hades", "Athena", "Apollo", "Artemis",
          "Ares", "Aphrodite", "Hephaestus", "Hermes", "Demeter", "Dionysus", "Hestia",
          "Persephone", "Cronus", "Kronos", "Rhea", "Gaia", "Uranus", "Prometheus", "Pandora",
          "Atlas", "Hecate", "Eros", "Nyx", "Helios", "Selene", "Eos", "Pan", "Nemesis",
          "Heracles", "Herakles", "Hercules", "Perseus", "Theseus", "Jason", "Argonauts",
          "Achilles", "Odysseus", "Ulysses", "Hector", "Paris", "Helen", "Agamemnon",
          "Orpheus", "Eurydice", "Oedipus", "Antigone", "Medea", "Medusa", "Gorgon", "Minotaur",
          "Cerberus", "Hydra", "Chimera", "Pegasus", "Cyclops", "Polyphemus", "Sirens", "Siren",
          "Scylla", "Charybdis", "Typhon", "Echidna", "Harpies", "Centaur", "Centaurs", "Satyr",
          "Nymph", "Nymphs", "Muses", "Fates", "Moirai", "Furies", "Erinyes", "Amazons",
          "Narcissus", "Echo", "Arachne", "Icarus", "Daedalus", "Midas", "Sisyphus",
          "Tantalus", "Atalanta", "Ariadne", "Cassandra", "Penelope", "Circe", "Calypso",
          "Bellerophon", "Adonis", "Ganymede", "Tartarus", "Olympus", "Troy", "Trojan",
          "Iliad", "Odyssey", "Hesiod", "Theogony", "Homer", "Eleusinian", "Delphi", "Oracle"]

ROMANA = ["Roman", "Romans", "Rome", "Jupiter", "Juno", "Neptune", "Pluto", "Minerva", "Mars", "Venus",
          "Vulcan", "Mercury", "Ceres", "Bacchus", "Vesta", "Diana", "Janus", "Saturn", "Cupid",
          "Aeneas", "Aeneid", "Romulus", "Remus", "Ovid", "Metamorphoses", "Virgil", "Lares"]

NORDICA = ["Norse", "Viking", "Vikings", "Odin", "Thor", "Loki", "Freya", "Freyja", "Frigg",
           "Freyr", "Baldr", "Balder", "Baldur", "Tyr", "Heimdall", "Hel", "Njord", "Idun",
           "Bragi", "Ymir", "Fenrir", "Jormungandr", "Sleipnir", "Ragnarok", "Ragnarök",
           "Valhalla", "Valkyrie", "Valkyries", "Yggdrasil", "Asgard", "Midgard", "Jotunheim",
           "Aesir", "Æsir", "Vanir", "Jotnar", "Giants", "Norns", "Edda", "Eddas", "Havamal",
           "Hávamál", "Voluspa", "Völuspá", "Sigurd", "Volsung", "Völsunga", "Mjolnir",
           "Huginn", "Muninn", "Dwarves", "Elves", "Seidr", "Berserker", "Berserkers"]

# Palabras ambiguas: pueden ser planetas, historia, ciudades... Si un título SOLO tiene
# palabras de esta lista, se exige además una palabra general (myth, god, legend...).
AMBIGUAS = {"greek", "greeks", "greece", "roman", "romans", "rome", "viking", "vikings", "mars", "mercury", "venus",
            "saturn", "pluto", "jupiter", "neptune", "uranus", "paris", "helen", "echo", "atlas",
            "oracle", "delphi", "giants", "elves", "dwarves", "hel", "pan", "jason", "diana",
            "cupid", "trojan", "troy", "homer", "fates", "nemesis", "siren", "sirens", "hydra",
            "olympus", "eros", "nyx", "selene", "helios", "eos", "rhea", "gaia", "tyr", "lares"}

# Si el título contiene alguna de estas palabras, se descarta
EXCLUIR = ["Egypt", "Egyptian", "Aztec", "Maya", "Mayan", "Inca", "Hindu", "India", "Chinese",
           "China", "Japanese", "Japan", "Celtic", "Irish", "Slavic", "Mesopotamian", "Sumerian",
           "Babylonian", "Gilgamesh", "Persian", "African", "Yoruba", "Polynesian", "Hawaiian",
           "Maori", "Native American", "Aboriginal", "Korean", "Arthurian", "Bible", "Biblical",
           "Jesus", "Christian", "Islam", "Marvel", "MCU", "Percy Jackson", "God of War",
           "Assassin", "Disney", "livestream", "Live Stream", "Q&A", "Grammar", "Pronunciation",
           "Old Norse class", "Language", "Lesson", "Trailer", "Shorts", "#shorts", "Podcast",
           "Old Norse", "Icelandic", "Norwegian", "Swedish", "Danish", "Dialect", "Word", "Words",
           "Rune", "Runes", "Runic", "Futhark", "Translation", "Translating", "Reading",
           "Unboxing", "Patreon", "Announcement"]


def patron(palabras):
    """Expresión regular que busca cualquiera de las palabras como palabra completa."""
    return re.compile(r"\b(" + "|".join(re.escape(p) for p in palabras) + r")\b", re.IGNORECASE)


P_GENERALES, P_GRIEGA = patron(GENERALES), patron(GRIEGA)
P_ROMANA, P_NORDICA, P_EXCLUIR = patron(ROMANA), patron(NORDICA), patron(EXCLUIR)


def clasificar(titulo):
    """
    Devuelve la(s) mitología(s) del título ('griega', 'romana;griega', 'nordica'...)
    o None si el título no es de nuestras mitologías.
    """
    if P_EXCLUIR.search(titulo):
        return None
    encontradas = {m.lower() for p in (P_GRIEGA, P_ROMANA, P_NORDICA) for m in p.findall(titulo)}
    if encontradas and encontradas <= AMBIGUAS and not P_GENERALES.search(titulo):
        return None                     # p. ej. "What's on Mars?" o "The Fall of Rome"
    mitos = []
    if P_GRIEGA.search(titulo):
        mitos.append("griega")
    if P_ROMANA.search(titulo):
        mitos.append("romana")
    if P_NORDICA.search(titulo):
        mitos.append("nordica")
    return ";".join(mitos) if mitos else None


def listar_canal(canal, url):
    """
    Lista los vídeos de un canal con yt-dlp, sin descargar nada.
    El listado se guarda en data/raw/youtube/listados/ para no repetirlo si se vuelve a
    ejecutar el script (por ejemplo, tras interrumpirlo con Ctrl+C).
    """
    cache = CARPETA_SALIDA / "listados" / (re.sub(r"\W+", "_", canal).strip("_") + ".csv")
    if cache.exists():
        print("  (ya listado antes: se lee del disco)")
        return pd.read_csv(cache).to_dict("records")

    opciones = {"extract_flat": "in_playlist", "quiet": True, "skip_download": True,
                "ignoreerrors": True, "socket_timeout": 30, "extractor_retries": 3}
    with YoutubeDL(opciones) as ydl:
        info = ydl.extract_info(url, download=False)
    videos = [{"id": e.get("id"), "title": e.get("title"), "duration": e.get("duration"),
               "view_count": e.get("view_count")}
              for e in (info or {}).get("entries", []) if e]
    if videos:                                   # solo se guarda si el listado salió bien
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(videos).to_csv(cache, index=False)
    return videos


def main():
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    todos, elegidos, resumen = [], [], []

    for c in CANALES:
        print(f"Listando {c['canal']}... (los canales grandes pueden tardar varios minutos)")
        inicio = time.time()
        try:
            videos = listar_canal(c["canal"], c["url"])
        except Exception as e:
            print(f"  ERROR: no se pudo listar ({e}). Revisa la URL del canal.")
            resumen.append({"canal": c["canal"], "listados": 0, "pasan_filtro": 0, "elegidos": 0})
            continue

        filas = []
        for v in videos:
            titulo = v.get("title") or ""
            duracion = v.get("duration")
            duracion = None if pd.isna(duracion) else duracion
            vistas = v.get("view_count")
            vistas = None if pd.isna(vistas) else vistas
            mitologia = clasificar(titulo)
            if c.get("exigir") and c["exigir"].lower() not in titulo.lower():
                mitologia = None
            ok_duracion = duracion is None or DURACION_MIN <= duracion <= DURACION_MAX
            fila = {"canal": c["canal"], "nivel": c["nivel"], "video_id": v.get("id"),
                    "titulo": titulo, "duracion_min": round(duracion / 60, 1) if duracion else None,
                    "visualizaciones": vistas, "mitologia": mitologia,
                    "url": f"https://www.youtube.com/watch?v={v.get('id')}",
                    "pasa": bool(mitologia) and ok_duracion}
            filas.append(fila)
        todos += filas

        # Se quedan los MAX_POR_CANAL más vistos de los que pasan el filtro
        pasan = sorted([f for f in filas if f["pasa"]],
                       key=lambda f: f["visualizaciones"] or 0, reverse=True)
        elegidos += pasan[:MAX_POR_CANAL]
        resumen.append({"canal": c["canal"], "listados": len(filas),
                        "pasan_filtro": len(pasan), "elegidos": min(len(pasan), MAX_POR_CANAL)})
        print(f"  {len(filas)} vídeos, {len(pasan)} pasan el filtro ({time.time() - inicio:.0f} s)")

    pd.DataFrame(todos).to_csv(CARPETA_SALIDA / "todos.csv", index=False)
    df = pd.DataFrame(elegidos).drop(columns="pasa")
    df.to_csv(CARPETA_SALIDA / "candidatos.csv", index=False)

    # --- Resumen ---
    print("\n" + "=" * 70)
    print(pd.DataFrame(resumen).to_string(index=False))
    print("=" * 70)
    print(f"TOTAL elegidos: {len(df)}  ->  {CARPETA_SALIDA / 'candidatos.csv'}")
    if len(df):
        df = df.copy()
        print(f"Horas de vídeo: {df['duracion_min'].sum() / 60:.1f}")
        print("\nPor nivel y mitología:")
        print(df.groupby(["nivel", "mitologia"]).size().to_string())


if __name__ == "__main__":
    main()