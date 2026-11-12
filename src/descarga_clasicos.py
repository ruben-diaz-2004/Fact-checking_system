"""
Descarga las fuentes clásicas del corpus, tal cual, sin procesar:

  - Libros de Project Gutenberg (texto plano): Hesíodo, Homero, Ovidio, las dos Eddas
    y Bulfinch.
  - Apolodoro (Biblioteca y Epítome), traducción de Frazer, en XML (TEI) desde el
    repositorio de la Perseus Digital Library en GitHub.

Los ficheros se guardan en data/raw/clasicos/. Si un fichero ya existe, no se vuelve
a descargar. La división de cada obra en documentos se hace en otro script.

Uso (desde la carpeta principal del repositorio, con el entorno activado):
    python src/descarga_clasicos.py
"""

import time
from pathlib import Path

import requests

CARPETA_SALIDA = Path(__file__).resolve().parents[1] / "data" / "raw" / "clasicos"

# Libros de Gutenberg: nombre de fichero -> número de ebook
GUTENBERG = {
    "hesiodo_himnos_homericos": 348,   # Hesíodo + Himnos homéricos (Evelyn-White)
    "iliada": 2199,                    # Homero, Ilíada (Butler, prosa)
    "odisea": 1727,                    # Homero, Odisea (Butler, prosa)
    "ovidio_metamorfosis_1": 21765,    # Ovidio, Metamorfosis libros I-VII (Riley)
    "ovidio_metamorfosis_2": 26073,    # Ovidio, Metamorfosis libros VIII-XV (Riley)
    "edda_poetica": 73533,             # Edda poética (Bellows)
    "edda_prosa": 18947,               # Edda en prosa / Younger Edda
    "bulfinch_age_of_fable": 3327,     # Bulfinch, The Age of Fable (divulgativa)
}
URL_GUTENBERG = "https://www.gutenberg.org/ebooks/{}.txt.utf-8"

# Apolodoro en Perseus (traducción inglesa de Frazer, 1921)
URL_PERSEUS = ("https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/"
               "data/tlg0548/{obra}/tlg0548.{obra}.perseus-eng2.xml")
PERSEUS = {
    "apolodoro_biblioteca": "tlg001",
    "apolodoro_epitome": "tlg002",
}


def descargar(sesion, url, ruta):
    """Descarga una URL a un fichero, salvo que ya exista. Devuelve True si todo fue bien."""
    if ruta.exists():
        print(f"  ya existe, se salta: {ruta.name}")
        return True
    try:
        resp = sesion.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR en {ruta.name}: {e}")
        return False
    ruta.write_bytes(resp.content)
    print(f"  descargado: {ruta.name}  ({len(resp.content) / 1024:.0f} KB)")
    time.sleep(1)  # pausa de cortesía entre descargas
    return True


def main():
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    sesion = requests.Session()
    sesion.headers["User-Agent"] = "FactCheckingMitologia/0.2 (proyecto UC3M)"

    errores = 0
    print("Project Gutenberg:")
    for nombre, numero in GUTENBERG.items():
        ruta = CARPETA_SALIDA / f"{nombre}.txt"
        errores += not descargar(sesion, URL_GUTENBERG.format(numero), ruta)

    print("\nPerseus (Apolodoro):")
    for nombre, obra in PERSEUS.items():
        ruta = CARPETA_SALIDA / f"{nombre}.xml"
        errores += not descargar(sesion, URL_PERSEUS.format(obra=obra), ruta)

    print("\n" + "=" * 60)
    print("Todo descargado sin errores." if errores == 0
          else f"Hubo {errores} errores: vuelve a ejecutar el script.")
    print(f"Ficheros en: {CARPETA_SALIDA}")


if __name__ == "__main__":
    main()