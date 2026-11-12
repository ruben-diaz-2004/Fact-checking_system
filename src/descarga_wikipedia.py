"""
Descarga el texto de los artículos de Wikipedia listados por lista_wikipedia.py.

Para cada artículo guarda el texto plano completo, con los títulos de sección
marcados como en Wikipedia (== Sección ==), más su URL y su número de revisión
(para saber exactamente qué versión del artículo usamos).

Aquí NO se filtra nada: se guarda todo tal cual. Los filtros (tema y longitud)
se aplican después, al construir el corpus, para poder ajustarlos sin volver
a descargar.

La descarga se puede interrumpir (Ctrl+C) y retomar: al volver a ejecutar el
script, se salta los artículos que ya estén descargados.

Uso (desde la carpeta principal del repositorio, con el entorno activado):
    python src/descarga_wikipedia.py --contacto tu_correo@ejemplo.com
"""

import argparse
import json
import time

import pandas as pd
import requests
from tqdm import tqdm

# Reutilizamos la función de peticiones con reintentos y la carpeta de lista_wikipedia.py
from lista_wikipedia import CARPETA_SALIDA, PAUSA, peticion

RUTA_LISTA = CARPETA_SALIDA / "lista_articulos.csv"
RUTA_SALIDA = CARPETA_SALIDA / "articulos.jsonl"   # un artículo por línea


def descargar_articulo(sesion, pageid):
    """
    Pide a la API el texto plano completo de un artículo, su URL y su revisión.
    Devuelve un diccionario, o None si no se pudo descargar.
    """
    params = {
        "action": "query",
        "prop": "extracts|info|revisions",
        "pageids": pageid,
        "explaintext": 1,            # texto plano, sin HTML
        "exsectionformat": "wiki",   # secciones marcadas como == Sección ==
        "inprop": "url",
        "rvprop": "ids",
        "format": "json",
        "maxlag": "5",
    }
    for _ in range(3):
        datos = peticion(sesion, params)
        time.sleep(PAUSA)
        if datos is None:
            return None
        if "error" in datos:  # p. ej. "maxlag": los servidores van cargados
            time.sleep(5)
            continue
        pagina = datos["query"]["pages"][str(pageid)]
        if "missing" in pagina:  # el artículo ya no existe
            return None
        return {
            "pageid": pageid,
            "titulo": pagina["title"],
            "url": pagina.get("fullurl", ""),
            "revid": pagina.get("revisions", [{}])[0].get("revid"),
            "texto": pagina.get("extract", ""),
        }
    return None


def ya_descargados():
    """Devuelve los pageid que ya están en el fichero de salida (para poder retomar)."""
    if not RUTA_SALIDA.exists():
        return set()
    with open(RUTA_SALIDA, encoding="utf-8") as f:
        return {json.loads(linea)["pageid"] for linea in f}


def main():
    parser = argparse.ArgumentParser(description="Descarga de los artículos de Wikipedia")
    parser.add_argument("--contacto", required=True,
                        help="Correo de contacto (Wikipedia lo pide en el User-Agent)")
    args = parser.parse_args()

    sesion = requests.Session()
    sesion.headers["User-Agent"] = f"FactCheckingMitologia/0.2 (proyecto UC3M; {args.contacto})"

    lista = pd.read_csv(RUTA_LISTA)
    hechos = ya_descargados()
    pendientes = lista[~lista["pageid"].isin(hechos)]
    print(f"Artículos en la lista: {len(lista)} | ya descargados: {len(hechos)} "
          f"| pendientes: {len(pendientes)}\n")

    fallidos = []
    # Se abre en modo "a" (añadir) para no perder lo descargado si se interrumpe
    with open(RUTA_SALIDA, "a", encoding="utf-8") as f:
        for fila in tqdm(pendientes.itertuples(), total=len(pendientes), desc="Descargando"):
            articulo = descargar_articulo(sesion, int(fila.pageid))
            if articulo is None:
                fallidos.append(fila.titulo)
                continue
            # Añadimos los datos de la lista (mitología y categoría de origen)
            articulo["mitologias"] = fila.mitologias
            articulo["categoria"] = fila.categoria
            f.write(json.dumps(articulo, ensure_ascii=False) + "\n")
            f.flush()  # se escribe en disco al momento

    # --- Resumen ---
    df = pd.read_json(RUTA_SALIDA, lines=True)
    palabras = df["texto"].str.split().str.len()
    print("\n" + "=" * 60)
    print(f"Descargados en total: {len(df)}  ->  {RUTA_SALIDA}")
    print(f"Fallidos en esta ejecución: {len(fallidos)}")
    if fallidos:
        print("  (vuelve a ejecutar el script para reintentarlos)")
        print("  Ejemplos:", fallidos[:10])
    print("=" * 60)
    print("\nLongitud de los artículos (palabras):")
    print(palabras.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(0).to_string())
    print("\nArtículos por debajo de cada longitud:")
    for umbral in [50, 100, 150, 200, 300]:
        n = (palabras < umbral).sum()
        print(f"  < {umbral:3d} palabras: {n:5d}  ({100 * n / len(df):.1f} %)")
    print("\nTotal de palabras del corpus de Wikipedia:", f"{palabras.sum():,}")


if __name__ == "__main__":
    main()