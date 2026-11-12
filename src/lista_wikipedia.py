"""
Lista de artículos de Wikipedia (en inglés) sobre mitología griega, romana y nórdica.

Recorre el árbol de categorías de Wikipedia con su API, empezando en las categorías
raíz de cada mitología y bajando hasta una profundidad máxima. Se salta las
subcategorías de cultura pop, arte, geografía y arqueología, y descarta los
artículos cuyo título indica una obra moderna (cómics, películas, cuadros...).

Guarda dos CSV para revisarlos antes de descargar los textos:
    data/raw/wikipedia/lista_articulos.csv   -> artículos que entran
    data/raw/wikipedia/descartados.csv       -> artículos descartados por el título

Uso (desde la carpeta principal del repositorio, con el entorno activado):
    python src/lista_wikipedia.py --contacto tu_correo@ejemplo.com
    python src/lista_wikipedia.py --contacto tu_correo@ejemplo.com --profundidad 2
"""

import argparse
import re
import time
from collections import Counter, deque
from pathlib import Path

import pandas as pd
import requests

# --- Configuración ------------------------------------------------------------

API_URL = "https://en.wikipedia.org/w/api.php"

# Categoría raíz de cada mitología
RAICES = {
    "griega": "Category:Greek mythology",
    "romana": "Category:Roman mythology",
    "nordica": "Category:Norse mythology",
}

# Si el nombre de una subcategoría contiene alguna de estas palabras, no se entra en ella.
EXCLUIR_CATEGORIAS = [
    # cultura pop y obras modernas
    "popular culture", "in art", "arts in", "in fiction", "in literature", "works based",
    "adaptations", "films", "television", "video games", "novels", "comics",
    "operas", "ballets", "plays", "paintings", "sculptures", "music", "songs",
    "albums", "fictional", "modern", "neopagan", "heathenry",
    # geografía, arqueología y temas que se alejan del mito
    "by region", "lgbtq", "pompeii", "herculaneum", "archaeolog",
    # mantenimiento de Wikipedia y estudiosos modernos
    "wikipedia", "stubs", "images", "templates", "scholars", "translators",
]

# Si la aclaración entre paréntesis del título contiene alguna de estas palabras,
# el artículo es una obra moderna y se descarta. Ej.: "Selene (comics)".
EXCLUIR_TITULOS = [
    # tipos de obra moderna (no se incluye "play": las tragedias griegas sí nos interesan)
    "comics", "film", "painting", "sculpture", "statue", "opera", "ballet",
    "novel", "album", "song", "band", "musical", "video game", "tv series",
    "television", "character", "marvel", "dc ", "manga", "anime", "ship", "asteroid",
    "crater", "moon", "planet", "spacecraft", "software", "company", "magazine",
    # artistas cuyos cuadros y esculturas mitológicas tienen artículo propio
    "rodin", "titian", "rubens", "canova", "bernini", "botticelli", "waterhouse",
    "moreau", "poussin", "caravaggio", "rembrandt", "michelangelo", "velázquez",
    "goya", "böcklin", "dalí", "picasso", "klimt", "tintoretto", "veronese",
    "correggio", "ingres", "delacroix", "burne-jones", "leighton", "thorvaldsen",
]

PAUSA = 0.2          # segundos entre peticiones a la API
MAX_INTENTOS = 5     # reintentos si Wikipedia nos frena o falla la conexión

# Carpeta de salida (data/raw/wikipedia, dentro del repositorio)
CARPETA_SALIDA = Path(__file__).resolve().parents[1] / "data" / "raw" / "wikipedia"


# --- Funciones ----------------------------------------------------------------

def peticion(sesion, params):
    """
    Hace una petición a la API con reintentos.
    Si Wikipedia responde 429 (demasiadas peticiones), espera lo que indica
    la cabecera Retry-After o, si no la hay, cada vez más (5, 10, 20... s).
    Devuelve el JSON, o None si fallan todos los intentos.
    """
    for intento in range(MAX_INTENTOS):
        try:
            resp = sesion.get(API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                espera = int(resp.headers.get("Retry-After", 5 * 2 ** intento))
                print(f"  Wikipedia pide esperar: {espera} s (intento {intento + 1})")
                time.sleep(espera)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            espera = 5 * 2 ** intento
            print(f"  Error de conexión: {e}. Reintento en {espera} s")
            time.sleep(espera)
    return None


def miembros_categoria(sesion, categoria, fallidas):
    """Devuelve todas las páginas y subcategorías de una categoría (sigue la paginación)."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": categoria,
        "cmtype": "page|subcat",
        "cmlimit": "500",
        "format": "json",
        "maxlag": "5",  # si los servidores van cargados, la API nos pide esperar
    }
    miembros = []
    while True:
        datos = peticion(sesion, params)
        time.sleep(PAUSA)
        if datos is None or "query" not in datos:
            fallidas.append(categoria)  # la anotamos para avisar al final
            return miembros
        miembros += datos["query"]["categorymembers"]
        if "continue" not in datos:
            return miembros
        params.update(datos["continue"])  # siguiente página de resultados


def categoria_excluida(categoria):
    """True si la subcategoría es de cultura pop, arte, geografía, etc."""
    nombre = categoria.lower()
    return any(palabra in nombre for palabra in EXCLUIR_CATEGORIAS)


def titulo_excluido(titulo):
    """True si la aclaración entre paréntesis del título indica una obra moderna."""
    aclaracion = re.search(r"\(([^)]*)\)\s*$", titulo)
    if not aclaracion:
        return False
    texto = aclaracion.group(1).lower()
    return any(palabra in texto for palabra in EXCLUIR_TITULOS)


def recorrer(sesion, mitologia, raiz, profundidad_max, articulos, descartados, fallidas):
    """
    Recorre el árbol de categorías desde 'raiz' (búsqueda en anchura) y añade
    los artículos encontrados a 'articulos' (o a 'descartados' si el título no pasa el filtro).
    Devuelve cuántas categorías se visitaron y cuántas se saltaron.
    """
    cola = deque([(raiz, 0, "(raíz)")])  # (categoría, profundidad, rama de primer nivel)
    visitadas, saltadas = {raiz}, 0

    while cola:
        categoria, prof, rama = cola.popleft()
        for m in miembros_categoria(sesion, categoria, fallidas):
            titulo = m["title"]

            if m["ns"] == 0:  # es un artículo
                if titulo_excluido(titulo):
                    descartados[titulo] = {"titulo": titulo, "categoria": categoria}
                    continue
                if titulo not in articulos:
                    articulos[titulo] = {
                        "titulo": titulo,
                        "pageid": m["pageid"],
                        "mitologias": set(),
                        "profundidad": prof,
                        "rama": rama,
                        "categoria": categoria,
                    }
                articulos[titulo]["mitologias"].add(mitologia)

            elif m["ns"] == 14 and titulo not in visitadas:  # es una subcategoría
                visitadas.add(titulo)
                if categoria_excluida(titulo):
                    saltadas += 1
                elif prof + 1 <= profundidad_max:
                    # La "rama" es la subcategoría de primer nivel por la que se llegó
                    nueva_rama = titulo if prof == 0 else rama
                    cola.append((titulo, prof + 1, nueva_rama))

    return len(visitadas), saltadas


# --- Programa principal ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lista de artículos de mitología en Wikipedia")
    parser.add_argument("--contacto", required=True,
                        help="Correo de contacto (Wikipedia lo pide en el User-Agent)")
    parser.add_argument("--profundidad", type=int, default=3,
                        help="Niveles de subcategorías a recorrer (por defecto: 3)")
    args = parser.parse_args()

    sesion = requests.Session()
    sesion.headers["User-Agent"] = f"FactCheckingMitologia/0.2 (proyecto UC3M; {args.contacto})"

    articulos, descartados, fallidas = {}, {}, []
    for mitologia, raiz in RAICES.items():
        print(f"Recorriendo {raiz} (profundidad {args.profundidad})...")
        n_cat, n_saltadas = recorrer(sesion, mitologia, raiz, args.profundidad,
                                     articulos, descartados, fallidas)
        n_art = sum(mitologia in a["mitologias"] for a in articulos.values())
        print(f"  {n_cat} categorías vistas, {n_saltadas} saltadas, {n_art} artículos\n")

    # Pasamos a tablas y guardamos
    df = pd.DataFrame(articulos.values())
    df["mitologias"] = df["mitologias"].apply(lambda s: ";".join(sorted(s)))
    df = df.sort_values(["mitologias", "titulo"]).reset_index(drop=True)
    df_desc = pd.DataFrame(descartados.values(), columns=["titulo", "categoria"])

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_SALIDA / "lista_articulos.csv"
    ruta_desc = CARPETA_SALIDA / "descartados.csv"
    df.to_csv(ruta, index=False)
    df_desc.to_csv(ruta_desc, index=False)

    # --- Resumen para revisar la lista ---
    print("=" * 60)
    print(f"TOTAL: {len(df)} artículos únicos  ->  {ruta}")
    print(f"Descartados por el título: {len(df_desc)}  ->  {ruta_desc}")
    if fallidas:
        print(f"ATENCIÓN: {len(fallidas)} categorías no se pudieron leer: {fallidas}")
    else:
        print("Todas las categorías se leyeron sin errores.")
    print("=" * 60)
    print("\nArtículos por mitología (uno puede estar en varias):")
    print(df["mitologias"].value_counts().to_string())
    print("\nArtículos por profundidad:")
    print(df["profundidad"].value_counts().sort_index().to_string())
    print("\nLas 20 ramas que más artículos aportan (para detectar basura):")
    for rama, n in Counter(df["rama"]).most_common(20):
        print(f"  {n:5d}  {rama}")
    print("\nMuestra aleatoria de 20 títulos que ENTRAN:")
    for t in df["titulo"].sample(min(20, len(df)), random_state=0):
        print(f"  - {t}")
    print("\nMuestra aleatoria de 15 títulos DESCARTADOS:")
    for t in df_desc["titulo"].sample(min(15, len(df_desc)), random_state=0):
        print(f"  - {t}")


if __name__ == "__main__":
    main()