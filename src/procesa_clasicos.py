"""
Parte las fuentes clásicas descargadas por descarga_clasicos.py en DOCUMENTOS
(cada canto, libro, poema, himno, fábula o capítulo es un documento) y quita
todo lo que no es el texto de la obra: prólogos, introducciones, notas a pie de
página y comentarios de los traductores, índices y licencias de Gutenberg.

Entrada:  data/raw/clasicos/*.txt y *.xml
Salida:   data/processed/clasicos.jsonl  (un documento por línea)

Uso (desde la carpeta principal del repositorio, con el entorno activado):
    python src/procesa_clasicos.py
"""

import json
import re
import string
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_RAW = RAIZ / "data" / "raw" / "clasicos"
RUTA_SALIDA = RAIZ / "data" / "processed" / "clasicos.jsonl"

URL_GUTENBERG = "https://www.gutenberg.org/ebooks/{}"
MIN_PALABRAS = 30   # los documentos más cortos (títulos sueltos, fragmentos mínimos) se descartan


# --- Utilidades comunes --------------------------------------------------------

def leer_gutenberg(nombre):
    """Lee un libro de Gutenberg y devuelve sus líneas, sin la cabecera ni la licencia."""
    texto = (CARPETA_RAW / f"{nombre}.txt").read_text(encoding="utf-8")
    inicio = texto.index("*** START OF")
    inicio = texto.index("\n", inicio) + 1
    fin = texto.index("*** END OF")
    return texto[inicio:fin].split("\n")


def parrafos(lineas, solo_sin_sangria=False):
    """
    Une las líneas en párrafos (separados por líneas vacías) y los devuelve
    como lista de cadenas, cada una en una sola línea.
    Con solo_sin_sangria=True se quedan solo los párrafos que no empiezan con
    espacios (en Ovidio, los sangrados son resúmenes y notas del traductor).
    """
    resultado, actual = [], []
    for linea in lineas + [""]:
        if linea.strip():
            actual.append(linea)
        elif actual:
            if not (solo_sin_sangria and actual[0].startswith(" ")):
                resultado.append(" ".join(" ".join(actual).split()))
            actual = []
    return resultado


def documento(doc_id, titulo, obra, autor, tipo, mitologia, url, pars):
    """Crea el diccionario de un documento. El texto son los párrafos separados por línea en blanco."""
    # Limpieza común: fuera las notas a pie de página sueltas y las marcas de laguna
    pars = [re.sub(r"\(\(LACUNA[^)]*\)\)", "[...]", p) for p in pars if p and not p.startswith("[Footnote")]
    texto = "\n\n".join(pars)
    return {
        "doc_id": doc_id,
        "titulo": titulo,
        "obra": obra,
        "autor": autor,
        "tipo_fuente": tipo,       # primaria | divulgativa
        "mitologia": mitologia,    # griega | romana | nordica (puede haber varias, separadas por ;)
        "url": url,
        "texto": texto,
    }


def titulo_bonito(titulo):
    """Pasa un título en mayúsculas a formato normal, dejando los números romanos en mayúsculas."""
    m = re.match(r"^([IVXL]+\.)\s*(.*)$", titulo)   # p. ej. "XIII. TO DEMETER"
    if m:
        return f"{m.group(1)} {string.capwords(m.group(2))}"
    return string.capwords(titulo)


# --- Homero: Ilíada y Odisea (un documento por canto) ---------------------------

def homero(nombre, obra, numero, prefijo):
    lineas = leer_gutenberg(nombre)
    # La Odisea trae al final las notas de Butler: se corta en "FOOTNOTES:"
    fin = next((i for i, l in enumerate(lineas) if l.startswith("FOOTNOTES:")), len(lineas))
    lineas = lineas[:fin]
    # Los cantos empiezan con "BOOK I." pegado al margen (el índice va sangrado)
    cabeceras = [i for i, l in enumerate(lineas) if re.fullmatch(r"BOOK [IVXL]+\.?", l.strip())
                 and not l.startswith(" ")]
    docs = []
    for k, ini in enumerate(cabeceras):
        fin = cabeceras[k + 1] if k + 1 < len(cabeceras) else len(lineas)
        canto = lineas[ini].strip().rstrip(".").split()[1]
        pars = [re.sub(r"\[\d+\]", "", p) for p in parrafos(lineas[ini + 1:fin])]  # quita llamadas a notas [12]
        docs.append(documento(f"{prefijo}_{k + 1:02d}", f"{obra}, Book {canto}", obra, "Homero",
                              "primaria", "griega", URL_GUTENBERG.format(numero), pars))
    return docs


# --- Hesíodo, Himnos homéricos y Ciclo épico ------------------------------------

def hesiodo():
    lineas = leer_gutenberg("hesiodo_himnos_homericos")
    # El texto de las obras va desde "HESIOD’S WORKS AND DAYS" hasta las notas finales
    ini = next(i for i, l in enumerate(lineas) if l.startswith("HESIOD’S WORKS AND DAYS"))
    fin = next(i for i in range(ini, len(lineas)) if lineas[i].strip() == "ENDNOTES")
    lineas = lineas[ini:fin]

    # Cabeceras: líneas en mayúsculas pegadas al margen (salvo las marcas de laguna)
    cabeceras = [i for i, l in enumerate(lineas)
                 if l.strip() and l == l.upper() and not l.startswith(" ")
                 and re.search(r"[A-Z]{3}", l) and "LACUNA" not in l and len(l) < 70]

    # Bloques que no son mitología (epigramas, parodias, la anécdota del certamen)
    excluir = ("HOMER’S EPIGRAMS", "THE PHOCAIS", "THE MARGITES", "THE CERCOPES",
               "THE BATTLE OF FROGS AND MICE", "OF THE ORIGIN OF HOMER AND HESIOD",
               "FRAGMENTS OF THE EPIC CYCLE", "HOMERICA", "THE HOMERIC HYMNS")

    docs, autor = [], "Hesíodo"
    for k, c in enumerate(cabeceras):
        titulo = re.sub(r"\d+$", "", lineas[c].strip()).strip()  # quita la llamada a nota pegada
        # El autor cambia al llegar a cada bloque del libro
        if titulo == "THE HOMERIC HYMNS":
            autor = "Himnos homéricos"
        elif titulo == "HOMER’S EPIGRAMS":
            autor = "Epigramas"
        elif titulo == "FRAGMENTS OF THE EPIC CYCLE":
            autor = "Ciclo épico"
        elif titulo == "HOMERICA":
            autor = "Homérica"
        if titulo.startswith(excluir) or autor == "Epigramas":
            continue

        fin_doc = cabeceras[k + 1] if k + 1 < len(cabeceras) else len(lineas)
        pars = []
        for p in parrafos(lineas[c + 1:fin_doc]):
            p = re.sub(r"\(ll\. [\d\-]+\)\s*", "", p)        # marcas de verso: (ll. 1-25)
            p = re.sub(r"\s?\b[1-9]\d{3}\b", "", p)          # llamadas a notas: 1601
            pars.append(p.strip())
        docs.append(documento(f"hesiodo_{len(docs) + 1:02d}", titulo_bonito(titulo), "Hesiod, the Homeric Hymns, and Homerica",
                              autor, "primaria", "griega", URL_GUTENBERG.format(348), pars))
    return docs


# --- Ovidio, Metamorfosis (un documento por fábula) -----------------------------

def ovidio(nombre, numero, prefijo):
    lineas = leer_gutenberg(nombre)
    # Las cabeceras van pegadas al margen (al final hay notas del transcriptor sangradas que empiezan por FABLE)
    es_fabula = re.compile(r"^FABLES? [IVXL]+\.?")
    es_libro = re.compile(r"^\s*BOOK THE [A-Z]+\.?\s*$")
    es_explicacion = re.compile(r"^\s*EXPLANATIONS?\.?\s*$")

    fabulas = [i for i, l in enumerate(lineas) if es_fabula.match(l)]
    docs = []
    for k, ini in enumerate(fabulas):
        # La fábula termina en su EXPLANATION (comentario de Riley) o en la siguiente cabecera
        fin = len(lineas)
        for j in range(ini + 1, len(lineas)):
            if (es_explicacion.match(lineas[j]) or es_fabula.match(lineas[j]) or es_libro.match(lineas[j])
                    or lineas[j].startswith("Transcriber")):
                fin = j
                break
        cuerpo = lineas[ini + 1:fin]
        # Referencia de versos, en la misma línea o en la siguiente: [I.5-31]
        ref = re.search(r"\[([^\]]+)\]", lineas[ini] + " " + (cuerpo[0] if cuerpo else ""))
        ref = ref.group(1).replace(" ", "") if ref else ""
        # El resumen de Riley (párrafo sangrado) se usa como título; el texto son los párrafos sin sangría
        resumen = next((p for p in parrafos(cuerpo) if not p.startswith("[")), "")
        resumen_corto = resumen.split(".")[0][:120]
        pars = []
        for p in parrafos(cuerpo, solo_sin_sangria=True):
            if p.startswith("["):             # la referencia de versos suelta
                continue
            p = re.sub(r"\[\d+\]", "", p)     # llamadas a notas: [4]
            p = re.sub(r"[{}]", "", p)        # palabras añadidas por el traductor: {more}
            pars.append(p)
        docs.append(documento(f"{prefijo}_{k + 1:03d}", f"Metamorphoses {ref}: {resumen_corto}",
                              "Metamorphoses", "Ovidio", "primaria", "griega;romana",
                              URL_GUTENBERG.format(numero), pars))
    return docs


# --- Edda poética (un documento por poema, sin la introducción ni las notas de Bellows) ---

def edda_poetica():
    lineas = leer_gutenberg("edda_poetica")
    intros = [i for i, l in enumerate(lineas) if l.strip() == "INTRODUCTORY NOTE"]
    notas = [i for i, l in enumerate(lineas) if l.strip() in ("NOTES", "NOTE")]
    docs = []
    for a in intros:
        b = min(n for n in notas if n > a)
        # La introducción de Bellows termina en el primer hueco de 5 o más líneas vacías
        inicio, vacias = None, 0
        for j in range(a + 1, b):
            if not lineas[j].strip():
                vacias += 1
            else:
                if vacias >= 5:
                    inicio = j
                    break
                vacias = 0
        if inicio is None:
            continue
        # Encima de "INTRODUCTORY NOTE" están el título inglés y, antes, el nórdico
        encima = [l.strip() for l in lineas[a - 8:a] if l.strip()]
        titulo_nordico, titulo_ingles = encima[-2], encima[-1]
        pars = [p.replace(" | ", " ").replace("|", " ") for p in parrafos(lineas[inicio:b])]
        pars = [" ".join(p.split()) for p in pars]
        docs.append(documento(f"edda_poetica_{len(docs) + 1:02d}",
                              f"{string.capwords(titulo_ingles)} ({string.capwords(titulo_nordico)})",
                              "Poetic Edda", "Edda poética", "primaria", "nordica",
                              URL_GUTENBERG.format(73533), pars))
    return docs


# --- Edda en prosa (un documento por capítulo o relato) ------------------------

def edda_prosa():
    lineas = leer_gutenberg("edda_prosa")
    # El texto empieza en el segundo "THE FOOLING OF GYLFE." (el primero está en el índice)
    ini = [i for i, l in enumerate(lineas) if l.strip() == "THE FOOLING OF GYLFE."][1]
    fin = next(i for i in range(ini, len(lineas)) if lineas[i].strip() == "NOTES.")
    lineas = lineas[ini:fin]

    es_cabecera = lambda l: (l.strip() and l == l.upper() and not l.startswith(" ")
                             and re.search(r"[A-Z]{3}", l) and len(l) < 70)
    partes = {"THE FOOLING OF GYLFE.": "Gylfaginning", "BRAGE’S TALK.": "Bragaraedur",
              "EXTRACTS FROM": "Skaldskaparmal"}
    docs, parte, titulo, cuerpo, saltar = [], "Gylfaginning", None, [], False

    def cerrar():
        if titulo and not saltar:
            pars = [re.sub(r"\[\d+\]", "", p) for p in parrafos(cuerpo)]
            docs.append(documento(f"edda_prosa_{len(docs) + 1:02d}", f"{parte}: {titulo}",
                                  "Prose Edda", "Edda en prosa (Snorri)", "primaria", "nordica",
                                  URL_GUTENBERG.format(18947), pars))

    i = 0
    while i < len(lineas):
        l = lineas[i]
        if es_cabecera(l):
            nombre = l.strip()
            if nombre in partes:                       # empieza una de las tres partes
                cerrar()
                parte, titulo, cuerpo, saltar = partes[nombre], None, [], False
            elif nombre.startswith("CHAPTER"):         # "CHAPTER I." + línea con el título
                cerrar()
                j = i + 1
                while not lineas[j].strip():
                    j += 1
                titulo = string.capwords(re.sub(r"\[\d+\]", "", lineas[j].strip()))
                cuerpo, saltar, i = [], False, j
            elif nombre == "AFTERWORD":                # comentario del traductor: se salta
                cerrar()
                titulo, cuerpo, saltar = "Afterword", [], True
            elif parte == "Skaldskaparmal" and not nombre.startswith(("THE POETICAL", "(SKALD")):
                cerrar()                               # en esta parte cada relato tiene su título
                titulo, cuerpo, saltar = string.capwords(re.sub(r"\[\d+\]", "", nombre)), [], False
        else:
            cuerpo.append(l)
        i += 1
    cerrar()
    return docs


# --- Apolodoro (Perseus, XML): un documento por capítulo, sin las notas de Frazer ---

def texto_sin_notas(elemento):
    """Devuelve el texto de un elemento XML saltándose las notas (<note>)."""
    partes = []

    def recorrer(e):
        if e.tag.split("}")[-1] == "note":
            if e.tail:
                partes.append(e.tail)
            return
        if e.text:
            partes.append(e.text)
        for hijo in e:
            recorrer(hijo)
        if e.tail:
            partes.append(e.tail)

    for hijo in elemento:
        recorrer(hijo)
    return " ".join(" ".join(partes).split())


def apolodoro():
    tei = "{http://www.tei-c.org/ns/1.0}div"
    url = "https://catalog.perseus.org/catalog/urn:cts:greekLit:tlg0548.{}"
    docs = []
    for fichero, obra, codigo in [("apolodoro_biblioteca", "Library", "tlg001"),
                                  ("apolodoro_epitome", "Epitome", "tlg002")]:
        raiz = ET.parse(CARPETA_RAW / f"{fichero}.xml").getroot()
        libro = None
        for div in raiz.iter(tei):
            if div.get("subtype") == "book":
                libro = div.get("n")
            if div.get("subtype") != "chapter":
                continue
            capitulo = div.get("n")
            ref = f"{libro}.{capitulo}" if obra == "Library" else capitulo
            # Cada sección del capítulo es un párrafo
            secciones = [d for d in div if d.tag == tei and d.get("subtype") == "section"]
            pars = [texto_sin_notas(s) for s in secciones] or [texto_sin_notas(div)]
            docs.append(documento(f"apolodoro_{len(docs) + 1:02d}", f"Apollodorus, {obra} {ref}",
                                  f"Apollodorus, {obra}", "Apolodoro", "primaria", "griega",
                                  url.format(codigo), pars))
    return docs


# --- Bulfinch (divulgativa): un documento por capítulo de mitología grecorromana o nórdica ---

def bulfinch():
    lineas = leer_gutenberg("bulfinch_age_of_fable")
    cabecera = re.compile(r"(Chapter|CHAPTER) [IVXL]+")
    caps = [i for i, l in enumerate(lineas) if cabecera.fullmatch(l.rstrip()) and not lineas[i - 1].strip()]
    # El índice también tiene "Chapter ..."; el texto empieza en el capítulo I cuyo título es "Introduction"
    def titulo_de(i):
        j = i + 1
        while not lineas[j].strip():
            j += 1
        return j, lineas[j].strip()
    primero = next(k for k, i in enumerate(caps) if titulo_de(i)[1] == "Introduction")
    caps = caps[primero:]
    fin_libro = next(i for i, l in enumerate(lineas) if "SKETCH OF THE HISTORY OF GREEK SCULPTURE" in l)

    # Capítulos que no son de mitología griega, romana o nórdica
    excluir = {"XXVII", "XXVIII", "XXIX", "XXX", "XXXIV"}
    nordicos = {"XXXI", "XXXII", "XXXIII"}

    docs = []
    for k, ini in enumerate(caps):
        num = lineas[ini].split()[1]
        if num in excluir:
            continue
        fin = caps[k + 1] if k + 1 < len(caps) else fin_libro
        j, titulo = titulo_de(ini)
        pars = parrafos(lineas[j + 1:fin])
        mitologia = "nordica" if num in nordicos else "griega;romana"
        docs.append(documento(f"bulfinch_{len(docs) + 1:02d}", f"Bulfinch, Chapter {num}: {' '.join(titulo.split())}",
                              "Bulfinch's Mythology: The Age of Fable", "Bulfinch", "divulgativa",
                              mitologia, URL_GUTENBERG.format(3327), pars))
    return docs


# --- Programa principal ---------------------------------------------------------

def main():
    obras = {
        "Ilíada": lambda: homero("iliada", "Iliad", 2199, "iliada"),
        "Odisea": lambda: homero("odisea", "Odyssey", 1727, "odisea"),
        "Hesíodo e himnos": hesiodo,
        "Metamorfosis I-VII": lambda: ovidio("ovidio_metamorfosis_1", 21765, "ovidio1"),
        "Metamorfosis VIII-XV": lambda: ovidio("ovidio_metamorfosis_2", 26073, "ovidio2"),
        "Edda poética": edda_poetica,
        "Edda en prosa": edda_prosa,
        "Apolodoro": apolodoro,
        "Bulfinch": bulfinch,
    }

    todos = []
    for nombre, funcion in obras.items():
        docs = funcion()
        todos += docs
        print(f"{nombre:22s} {len(docs):4d} documentos")

    df = pd.DataFrame(todos)
    df["palabras"] = df["texto"].str.split().str.len()
    cortos = df[df["palabras"] < MIN_PALABRAS]
    df = df[df["palabras"] >= MIN_PALABRAS]

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        for doc in df.drop(columns="palabras").to_dict("records"):
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # --- Resumen ---
    print("\n" + "=" * 60)
    print(f"TOTAL: {len(df)} documentos  ->  {RUTA_SALIDA}")
    print(f"Descartados por cortos (< {MIN_PALABRAS} palabras): {len(cortos)}"
          + (f"  {list(cortos['titulo'])}" if len(cortos) else ""))
    print("=" * 60)
    print("\nPor autor:")
    resumen = df.groupby("autor")["palabras"].agg(documentos="count", palabras="sum", mediana="median")
    print(resumen.astype(int).to_string())
    print(f"\nTotal de palabras: {df['palabras'].sum():,}")
    print("\nEjemplos (primer documento de cada autor):")
    for _, fila in df.groupby("autor").head(1).iterrows():
        print(f"\n  [{fila['doc_id']}] {fila['titulo']}")
        print(f"  {fila['texto'][:200]}...")


if __name__ == "__main__":
    main()