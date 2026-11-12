"""
Troceo (chunking) de documentos en fragmentos para indexar.

Cada documento se parte en fragmentos de unas OBJETIVO palabras, sin cortar
frases por la mitad, con un solapamiento de un SOLAPE (15 %) entre fragmentos
consecutivos para no perder contexto en los bordes.

- Si el texto tiene secciones de Wikipedia (== Sección ==), se intenta empezar
  un fragmento nuevo en cada sección (salvo que el fragmento actual sea aún muy
  corto, para no crear fragmentos diminutos), y cada fragmento guarda el nombre
  de su sección.
- Cada fragmento conserva el id de su documento y tiene uno propio: doc_123_c07.

Uso como script (prueba sobre un fichero de documentos .jsonl):
    python src/troceo.py data/processed/clasicos.jsonl

Uso desde otro código o notebook:
    from troceo import trocear_documento
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

OBJETIVO = 280          # palabras por fragmento (el profesor sugiere 200-400)
SOLAPE = 0.15           # 15 % de solapamiento entre fragmentos consecutivos
MINIMO = 0.5            # no se abre fragmento nuevo en una sección si el actual tiene < 50 % del objetivo
MAXIMO = 1.4            # el último fragmento se une al anterior si juntos no pasan del 140 % del objetivo

CABECERA = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$")         # == Sección ==
FIN_FRASE = re.compile(r"(?<=[.!?;])\s+(?=[\"“‘'(\[A-Z0-9])")


def frases_con_seccion(texto):
    """
    Recorre el texto y devuelve una lista de (sección, frase).
    Las cabeceras == Sección == no se devuelven como frases: solo cambian la sección.
    Las frases muy largas (sin puntos, p. ej. versos) se parten en trozos de OBJETIVO palabras.
    """
    resultado, seccion = [], ""
    for parrafo in re.split(r"\n\s*\n", texto):
        lineas_texto = []
        for linea in parrafo.split("\n"):
            m = CABECERA.match(linea.strip())
            if m:                              # cabecera: cambia la sección
                if lineas_texto:
                    resultado += [(seccion, f) for f in partir_frases(" ".join(lineas_texto))]
                    lineas_texto = []
                seccion = m.group(2)
            elif linea.strip():
                lineas_texto.append(linea.strip())
        if lineas_texto:
            resultado += [(seccion, f) for f in partir_frases(" ".join(lineas_texto))]
    return resultado


def partir_frases(parrafo):
    """Parte un párrafo en frases; las que superan OBJETIVO palabras se cortan por palabras."""
    frases = []
    for f in FIN_FRASE.split(parrafo):
        palabras = f.split()
        for i in range(0, len(palabras), OBJETIVO):
            frases.append(" ".join(palabras[i:i + OBJETIVO]))
    return [f for f in frases if f]


def trocear_documento(doc_id, texto):
    """
    Parte un documento en fragmentos. Devuelve una lista de diccionarios con
    chunk_id, doc_id, seccion, texto y n_palabras.
    """
    frases = frases_con_seccion(texto)
    if not frases:
        return []
    n = [len(f.split()) for _, f in frases]

    trozos = []                     # cada trozo es una lista de índices de frases
    actual, palabras = [], 0
    for i, (seccion, _) in enumerate(frases):
        cambio_seccion = actual and seccion != frases[actual[-1]][0]
        # Se cierra el trozo si con la frase nueva pasa del objetivo... salvo que el trozo
        # sea aún muy corto y quepa sin pasar del máximo (así no salen trozos diminutos
        # cuando detrás viene una frase muy larga)
        excede = palabras + n[i] > OBJETIVO
        cerrar = excede and (palabras >= MINIMO * OBJETIVO or palabras + n[i] > MAXIMO * OBJETIVO)
        if actual and (cerrar or (cambio_seccion and palabras >= MINIMO * OBJETIVO)):
            trozos.append(actual)
            # Solapamiento: el nuevo trozo empieza con las últimas frases del anterior
            # (solo si seguimos en la misma sección, y nunca con frases muy largas,
            # para no repetir medio trozo)
            solape, p = [], 0
            if not cambio_seccion:
                for j in reversed(actual):
                    if p >= SOLAPE * OBJETIVO or len(solape) + 1 >= len(actual):
                        break
                    if n[j] > 1.5 * SOLAPE * OBJETIVO:
                        break
                    solape.insert(0, j)
                    p += n[j]
            actual, palabras = solape, p
        actual.append(i)
        palabras += n[i]
    if actual:
        trozos.append(actual)

    # Si el último trozo es muy corto, se une al anterior (sin repetir las frases del solape)
    if len(trozos) > 1:
        ultimo = [j for j in trozos[-1] if j not in trozos[-2]]
        if sum(n[j] for j in trozos[-2] + ultimo) <= MAXIMO * OBJETIVO:
            trozos[-2] += ultimo
            trozos.pop()

    fragmentos = []
    for k, trozo in enumerate(trozos, start=1):
        texto_trozo = " ".join(frases[j][1] for j in trozo)
        fragmentos.append({
            "chunk_id": f"{doc_id}_c{k:02d}",
            "doc_id": doc_id,
            "seccion": frases[trozo[0]][0],
            "texto": texto_trozo,
            "n_palabras": len(texto_trozo.split()),
        })
    return fragmentos


def main():
    """Prueba: trocea un .jsonl de documentos, guarda los fragmentos y muestra un resumen."""
    if len(sys.argv) < 2:
        print("Uso: python src/troceo.py ruta/a/documentos.jsonl")
        sys.exit(1)
    entrada = Path(sys.argv[1])
    salida = entrada.with_name(entrada.stem + "_chunks.jsonl")

    docs = pd.read_json(entrada, lines=True)
    fragmentos = []
    for doc in docs.itertuples():
        fragmentos += trocear_documento(doc.doc_id, doc.texto)

    with open(salida, "w", encoding="utf-8") as f:
        for frag in fragmentos:
            f.write(json.dumps(frag, ensure_ascii=False) + "\n")

    df = pd.DataFrame(fragmentos)
    por_doc = df.groupby("doc_id").size()
    print(f"Documentos: {len(docs)}  ->  fragmentos: {len(df)}  ({salida})")
    print(f"Fragmentos por documento: mediana {por_doc.median():.0f}, máximo {por_doc.max()}")
    print("\nPalabras por fragmento:")
    print(df["n_palabras"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(0).to_string())
    print(f"\nFragmentos con menos de 100 palabras: {(df['n_palabras'] < 100).sum()}"
          " (suelen ser documentos cortos enteros, p. ej. himnos breves)")

    # Ejemplo: dos fragmentos consecutivos para ver el solapamiento
    ejemplo = df[df["doc_id"] == por_doc.idxmax()].head(2)
    print("\nEjemplo de dos fragmentos consecutivos (fíjate en el solape):")
    for frag in ejemplo.itertuples():
        print(f"\n[{frag.chunk_id}] ({frag.n_palabras} palabras)")
        print(f"  INICIO: {frag.texto[:160]}...")
        print(f"  FINAL:  ...{frag.texto[-160:]}")


if __name__ == "__main__":
    main()