"""
Descarga los subtítulos en inglés de los vídeos de data/youtube_seleccion.csv y los
convierte en documentos del corpus. No descarga ningún vídeo, solo los subtítulos.

- Si el vídeo tiene subtítulos hechos por personas, se usan esos; si no, los
  automáticos de YouTube (peor calidad: sin puntuación y con algún error).
- Cada documento es un vídeo. El texto se divide en bloques de unos 30 segundos
  marcados como secciones "== t=125 ==" (segundo 125). Así troceo.py sabe en qué
  segundo empieza cada fragmento y el sistema podrá citar "minuto 2:05".
- Se puede interrumpir con Ctrl+C y retomar: los subtítulos ya descargados se saltan.

IMPORTANTE: ejecutar en un ordenador personal (YouTube bloquea las IP de la nube).

Uso (desde la carpeta principal del repositorio, con el entorno activado):
    python src/descarga_youtube.py

Salida:
    data/raw/youtube/subs/<video_id>.en.json3   -> subtítulos tal cual
    data/raw/youtube/subs/<video_id>.meta.json  -> si son manuales o automáticos
    data/processed/youtube.jsonl                -> documentos (mismo formato que el resto)
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from yt_dlp import YoutubeDL

RAIZ = Path(__file__).resolve().parents[1]
RUTA_SELECCION = RAIZ / "data" / "youtube_seleccion.csv"
CARPETA_SUBS = RAIZ / "data" / "raw" / "youtube" / "subs"
RUTA_SALIDA = RAIZ / "data" / "processed" / "youtube.jsonl"

SEGUNDOS_POR_BLOQUE = 30
PAUSA = 2   # segundos entre vídeos, para no saturar a YouTube


def descargar_subtitulos(video_id):
    """
    Descarga los subtítulos en inglés de un vídeo (formato json3, con marcas de tiempo).
    Devuelve "manuales", "automaticos" o None si el vídeo no tiene subtítulos en inglés.
    """
    opciones = {
        "skip_download": True,               # nada de vídeo
        "writesubtitles": True,              # subtítulos hechos por personas...
        "writeautomaticsub": True,           # ...y, si no los hay, los automáticos
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "json3",
        "outtmpl": str(CARPETA_SUBS / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "sleep_interval_subtitles": 1,
    }
    with YoutubeDL(opciones) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    manuales = set((info or {}).get("subtitles") or {})
    pedidos = set((info or {}).get("requested_subtitles") or {})
    if not pedidos:
        return None
    return "manuales" if pedidos & manuales else "automaticos"


def fichero_subs(video_id):
    """Ruta del fichero de subtítulos descargado (el idioma puede ser en, en-US o en-GB)."""
    encontrados = sorted(CARPETA_SUBS.glob(f"{video_id}.en*.json3"))
    return encontrados[0] if encontrados else None


def leer_json3(ruta):
    """Lee un json3 de YouTube y devuelve una lista de (segundo, texto)."""
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    frases = []
    for evento in datos.get("events", []):
        texto = "".join(seg.get("utf8", "") for seg in evento.get("segs", []) or [])
        texto = re.sub(r"\[[^\]]*\]", " ", texto)          # [Music], [Applause]...
        texto = " ".join(texto.replace("\n", " ").split())
        if texto:
            frases.append((evento.get("tStartMs", 0) / 1000, texto))
    return frases


def a_texto_con_bloques(frases):
    """
    Junta los subtítulos en bloques de unos SEGUNDOS_POR_BLOQUE segundos, cada uno con
    una cabecera "== t=SEGUNDO ==" para que el troceo sepa dónde empieza cada fragmento.
    """
    bloques, actual, inicio = [], [], None
    for segundo, texto in frases:
        if inicio is None:
            inicio = segundo
        if segundo - inicio >= SEGUNDOS_POR_BLOQUE and actual:
            bloques.append(f"== t={int(inicio)} ==\n{' '.join(actual)}")
            actual, inicio = [], segundo
        actual.append(texto)
    if actual:
        bloques.append(f"== t={int(inicio)} ==\n{' '.join(actual)}")
    return "\n\n".join(bloques)


def main():
    seleccion = pd.read_csv(RUTA_SELECCION)
    CARPETA_SUBS.mkdir(parents=True, exist_ok=True)

    # 1. Descarga (se salta lo ya descargado)
    sin_subs, errores = [], []
    for v in tqdm(seleccion.itertuples(), total=len(seleccion), desc="Subtítulos"):
        meta = CARPETA_SUBS / f"{v.video_id}.meta.json"
        if meta.exists():
            continue
        try:
            tipo = descargar_subtitulos(v.video_id)
        except Exception as e:                       # vídeo privado, borrado, bloqueo...
            errores.append((v.video_id, v.titulo, str(e)[:120]))
            time.sleep(PAUSA)
            continue
        meta.write_text(json.dumps({"subtitulos": tipo}), encoding="utf-8")
        if tipo is None:
            sin_subs.append(v.titulo)
        time.sleep(PAUSA)

    # 2. Conversión a documentos
    docs = []
    for v in seleccion.itertuples():
        ruta, meta = fichero_subs(v.video_id), CARPETA_SUBS / f"{v.video_id}.meta.json"
        if ruta is None or not meta.exists():
            continue
        texto = a_texto_con_bloques(leer_json3(ruta))
        if len(texto.split()) < 50:
            continue
        docs.append({
            "doc_id": f"yt_{v.video_id}",
            "titulo": v.titulo,
            "obra": v.canal,
            "autor": v.canal,
            "tipo_fuente": "divulgativa",
            "nivel": v.nivel,                       # experta | popular
            "mitologia": v.mitologia,
            "url": v.url,
            "subtitulos": json.loads(meta.read_text())["subtitulos"],
            "texto": texto,
        })

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # --- Resumen ---
    df = pd.DataFrame(docs)
    print("\n" + "=" * 60)
    print(f"Vídeos seleccionados: {len(seleccion)}  ->  documentos: {len(df)}  ({RUTA_SALIDA})")
    print(f"Sin subtítulos en inglés (esta ejecución): {len(sin_subs)}")
    print(f"Errores (esta ejecución): {len(errores)}"
          + ("  -> vuelve a ejecutar para reintentarlos" if errores else ""))
    for e in errores[:10]:
        print(f"   {e[0]}  {e[1][:50]}  |  {e[2]}")
    print("=" * 60)
    if len(df):
        df["palabras"] = df["texto"].str.split().str.len()
        print("\nPor tipo de subtítulos:")
        print(df["subtitulos"].value_counts().to_string())
        print("\nPor nivel:")
        print(df.groupby("nivel")["palabras"].agg(videos="count", palabras="sum").to_string())
        print(f"\nTotal de palabras: {df['palabras'].sum():,}")
        ejemplo = df.iloc[0]
        print(f"\nEjemplo [{ejemplo['doc_id']}] {ejemplo['titulo']} ({ejemplo['subtitulos']}):")
        print(ejemplo["texto"][:400] + "...")


if __name__ == "__main__":
    main()