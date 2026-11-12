# Verificador de hechos sobre mitología

Proyecto final de Procesamiento del Lenguaje Natural (Máster en IA Aplicada, UC3M).

Sistema RAG que verifica afirmaciones sobre mitología grecorromana y nórdica.
Para cada afirmación da un veredicto (verdadero, falso, depende de la fuente o
sin información suficiente) y cita los fragmentos del corpus que lo justifican.

## Corpus

| Fuente | Tipo |
|---|---|
| Wikipedia (artículos de mitología griega, romana y nórdica) | Enciclopédica |
| Hesíodo, Homero, Ovidio, Eddas, Apolodoro | Primaria |
| World History Encyclopedia, Bulfinch | Divulgativa |
| Transcripciones de vídeos de YouTube | Divulgativa |

Los textos no se suben al repositorio: se generan con los scripts de `src/`.

## Estructura

- `src/`: código (descarga, troceo, preprocesado, recuperación, métricas)
- `notebooks/`: notebook ejecutado de cada hito
- `referencia/`: notebooks de la asignatura
- `queries.jsonl`: consultas de evaluación
- `MILESTONE_N.md`: resumen de cada hito

## Estado

- [ ] Hito 1: baseline de recuperación (TF-IDF y BM25)
- [ ] Hito 2: sistema RAG
- [ ] Hito 3: extensión
