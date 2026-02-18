# botsentencias

Sistema RAG en Python para descargar fallos de la CSJN (PDF), indexarlos en una base vectorial y responder preguntas jurídicas con citas de fuentes.

## Requisitos

- Python 3.10+
- Instalar dependencias: `python -m pip install -r requirements.txt`

## Configuración

1. Copiar variables:

```bash
cp .env.example .env
```

2. Ajustar `.env` según backend de embeddings/LLM.

### Backends soportados

- **Embeddings**
  - `sentence-transformers` (local, recomendado para pruebas)
  - `openai`
- **LLM**
  - `local-echo` (no genera respuesta doctrinal completa, solo diagnóstico)
  - `ollama` (si hay servidor local)
  - `openai`
  - `anthropic`

## Flujo de trabajo

### 1) Scraping y descarga de PDFs

```bash
python -m botsentencias.cli scrape
```

Opcional para pruebas:

```bash
python -m botsentencias.cli scrape --limit 10
```

Modo descubrimiento sin descarga:

```bash
python -m botsentencias.cli scrape --discover-only --limit 20
```

Qué hace:
- Crawlea enlaces desde `BASE_URL` y subpáginas compatibles.
- Detecta PDFs, filtra por `MIN_YEAR` (>=1994 por defecto).
- Descarga con reintentos exponenciales, delay entre requests y registro de errores.
- Guarda PDFs en `PDF_DIR/{año}/archivo.pdf`.
- Registra estado en SQLite (`pdf_registry`) y exporta CSV (`data/pdf_registry.csv`).

### 2) Procesamiento e indexado vectorial

```bash
python -m botsentencias.cli process
```

Qué hace:
- Lee PDFs descargados.
- Extrae texto con **PyMuPDF**.
- Segmenta en chunks (máx tokens + overlap configurables).
- Genera embeddings.
- Inserta en **ChromaDB persistente**.
- Registra avance incremental para reanudar sin reprocesar (`processing_registry`).

### 3) Consultas al agente RAG (CLI)

```bash
python -m botsentencias.cli ask "¿Cuál es el criterio de la Corte sobre prescripción adquisitiva administrativa?"
```

Salida:
- Respuesta del agente.
- Lista JSON de fuentes citadas con metadatos y score.

## Estructura

- `src/botsentencias/scraper.py`: crawl + descarga robusta.
- `src/botsentencias/processing.py`: extracción, chunking, embeddings, Chroma.
- `src/botsentencias/rag.py`: retrieval híbrido (vector + BM25) + generación.
- `src/botsentencias/registry.py`: SQLite para estado incremental.
- `src/botsentencias/cli.py`: comandos `scrape`, `process`, `ask`.

## Manejo de errores

- Reintentos con backoff exponencial en red.
- Registro de errores por PDF en SQLite.
- PDFs faltantes/corruptos se marcan como error y no detienen todo el pipeline.
- Si no hay evidencia suficiente, el agente lo indica explícitamente.

## Notas de operación responsable

- Mantener `REQUEST_DELAY_SECONDS >= 1.0` para no sobrecargar el sitio.
- Evitar paralelismo masivo contra CSJN.
- El primer procesamiento puede tardar horas según volumen y backend.
