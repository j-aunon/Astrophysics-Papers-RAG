# Astrophysics-Papers-RAG

Production-ready, fully local multimodal RAG system for answering scientific questions over astrophysics research PDFs.

This repo:
- Ingests PDFs, extracts text and figures, renders pages, and stores metadata in SQLite
- Indexes text chunks in persistent ChromaDB using `BAAI/bge-m3`
- Retrieves visually relevant pages using ColPali (page-level candidates only)
- Uses Tesseract OCR for figure text extraction
- Uses Qwen3-VL for figure captioning (caption + entities + scientific bullets)
- Uses Qwen2.5-7B-Instruct for final answer generation with strict citations

All logs, prompts, CLI output, and generated answers are English-only. Any non-English output triggers:
`Language policy violation: output must be English.`

## Quickstart

### 1) System dependencies (Linux)

Required for OCR:
- `tesseract-ocr`

Optional but recommended (PDF rendering speed / font support may vary by distro):
- `poppler-utils` (not required for PyMuPDF)

### 2) Python environment

Python 3.10+ recommended.

Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 3) Configure

Copy and edit:
```bash
cp config.example.yaml config.yaml
```

Key settings in `config.yaml`:
- `language.output_language` must be `en` (startup will error otherwise)
- `models.text_embedding_model` must be `BAAI/bge-m3`
- `models.vlm_model` must be a local Qwen3-VL model identifier
- `models.llm_model` must be `Qwen/Qwen2.5-7B-Instruct`
- Retrieval and chunking defaults are set in code (not configurable via YAML).

### 4) Ingest + index

Put PDFs in `documents/` (this repo already contains a few example PDFs), then run:
```bash
python scripts/ingest_index.py --pdf-dir documents
```

### 5) Query
```bash
python scripts/query.py --question "What evidence is presented for the main conclusion?"
```

## CLI

Use the consolidated CLI:
```bash
python -m src.cli --help
```

Key commands:
- `python -m src.cli ingest --pdf PATH_OR_DIR`
- `python -m src.cli index --pdf PATH_OR_DIR`
- `python -m src.cli query --question "..."`

## Data layout

Runtime artifacts are stored in `data/` by default:
- `data/metadata.sqlite3` (SQLite relationships)
- `data/chroma/` (ChromaDB persistent index)
- `data/pages/` (rendered page images)
- `data/figures/` (extracted figure images)
- `data/colpali/` (visual index artifacts)

## Model downloads

This project uses Hugging Face models locally. The first run will download weights unless you pre-populate a local cache.
Set:
- `HF_HOME=hf_cache` (or any local path)

## Notes on ColPali and Qwen3-VL

ColPali and Qwen3-VL must be available as local Python dependencies and models must be present in your local Hugging Face cache.
If a component is missing, the system will continue running with a stub (visual retrieval returns no results; VLM captioning is skipped) and logs a clear warning.

## Citations

The system uses strict citations:
- Text chunks: `[doc_id:page:chunk_id]`
- Figures: `[doc_id:page:figure_id]`

`doc_id` is the SQLite document id assigned during ingestion.

## Troubleshooting

- If you see `Language policy violation: output must be English.`: a model or OCR output contained non-English script; switch to English-only models/papers or disable the offending component and re-run.
- If OCR is disabled: install `tesseract-ocr` and verify `tesseract --version` works.
- If models fail to load: pre-download the weights into your local Hugging Face cache and set `HF_HOME` to a local directory.

## Security & privacy

No external APIs. No SaaS. Everything runs locally.
