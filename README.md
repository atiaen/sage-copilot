# Sage Copilot

A lightweight, self-hosted RAG (Retrieval-Augmented Generation) system for personal document Q&A. Runs entirely on local hardware — designed for home servers and edge devices.

---
## Journey to Build This
I've documented a large portion of building this [here](https://adeayo.dev/programmers_trap/)

---

## What It Does

Sage Copilot lets you chat with your documents. Point it at a folder of files, and it will:

1. **Ingest** — Parse PDFs, Word docs, text files, and more
2. **Index** — Create searchable embeddings stored locally
3. **Answer** — Respond to questions using retrieved context + a local LLM

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Streamlit  │────▶│  RAG Chain   │────▶│  Ollama (LLM)   │
│    (UI)     │     │(LangChain)   │     │  qwen3:0.6b     │
└─────────────┘     └──────────────┘     └─────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  ChromaDB    │
                    │ (Embeddings) │
                    └──────────────┘
                           ▲
                           │
┌─────────────┐     ┌──────────────┐
│  Nextcloud  │────▶│ FastAPI      │
│  Webhooks   │     │ (webhooks.py)│
└─────────────┘     └──────────────┘
```

---

## Key Components

| Module | Purpose |
|--------|---------|
| `app.py` | Streamlit chat interface |
| `src/embeddings.py` | Document parsing, chunking, indexing |
| `src/llm_query.py` | Multi-query RAG pipeline |
| `src/get_vector_db.py` | ChromaDB initialization |
| `webhooks.py` | FastAPI server for file-change events |
| `src/config.py` | Centralized configuration |

---

## Design Decisions

### Local-First
- **Ollama** for LLM inference — no API keys, no external dependencies
- **ChromaDB** with local persistence — data stays on your machine
- Lightweight models (`qwen3:0.6b`, `all-minilm`) for resource-constrained environments

### Multi-Query Retrieval
The system generates up to 5 question variations per user query to improve retrieval coverage and overcome limitations of simple similarity search.

### Modular Ingestion
- `unstructured` library handles 10+ file formats (PDF, DOCX, TXT, HTML, etc.)
- Recursive chunking with overlap preserves context across chunk boundaries
- Webhook endpoint enables real-time indexing from external sources (Nextcloud)

---

## Quick Start

### Prerequisites
- Python 3.10+
- Ollama installed and running locally

### Install

```bash
# Clone and setup
pip install -r requirements.txt

# Pull required models
ollama pull qwen3:0.6b
ollama pull all-minilm
```

### Configure

Create a `.env` file:

```env
NEXTCLOUD_PATH=/path/to/your/documents
OLLAMA_MODEL=qwen3:0.6b
EMBEDDING_MODEL=all-minilm
CHROMA_PATH=./chroma_db
COLLECTION_NAME=documents
```

### Run

```bash
# Start the chat UI
streamlit run app.py

# Or start the webhook server (optional)
python webhooks.py
```

---

## Usage

1. **Index documents**: The embedder automatically processes files from `NEXTCLOUD_PATH` on startup
2. **Ask questions**: Type queries in the Streamlit interface
3. **Get answers**: Responses are grounded in your indexed documents with source metadata

---

## Webhook Integration

The FastAPI server (`webhooks.py`) accepts webhooks from Nextcloud to auto-index files on upload:

```bash
# Start the webhook server
uvicorn webhooks:app --host 0.0.0.0 --port 8000
```

Endpoint: `POST /webhook/nextcloud`

---

## File Support

| Format | Status |
|--------|--------|
| PDF | Supported |
| DOCX/DOC | Supported |
| TXT / MD | Supported |
| HTML | Supported |
| CSV / XLSX | Supported |
| PPTX | Supported |

---

## Tech Stack

- **LangChain** — RAG pipeline, retrieval, prompting
- **Ollama** — Local LLM inference
- **ChromaDB** — Vector storage
- **Unstructured** — Document parsing
- **Streamlit** — UI
- **FastAPI** — Webhook server

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXTCLOUD_PATH` | `/home/deck/Documents/Books` | Directory to scan for documents |
| `OLLAMA_MODEL` | `qwen3:0.6b` | LLM model name |
| `EMBEDDING_MODEL` | `all-minilm` | Embedding model name |
| `CHROMA_PATH` | `./chroma_db` | Vector DB storage location |
| `COLLECTION_NAME` | `documents` | ChromaDB collection name |
