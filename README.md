# RAG Study Notes System

A **Retrieval-Augmented Generation** system for chatting with your course notes, lecture slides, and study materials. Built for finals prep — ask questions, get accurate answers with source citations.

<p align="center">
  <img src="assets/chat_ui.png" width="48%" />
  <img src="assets/documents_ui.png" width="48%" />
</p>

**High-Performance Local Inference** — powered by Gemma4 26B via LM Studio networked GPU offloading. Complete privacy, zero cloud costs, maximum speed.


## Features

- 🔍 **Hybrid Search** — Semantic (vector) + BM25 keyword search with Reciprocal Rank Fusion
- 🎯 **Doc-Type-Aware Retrieval** — Explanatory queries boost lectures/resources and penalize exam keyword hits
- 📚 **Multi-Format Ingestion** — PDF, DOCX, Markdown, TXT, HTML support
- 🧠 **Smart Chunking** — Heading-aware splitting for markdown, page-based for PDFs, sentence-boundary for text
- 👁️ **OCR Fallback** — Scanned PDFs auto-detected via grayscale stddev; routed through local LightOnOCR-2-1B (Apple Silicon MPS), with Gemma4 vision as a second fallback
- 💬 **Streaming Chat** — Real-time NDJSON streaming with live token render
- 🔢 **Inline `[N]` Citations** — Numbered markers in the answer link/scroll to the matching source card
- 🗣️ **Multi-Turn Memory** — Session-scoped history (last 6 turns) so follow-ups like *"what's its time complexity?"* resolve against prior context
- 🛑 **Honest Empty-State** — Low-agreement retrievals render a distinct "I don't see this in your notes" card instead of a hallucinated refusal
- 📎 **Source Citations** — Every answer links back to the exact file and page
- 🔄 **Incremental Sync** — Auto-detects new/modified/deleted files via SHA-256 manifest
- 📁 **Drag & Drop Upload** — Add new files directly from the browser
- 🎯 **Course Filtering** — Persistent course pill; switching courses clears chat history atomically
- ⚡ **GPU Offloading** — Heavy generation and multimodal OCR bypass Mac RAM entirely and execute over the network on a dedicated Windows GPU.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Next.js Frontend (React 19)               │
│  Chat UI · Course Pill · Inline [N] Citations · Sources   │
└─────────────────────────────┬────────────────────────────┘
                              │ REST API (NDJSON streaming)
┌─────────────────────────────▼────────────────────────────┐
│                  FastAPI Backend (Python)                  │
│                                                            │
│  Ingestion:  Load → OCR (if scanned) → Chunk → Embed      │
│  Retrieval:  Vector + BM25 → RRF + Doc-Type Boost         │
│  Chain:      Classify (empty/weak/good) → History-Aware    │
│              Prompt → Gemma4 (streaming)                   │
│                                                            │
│  OCR (local): LightOnOCR-2-1B on Apple Silicon MPS         │
│               ↳ Gemma4 multimodal fallback                 │
│                                                            │
│  LM Studio (remote Windows GPU box)                        │
│  ├── google/gemma-4-26b-a4b  → generation & vision        │
│  └── nomic-embed-text-v1.5   → embeddings (768-dim)       │
└───────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Gemma4 26B via LM Studio (OpenAI endpoint) |
| **Embeddings** | nomic-embed-text-v1.5 via LM Studio |
| **Vector DB** | ChromaDB (persistent, local) |
| **Keyword Search** | BM25 via rank-bm25 |
| **Reranking** | Reciprocal Rank Fusion (no model needed) |
| **Backend** | Python, FastAPI, uvicorn |
| **Frontend** | Next.js 16, React, TypeScript |
| **PDF Processing** | PyMuPDF (fitz) |
| **Doc Organization** | Custom Python script → Obsidian vault |

## Quick Start

### Prerequisites

- [LM Studio](https://lmstudio.ai/) running as a local server on a dedicated GPU machine:
  - Load `google/gemma-4-26b` for Vision/Chat
  - Load `nomic-embed-text-v1.5` for Embeddings
- Python 3.9+
- Node.js 18+

### Setup

```bash
# Clone
git clone https://github.com/erjon-musa/local-study-rag.git
cd local-study-rag

# Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Configure
cp .env.example .env
# Edit .env to set your VAULT_PATH and LMSTUDIO_BASE_URL

# Frontend
cd frontend && npm install && cd ..
```

### Run

```bash
# Terminal 1: Backend
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev -- -p 3001
```

Open [http://localhost:3001](http://localhost:3001) to start chatting with your notes.

### First-Time Ingestion

1. Place your study materials in the vault directory (configured in `.env`)
2. Click **Sync Vault** on the Documents page, or hit the API:
   ```bash
   curl -X POST http://localhost:8000/api/documents/sync
   ```
3. Start asking questions!

## Adding New Files

Three ways to add content:

1. **Drop files in the vault folder** → Auto-detected on next query
2. **Drag & drop in the UI** → Documents page upload zone
3. **Click Sync** → Manual re-scan of vault directory

The system uses a SHA-256 manifest to track what's been ingested. Only new/modified files are processed — unchanged files are skipped instantly.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Streaming chat — accepts `question`, `course`, `top_k`, `history[]` (last 6 turns) |
| `POST` | `/api/chat/simple` | Non-streaming chat (for testing) |
| `POST` | `/api/documents/sync` | Scan vault for changes and ingest |
| `POST` | `/api/documents/upload` | Upload file to vault + auto-ingest |
| `GET` | `/api/documents` | List indexed documents (includes OCR stats per file) |
| `GET` | `/api/documents/stats` | Index statistics |
| `GET` | `/api/courses` | List courses with counts |
| `GET` | `/api/health` | Health check (doesn't load models) |

### Streaming Protocol (NDJSON)

`POST /api/chat` emits newline-delimited JSON events in order:

1. `{"type": "sources", "data": [...]}` — source metadata arrives first
2. `{"type": "token", "data": "..."}` — individual tokens as they generate
3. `{"type": "done"}` — stream complete

## Project Structure

```
local-study-rag/
├── backend/
│   ├── main.py                      # FastAPI entry point
│   ├── ingestion/                   # Load → Chunk → Embed → Store
│   │   ├── loader.py                # Multi-format readers + OCR trigger
│   │   ├── chunker.py               # Smart text chunking
│   │   ├── embedder.py              # nomic-embed-text via LM Studio
│   │   ├── local_ocr.py             # LightOnOCR-2-1B on Apple Silicon MPS
│   │   └── pipeline.py              # Incremental ingestion orchestrator
│   ├── retrieval/                   # Hybrid search engine
│   │   ├── vector_search.py         # ChromaDB semantic search
│   │   ├── keyword_search.py        # BM25 keyword search
│   │   ├── reranker.py              # Reciprocal Rank Fusion + doc-type boost
│   │   └── retriever.py             # Unified interface + diagnostics
│   ├── generation/                  # LLM generation
│   │   ├── llm.py                   # LM Studio client (sync + async streaming)
│   │   ├── prompts.py                # System prompt + empty-state template
│   │   │                             #   + history citation neutralization
│   │   └── chain.py                 # Retrieve → classify → generate chain
│   └── api/                         # REST endpoints
│       ├── chat.py                  # Streaming chat + history forwarding
│       ├── documents.py             # Document management
│       └── courses.py               # Course listing
├── frontend/
│   └── src/
│       ├── app/page.tsx             # Chat page (messages state + abort)
│       ├── components/chat/
│       │   ├── CourseSelectorPill.tsx  # Persistent course scope pill
│       │   ├── ChatMessage.tsx      # Inline [N] citation rendering
│       │   └── SourceCard.tsx       # Numbered, jump-to-source cards
│       └── lib/
│           ├── api.ts               # streamChat (NDJSON consumer)
│           └── citations.ts         # Buffered [N] token parser
├── scripts/
│   ├── build_vault.py               # Build the Obsidian-style vault
│   ├── rebuild_manifest.py          # Reconstruct manifest from ChromaDB
│   ├── force_reocr.py               # Re-run OCR for flagged files
│   └── calibrate_classification.py  # Tune retrieval thresholds
├── tests/
│   └── test_smoke_queries.py        # 6 golden queries + multi-turn memory
└── data/                            # ChromaDB + manifest (gitignored)
```

## Testing

Golden-query smoke tests live in `tests/test_smoke_queries.py`. They run 2 queries per course across 3 courses, plus a multi-turn memory test (pronoun resolution across turns):

```bash
source .venv/bin/activate
PYTHONPATH=. python tests/test_smoke_queries.py
```

Tests skip (rather than fail) when LM Studio is unreachable — that's a local-environment concern, not a correctness regression.

## Purpose

This project was built to explore and demonstrate **RAG architecture from scratch** — without relying on frameworks like LangChain. Every component (ingestion, chunking, embedding, retrieval, reranking, generation) is implemented manually to understand the full pipeline.

Built with a multi-agent workflow using Claude Code.

## License

MIT
