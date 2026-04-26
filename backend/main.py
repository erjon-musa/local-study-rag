"""
RAG Study Notes System — FastAPI Backend

Entry point for the API server. Run with:
    uvicorn backend.main:app --reload --port 8000   # from project root
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# `backend.config` loads .env on import — no manual load_dotenv needed here.
from .api.chat import router as chat_router
from .api.courses import router as courses_router
from .api.documents import router as documents_router
from .api.graph_api import router as graph_router
from .config import settings
from .generation.llm import check_health

app = FastAPI(
    title="RAG Study Notes",
    description="Chat with your course notes using RAG + Gemma4",
    version="1.0.0",
)

# CORS — allow the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(courses_router)
app.include_router(graph_router)

# Serve the static graph visualizer at /graph
frontend_graph_path = settings.project_root / "frontend" / "graph"
if frontend_graph_path.exists():
    app.mount("/graph", StaticFiles(directory=frontend_graph_path, html=True), name="graph")


@app.get("/api/health")
async def health():
    """
    Health check — verifies LM Studio is reachable.
    Does NOT load any model into Mac RAM.
    """
    llm_status = check_health()
    vault_exists = settings.vault_path.exists()

    return {
        "status": "ok" if llm_status["status"] == "ok" else "degraded",
        "llm": llm_status,
        "vault_path": str(settings.vault_path),
        "vault_exists": vault_exists,
    }
