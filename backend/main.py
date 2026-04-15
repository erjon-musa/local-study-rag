"""
RAG Study Notes System — FastAPI Backend

Entry point for the API server. Run with:
    cd backend && uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from .api.chat import router as chat_router
from .api.courses import router as courses_router
from .api.documents import router as documents_router
from .api.graph_api import router as graph_router
from .generation.llm import check_ollama_health
from fastapi.staticfiles import StaticFiles

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
frontend_graph_path = Path(__file__).parent.parent / "frontend" / "graph"
if frontend_graph_path.exists():
    app.mount("/graph", StaticFiles(directory=frontend_graph_path, html=True), name="graph")


@app.get("/api/health")
async def health():
    """
    Health check — verifies Ollama is running.
    Does NOT load any model into RAM.
    """
    ollama_status = check_ollama_health()
    
    vault_path = os.getenv("VAULT_PATH", "")
    vault_exists = Path(vault_path).expanduser().exists() if vault_path else False

    return {
        "status": "ok" if ollama_status["status"] == "ok" else "degraded",
        "ollama": ollama_status,
        "vault_path": vault_path,
        "vault_exists": vault_exists,
    }
