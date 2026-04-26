"""
Centralized configuration for local-study-rag.

Single source of truth for env-driven settings. Replaces ad-hoc
`os.getenv()` calls that used to live in `llm.py`, `embedder.py`,
`loader.py`, `graph.py`, `pipeline.py`, `retriever.py`, and several
scripts — each with its own copy of the defaults, which is exactly
how the model-identifier drift bug crept in.

Usage:
    from backend.config import settings

    print(settings.lmstudio_model)
    print(settings.chroma_persist_dir)

The `settings` singleton is constructed once at import time. Don't
instantiate `Settings` yourself; tests that need to override values
should monkeypatch attributes on the singleton or set env vars before
importing this module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of backend/. Load .env from there so callers
# don't have to load_dotenv themselves.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    """
    All env-driven config in one place. Frozen so callers can't mutate
    it accidentally — the values are baked at import time.
    """

    # ── LM Studio ───────────────────────────────────────────────────
    lmstudio_base_url: str = _env("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    lmstudio_api_key: str = _env("LMSTUDIO_API_KEY", "lmstudio-link")
    lmstudio_model: str = _env("LMSTUDIO_MODEL", "google/gemma-3-27b-it")
    embedding_model: str = _env(
        "EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"
    )

    # ── Vault + storage ─────────────────────────────────────────────
    vault_path: Path = Path(
        _env("VAULT_PATH", str(Path.home() / "Documents" / "StudyVault"))
    ).expanduser()
    chroma_persist_dir: str = _env(
        "CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma")
    )
    manifest_path: Path = Path(
        _env("MANIFEST_PATH", str(PROJECT_ROOT / "data" / "manifest.json"))
    )
    graph_output_path: Path = Path(
        _env("GRAPH_OUTPUT_PATH", str(PROJECT_ROOT / "data" / "knowledge_graph.json"))
    )

    # ── ChromaDB ────────────────────────────────────────────────────
    chroma_collection_name: str = "study_notes"

    # ── Chunking ────────────────────────────────────────────────────
    chunk_size: int = _env_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 100)

    # ── OCR (loader.py) ─────────────────────────────────────────────
    use_local_ocr: bool = _env_bool("USE_LOCAL_OCR", True)
    local_ocr_model: str = _env("LOCAL_OCR_MODEL", "lightonai/LightOnOCR-2-1B")
    ocr_max_image_dim: int = _env_int("OCR_MAX_IMAGE_DIM", 1024)
    # Hard kill-switch for dev iterations where we don't want MPS/Gemma OCR
    # churn. Set DISABLE_OCR=1 in the env before importing backend.config.
    disable_ocr: bool = _env_bool("DISABLE_OCR", False)

    # ── Project ─────────────────────────────────────────────────────
    project_root: Path = PROJECT_ROOT


# Module-level singleton — import this, don't construct Settings yourself.
settings = Settings()
