"""
LM Studio client factories.

Single place to construct `openai.OpenAI` / `openai.AsyncOpenAI`
clients pointed at LM Studio. Replaces ad-hoc client construction
that used to live in llm.py, embedder.py, loader.py, graph.py, and
several scripts.

Pulls config from `backend.config.settings`. This module never reads
env vars directly — that's `config.py`'s job.

Usage:
    from backend.lm_client import get_sync_client, get_async_client

    client = get_sync_client()
    response = client.chat.completions.create(...)
"""
from __future__ import annotations

from functools import lru_cache

import openai

from .config import settings


@lru_cache(maxsize=1)
def get_sync_client() -> openai.OpenAI:
    """
    Shared sync OpenAI client pointed at LM Studio.
    Cached so repeated callers reuse the same connection pool.
    """
    return openai.OpenAI(
        base_url=settings.lmstudio_base_url,
        api_key=settings.lmstudio_api_key,
    )


@lru_cache(maxsize=1)
def get_async_client() -> openai.AsyncOpenAI:
    """
    Shared async OpenAI client pointed at LM Studio.
    Cached so repeated callers reuse the same connection pool.
    """
    return openai.AsyncOpenAI(
        base_url=settings.lmstudio_base_url,
        api_key=settings.lmstudio_api_key,
    )
