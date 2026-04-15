"""
Embedding generator using nomic-embed-text via Ollama.

All requests use keep_alive="0" to immediately unload the model
from RAM after each batch. nomic-embed-text is ~275MB so load/unload
is nearly instant.
"""
from __future__ import annotations

import os
from typing import List

import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "0")


async def embed_texts(texts: List[str], model: str = None) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Ollama.
    
    Uses keep_alive="0" to unload the model immediately after use.
    Returns a list of embedding vectors (768 dimensions for nomic-embed-text).
    """
    model = model or EMBEDDING_MODEL
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": model,
                "input": texts,
                "keep_alive": KEEP_ALIVE,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError(f"No embeddings returned. Response: {data}")
        
        return embeddings


def embed_texts_sync(texts: List[str], model: str = None) -> List[List[float]]:
    """
    Synchronous version of embed_texts for use outside async contexts.
    """
    model = model or EMBEDDING_MODEL
    
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": model,
                "input": texts,
                "keep_alive": KEEP_ALIVE,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError(f"No embeddings returned. Response: {data}")
        
        return embeddings


def embed_single(text: str, model: str = None) -> List[float]:
    """Embed a single text string. Returns one embedding vector."""
    results = embed_texts_sync([text], model=model)
    return results[0]


async def embed_single_async(text: str, model: str = None) -> List[float]:
    """Async version: embed a single text string."""
    results = await embed_texts(texts=[text], model=model)
    return results[0]


def embed_in_batches(texts: List[str], batch_size: int = 32, model: str = None) -> List[List[float]]:
    """
    Embed a large list of texts in batches.
    
    nomic-embed-text is fast, but batching avoids sending huge payloads.
    Model loads once for the first batch and unloads after the last batch
    (keep_alive="0" on every call means Ollama manages the lifecycle).
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embed_texts_sync(batch, model=model)
        all_embeddings.extend(batch_embeddings)
        
        # Progress feedback for large batches
        done = min(i + batch_size, len(texts))
        if len(texts) > batch_size:
            print(f"  Embedded {done}/{len(texts)} chunks...")
    
    return all_embeddings
