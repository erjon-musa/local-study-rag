"""
Embedding generator using nomic-embed-text via LM Studio.

Routes all embedding requests through LM Studio's OpenAI-compatible
/v1/embeddings endpoint, which handles routing to the PC GPU via LM Link.

Config (model id, base URL, API key) is centralized in `backend.config`.
Clients come from `backend.lm_client` so we share connection pools.
"""
from __future__ import annotations

from typing import List

from ..config import settings
from ..lm_client import get_async_client, get_sync_client


async def embed_texts(texts: List[str], model: str = None) -> List[List[float]]:
    """
    Generate embeddings for a list of texts via LM Studio.
    Returns a list of embedding vectors.
    """
    model = model or settings.embedding_model
    client = get_async_client()

    response = await client.embeddings.create(
        model=model,
        input=texts,
    )

    return [item.embedding for item in response.data]


def embed_texts_sync(texts: List[str], model: str = None) -> List[List[float]]:
    """
    Synchronous version of embed_texts.
    """
    model = model or settings.embedding_model
    client = get_sync_client()

    response = client.embeddings.create(
        model=model,
        input=texts,
    )

    return [item.embedding for item in response.data]


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

    Batching avoids sending huge payloads to the API.
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
