"""
Embedding generator using nomic-embed-text via LM Studio.

Routes all embedding requests through LM Studio's OpenAI-compatible
/v1/embeddings endpoint, which handles routing to the PC GPU via LM Link.
"""
from __future__ import annotations

import os
from typing import List

import openai

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")


def _get_client() -> openai.OpenAI:
    return openai.OpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key="lmstudio-link",
    )


def _get_async_client() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key="lmstudio-link",
    )


async def embed_texts(texts: List[str], model: str = None) -> List[List[float]]:
    """
    Generate embeddings for a list of texts via LM Studio.
    Returns a list of embedding vectors.
    """
    model = model or EMBEDDING_MODEL
    client = _get_async_client()

    response = await client.embeddings.create(
        model=model,
        input=texts,
    )

    return [item.embedding for item in response.data]


def embed_texts_sync(texts: List[str], model: str = None) -> List[List[float]]:
    """
    Synchronous version of embed_texts.
    """
    model = model or EMBEDDING_MODEL
    client = _get_client()

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
