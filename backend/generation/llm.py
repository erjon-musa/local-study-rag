"""
Ollama Gemma4 client with streaming and RAM management.

All requests use keep_alive="0" to immediately unload the model
from RAM after each generation. Gemma4 (8B) uses ~10GB RAM, so
this is critical on a Mac.
"""
from __future__ import annotations

import json
import os
from typing import AsyncIterator, Iterator, Optional

import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemma4:latest")
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "0")


def generate_stream(
    prompt: str,
    system: str = "",
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> Iterator[str]:
    """
    Stream a response from Gemma4 via Ollama.
    
    Yields text chunks as they arrive.
    Uses keep_alive="0" to unload from RAM when done.
    """
    model = model or GENERATION_MODEL

    with httpx.Client(timeout=300.0) as client:
        with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": True,
                "keep_alive": KEEP_ALIVE,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue


async def generate_stream_async(
    prompt: str,
    system: str = "",
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> AsyncIterator[str]:
    """
    Async streaming generation from Gemma4 via Ollama.
    
    Yields text chunks as they arrive.
    Uses keep_alive="0" to unload from RAM when done.
    """
    model = model or GENERATION_MODEL

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": True,
                "keep_alive": KEEP_ALIVE,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue


def generate(
    prompt: str,
    system: str = "",
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """
    Non-streaming generation. Returns the full response as a string.
    Uses keep_alive="0" to unload from RAM when done.
    """
    tokens = []
    for token in generate_stream(prompt, system, model, temperature, max_tokens):
        tokens.append(token)
    return "".join(tokens)


def check_ollama_health() -> dict:
    """
    Check if Ollama is running WITHOUT loading any model.
    Just hits the /api/tags endpoint to list available models.
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"status": "ok", "models": models}
    except Exception as e:
        return {"status": "error", "error": str(e)}
