"""
LLM client — routes all generation to PC GPU via LM Studio LM Link.

No local model loading. If LM Studio is unreachable, we error
instead of silently loading a 10GB model into Mac RAM.
"""
from __future__ import annotations

import os
from typing import AsyncIterator, Iterator

import httpx
import openai

# LM Studio Config (via LM Link proxy on localhost)
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "google/gemma-4-26b-a4b")


def generate_stream(
    prompt: str,
    system: str = "",
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> Iterator[str]:
    """
    Stream a response from the PC GPU via LM Studio.
    Yields text chunks as they arrive.
    """
    client = openai.OpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key="lmstudio-link",
    )

    system_prompt = system + "\n\nDo not use internal reasoning. Respond directly."

    stream = client.chat.completions.create(
        model=model or LMSTUDIO_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=0.3,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        token = delta.content or ""
        reasoning = getattr(delta, "reasoning_content", None) or ""
        combined = token or reasoning
        if combined:
            yield combined


async def generate_stream_async(
    prompt: str = None,
    system: str = "",
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    messages: list = None,
) -> AsyncIterator[str]:
    """
    Async stream from the PC GPU via LM Studio.
    Yields text chunks as they arrive.

    Two calling modes (exactly one must be provided):
      - Legacy single-turn:  prompt="..." [+ system="..."]
        Internally wraps into a messages list with the system + user turns.
      - Multi-turn / history: messages=[{"role": "system"|"user"|"assistant",
                                          "content": "..."}, ...]
        Passed through unchanged to LM Studio's OpenAI-compatible API.
        Callers are responsible for building the full messages list
        (system prompt + neutralized history + current user turn).
    """
    if (prompt is None) == (messages is None):
        raise ValueError(
            "generate_stream_async: provide exactly one of `prompt` or `messages`"
        )

    client = openai.AsyncOpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key="lmstudio-link",
    )

    if messages is None:
        # Legacy path: build the simple system+user messages list.
        system_prompt = system + "\n\nDo not use internal reasoning. Respond directly."
        request_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        # Multi-turn path: pass through as-is. Caller owns message construction.
        request_messages = messages

    stream = await client.chat.completions.create(
        model=model or LMSTUDIO_MODEL,
        messages=request_messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=0.3,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # We intentionally IGNORE reasoning_content so the user only sees the final answer.
        # The frontend will just show '...' while the model thinks.

        # Append actual content
        if getattr(delta, "content", None):
            yield delta.content


def generate(
    prompt: str,
    system: str = "",
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """Non-streaming generation. Returns the full response as a string."""
    tokens = []
    for token in generate_stream(prompt, system, model, temperature, max_tokens):
        tokens.append(token)
    return "".join(tokens)


def check_health() -> dict:
    """Check if LM Studio is running and the model is loaded."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{LMSTUDIO_BASE_URL}/models")
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            return {"status": "ok", "backend": "lmstudio", "models": models}
    except Exception as e:
        return {"status": "error", "backend": "lmstudio", "error": str(e)}
