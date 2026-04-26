"""
LLM client — routes all generation to PC GPU via LM Studio LM Link.

No local model loading. If LM Studio is unreachable, we error
instead of silently loading a 10GB model into Mac RAM.

Sync vs async asymmetry (deliberate)
------------------------------------
`generate_stream` (sync)        — yields BOTH `content` and `reasoning_content`.
                                  Used by the legacy non-streaming `RAGChain.answer`
                                  path and by `graph.py` entity extraction, where
                                  reasoning text is acceptable.
`generate_stream_async` (async) — yields ONLY `content`, and additionally strips
                                  any `<think>…</think>` / `<reasoning>…</reasoning>`
                                  blocks that leak into `content` itself. This is
                                  the path used by the chat UI, where leaked
                                  reasoning would appear as garbage to the user.
"""
from __future__ import annotations

import re
from typing import AsyncIterator, Iterator

import httpx

from ..config import settings
from ..lm_client import get_async_client, get_sync_client


# ---------------------------------------------------------------------------
# Streaming reasoning-tag filter
# ---------------------------------------------------------------------------
# Some local models (and some LM Studio versions) ignore the OpenAI-style
# `reasoning_content` channel and emit raw `<think>…</think>` blocks inside
# the content stream itself. We strip those so the chat UI never shows them.
#
# Tags can be split across token boundaries, so we buffer up to TAG_BUF_LEN
# trailing chars and only yield content we know isn't a partial tag prefix.

_OPEN_TAG = re.compile(r"<(?:think|reasoning)>", re.IGNORECASE)
_CLOSE_TAG = re.compile(r"</(?:think|reasoning)>", re.IGNORECASE)
TAG_BUF_LEN = len("</reasoning>")  # 12 — longest tag we need to recognize


async def _strip_reasoning_tags(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """
    Wrap a token stream and drop any <think>…</think> / <reasoning>…</reasoning>
    blocks. Tag boundaries split across tokens are handled by buffering the last
    TAG_BUF_LEN chars before yielding.
    """
    buf = ""
    dropping = False

    async for token in stream:
        if not token:
            continue
        buf += token

        # Each iteration: peel off as many open/close tag transitions as we can
        # find in the buffer; what remains gets safely yielded or held back.
        while True:
            if dropping:
                m = _CLOSE_TAG.search(buf)
                if m:
                    buf = buf[m.end():]
                    dropping = False
                    continue
                # No close tag yet — discard everything except the trailing
                # window that might contain a split close tag.
                if len(buf) > TAG_BUF_LEN:
                    buf = buf[-TAG_BUF_LEN:]
                break

            m = _OPEN_TAG.search(buf)
            if m:
                safe = buf[: m.start()]
                if safe:
                    yield safe
                buf = buf[m.end():]
                dropping = True
                continue

            # No open tag visible. Yield everything except the trailing
            # window that might contain a split open tag.
            if len(buf) > TAG_BUF_LEN:
                yield buf[:-TAG_BUF_LEN]
                buf = buf[-TAG_BUF_LEN:]
            break

    # End of stream — flush whatever's left if we're not mid-block.
    if not dropping and buf:
        yield buf


def generate_stream(
    prompt: str,
    system: str = "",
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> Iterator[str]:
    """
    Sync streaming generation.

    Used by the legacy non-streaming `RAGChain.answer` (single-shot) and by
    `graph.py` entity extraction — callers where reasoning content is fine.
    The system prompt passed in is used as-is; no automatic appends. The
    "respond directly" instruction now lives in STUDY_ASSISTANT_PROMPT itself.
    """
    client = get_sync_client()

    stream = client.chat.completions.create(
        model=model or settings.lmstudio_model,
        messages=[
            {"role": "system", "content": system},
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
    Async streaming generation — used by the chat UI.

    Yields ONLY `content` tokens. `reasoning_content` is intentionally dropped,
    and any in-content <think>/<reasoning> blocks are stripped via
    _strip_reasoning_tags so the user never sees them.

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

    client = get_async_client()

    if messages is None:
        # Legacy path: build a simple system+user messages list. The "respond
        # directly" instruction lives in the caller's system prompt now.
        request_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    else:
        # Multi-turn path: pass through as-is. Caller owns message construction.
        request_messages = messages

    async def _raw_stream() -> AsyncIterator[str]:
        stream = await client.chat.completions.create(
            model=model or settings.lmstudio_model,
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

    async for token in _strip_reasoning_tags(_raw_stream()):
        yield token


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
            response = client.get(f"{settings.lmstudio_base_url}/models")
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            return {"status": "ok", "backend": "lmstudio", "models": models}
    except Exception as e:
        return {"status": "error", "backend": "lmstudio", "error": str(e)}
