"""
Chat API endpoint with streaming responses and source citations.
"""
from __future__ import annotations

import json
from typing import List, Literal, Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..generation.chain import RAGChain
from ..ingestion.pipeline import IngestionPipeline

# Cap on history turns the server will forward to the chain. Matches the
# plan's "last 6 turns" guardrail. The client may send more; we trim here
# so malicious/buggy clients can't blow the token budget.
MAX_HISTORY_TURNS = 6

router = APIRouter(prefix="/api", tags=["chat"])

# Lazy-initialized singletons
_chain: Optional[RAGChain] = None
_pipeline: Optional[IngestionPipeline] = None


def get_chain() -> RAGChain:
    global _chain
    if _chain is None:
        _chain = RAGChain()
    return _chain


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


class HistoryMessage(BaseModel):
    """A single prior turn in the conversation."""
    role: Literal["user", "assistant"]
    content: str
    # Optional per-turn sources — the chain uses them to neutralize [N] citation
    # markers when folding assistant turns back into the prompt. Absent on user
    # turns; may be absent on old assistant turns too, which is fine.
    sources: Optional[List[dict]] = None


class ChatRequest(BaseModel):
    question: str
    course: Optional[str] = None
    top_k: int = 10
    # Prior conversation turns. Server caps to MAX_HISTORY_TURNS regardless of
    # what the client sends.
    history: List[HistoryMessage] = Field(default_factory=list)


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Ask a question and get a streamed answer with source citations.
    
    Auto-scans the vault for new files before answering.
    Streams the response as newline-delimited JSON events.
    """
    chain = get_chain()
    pipeline = get_pipeline()

    # Quick scan for new files (fast if nothing changed)
    # Removing auto-ingest here so it doesn't block the chat UI.
    # We will trigger ingest manually.

    # Cap history server-side — don't trust client counts.
    capped_history = [
        {"role": m.role, "content": m.content, "sources": m.sources}
        for m in request.history[-MAX_HISTORY_TURNS:]
    ]

    # Get streaming answer + sources asynchronously
    token_stream, sources = await chain.answer_stream_async(
        question=request.question,
        course=request.course,
        top_k=request.top_k,
        history=capped_history,
    )

    # Serialize sources
    sources_data = [
        {
            "filename": s.filename,
            "page": s.page,
            "course": s.course,
            "category": s.category,
            "relevance_score": s.relevance_score,
            "text_preview": s.text_preview,
        }
        for s in sources
    ]

    async def event_stream():
        # First, send sources as a JSON event
        yield json.dumps({"type": "sources", "data": sources_data}) + "\n"

        try:
            # Then stream the answer tokens using async iteration
            async for token in token_stream:
                yield json.dumps({"type": "token", "data": token}) + "\n"
        except Exception as e:
            error_msg = f"\n\n[Model Error: {str(e)}]"
            yield json.dumps({"type": "token", "data": error_msg}) + "\n"

        # Signal completion
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
    )


@router.post("/chat/simple")
async def chat_simple(request: ChatRequest):
    """
    Non-streaming chat endpoint. Returns the full response at once.
    Useful for testing.
    """
    chain = get_chain()
    response = chain.answer(
        question=request.question,
        course=request.course,
        top_k=request.top_k,
    )

    return {
        "answer": response.answer,
        "sources": [
            {
                "filename": s.filename,
                "page": s.page,
                "course": s.course,
                "category": s.category,
                "relevance_score": s.relevance_score,
                "text_preview": s.text_preview,
            }
            for s in response.sources
        ],
    }
