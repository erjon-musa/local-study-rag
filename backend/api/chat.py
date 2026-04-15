"""
Chat API endpoint with streaming responses and source citations.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..generation.chain import RAGChain
from ..ingestion.pipeline import IngestionPipeline

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


class ChatRequest(BaseModel):
    question: str
    course: Optional[str] = None
    top_k: int = 8


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
    scan = pipeline.scan()
    if scan.has_changes:
        print(f"  Auto-scan: {scan.summary()}")
        ingest_result = pipeline.ingest()
        print(f"  Auto-ingested: {ingest_result.new} new, {ingest_result.updated} updated")

    # Get streaming answer + sources
    token_stream, sources = chain.answer_stream(
        question=request.question,
        course=request.course,
        top_k=request.top_k,
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

        # Then stream the answer tokens
        for token in token_stream:
            yield json.dumps({"type": "token", "data": token}) + "\n"

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
