"""
RAG chain: ties retrieval → context formatting → LLM generation.

This is the main entry point for answering questions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, List, Optional

from ..retrieval.retriever import HybridRetriever
from .llm import generate_stream, generate_stream_async
from .prompts import STUDY_ASSISTANT_PROMPT, build_rag_prompt


@dataclass
class Source:
    """A cited source from the RAG response."""
    filename: str
    page: str
    course: str
    category: str
    relevance_score: float
    text_preview: str = ""


@dataclass
class RAGResponse:
    """Complete RAG response with answer text and sources."""
    answer: str
    sources: List[Source] = field(default_factory=list)


class RAGChain:
    """
    Full RAG pipeline: question → retrieve → generate → answer with citations.
    """

    def __init__(self, chroma_persist_dir: str = None):
        self.retriever = HybridRetriever(chroma_persist_dir=chroma_persist_dir)

    def _extract_sources(self, results) -> List[Source]:
        """Extract source citations from retrieval results."""
        seen = set()
        sources = []

        for result in results:
            meta = result.metadata
            filename = meta.get("source", "Unknown")
            page = meta.get("page", "")
            key = f"{filename}:{page}"

            if key in seen:
                continue
            seen.add(key)

            sources.append(Source(
                filename=filename,
                page=str(page),
                course=meta.get("course", ""),
                category=meta.get("category", ""),
                relevance_score=round(result.rrf_score, 4),
                text_preview=result.text[:200] + "..." if len(result.text) > 200 else result.text,
            ))

        return sources

    def answer(
        self,
        question: str,
        course: Optional[str] = None,
        top_k: int = 8,
    ) -> RAGResponse:
        """
        Answer a question using the RAG pipeline (non-streaming).
        """
        # Retrieve
        results = self.retriever.retrieve(query=question, top_k=top_k, course=course)

        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant information in your notes for this question. "
                       "Make sure your notes have been ingested and try rephrasing your question.",
                sources=[],
            )

        # Build prompt
        prompt = build_rag_prompt(question, results)

        # Generate
        answer = ""
        for token in generate_stream(prompt, system=STUDY_ASSISTANT_PROMPT):
            answer += token

        # Extract sources
        sources = self._extract_sources(results)

        return RAGResponse(answer=answer, sources=sources)

    def answer_stream(
        self,
        question: str,
        course: Optional[str] = None,
        top_k: int = 8,
    ) -> tuple:
        """
        Stream an answer using the RAG pipeline.
        
        Returns (token_iterator, sources) where sources are available immediately
        from the retrieval step.
        """
        # Retrieve
        results = self.retriever.retrieve(query=question, top_k=top_k, course=course)

        if not results:
            def empty_stream():
                yield "I couldn't find any relevant information in your notes for this question. "
                yield "Make sure your notes have been ingested and try rephrasing your question."
            return empty_stream(), []

        # Build prompt
        prompt = build_rag_prompt(question, results)

        # Extract sources (available immediately, before generation)
        sources = self._extract_sources(results)

        # Return the stream and sources
        token_stream = generate_stream(prompt, system=STUDY_ASSISTANT_PROMPT)
        return token_stream, sources

    async def answer_stream_async(
        self,
        question: str,
        course: Optional[str] = None,
        top_k: int = 4,
    ) -> tuple:
        """
        Async version of answer_stream.
        
        Returns (async_token_iterator, sources).
        """
        # Retrieve (sync — retrieval is fast)
        results = self.retriever.retrieve(query=question, top_k=top_k, course=course)

        if not results:
            async def empty_stream():
                yield "I couldn't find any relevant information in your notes for this question. "
                yield "Make sure your notes have been ingested and try rephrasing your question."
            return empty_stream(), []

        # Build prompt
        prompt = build_rag_prompt(question, results)

        # Extract sources
        sources = self._extract_sources(results)

        # Return the async stream and sources
        token_stream = generate_stream_async(prompt, system=STUDY_ASSISTANT_PROMPT)
        return token_stream, sources
