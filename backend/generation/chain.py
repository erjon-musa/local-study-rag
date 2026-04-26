"""
RAG chain: ties retrieval → context formatting → LLM generation.

This is the main entry point for answering questions.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from ..retrieval.retriever import HybridRetriever, RetrievalDiagnostics
from .llm import generate_stream, generate_stream_async
from .prompts import (
    STUDY_ASSISTANT_PROMPT,
    build_rag_prompt,
    format_context,
    format_history_for_prompt,
    render_empty_state,
)

log = logging.getLogger(__name__)

# Retrieval-quality thresholds (plan Task 7). These are the calibrated dials;
# the plan calls them out explicitly so Task 10 can tune them against the
# golden queries without hunting through the code.
# Calibrated 2026-04-18 against real retrieval data: with
# text-embedding-nomic-embed-text-v1.5, even fully out-of-domain queries return
# top_sim >= 0.75 (the embedding model always finds *something* vaguely related
# in the vault). The signal that cleanly separates in-domain vs out-of-domain
# is cross-retriever agreement: for every in-domain golden query, vector-top-10
# and keyword-top-10 overlap on >= 1 doc; for every OOD probe, overlap = 0.
# So: weak = retrievers disagree (overlap == 0) AND top_sim below the ceiling
# observed for OOD queries.
WEAK_TOP_SIM_THRESHOLD = 0.85        # OOD top_sim maxed at 0.83; golden min was 0.855
WEAK_OVERLAP_THRESHOLD = 1           # require ≥1 overlap to avoid "weak"

# Token budget. 1 token ~ 4 chars in English is the OpenAI tokenizer rule of
# thumb; good enough for guarding a context window without pulling tiktoken.
MAX_PROMPT_TOKENS = 12_000
CHARS_PER_TOKEN = 4


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


def _estimate_tokens(text: str) -> int:
    """Cheap char-based token estimator. ~4 chars/token for English."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _messages_total_chars(messages: List[dict]) -> int:
    """Sum of all message content lengths for token estimation."""
    return sum(len(m.get("content", "") or "") for m in messages)


def _enforce_token_budget(
    system_prompt: str,
    history_messages: List[dict],
    final_user_content: str,
) -> List[dict]:
    """
    Assemble the full messages list and drop oldest history turns one at a time
    until the estimated token count fits inside MAX_PROMPT_TOKENS. The system
    prompt and current user turn are non-negotiable — only history is shed.

    Returns the trimmed history_messages (may be [] if the budget is tight).
    """
    def size_for(history: List[dict]) -> int:
        total_chars = len(system_prompt) + _messages_total_chars(history) + len(final_user_content)
        return max(1, total_chars // CHARS_PER_TOKEN)

    trimmed = list(history_messages)
    dropped: List[dict] = []
    while trimmed and size_for(trimmed) > MAX_PROMPT_TOKENS:
        dropped.append(trimmed.pop(0))  # oldest first

    if dropped:
        log.info(
            "chain: dropped %d oldest history turn(s) to fit token budget "
            "(kept %d, est tokens=%d, cap=%d)",
            len(dropped),
            len(trimmed),
            size_for(trimmed),
            MAX_PROMPT_TOKENS,
        )

    return trimmed


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

    def _classify_retrieval(
        self,
        results: list,
        diagnostics: RetrievalDiagnostics,
    ) -> str:
        """
        Quality classification (plan Task 7, calibrated 2026-04-18):

          empty:  no results at all, OR no vector hits
          weak:   retrievers disagree (vector/keyword top-10 overlap == 0)
                  AND top vector cosine similarity below OOD ceiling (0.85)
          good:   everything else (retrievers agreed on >=1 doc, OR semantic
                  match is very strong)

        The overlap == 0 guard is the primary signal — on real data, every
        in-domain golden query produced overlap >= 1 while every OOD probe
        produced overlap == 0. The similarity threshold is a safety net for
        the (rare) case where a legitimate query happens to get overlap == 0
        but still has a near-exact semantic match.
        """
        if not results or diagnostics.vector_hits == 0:
            return "empty"
        if (
            diagnostics.vector_keyword_overlap < WEAK_OVERLAP_THRESHOLD
            and diagnostics.top_vector_similarity < WEAK_TOP_SIM_THRESHOLD
        ):
            return "weak"
        return "good"

    def _closest_topics_from(self, results: list, limit: int = 3) -> List[str]:
        """Grab up to `limit` distinct source labels to surface as 'closest matches'."""
        topics: List[str] = []
        seen = set()
        for r in results[:10]:
            meta = r.metadata or {}
            rel = meta.get("rel_path") or meta.get("source") or ""
            if not rel or rel in seen:
                continue
            seen.add(rel)
            # Strip path + extension to get a readable topic name.
            basename = rel.rsplit("/", 1)[-1]
            stem = basename.rsplit(".", 1)[0]
            topics.append(stem)
            if len(topics) >= limit:
                break
        return topics

    def answer(
        self,
        question: str,
        course: Optional[str] = None,
        top_k: int = 10,
    ) -> RAGResponse:
        """
        Answer a question using the RAG pipeline (non-streaming).
        Kept simple: no history, no classification — used by /chat/simple.
        """
        results = self.retriever.retrieve(query=question, top_k=top_k, course=course)

        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant information in your notes for this question. "
                       "Make sure your notes have been ingested and try rephrasing your question.",
                sources=[],
            )

        prompt = build_rag_prompt(question, results)
        answer = ""
        for token in generate_stream(prompt, system=STUDY_ASSISTANT_PROMPT):
            answer += token

        sources = self._extract_sources(results)
        return RAGResponse(answer=answer, sources=sources)

    def answer_stream(
        self,
        question: str,
        course: Optional[str] = None,
        top_k: int = 10,
    ) -> tuple:
        """
        Stream an answer using the RAG pipeline (sync).

        Returns (token_iterator, sources) where sources are available immediately
        from the retrieval step.
        """
        results = self.retriever.retrieve(query=question, top_k=top_k, course=course)

        if not results:
            def empty_stream():
                yield "I couldn't find any relevant information in your notes for this question. "
                yield "Make sure your notes have been ingested and try rephrasing your question."
            return empty_stream(), []

        prompt = build_rag_prompt(question, results)
        sources = self._extract_sources(results)
        token_stream = generate_stream(prompt, system=STUDY_ASSISTANT_PROMPT)
        return token_stream, sources

    async def answer_stream_async(
        self,
        question: str,
        course: Optional[str] = None,
        top_k: int = 10,
        history: Optional[List[dict]] = None,
    ) -> tuple:
        """
        Async streaming with multi-turn history, retrieval classification, and
        synthetic empty-state streaming.

        Returns (async_token_iterator, sources_list).

        Flow:
          1. Retrieve + diagnostics.
          2. Classify empty / weak / good.
          3. Empty or weak → yield EMPTY_STATE_TEMPLATE as a synthetic token
             stream. Attach top-3 sources if any (so the UI can still show
             "closest matches" cards even when we refused to answer).
          4. Good → build OpenAI messages [system, ...history, user], enforce
             the token budget, call generate_stream_async(messages=...).
        """
        history = history or []

        # 1. Retrieve
        results, diagnostics = self.retriever.retrieve_with_diagnostics(
            query=question, top_k=top_k, course=course,
        )

        # 2. Classify
        quality = self._classify_retrieval(results, diagnostics)
        log.info(
            "chain: retrieval quality=%s top_sim=%.3f overlap=%d hits=(v=%d,k=%d)",
            quality,
            diagnostics.top_vector_similarity,
            diagnostics.vector_keyword_overlap,
            diagnostics.vector_hits,
            diagnostics.keyword_hits,
        )

        # 3. Empty or weak → synthetic empty-state stream, no LLM call.
        if quality in ("empty", "weak"):
            closest = self._closest_topics_from(results, limit=3) if results else []
            message = render_empty_state(
                topic=question,
                course=course,
                closest_topics=closest,
                suggested_sources=closest,  # Same list — they're the suggestion surface
            )
            # Attach top 3 sources so the UI still has something to render.
            sources = self._extract_sources(results[:3]) if results else []
            return _synthetic_token_stream(message), sources

        # 4. Good → build messages and hit the LLM.
        context_block = format_context(results)
        final_user_content = (
            "Here are relevant excerpts from your course materials, numbered by source:\n\n"
            f"{context_block}\n\n"
            "---\n\n"
            f"Question: {question}\n\n"
            "Answer using the sources above. Cite inline with [N] markers matching "
            "the numbers in the context block. Synthesize across sources when helpful."
        )

        # Plan caps history at 6 turns. Truncation/neutralization happens inside
        # format_history_for_prompt.
        capped_history = history[-6:]
        history_messages = format_history_for_prompt(capped_history)

        # Token budget: drop oldest history turns if the prompt is too big.
        history_messages = _enforce_token_budget(
            system_prompt=STUDY_ASSISTANT_PROMPT,
            history_messages=history_messages,
            final_user_content=final_user_content,
        )

        messages = (
            [{"role": "system", "content": STUDY_ASSISTANT_PROMPT}]
            + history_messages
            + [{"role": "user", "content": final_user_content}]
        )

        sources = self._extract_sources(results)
        token_stream = generate_stream_async(messages=messages)
        return token_stream, sources


async def _synthetic_token_stream(message: str) -> AsyncIterator[str]:
    """
    Yield `message` as a word-by-word async stream so the frontend's streaming
    UX still feels live even though we never hit the LLM.

    Tiny sleep between chunks is purely cosmetic — keeps the text from
    arriving as one lump and makes the "typing" animation work.
    """
    # Split keeping the whitespace so reassembly is lossless.
    pieces: List[str] = []
    buf = ""
    for ch in message:
        if ch == " ":
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(" ")
        else:
            buf += ch
    if buf:
        pieces.append(buf)

    for piece in pieces:
        yield piece
        await asyncio.sleep(0.01)
