"""
Unified hybrid retriever.

Combines semantic vector search + BM25 keyword search
with Reciprocal Rank Fusion for the best of both worlds.

Includes course alias resolution so users can type "223" or "CMPE223"
instead of the full course name.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..config import settings
from .keyword_search import KeywordSearch
from .reranker import RankedResult, reciprocal_rank_fusion
from .vector_search import VectorSearch


@dataclass
class RetrievalDiagnostics:
    """Signals about retrieval quality. Used by the chain to classify the
    response path (good / weak / empty) without re-running the searches."""
    top_vector_similarity: float = 0.0  # 0..1; higher = better match
    vector_keyword_overlap: int = 0     # # of chunks in both vector top-N and keyword top-N
    vector_hits: int = 0
    keyword_hits: int = 0

# Course aliases: maps shorthand inputs to the full course name used in metadata.
# Case-insensitive matching is applied at lookup time.
COURSE_ALIASES = {
    # CMPE 223
    "223": "CMPE 223 - Software Specification",
    "cmpe223": "CMPE 223 - Software Specification",
    "cmpe 223": "CMPE 223 - Software Specification",
    "cisc223": "CMPE 223 - Software Specification",
    "cisc 223": "CMPE 223 - Software Specification",
    "cisc/cmpe 223": "CMPE 223 - Software Specification",
    "software specification": "CMPE 223 - Software Specification",
    # ELEC 472
    "472": "ELEC 472 - Artificial Intelligence",
    "elec472": "ELEC 472 - Artificial Intelligence",
    "elec 472": "ELEC 472 - Artificial Intelligence",
    "artificial intelligence": "ELEC 472 - Artificial Intelligence",
    "ai": "ELEC 472 - Artificial Intelligence",
    # ELEC 477
    "477": "ELEC 477 - Distributed Systems",
    "elec477": "ELEC 477 - Distributed Systems",
    "elec 477": "ELEC 477 - Distributed Systems",
    "distributed systems": "ELEC 477 - Distributed Systems",
    "dds": "ELEC 477 - Distributed Systems",
}


def resolve_course(course_input: Optional[str]) -> Optional[str]:
    """
    Resolve a user's course input to the full course name.

    Handles shortcuts like '223', 'CMPE223', 'ai', etc.
    Returns None if input is None or unrecognized (no filter).
    """
    if not course_input:
        return None

    normalized = course_input.strip().lower()

    # Direct alias match
    if normalized in COURSE_ALIASES:
        return COURSE_ALIASES[normalized]

    # Check if it's already a full course name
    for full_name in COURSE_ALIASES.values():
        if normalized == full_name.lower():
            return full_name

    # Partial match: check if the input is a substring of any course name
    for full_name in set(COURSE_ALIASES.values()):
        if normalized in full_name.lower():
            return full_name

    # Unrecognized — return as-is (ChromaDB will filter, may return nothing)
    return course_input


def extract_doc_type_hint(query: str) -> Optional[str]:
    """
    Extract a document type hint from the user's query.

    If the query mentions 'exam', 'midterm', 'assignment', etc.,
    return the doc_type to use as a metadata filter.
    """
    q = query.lower()

    if any(w in q for w in ["exam", "final exam", "midterm", "quiz", "test"]):
        return "exam"
    if any(w in q for w in ["assignment", "homework", "problem set"]):
        return "assignment"
    if "lab" in q:
        return "lab"
    if any(w in q for w in ["lecture", "slide", "notes"]):
        return "lecture"

    return None


class HybridRetriever:
    """
    Hybrid retrieval engine combining semantic + keyword search.
    
    Usage:
        retriever = HybridRetriever()
        results = retriever.retrieve("How does A* search work?", course="472")
    """

    def __init__(self, chroma_persist_dir: str = None):
        persist_dir = chroma_persist_dir or settings.chroma_persist_dir
        self.vector_search = VectorSearch(chroma_persist_dir=persist_dir)
        self.keyword_search = KeywordSearch(chroma_persist_dir=persist_dir)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        course: Optional[str] = None,
        vector_weight: int = 1,
        keyword_weight: int = 1,
    ) -> List[RankedResult]:
        """
        Retrieve relevant chunks using hybrid search.

        Doc-type awareness (new):
        - If the query explicitly hints at a doc_type (e.g. "the 2023 exam"),
          filter strictly to that doc_type after RRF.
        - Otherwise, for explanatory queries ("explain", "what is",
          "how does", "define", "describe"), apply a multiplicative rerank:
          lecture/resource get a 1.3x boost, exam gets 0.5x. This pushes
          the most useful teaching material up without hiding exams
          when the user is actually asking about an exam.

        Args:
            query: The user's question
            top_k: Number of results to return (default 10)
            course: Optional course filter — supports shortcuts like "223", "ai"
            vector_weight: How many results to fetch from vector search (multiplied by top_k)
            keyword_weight: How many results to fetch from keyword search

        Returns:
            Reranked results combining both search strategies
        """
        # Resolve course alias
        resolved_course = resolve_course(course)

        # Decide doc-type handling up front — `extract_doc_type_hint` is
        # defined above and was never being called before this change.
        doc_type_hint = extract_doc_type_hint(query)

        # Fetch more candidates than needed, then rerank.
        # When we plan to filter post-RRF, pull extra so we don't run
        # dry after the filter.
        fetch_k = top_k * 2
        if doc_type_hint is not None:
            fetch_k = max(fetch_k, top_k * 4)

        # Run both searches
        vector_results = self.vector_search.search(
            query=query,
            top_k=fetch_k,
            course_filter=resolved_course,
        )

        keyword_results = self.keyword_search.search(
            query=query,
            top_k=fetch_k,
            course_filter=resolved_course,
        )

        # Fuse results. Pull more than top_k so post-RRF filtering/boosting
        # has room to work without truncating prematurely.
        fused = reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            top_k=fetch_k,
        )

        if doc_type_hint is not None:
            # Explicit hint → strict filter. If the user asks about "the 2023
            # exam", they want exam chunks only; keep the RRF order.
            filtered = [r for r in fused if r.metadata.get("doc_type") == doc_type_hint]
            # Fall back to the unfiltered fused list if the filter wipes
            # everything out (e.g. ChromaDB has no exam chunks for this course)
            # — returning nothing would be worse than returning near-misses.
            return (filtered or fused)[:top_k]

        # No explicit hint. Check for explanatory queries and apply a
        # post-RRF boost. Substring check is fine: "describe" catches both
        # "describe" and "described", which is the intent.
        q_lower = query.lower()
        explanatory_signals = ("explain", "what is", "how does", "define", "describe")
        is_explanatory = any(signal in q_lower for signal in explanatory_signals)

        if is_explanatory:
            boosted: List[RankedResult] = []
            for r in fused:
                dt = r.metadata.get("doc_type")
                if dt in {"lecture", "resource"}:
                    multiplier = 1.3
                elif dt == "exam":
                    multiplier = 0.5
                else:
                    multiplier = 1.0
                # Rebuild with an adjusted score so downstream sees the
                # actual ranking signal, not the pre-boost score.
                boosted.append(RankedResult(
                    text=r.text,
                    metadata=r.metadata,
                    rrf_score=r.rrf_score * multiplier,
                    chunk_id=r.chunk_id,
                    vector_rank=r.vector_rank,
                    keyword_rank=r.keyword_rank,
                ))
            boosted.sort(key=lambda x: x.rrf_score, reverse=True)
            return boosted[:top_k]

        return fused[:top_k]

    def retrieve_with_diagnostics(
        self,
        query: str,
        top_k: int = 10,
        course: Optional[str] = None,
    ) -> Tuple[List[RankedResult], RetrievalDiagnostics]:
        """
        Retrieve + return diagnostics the chain needs to classify retrieval
        quality (top vector similarity, vector/keyword top-10 overlap).

        This parallels `retrieve()` but runs its own searches so it can surface
        the raw cosine similarity (which `retrieve()`'s RRF output discards) and
        the top-10 overlap (which is otherwise only implicit).

        Doc-type handling (explicit filter vs. explanatory boost) mirrors
        `retrieve()` so the chain gets the same final ordering it would have
        gotten from `retrieve()`.

        Returns:
            (results, diagnostics) — results is identical in shape to retrieve().
        """
        resolved_course = resolve_course(course)
        doc_type_hint = extract_doc_type_hint(query)

        fetch_k = top_k * 2
        if doc_type_hint is not None:
            fetch_k = max(fetch_k, top_k * 4)

        vector_results = self.vector_search.search(
            query=query,
            top_k=fetch_k,
            course_filter=resolved_course,
        )
        keyword_results = self.keyword_search.search(
            query=query,
            top_k=fetch_k,
            course_filter=resolved_course,
        )

        # Top vector similarity is the first (highest-ranked) result's score.
        top_vec_sim = vector_results[0].score if vector_results else 0.0

        # Overlap between the top-10 of each search (by chunk_id). This is the
        # agreement signal: high overlap = both retrievers agree = confident.
        vec_top10_ids = {r.chunk_id for r in vector_results[:10]}
        kw_top10_ids = {r.chunk_id for r in keyword_results[:10]}
        overlap = len(vec_top10_ids & kw_top10_ids)

        diagnostics = RetrievalDiagnostics(
            top_vector_similarity=top_vec_sim,
            vector_keyword_overlap=overlap,
            vector_hits=len(vector_results),
            keyword_hits=len(keyword_results),
        )

        fused = reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            top_k=fetch_k,
        )

        if doc_type_hint is not None:
            filtered = [r for r in fused if r.metadata.get("doc_type") == doc_type_hint]
            return (filtered or fused)[:top_k], diagnostics

        q_lower = query.lower()
        explanatory_signals = ("explain", "what is", "how does", "define", "describe")
        is_explanatory = any(signal in q_lower for signal in explanatory_signals)

        if is_explanatory:
            boosted: List[RankedResult] = []
            for r in fused:
                dt = r.metadata.get("doc_type")
                if dt in {"lecture", "resource"}:
                    multiplier = 1.3
                elif dt == "exam":
                    multiplier = 0.5
                else:
                    multiplier = 1.0
                boosted.append(RankedResult(
                    text=r.text,
                    metadata=r.metadata,
                    rrf_score=r.rrf_score * multiplier,
                    chunk_id=r.chunk_id,
                    vector_rank=r.vector_rank,
                    keyword_rank=r.keyword_rank,
                ))
            boosted.sort(key=lambda x: x.rrf_score, reverse=True)
            return boosted[:top_k], diagnostics

        return fused[:top_k], diagnostics

    def retrieve_vector_only(
        self,
        query: str,
        top_k: int = 8,
        course: Optional[str] = None,
    ) -> list:
        """Retrieve using only semantic search."""
        resolved_course = resolve_course(course)
        return self.vector_search.search(query=query, top_k=top_k, course_filter=resolved_course)

    def retrieve_keyword_only(
        self,
        query: str,
        top_k: int = 8,
        course: Optional[str] = None,
    ) -> list:
        """Retrieve using only BM25 keyword search."""
        resolved_course = resolve_course(course)
        return self.keyword_search.search(query=query, top_k=top_k, course_filter=resolved_course)
