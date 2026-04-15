"""
Unified hybrid retriever.

Combines semantic vector search + BM25 keyword search
with Reciprocal Rank Fusion for the best of both worlds.

Includes course alias resolution so users can type "223" or "CMPE223"
instead of the full course name.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

from .keyword_search import KeywordSearch
from .reranker import RankedResult, reciprocal_rank_fusion
from .vector_search import VectorSearch

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
        persist_dir = chroma_persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        self.vector_search = VectorSearch(chroma_persist_dir=persist_dir)
        self.keyword_search = KeywordSearch(chroma_persist_dir=persist_dir)

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        course: Optional[str] = None,
        vector_weight: int = 1,
        keyword_weight: int = 1,
    ) -> List[RankedResult]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: The user's question
            top_k: Number of results to return
            course: Optional course filter — supports shortcuts like "223", "ai"
            vector_weight: How many results to fetch from vector search (multiplied by top_k)
            keyword_weight: How many results to fetch from keyword search
        
        Returns:
            Reranked results combining both search strategies
        """
        # Resolve course alias
        resolved_course = resolve_course(course)

        # Fetch more candidates than needed, then rerank
        fetch_k = top_k * 2

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

        # Fuse results
        fused = reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            top_k=top_k,
        )

        return fused

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
