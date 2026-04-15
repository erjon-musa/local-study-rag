"""
Unified hybrid retriever.

Combines semantic vector search + BM25 keyword search
with Reciprocal Rank Fusion for the best of both worlds.
"""
from __future__ import annotations

import os
from typing import List, Optional

from .keyword_search import KeywordSearch
from .reranker import RankedResult, reciprocal_rank_fusion
from .vector_search import VectorSearch


class HybridRetriever:
    """
    Hybrid retrieval engine combining semantic + keyword search.
    
    Usage:
        retriever = HybridRetriever()
        results = retriever.retrieve("How does A* search work?", course="ELEC 472")
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
            course: Optional course filter (e.g. "ELEC 472 - Artificial Intelligence")
            vector_weight: How many results to fetch from vector search (multiplied by top_k)
            keyword_weight: How many results to fetch from keyword search
        
        Returns:
            Reranked results combining both search strategies
        """
        # Fetch more candidates than needed, then rerank
        fetch_k = top_k * 2

        # Run both searches
        vector_results = self.vector_search.search(
            query=query,
            top_k=fetch_k,
            course_filter=course,
        )

        keyword_results = self.keyword_search.search(
            query=query,
            top_k=fetch_k,
            course_filter=course,
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
        return self.vector_search.search(query=query, top_k=top_k, course_filter=course)

    def retrieve_keyword_only(
        self,
        query: str,
        top_k: int = 8,
        course: Optional[str] = None,
    ) -> list:
        """Retrieve using only BM25 keyword search."""
        return self.keyword_search.search(query=query, top_k=top_k, course_filter=course)
