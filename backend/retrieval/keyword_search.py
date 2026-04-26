"""
BM25 keyword search for exact term matching.

Complements semantic search for technical terms like
"DDS", "CycloneDDS", "BFS", "A* search", etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from rank_bm25 import BM25Okapi

import chromadb

from ..config import settings


@dataclass
class KeywordResult:
    """A BM25 keyword search result."""
    text: str
    metadata: dict
    score: float
    chunk_id: str = ""


class KeywordSearch:
    """BM25-based keyword search over indexed chunks."""

    def __init__(self, chroma_persist_dir: str = None):
        self.client = chromadb.PersistentClient(
            path=chroma_persist_dir or settings.chroma_persist_dir
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._index = None
        self._docs = None
        self._metadatas = None
        self._ids = None

    def _build_index(self, course_filter: Optional[str] = None):
        """Build/rebuild the BM25 index from ChromaDB documents."""
        where = None
        if course_filter:
            where = {"course": {"$eq": course_filter}}

        # Fetch all documents from ChromaDB
        count = self.collection.count()
        if count == 0:
            self._index = None
            self._docs = []
            self._metadatas = []
            self._ids = []
            return

        results = self.collection.get(
            where=where,
            include=["documents", "metadatas"],
            limit=count,
        )

        self._docs = results["documents"] or []
        self._metadatas = results["metadatas"] or []
        self._ids = results["ids"] or []

        if not self._docs:
            self._index = None
            return

        # Tokenize for BM25 — split on whitespace AND punctuation
        # so 'CISC/CMPE-223' → ['cisc', 'cmpe', '223']
        import re
        tokenized = [re.split(r'[\s/\-_,;:()]+', doc.lower()) for doc in self._docs]
        # Filter empty tokens
        tokenized = [[t for t in tokens if t] for tokens in tokenized]
        self._index = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        top_k: int = 10,
        course_filter: Optional[str] = None,
    ) -> List[KeywordResult]:
        """
        Search using BM25 keyword matching.
        
        Rebuilds the index each time to reflect the latest data.
        This is fast since the corpus is small (<10K chunks).
        """
        self._build_index(course_filter=course_filter)

        if not self._index or not self._docs:
            return []

        import re
        tokenized_query = [t for t in re.split(r'[\s/\-_,;:()]+', query.lower()) if t]
        scores = self._index.get_scores(tokenized_query)

        # Get top-k indices
        scored_indices = [(i, float(s)) for i, s in enumerate(scores) if s > 0]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = scored_indices[:top_k]

        results = []
        for idx, score in top_indices:
            results.append(KeywordResult(
                text=self._docs[idx],
                metadata=self._metadatas[idx] if idx < len(self._metadatas) else {},
                score=score,
                chunk_id=self._ids[idx] if idx < len(self._ids) else "",
            ))

        return results
