"""
Semantic vector search using ChromaDB.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass

import chromadb

from ..ingestion.embedder import embed_single


@dataclass
class SearchResult:
    """A single search result with text, metadata, and score."""
    text: str
    metadata: dict
    score: float  # 0-1, higher is more relevant
    chunk_id: str = ""


class VectorSearch:
    """Semantic search over the ChromaDB collection."""

    def __init__(self, chroma_persist_dir: str = "./data/chroma"):
        self.client = chromadb.PersistentClient(path=chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="study_notes",
            metadata={"hnsw:space": "cosine"},
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        course_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for chunks semantically similar to the query.
        
        Optionally filter by course name.
        Returns results sorted by relevance (highest first).
        """
        # Embed the query
        query_embedding = embed_single(query)

        # Build where filter
        where = None
        if course_filter:
            where = {"course": {"$eq": course_filter}}

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult objects
        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
                # Convert to similarity score (1 = identical, 0 = orthogonal)
                distance = results["distances"][0][i]
                score = 1 - (distance / 2)

                search_results.append(SearchResult(
                    text=doc,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=score,
                    chunk_id=results["ids"][0][i] if results["ids"] else "",
                ))

        return search_results
