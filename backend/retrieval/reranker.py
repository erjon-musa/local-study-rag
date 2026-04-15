"""
Reciprocal Rank Fusion (RRF) reranker.

Merges results from multiple search strategies (vector + keyword)
into a single ranked list. No external model needed.

Formula: score(d) = Σ 1/(k + rank_i(d))
where k is a constant (default 60) and rank_i is the rank in list i.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RankedResult:
    """A result after RRF reranking."""
    text: str
    metadata: dict
    rrf_score: float
    chunk_id: str = ""
    vector_rank: int = -1  # -1 = not in vector results
    keyword_rank: int = -1  # -1 = not in keyword results


def reciprocal_rank_fusion(
    vector_results: list,
    keyword_results: list,
    top_k: int = 10,
    k: int = 60,
) -> List[RankedResult]:
    """
    Merge vector and keyword search results using RRF.
    
    Args:
        vector_results: Results from semantic search (must have .text, .metadata, .chunk_id)
        keyword_results: Results from BM25 search (must have .text, .metadata, .chunk_id)
        top_k: Number of results to return
        k: RRF constant (higher = more weight to lower-ranked results)
    
    Returns:
        Merged and reranked results
    """
    # Build a map of chunk_id → combined info
    combined: Dict[str, dict] = {}

    # Process vector results
    for rank, result in enumerate(vector_results):
        cid = result.chunk_id
        if cid not in combined:
            combined[cid] = {
                "text": result.text,
                "metadata": result.metadata,
                "chunk_id": cid,
                "rrf_score": 0.0,
                "vector_rank": -1,
                "keyword_rank": -1,
            }
        combined[cid]["rrf_score"] += 1.0 / (k + rank + 1)
        combined[cid]["vector_rank"] = rank + 1

    # Process keyword results
    for rank, result in enumerate(keyword_results):
        cid = result.chunk_id
        if cid not in combined:
            combined[cid] = {
                "text": result.text,
                "metadata": result.metadata,
                "chunk_id": cid,
                "rrf_score": 0.0,
                "vector_rank": -1,
                "keyword_rank": -1,
            }
        combined[cid]["rrf_score"] += 1.0 / (k + rank + 1)
        combined[cid]["keyword_rank"] = rank + 1

    # Sort by RRF score (descending) and take top_k
    sorted_results = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)

    return [
        RankedResult(
            text=r["text"],
            metadata=r["metadata"],
            rrf_score=r["rrf_score"],
            chunk_id=r["chunk_id"],
            vector_rank=r["vector_rank"],
            keyword_rank=r["keyword_rank"],
        )
        for r in sorted_results[:top_k]
    ]
