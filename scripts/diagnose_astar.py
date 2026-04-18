#!/usr/bin/env python3
"""
One-off diagnostic for the 'A* search algorithm' query.
Shows what chunks retrieval actually returns so we can see why the model refused.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from backend.retrieval.retriever import HybridRetriever
from backend.ingestion.pipeline import IngestionPipeline


QUERY = "Explain the A* search algorithm"


def dump_results(results, preview_chars: int = 500):
    for i, r in enumerate(results):
        meta = r.metadata
        print(f"\n--- rank {i+1} | RRF={r.rrf_score:.4f} | vec_rank={r.vector_rank} | kw_rank={r.keyword_rank} ---")
        print(f"  source: {meta.get('source', '?')}")
        print(f"  page:   {meta.get('page', '?')}")
        print(f"  course: {meta.get('course', '?')}")
        print(f"  doc_type: {meta.get('doc_type', '?')}")
        print(f"  length: {len(r.text)} chars")
        preview = r.text[:preview_chars].replace("\n", " ⏎ ")
        print(f"  text: {preview}{'...' if len(r.text) > preview_chars else ''}")


def check_astar_content(results) -> dict:
    """Does any chunk actually explain A* (vs just mentioning it)?"""
    findings = {"total": len(results), "mentions_astar": 0, "likely_explains": 0, "exam_or_lab_cover": 0}
    for r in results:
        text_lower = r.text.lower()
        if "a*" in text_lower or "a-star" in text_lower or "a star" in text_lower:
            findings["mentions_astar"] += 1
        # Heuristic for actual explanation: has heuristic/f(n)/g(n)/h(n) terminology
        signals = ["f(n)", "g(n)", "h(n)", "heuristic", "admissible", "open list", "closed list", "priority queue", "consistent"]
        if sum(1 for s in signals if s in text_lower) >= 2:
            findings["likely_explains"] += 1
        doc_type = (r.metadata.get("doc_type") or "").lower()
        source = (r.metadata.get("source") or "").lower()
        if "exam" in doc_type or "exam" in source or ("lab" in source and int(r.metadata.get("page", 99) or 99) <= 2):
            findings["exam_or_lab_cover"] += 1
    return findings


def search_vault_for_astar():
    """Independent of retrieval — is A* actually explained anywhere in the indexed text?"""
    print("\n" + "=" * 70)
    print("GROUND TRUTH: Scanning all indexed chunks for A* explanations")
    print("=" * 70)
    retriever = HybridRetriever()
    # Access the raw chunks via ChromaDB
    all_docs = retriever.vector_search.collection.get(include=["documents", "metadatas"])
    docs = all_docs["documents"]
    metas = all_docs["metadatas"]

    explains = []
    for doc, meta in zip(docs, metas):
        lower = doc.lower()
        if "a*" not in lower and "a-star" not in lower:
            continue
        signals = ["f(n)", "g(n)", "h(n)", "heuristic", "admissible", "open list", "closed list", "consistent"]
        score = sum(1 for s in signals if s in lower)
        if score >= 2:
            explains.append((score, meta.get("source", "?"), meta.get("page", "?"), meta.get("course", "?"), doc[:300]))

    explains.sort(reverse=True)
    print(f"\nFound {len(explains)} chunks that LIKELY contain an A* explanation (>=2 signal terms):\n")
    for score, src, page, course, preview in explains[:10]:
        print(f"  signals={score} | {course} | {src} p.{page}")
        print(f"    {preview.replace(chr(10), ' ⏎ ')[:200]}...")
        print()
    return len(explains)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(f"QUERY: {QUERY!r}")
    print("=" * 70)

    retriever = HybridRetriever()

    # Top-4 (what chat actually uses)
    print("\n\n### TOP 4 (what the chat UI passed to the model)")
    print("=" * 70)
    r4 = retriever.retrieve(query=QUERY, top_k=4)
    dump_results(r4, preview_chars=400)
    print("\nFindings for top-4:", check_astar_content(r4))

    # Top-12 (what a better default would return)
    print("\n\n### TOP 12 (what a better default would return)")
    print("=" * 70)
    r12 = retriever.retrieve(query=QUERY, top_k=12)
    dump_results(r12, preview_chars=300)
    print("\nFindings for top-12:", check_astar_content(r12))

    # Ground truth: is A* actually in the vault?
    found_count = search_vault_for_astar()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"  Top-4 contains explanatory content?   {check_astar_content(r4)['likely_explains']}/4")
    print(f"  Top-12 contains explanatory content?  {check_astar_content(r12)['likely_explains']}/12")
    print(f"  Total explanatory chunks in vault:    {found_count}")
    print()
