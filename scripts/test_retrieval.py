#!/usr/bin/env python3
"""
RAG System Retrieval Quality Test Suite.

Tests the full retrieval pipeline with realistic user stories
to verify that search actually returns the right documents.

Usage:
    cd RAG_System
    source .venv/bin/activate
    python scripts/test_retrieval.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from backend.retrieval.retriever import HybridRetriever, resolve_course
from backend.ingestion.pipeline import IngestionPipeline


def print_results(results, max_preview: int = 100):
    """Pretty-print retrieval results."""
    for i, r in enumerate(results):
        source = r.metadata.get("source", "?")
        page = r.metadata.get("page", "?")
        course = r.metadata.get("course_code", r.metadata.get("course", "?"))
        doc_type = r.metadata.get("doc_type", "?")
        method = r.metadata.get("extraction_method", "?")
        preview = r.text[:max_preview].replace("\n", " ")
        print(f"  {i+1}. [{course} - {doc_type}] {source} p.{page} "
              f"(RRF={r.rrf_score:.4f}, vec={r.vector_rank}, kw={r.keyword_rank}, method={method})")
        print(f"     {preview}...")
        print()


def test_course_alias_resolution():
    """Test that course aliases resolve correctly."""
    print("=" * 70)
    print("TEST 1: Course Alias Resolution")
    print("=" * 70)

    test_cases = [
        ("223", "CMPE 223 - Software Specification"),
        ("CMPE223", "CMPE 223 - Software Specification"),
        ("cmpe 223", "CMPE 223 - Software Specification"),
        ("472", "ELEC 472 - Artificial Intelligence"),
        ("ai", "ELEC 472 - Artificial Intelligence"),
        ("477", "ELEC 477 - Distributed Systems"),
        ("dds", "ELEC 477 - Distributed Systems"),
        (None, None),
    ]

    passed = 0
    for input_val, expected in test_cases:
        result = resolve_course(input_val)
        ok = result == expected
        status = "✅" if ok else "❌"
        print(f"  {status} resolve_course({input_val!r}) → {result!r}")
        if not ok:
            print(f"     Expected: {expected!r}")
        passed += ok

    print(f"\n  {passed}/{len(test_cases)} passed\n")
    return passed == len(test_cases)


def test_retrieval_stories(retriever: HybridRetriever):
    """Run user story retrieval tests."""
    stories = [
        {
            "name": "Find CMPE 223 exam content",
            "query": "question 3 from the 223 final exam",
            "course": "223",
            "expect_in_sources": ["exam", "223", "cmpe"],
            "expect_doc_type": "exam",
        },
        {
            "name": "Find context-free grammar content",
            "query": "explain what a context-free grammar is",
            "course": "223",
            "expect_in_sources": [],
            "expect_doc_type": None,
        },
        {
            "name": "Find DDS content filtering",
            "query": "how does DDS content filtering work?",
            "course": "477",
            "expect_in_sources": [],
            "expect_doc_type": None,
        },
        {
            "name": "Find ML preprocessing steps",
            "query": "what are the key ML preprocessing steps?",
            "course": "472",
            "expect_in_sources": ["preprocessing"],
            "expect_doc_type": None,
        },
        {
            "name": "Find A* search algorithm",
            "query": "explain the A* search algorithm",
            "course": None,
            "expect_in_sources": ["search"],
            "expect_doc_type": None,
        },
    ]

    passed = 0
    for story in stories:
        print("=" * 70)
        print(f"STORY: {story['name']}")
        print(f"  Query: {story['query']!r}")
        print(f"  Course filter: {story['course']!r}")
        print("=" * 70)

        start = time.time()
        results = retriever.retrieve(
            query=story["query"],
            course=story["course"],
            top_k=5,
        )
        elapsed = time.time() - start

        print(f"\n  Found {len(results)} results in {elapsed:.2f}s:\n")

        if results:
            print_results(results)

            # Check if expected terms appear in top results
            all_ok = True
            if story["expect_in_sources"]:
                top_text = " ".join(r.text.lower() for r in results[:3])
                top_meta = " ".join(
                    f"{r.metadata.get('source', '')} {r.metadata.get('course', '')} "
                    f"{r.metadata.get('doc_type', '')}".lower()
                    for r in results[:3]
                )
                combined = top_text + " " + top_meta
                for term in story["expect_in_sources"]:
                    if term.lower() not in combined:
                        print(f"  ⚠️  Expected term '{term}' not found in top 3 results")
                        all_ok = False

            if all_ok:
                print(f"  ✅ PASS")
                passed += 1
            else:
                print(f"  ❌ FAIL — expected content not in top results")
        else:
            print(f"  ❌ FAIL — no results returned")

        print()

    return passed, len(stories)


def test_index_stats():
    """Show the current index statistics."""
    print("=" * 70)
    print("INDEX STATISTICS")
    print("=" * 70)

    pipeline = IngestionPipeline()
    stats = pipeline.get_stats()

    print(f"  Total files: {stats['total_files']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  ChromaDB count: {stats['collection_count']}")
    print()

    for course, info in sorted(stats.get("courses", {}).items()):
        print(f"  📁 {course}: {info['files']} files, {info['chunks']} chunks")

    # Check for files with 0 chunks
    zero_chunk_files = [
        path for path, info in pipeline.manifest.get("files", {}).items()
        if info.get("chunks", 0) == 0
    ]
    if zero_chunk_files:
        print(f"\n  ⚠️  {len(zero_chunk_files)} files with 0 chunks:")
        for f in zero_chunk_files[:10]:
            print(f"     - {f}")
        if len(zero_chunk_files) > 10:
            print(f"     ... and {len(zero_chunk_files) - 10} more")

    print()
    return stats


if __name__ == "__main__":
    print("\n🧪 RAG System Retrieval Quality Test Suite\n")

    # Test 1: Course aliases
    alias_ok = test_course_alias_resolution()

    # Test 2: Index stats
    stats = test_index_stats()

    # Test 3: User stories
    print("\nInitializing retriever...\n")
    retriever = HybridRetriever()
    story_passed, story_total = test_retrieval_stories(retriever)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Course aliases:  {'✅ PASS' if alias_ok else '❌ FAIL'}")
    print(f"  Index:           {stats['total_files']} files, {stats['total_chunks']} chunks")
    print(f"  User stories:    {story_passed}/{story_total} passed")

    zero_chunks = sum(1 for info in IngestionPipeline().manifest.get("files", {}).values()
                      if info.get("chunks", 0) == 0)
    if zero_chunks:
        print(f"  ⚠️  {zero_chunks} files still have 0 chunks (need OCR or re-ingest)")

    print()
