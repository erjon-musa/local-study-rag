#!/usr/bin/env python3
"""
build_graph.py — populate data/knowledge_graph.json from ingested chunks.

Pulls every chunk currently in the ChromaDB `study_notes` collection, runs
the Gemma-based entity/relationship extractor on each, and saves the result
to data/knowledge_graph.json so the /api/graph endpoints have something to
serve.

This script is intentionally thin — all the heavy logic lives in
backend/ingestion/graph.py. Run it after ingestion has populated ChromaDB.

Usage:
    python scripts/build_graph.py
    python scripts/build_graph.py --max-chunks 50    # smoke-test on a subset
    python scripts/build_graph.py --course "ELEC 472"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the backend package importable when run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb  # noqa: E402

from backend.config import settings  # noqa: E402
from backend.ingestion.graph import build_graph, save_graph  # noqa: E402


def load_chunks_from_chroma(course_filter: str | None) -> list[dict]:
    """
    Pull all chunks (or a course-filtered subset) out of ChromaDB and shape
    them into the dicts that build_graph() expects.
    """
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        collection = client.get_collection(settings.chroma_collection_name)
    except Exception as exc:
        raise SystemExit(
            f"ChromaDB collection '{settings.chroma_collection_name}' not found "
            f"at {settings.chroma_persist_dir}. Run ingestion first."
        ) from exc

    where = {"course": course_filter} if course_filter else None
    data = collection.get(where=where, include=["documents", "metadatas"])

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    chunks: list[dict] = []
    for text, meta in zip(documents, metadatas):
        meta = meta or {}
        chunks.append(
            {
                "text": text or "",
                "course": meta.get("course", "Unknown"),
                "category": meta.get("category", meta.get("doc_type", "Unknown")),
                "filename": meta.get("source", "Unknown"),
            }
        )
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the knowledge graph from ingested ChromaDB chunks.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Cap the number of chunks processed (useful for smoke-testing).",
    )
    parser.add_argument(
        "--course",
        default=None,
        help="Only build the graph for chunks tagged with this course.",
    )
    args = parser.parse_args()

    print(f"build_graph: chroma_dir={settings.chroma_persist_dir}")
    chunks = load_chunks_from_chroma(args.course)
    print(f"build_graph: loaded {len(chunks)} chunks from ChromaDB")

    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]
        print(f"build_graph: capped to first {len(chunks)} chunks (--max-chunks)")

    if not chunks:
        print("build_graph: nothing to process — exiting.")
        return 1

    print(
        "build_graph: extracting entities + relations "
        "(this calls LM Studio once per chunk; expect minutes-to-hours)…"
    )
    t0 = time.time()

    def _progress(current: int, total: int, msg: str) -> None:
        if current == 1 or current == total or current % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{current}/{total}] {msg}  ({elapsed:.0f}s elapsed)")

    graph = build_graph(chunks, progress_callback=_progress)

    save_graph(graph)
    elapsed = time.time() - t0
    print(
        f"build_graph: done in {elapsed:.0f}s — "
        f"{len(graph.nodes)} nodes, {len(graph.edges)} edges, "
        f"{len(graph.errors)} chunk errors"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
