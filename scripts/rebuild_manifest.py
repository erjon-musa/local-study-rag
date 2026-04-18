"""
Rebuild `data/manifest.json` from the current ChromaDB state.

Why this exists
---------------
The ingestion pipeline tracks a SHA-256 manifest per file so it can do
incremental re-ingests. If that manifest is lost (or the app was started
without it while ChromaDB already had chunks), the pipeline no longer
knows what's indexed, and every run looks like a "full re-ingest".

This script reconstructs the manifest purely from state:

1. Pull every chunk's metadata out of ChromaDB.
2. Group by `rel_path` (the vault-relative source file).
3. For each file, compute the current on-disk SHA-256 so incremental
   re-ingest works next run.
4. Scan the vault for files that exist on disk but are NOT in ChromaDB
   and write them with `{chunks: 0, status: "missing"}` so we can see
   what still needs (re-)OCR without inventing work.

It never re-ingests anything. It only reconstructs bookkeeping.

Atomic write: always writes to `<manifest>.tmp` and then `os.replace`s
to the final path so an interrupted run can't leave a partial manifest.

Usage:
    python scripts/rebuild_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Make `backend` importable when run as `python scripts/rebuild_manifest.py`
# from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb  # noqa: E402

from backend.ingestion.loader import LOADERS  # noqa: E402

# Supported extensions mirror pipeline.py exactly so the "missing from index"
# list doesn't include files the pipeline would have skipped anyway.
SUPPORTED_EXTENSIONS = set(LOADERS.keys()) | {".csv", ".pub"}

VAULT_PATH = Path(os.getenv("VAULT_PATH", str(Path.home() / "Documents" / "StudyVault"))).expanduser()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
MANIFEST_PATH = Path(os.getenv("MANIFEST_PATH", "./data/manifest.json"))


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, streamed so we don't load giant PDFs into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def collect_indexed_files() -> Dict[str, dict]:
    """
    Pull every chunk's metadata from ChromaDB and group by `rel_path`.

    Returns a mapping of rel_path -> aggregated info:
        {
          "chunks": int,
          "chunk_ids": [str, ...],
          "doc_type": str | None,
          "course": str | None,
          "course_code": str | None,
          "category": str | None,
        }
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name="study_notes",
        metadata={"hnsw:space": "cosine"},
    )

    # include=["metadatas"] + ids are always returned. No embeddings / docs
    # so the pull stays light even on 5k+ chunk collections.
    got = collection.get(include=["metadatas"])
    ids: List[str] = got.get("ids") or []
    metadatas: List[dict] = got.get("metadatas") or []

    grouped: Dict[str, dict] = {}
    for chunk_id, meta in zip(ids, metadatas):
        if not meta:
            continue
        rel_path = meta.get("rel_path")
        if not rel_path:
            # Older chunks without rel_path fall back to source_path relative
            # to the vault, otherwise we skip them — no point guessing.
            source_path = meta.get("source_path")
            if source_path:
                try:
                    rel_path = str(Path(source_path).relative_to(VAULT_PATH))
                except ValueError:
                    continue
            else:
                continue

        entry = grouped.setdefault(
            rel_path,
            {
                "chunks": 0,
                "chunk_ids": [],
                "doc_type": None,
                "course": None,
                "course_code": None,
                "category": None,
            },
        )
        entry["chunks"] += 1
        entry["chunk_ids"].append(chunk_id)
        # Metadata values are strings in Chroma; first-seen wins (they should
        # be identical across chunks of the same file).
        for key in ("doc_type", "course", "course_code", "category"):
            if entry[key] is None and meta.get(key):
                entry[key] = meta.get(key)

    return grouped


def scan_vault_files() -> Dict[str, Path]:
    """
    Walk the vault and return {rel_path: absolute_path} for every supported file.

    Mirrors IngestionPipeline.scan() filtering rules so "missing from index"
    is an apples-to-apples comparison.
    """
    found: Dict[str, Path] = {}
    if not VAULT_PATH.exists():
        print(f"WARN: VAULT_PATH does not exist: {VAULT_PATH}")
        return found

    for root, dirs, files in os.walk(VAULT_PATH):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for filename in files:
            if filename.startswith(".") or filename.startswith("_"):
                continue
            filepath = Path(root) / filename
            if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            rel_path = str(filepath.relative_to(VAULT_PATH))
            found[rel_path] = filepath
    return found


def atomic_write_json(path: Path, data: dict) -> None:
    """Write `data` as JSON to `path` atomically via a .tmp + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def rebuild() -> dict:
    indexed = collect_indexed_files()
    on_disk = scan_vault_files()

    manifest_files: Dict[str, dict] = {}
    now = iso_now()

    total_chunks = 0
    missing_count = 0

    # 1) Every file ChromaDB knows about.
    for rel_path, info in indexed.items():
        abs_path = on_disk.get(rel_path)
        if abs_path is not None and abs_path.exists():
            sha = file_sha256(abs_path)
        else:
            sha = None  # file was removed from disk but chunks still live in Chroma

        entry: Dict[str, Optional[object]] = {
            "hash": sha,
            "sha256": sha,  # accept both names; some callers look for one or the other
            "chunks": info["chunks"],
            "chunk_ids": sorted(info["chunk_ids"]),
            "doc_type": info["doc_type"],
            "course": info["course"],
            "course_code": info["course_code"],
            "category": info["category"],
            "ingested_at": now,
        }
        if sha is None:
            entry["status"] = "stale_on_disk"  # in Chroma but file gone
        manifest_files[rel_path] = entry
        total_chunks += info["chunks"]

    # 2) Files on disk that ChromaDB has never seen (= still need ingest).
    for rel_path, abs_path in on_disk.items():
        if rel_path in manifest_files:
            continue
        try:
            sha = file_sha256(abs_path)
        except OSError as e:
            print(f"WARN: could not hash {rel_path}: {e}")
            sha = None
        manifest_files[rel_path] = {
            "hash": sha,
            "sha256": sha,
            "chunks": 0,
            "chunk_ids": [],
            "doc_type": None,
            "course": None,
            "course_code": None,
            "category": None,
            "status": "missing",
            "ingested_at": None,
        }
        missing_count += 1

    manifest = {
        "version": 1,
        "rebuilt_at": now,
        "source": "rebuild_manifest.py",
        "total_chunks": total_chunks,
        "files": manifest_files,
    }
    return manifest


def main() -> int:
    print(f"Rebuilding manifest from ChromaDB at {CHROMA_PERSIST_DIR}")
    print(f"Vault: {VAULT_PATH}")
    print(f"Output: {MANIFEST_PATH}")

    manifest = rebuild()

    atomic_write_json(MANIFEST_PATH, manifest)

    files = manifest["files"]
    n_files = len(files)
    total_chunks = manifest["total_chunks"]
    k_missing = sum(1 for v in files.values() if v.get("status") == "missing")

    print(
        f"Reconstructed manifest: {n_files} files, "
        f"{total_chunks} total chunks, "
        f"{k_missing} files missing from index"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
