"""
Ingestion pipeline orchestrator.

Manages the full lifecycle:
1. Scan vault for files
2. Compare against manifest (detect new/modified/deleted)
3. Load → chunk → embed → store in ChromaDB
4. Update manifest

Incremental: only processes what changed since last scan.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb

from .chunker import Chunk, chunk_documents
from .embedder import embed_in_batches
from .loader import LOADERS, Document, PdfLoadStats, load_file_with_stats

# Supported file extensions (from loader.py)
SUPPORTED_EXTENSIONS = set(LOADERS.keys()) | {".csv", ".pub"}

VAULT_PATH = os.getenv("VAULT_PATH", str(Path.home() / "Documents" / "StudyVault"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "./data/manifest.json")


@dataclass
class ScanResult:
    """Result of scanning the vault against the manifest."""
    new_files: List[Path] = field(default_factory=list)
    modified_files: List[Path] = field(default_factory=list)
    deleted_files: List[str] = field(default_factory=list)  # relative paths
    unchanged_files: List[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.new_files or self.modified_files or self.deleted_files)

    def summary(self) -> dict:
        return {
            "new": len(self.new_files),
            "modified": len(self.modified_files),
            "deleted": len(self.deleted_files),
            "unchanged": len(self.unchanged_files),
        }


@dataclass
class IngestResult:
    """Result of an ingestion run."""
    new: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    total_chunks: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class IngestionPipeline:
    """
    Full ingestion pipeline with incremental processing.
    
    Tracks a manifest of ingested files so it only re-processes
    what has actually changed.
    """

    def __init__(
        self,
        vault_path: str = None,
        chroma_persist_dir: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        manifest_path: str = None,
    ):
        self.vault_path = Path(vault_path or VAULT_PATH).expanduser()
        self.chroma_persist_dir = chroma_persist_dir or CHROMA_PERSIST_DIR
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.manifest_path = Path(manifest_path or MANIFEST_PATH)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="study_notes",
            metadata={"hnsw:space": "cosine"},
        )

        # Load manifest
        self.manifest = self._load_manifest()

    # ── Manifest management ──────────────────────────────────

    def _load_manifest(self) -> dict:
        """Load the file manifest from disk."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"files": {}}

    def _save_manifest(self):
        """
        Save the file manifest to disk atomically.

        Writes to `<manifest>.tmp` first, then `os.replace` swaps it into
        place. Prevents partial-write corruption if the process is killed
        mid-write (e.g. during a long sync).
        """
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    # ── File hashing ─────────────────────────────────────────

    @staticmethod
    def _file_hash(path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ── Vault scanning ───────────────────────────────────────

    def _detect_course_and_category(self, rel_path: str) -> Tuple[str, str]:
        """Extract course and category from the relative path in the vault."""
        parts = Path(rel_path).parts

        course = parts[0] if len(parts) > 0 else "Unknown"
        category = parts[1] if len(parts) > 1 else "Resources"

        return course, category

    @staticmethod
    def _extract_course_code(course: str) -> str:
        """
        Extract short course code from full course name.
        'CMPE 223 - Software Specification' → 'CMPE 223'
        """
        if " - " in course:
            return course.split(" - ")[0].strip()
        return course

    @staticmethod
    def _extract_doc_type(category: str) -> str:
        """
        Normalize category folder name to a doc_type.
        'Lectures' → 'lecture', 'Exams' → 'exam'
        """
        mapping = {
            "Lectures": "lecture",
            "Exams": "exam",
            "Assignments": "assignment",
            "Labs": "lab",
            "Resources": "resource",
        }
        return mapping.get(category, "resource")

    @staticmethod
    def _extract_year(filename: str) -> str:
        """
        Try to extract a year from the filename.
        'Final Exam - April 2014.pdf' → '2014'
        'midtermSampleSolutionsW26.pdf' → '2026'
        'a1W26.pdf' → '2026'
        """
        import re

        # Match explicit 4-digit years (2000-2099)
        m = re.search(r"(20\d{2})", filename)
        if m:
            return m.group(1)

        # Match W26, W22 etc. (Queen's winter term notation)
        m = re.search(r"W(\d{2})", filename)
        if m:
            year_short = int(m.group(1))
            return str(2000 + year_short)

        return ""

    def scan(self) -> ScanResult:
        """
        Scan the vault directory and compare against the manifest.
        
        This is fast (<100ms) when nothing has changed because it only
        computes hashes for files not in the manifest or with different sizes.
        """
        result = ScanResult()
        current_files = {}

        # Walk the vault
        for root, dirs, files in os.walk(self.vault_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filename in files:
                if filename.startswith(".") or filename.startswith("_"):
                    continue

                filepath = Path(root) / filename
                ext = filepath.suffix.lower()

                # Only process supported file types
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                rel_path = str(filepath.relative_to(self.vault_path))
                current_files[rel_path] = filepath

                # Check manifest
                manifest_entry = self.manifest["files"].get(rel_path)

                if manifest_entry is None:
                    # New file
                    result.new_files.append(filepath)
                else:
                    # Check if modified (compare hash)
                    current_hash = self._file_hash(filepath)
                    if current_hash != manifest_entry.get("hash"):
                        result.modified_files.append(filepath)
                    else:
                        result.unchanged_files.append(rel_path)

        # Check for deleted files
        for rel_path in self.manifest["files"]:
            if rel_path not in current_files:
                result.deleted_files.append(rel_path)

        return result

    # ── Ingestion ────────────────────────────────────────────

    def _ingest_file(
        self, filepath: Path, force: bool = False
    ) -> Tuple[int, List[str], PdfLoadStats]:
        """
        Ingest a single file: load → chunk → embed → store.

        Returns (chunk_count, chunk_ids, stats). Stats include per-page
        OCR outcomes so the caller can record `ocr_pages`/`ocr_failed_pages`
        on the manifest and surface them via the API.

        `force=True` signals to callers (e.g. `scripts/force_reocr.py`) that
        the caller intentionally bypassed the manifest hash check. It does
        not change ingest behavior inside this method — both paths go
        through load → chunk → embed — but we record it on the manifest
        entry below for traceability.
        """
        rel_path = str(filepath.relative_to(self.vault_path))
        course, category = self._detect_course_and_category(rel_path)

        # Compute enriched metadata fields
        course_code = self._extract_course_code(course)
        doc_type = self._extract_doc_type(category)
        year = self._extract_year(filepath.name)

        # Load (with per-page OCR stats)
        docs, stats = load_file_with_stats(filepath, course=course, category=category)
        if not docs:
            return 0, [], stats

        # Inject enriched metadata into every loaded document
        for doc in docs:
            doc.metadata["course_code"] = course_code
            doc.metadata["doc_type"] = doc_type
            if year:
                doc.metadata["year"] = year

        # Chunk
        chunks = chunk_documents(docs, max_size=self.chunk_size, overlap=self.chunk_overlap)
        if not chunks:
            return 0, [], stats

        # Embed
        texts = [c.text for c in chunks]
        embeddings = embed_in_batches(texts, batch_size=32)

        # Store in ChromaDB
        ids = [f"{rel_path}:{i}" for i in range(len(chunks))]
        metadatas = []
        for chunk in chunks:
            meta = {k: str(v) for k, v in chunk.metadata.items()}
            meta["rel_path"] = rel_path
            metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return len(chunks), ids, stats

    def _remove_file(self, rel_path: str):
        """Remove a file's chunks from ChromaDB and manifest."""
        manifest_entry = self.manifest["files"].get(rel_path, {})
        chunk_ids = manifest_entry.get("chunk_ids", [])

        if chunk_ids:
            try:
                self.collection.delete(ids=chunk_ids)
            except Exception as e:
                # IDs might not exist anymore — log so it's visible but not fatal.
                print(
                    f"  ⚠ _remove_file: delete by id failed for {rel_path}: "
                    f"{type(e).__name__}: {e}"
                )

        # Belt-and-suspenders: also sweep by rel_path metadata in case the
        # manifest's chunk_ids list drifted from ChromaDB reality.
        try:
            self.collection.delete(where={"rel_path": rel_path})
        except Exception as e:
            print(
                f"  ⚠ _remove_file: delete by where failed for {rel_path}: "
                f"{type(e).__name__}: {e}"
            )

        # Remove from manifest
        self.manifest["files"].pop(rel_path, None)

    def ingest(self, force_full: bool = False) -> IngestResult:
        """
        Run the ingestion pipeline.
        
        Scans the vault, detects changes, and only processes what's new/modified.
        Set force_full=True to re-ingest everything.
        """
        start_time = time.time()
        result = IngestResult()

        if force_full:
            # Clear everything and start fresh
            self.manifest = {"files": {}}
            try:
                self.chroma_client.delete_collection("study_notes")
                self.collection = self.chroma_client.get_or_create_collection(
                    name="study_notes",
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                print(
                    f"  ⚠ force_full: could not reset collection: "
                    f"{type(e).__name__}: {e}"
                )

        # Scan for changes
        scan = self.scan()
        
        if not scan.has_changes and not force_full:
            result.skipped = len(scan.unchanged_files)
            result.duration_seconds = time.time() - start_time
            return result

        # Process deleted files
        for rel_path in scan.deleted_files:
            self._remove_file(rel_path)
            result.deleted += 1

        # Process modified files (delete old chunks, then re-ingest)
        for filepath in scan.modified_files:
            rel_path = str(filepath.relative_to(self.vault_path))
            self._remove_file(rel_path)

            try:
                chunk_count, chunk_ids, stats = self._ingest_file(filepath)
                self.manifest["files"][rel_path] = {
                    "hash": self._file_hash(filepath),
                    "chunks": chunk_count,
                    "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "chunk_ids": chunk_ids,
                    "ocr_pages": stats.ocr_pages,
                    "ocr_failed_pages": stats.ocr_failed_pages,
                    "skipped_blank_pages": stats.skipped_blank_pages,
                }
                result.updated += 1
                result.total_chunks += chunk_count
                # Forward per-page OCR failures up to IngestResult so sync API can surface them.
                for err in stats.errors:
                    result.errors.append(f"{rel_path}: {err}")
                print(f"  ↻ Updated: {rel_path} ({chunk_count} chunks)")
            except Exception as e:
                err_class = type(e).__name__
                result.errors.append(f"{rel_path}: {err_class}: {e}")
                print(f"  ✗ Skipped {rel_path}: {err_class}: {e}")

        # Process new files
        for filepath in scan.new_files:
            rel_path = str(filepath.relative_to(self.vault_path))

            try:
                chunk_count, chunk_ids, stats = self._ingest_file(filepath)
                self.manifest["files"][rel_path] = {
                    "hash": self._file_hash(filepath),
                    "chunks": chunk_count,
                    "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "chunk_ids": chunk_ids,
                    "ocr_pages": stats.ocr_pages,
                    "ocr_failed_pages": stats.ocr_failed_pages,
                    "skipped_blank_pages": stats.skipped_blank_pages,
                }
                result.new += 1
                result.total_chunks += chunk_count
                for err in stats.errors:
                    result.errors.append(f"{rel_path}: {err}")
                print(f"  ✓ Ingested: {rel_path} ({chunk_count} chunks)")
            except Exception as e:
                err_class = type(e).__name__
                result.errors.append(f"{rel_path}: {err_class}: {e}")
                print(f"  ✗ Skipped {rel_path}: {err_class}: {e}")

        result.skipped = len(scan.unchanged_files)
        result.duration_seconds = time.time() - start_time

        # Save manifest
        self._save_manifest()

        return result

    def get_stats(self) -> dict:
        """Get statistics about the indexed documents."""
        total_files = len(self.manifest["files"])
        total_chunks = sum(f.get("chunks", 0) for f in self.manifest["files"].values())

        # Per-course breakdown
        courses = {}
        for rel_path, info in self.manifest["files"].items():
            course = Path(rel_path).parts[0] if Path(rel_path).parts else "Unknown"
            if course not in courses:
                courses[course] = {"files": 0, "chunks": 0}
            courses[course]["files"] += 1
            courses[course]["chunks"] += info.get("chunks", 0)

        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "collection_count": self.collection.count(),
            "courses": courses,
        }
