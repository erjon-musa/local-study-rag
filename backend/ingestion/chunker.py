"""
Smart document chunking for the RAG pipeline.

Strategies:
- Heading-aware: splits markdown/text at heading boundaries
- Page-based: one chunk per PDF page (natural slide boundaries)
- Overlap: configurable overlap between chunks for context continuity
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .loader import Document


@dataclass
class Chunk:
    """A chunk of text with metadata for embedding and retrieval."""
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""


def chunk_by_headings(doc: Document, max_size: int = 800, overlap: int = 100) -> List[Chunk]:
    """
    Split a document at markdown headings, then further split
    any oversized sections by paragraph boundaries.
    """
    text = doc.text
    chunks = []

    # Split at markdown headings (# ## ### etc.)
    heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    sections = []
    last_end = 0
    current_heading = ""

    for match in heading_pattern.finditer(text):
        if last_end < match.start():
            section_text = text[last_end:match.start()].strip()
            if section_text:
                sections.append((current_heading, section_text))

        current_heading = match.group(2).strip()
        last_end = match.end()

    # Don't forget the last section
    remaining = text[last_end:].strip()
    if remaining:
        sections.append((current_heading, remaining))

    # If no headings found, treat the whole doc as one section
    if not sections:
        sections = [("", text)]

    # Now split oversized sections by paragraphs
    for heading, section_text in sections:
        if len(section_text) <= max_size:
            chunk_meta = {**doc.metadata, "heading": heading}
            chunks.append(Chunk(text=section_text, metadata=chunk_meta))
        else:
            # Split by paragraphs
            paragraphs = re.split(r"\n\s*\n", section_text)
            current_chunk = ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                if len(current_chunk) + len(para) + 2 > max_size and current_chunk:
                    chunk_meta = {**doc.metadata, "heading": heading}
                    chunks.append(Chunk(text=current_chunk, metadata=chunk_meta))

                    # Overlap: keep the tail of the previous chunk
                    if overlap > 0 and len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    current_chunk = (current_chunk + "\n\n" + para).strip()

            if current_chunk.strip():
                chunk_meta = {**doc.metadata, "heading": heading}
                chunks.append(Chunk(text=current_chunk, metadata=chunk_meta))

    return chunks


def chunk_by_page(doc: Document) -> List[Chunk]:
    """
    Use the document as-is (one chunk per page).
    Best for PDF pages that are already natural boundaries (slides).
    """
    chunk_meta = {**doc.metadata}
    return [Chunk(text=doc.text, metadata=chunk_meta)]


def chunk_by_fixed_size(doc: Document, max_size: int = 800, overlap: int = 100) -> List[Chunk]:
    """
    Split text into fixed-size chunks with overlap.
    Tries to split at sentence boundaries.
    """
    text = doc.text
    chunks = []

    if len(text) <= max_size:
        return [Chunk(text=text, metadata={**doc.metadata})]

    # Split at sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_size and current_chunk:
            chunks.append(Chunk(text=current_chunk.strip(), metadata={**doc.metadata}))

            # Overlap
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()

    if current_chunk.strip():
        chunks.append(Chunk(text=current_chunk.strip(), metadata={**doc.metadata}))

    return chunks


def chunk_document(doc: Document, max_size: int = 800, overlap: int = 100) -> List[Chunk]:
    """
    Choose the best chunking strategy based on document type.
    """
    file_type = doc.metadata.get("file_type", "")

    if file_type == "pdf":
        # PDF pages are already one-per-page from the loader
        # If the page is small enough, keep as-is; otherwise split further
        if len(doc.text) <= max_size:
            return chunk_by_page(doc)
        else:
            return chunk_by_fixed_size(doc, max_size, overlap)

    elif file_type in ("markdown", "html"):
        # Markdown and HTML benefit from heading-aware chunking
        return chunk_by_headings(doc, max_size, overlap)

    elif file_type == "text":
        # Plain text (captions) — fixed size with sentence splitting
        return chunk_by_fixed_size(doc, max_size, overlap)

    elif file_type == "docx":
        # Treat like markdown for heading detection
        return chunk_by_headings(doc, max_size, overlap)

    else:
        return chunk_by_fixed_size(doc, max_size, overlap)


def _build_context_header(metadata: dict) -> str:
    """
    Build a contextual header to prepend to each chunk.

    This dramatically improves retrieval for document-specific queries
    (e.g., "question 3 from the 223 exam") because the header gives
    the embedding model document-level context that raw chunk text lacks.

    Format: [CMPE 223 - Exam - filename.pdf, Page 3]
    """
    parts = []

    # Course (short form if possible)
    course = metadata.get("course", "")
    if course:
        # Extract short code: "CMPE 223 - Software Specification" → "CMPE 223"
        short_course = course.split(" - ")[0].strip() if " - " in course else course
        parts.append(short_course)

    # Document type from category
    category = metadata.get("category", "")
    if category:
        # Normalize: "Lectures" → "Lecture", "Exams" → "Exam"
        doc_type = category.rstrip("s") if category.endswith("s") else category
        parts.append(doc_type)

    # Filename
    source = metadata.get("source", "")
    if source:
        parts.append(source)

    # Page number
    page = metadata.get("page", "")
    if page:
        parts.append(f"Page {page}")

    if not parts:
        return ""

    return f"[{' - '.join(parts)}]\n"


def chunk_documents(docs: List[Document], max_size: int = 800, overlap: int = 100) -> List[Chunk]:
    """Chunk a list of documents, assigning unique IDs and contextual headers."""
    all_chunks = []

    for doc in docs:
        doc_chunks = chunk_document(doc, max_size, overlap)

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)

        # Build context header from document metadata
        context_header = _build_context_header(doc.metadata)

        for i, chunk in enumerate(doc_chunks):
            chunk.chunk_id = f"{source}:p{page}:c{i}"
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks_in_doc"] = len(doc_chunks)

            # Prepend context header to chunk text for better retrieval
            if context_header:
                chunk.text = context_header + chunk.text

            all_chunks.append(chunk)

    return all_chunks
