"""
Document loaders for the RAG ingestion pipeline.

Supports: PDF, Markdown, TXT, DOCX, HTML
Each loader returns a list of Document objects with text and metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Document:
    """A loaded document with text content and metadata."""
    text: str
    metadata: dict = field(default_factory=dict)


def load_pdf(path: Path, course: str = "", category: str = "") -> List[Document]:
    """Load a PDF file, returning one Document per page."""
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError("pymupdf is required: pip install pymupdf")

    docs = []
    pdf = fitz.open(str(path))

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text("text").strip()

        if not text:
            continue

        docs.append(Document(
            text=text,
            metadata={
                "source": path.name,
                "source_path": str(path),
                "page": page_num + 1,
                "total_pages": len(pdf),
                "course": course,
                "category": category,
                "file_type": "pdf",
            }
        ))

    pdf.close()
    return docs


def load_markdown(path: Path, course: str = "", category: str = "") -> List[Document]:
    """Load a markdown file as a single Document."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()

    if not text:
        return []

    return [Document(
        text=text,
        metadata={
            "source": path.name,
            "source_path": str(path),
            "course": course,
            "category": category,
            "file_type": "markdown",
        }
    )]


def load_text(path: Path, course: str = "", category: str = "") -> List[Document]:
    """Load a plain text file (e.g. lecture captions) as a single Document."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()

    if not text:
        return []

    return [Document(
        text=text,
        metadata={
            "source": path.name,
            "source_path": str(path),
            "course": course,
            "category": category,
            "file_type": "text",
        }
    )]


def load_docx(path: Path, course: str = "", category: str = "") -> List[Document]:
    """Load a DOCX file as a single Document."""
    try:
        from docx import Document as DocxDocument
    except ImportError:
        raise ImportError("python-docx is required: pip install python-docx")

    doc = DocxDocument(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)

    if not text:
        return []

    return [Document(
        text=text,
        metadata={
            "source": path.name,
            "source_path": str(path),
            "course": course,
            "category": category,
            "file_type": "docx",
        }
    )]


def load_html(path: Path, course: str = "", category: str = "") -> List[Document]:
    """Load an HTML file, extracting text content."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4")

    html_content = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer"]):
        element.decompose()

    text = soup.get_text(separator="\n", strip=True)

    if not text:
        return []

    return [Document(
        text=text,
        metadata={
            "source": path.name,
            "source_path": str(path),
            "course": course,
            "category": category,
            "file_type": "html",
        }
    )]


# Extension → loader mapping
LOADERS = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".txt": load_text,
    ".docx": load_docx,
    ".html": load_html,
}


def load_file(path: Path, course: str = "", category: str = "") -> List[Document]:
    """Load a file using the appropriate loader based on extension."""
    ext = path.suffix.lower()
    loader = LOADERS.get(ext)

    if loader is None:
        print(f"  ⚠ Unsupported file type: {ext} ({path.name})")
        return []

    try:
        return loader(path, course=course, category=category)
    except Exception as e:
        print(f"  ✗ Error loading {path.name}: {e}")
        return []
