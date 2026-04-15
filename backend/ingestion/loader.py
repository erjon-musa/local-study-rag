"""
Document loaders for the RAG ingestion pipeline.

Supports: PDF, Markdown, TXT, DOCX, HTML
Each loader returns a list of Document objects with text and metadata.

For scanned PDF pages (image-only, no extractable text), falls back to
Gemma4 multimodal OCR via LM Studio to extract text from the page image.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import openai

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "google/gemma-4-26b-a4b")

# OCR prompt for Gemma4 multimodal — tuned for academic documents
OCR_PROMPT = (
    "Extract ALL text from this document page exactly as written. "
    "Include headings, questions, formulas, and any mathematical notation. "
    "Preserve the structure (numbered questions, sub-parts like (i), (ii), etc.). "
    "Output only the extracted text, nothing else. "
    "Do not use internal reasoning. Respond directly."
)


@dataclass
class Document:
    """A loaded document with text content and metadata."""
    text: str
    metadata: dict = field(default_factory=dict)


def _ocr_page_with_gemma(page_png_bytes: bytes) -> str:
    """
    Send a page image to Gemma4 multimodal via LM Studio for OCR.

    Args:
        page_png_bytes: PNG image bytes of the page

    Returns:
        Extracted text, or empty string on failure
    """
    b64_image = base64.b64encode(page_png_bytes).decode("utf-8")

    try:
        client = openai.OpenAI(
            base_url=LMSTUDIO_BASE_URL,
            api_key="lmstudio-link",
        )

        response = client.chat.completions.create(
            model=LMSTUDIO_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=4096,
        )

        # Handle both content and reasoning_content (thinking mode)
        message = response.choices[0].message
        text = message.content or ""
        if not text:
            reasoning = getattr(message, "reasoning_content", None) or ""
            text = reasoning

        return text.strip()
    except Exception as e:
        print(f"    ⚠ OCR failed: {e}")
        return ""


def load_pdf(path: Path, course: str = "", category: str = "") -> List[Document]:
    """
    Load a PDF file, returning one Document per page.

    For pages with no extractable text (scanned images), falls back to
    Gemma4 multimodal OCR. Each document includes an 'extraction_method'
    metadata field ('text' or 'ocr_gemma4').
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError("pymupdf is required: pip install pymupdf")

    docs = []
    pdf = fitz.open(str(path))
    ocr_count = 0

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text("text").strip()
        extraction_method = "text"

        # If no text but page has images, try OCR via Gemma4 (unless disabled)
        if not text and page.get_images() and "DISABLE_OCR" not in os.environ:
            # Render page as PNG at 150 DPI (good balance of quality vs size)
            pixmap = page.get_pixmap(dpi=150)
            png_bytes = pixmap.tobytes("png")

            print(f"    🔍 OCR page {page_num + 1}/{len(pdf)} of {path.name}...")
            text = _ocr_page_with_gemma(png_bytes)
            extraction_method = "ocr_gemma4"

            if text:
                ocr_count += 1

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
                "extraction_method": extraction_method,
            }
        ))

    if ocr_count > 0:
        print(f"    ✓ OCR extracted {ocr_count} pages from {path.name}")

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
            "extraction_method": "text",
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
            "extraction_method": "text",
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
            "extraction_method": "text",
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
            "extraction_method": "text",
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
