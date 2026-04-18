"""
Unit tests for backend.generation.prompts._neutralize_citations.

These tests don't touch LM Studio or ChromaDB — they validate the pure
string-rewriting behavior used to prevent [N]-marker collisions across
turns in multi-turn chat history.

The function expects sources as either:
  - objects with a `.metadata` dict, OR
  - flat dicts with keys like `doc_type`, `category`, `page`, `filename`,
    `rel_path` (matches the shape chat.py serializes over the wire).

The spec sketch in the plan used a nested `{"metadata": {...}}` dict shape,
which doesn't match the actual implementation — we use the real shape here.
"""
from __future__ import annotations

from backend.generation.prompts import _neutralize_citations


# ---------------------------------------------------------------------------
# 1) Known sources: [1] must be replaced by a neutral ref, no raw [1] left.
# ---------------------------------------------------------------------------

def test_known_sources_flat_dict_shape():
    """Flat-dict sources (what the API actually serializes) should map [1]
    to a neutral parenthesized ref using the dict's doc_type / page fields."""
    sources = [
        {
            "filename": "Lecture Chapter 2 Search.pdf",
            "page": 104,
            "doc_type": "lecture",
            "category": "Lectures",
        },
    ]
    out = _neutralize_citations("A* uses f(n)=g(n)+h(n) [1].", sources)
    assert "[1]" not in out
    # Neutral refs are parenthesized, e.g. "(lecture, p.104)"
    assert "(" in out and ")" in out
    assert "lecture" in out.lower() or "104" in out


def test_known_sources_category_fallback():
    """`category` is the folder-based label (Lectures/Labs/…). If doc_type
    is missing, the helper falls back to `category`."""
    sources = [
        {
            "filename": "Lecture Week 1 - Distributed Systems.pdf",
            "page": 4,
            "category": "Lectures",
        },
    ]
    out = _neutralize_citations("Foo [1].", sources)
    assert "[1]" not in out
    assert "(" in out and ")" in out


# ---------------------------------------------------------------------------
# 2) No sources: strip all markers cleanly, prose survives.
# ---------------------------------------------------------------------------

def test_no_sources_strips():
    out = _neutralize_citations("A* uses f(n) [1] [2].", [])
    assert "[1]" not in out
    assert "[2]" not in out
    # Prose should still be intact and readable.
    assert "A*" in out
    assert "f(n)" in out


def test_no_sources_none_also_strips():
    """Passing `None` should behave like an empty list: strip markers."""
    out = _neutralize_citations("A* uses f(n) [1] [2].", None)
    assert "[1]" not in out
    assert "[2]" not in out


# ---------------------------------------------------------------------------
# 3) Out-of-range: [N] where N > len(sources) — must drop, not crash.
# ---------------------------------------------------------------------------

def test_out_of_range():
    sources = [{"filename": "a.pdf", "doc_type": "lecture", "page": 1}]
    out = _neutralize_citations("Claim [1] extra claim [5].", sources)
    # [1] got replaced; [5] was out of range → dropped, not left as-is.
    assert "[1]" not in out
    assert "[5]" not in out
    # Original prose survives (minus the markers).
    assert "Claim" in out
    assert "extra claim" in out


# ---------------------------------------------------------------------------
# 4) Multi-digit markers ([12]) are handled.
# ---------------------------------------------------------------------------

def test_multi_digit():
    sources = [
        {"filename": f"s{i}.pdf", "doc_type": "lecture", "page": i}
        for i in range(15)
    ]
    out = _neutralize_citations("Cite [12] here.", sources)
    assert "[12]" not in out
    # Prose is preserved around the replacement.
    assert "Cite" in out and "here" in out


# ---------------------------------------------------------------------------
# 5) Years like [2024] (3+ digits) must NOT be mangled.
# ---------------------------------------------------------------------------

def test_year_not_mangled_empty_sources():
    """The regex matches only 1-2 digit brackets. [2024] is 4 digits
    and must pass through untouched even with no sources provided."""
    out = _neutralize_citations("In [2024] they said X.", [])
    assert "[2024]" in out


def test_year_not_mangled_with_sources():
    sources = [{"filename": "a.pdf", "doc_type": "lecture", "page": 1}]
    out = _neutralize_citations("Published in [2024]. Cited [1].", sources)
    assert "[2024]" in out
    assert "[1]" not in out


# ---------------------------------------------------------------------------
# 6) Defensive: empty/None content is a no-op.
# ---------------------------------------------------------------------------

def test_empty_content_is_noop():
    assert _neutralize_citations("", []) == ""
    assert _neutralize_citations(None, None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Allow running the file directly without pytest (pytest isn't in the
# project's requirements.txt yet, so we provide a plain runner).
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys
    import traceback

    tests = [
        test_known_sources_flat_dict_shape,
        test_known_sources_category_fallback,
        test_no_sources_strips,
        test_no_sources_none_also_strips,
        test_out_of_range,
        test_multi_digit,
        test_year_not_mangled_empty_sources,
        test_year_not_mangled_with_sources,
        test_empty_content_is_noop,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception:
            failed += 1
            print(f"ERROR {t.__name__}:")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
