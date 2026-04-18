#!/usr/bin/env python3
"""
Calibration audit for retrieval-quality classification (Acceptance Criterion #11).

Runs the 6 golden queries + 3 deliberately-out-of-domain probes through
`HybridRetriever.retrieve_with_diagnostics` and classifies each using the
same logic as `RAGChain._classify_retrieval` (empty / weak / good).

Expected:
  - All 6 golden queries classify as `good`.
  - All 3 OOD probes classify as `weak` or `empty`.

If anything miscategorizes, the script REPORTS the offending case and
suggests a threshold range that would fix it. It does NOT auto-tune the
thresholds — tuning is a human decision.

Usage:
    source .venv/bin/activate
    python scripts/calibrate_classification.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

from backend.generation.chain import (  # noqa: E402
    WEAK_OVERLAP_THRESHOLD,
    WEAK_TOP_SIM_THRESHOLD,
)
from backend.retrieval.retriever import HybridRetriever  # noqa: E402


GOLDEN_QUERIES = [
    # (query, course)
    ("Explain the A* search algorithm", "ELEC 472"),
    ("What are the core steps in data preprocessing for machine learning?", "ELEC 472"),
    ("What is a context-free grammar?", "CMPE 223"),
    ("Explain regular expressions and their use in lexical analysis", "CMPE 223"),
    ("What is remote procedure call (RPC) and how does it work?", "ELEC 477"),
    ("Explain primary-backup replication in distributed systems", "ELEC 477"),
]

# Out-of-domain probes — topics that should not appear in any course vault.
OOD_PROBES = [
    ("quantum entanglement", "ELEC 472"),
    ("macroeconomic policy", "CMPE 223"),
    ("medieval french literature", "ELEC 477"),
]


def classify(results, diagnostics) -> str:
    """Mirror of RAGChain._classify_retrieval — keep in sync with chain.py."""
    if not results or diagnostics.vector_hits == 0:
        return "empty"
    if (
        diagnostics.top_vector_similarity < WEAK_TOP_SIM_THRESHOLD
        and diagnostics.vector_keyword_overlap < WEAK_OVERLAP_THRESHOLD
    ):
        return "weak"
    return "good"


def run_batch(retriever: HybridRetriever, batch: list, label: str, expected: set):
    print(f"\n=== {label} (expected: {expected}) ===\n")
    miscategorized = []
    for query, course in batch:
        results, diag = retriever.retrieve_with_diagnostics(
            query=query, top_k=10, course=course,
        )
        cls = classify(results, diag)
        top_sim = diag.top_vector_similarity
        overlap = diag.vector_keyword_overlap
        short_q = query if len(query) <= 50 else query[:47] + "..."
        mark = "OK  " if cls in expected else "MISS"
        print(
            f"  {mark}  [{cls:5s}] {short_q!r:55s}  course={course:12s}  "
            f"top_sim={top_sim:.3f}  overlap={overlap}  "
            f"hits=(v={diag.vector_hits},k={diag.keyword_hits})"
        )
        if cls not in expected:
            miscategorized.append(
                {
                    "query": query,
                    "course": course,
                    "got": cls,
                    "expected": expected,
                    "top_sim": top_sim,
                    "overlap": overlap,
                }
            )
    return miscategorized


def _suggest_thresholds(misses: list, batch_kind: str) -> list:
    """Given misclassifications, propose threshold adjustments. Non-prescriptive —
    reports ranges that WOULD change the classification for each miss."""
    suggestions = []
    for m in misses:
        if batch_kind == "golden" and m["got"] in ("weak", "empty"):
            # Golden query classified weak — we'd need to RELAX the thresholds.
            suggestions.append(
                f"  - Golden query {m['query']!r} classified {m['got']!r}: "
                f"top_sim={m['top_sim']:.3f}, overlap={m['overlap']}. "
                f"To rescue, drop WEAK_TOP_SIM_THRESHOLD below "
                f"{m['top_sim']:.3f} OR WEAK_OVERLAP_THRESHOLD to "
                f"{m['overlap']} (currently {WEAK_TOP_SIM_THRESHOLD} / "
                f"{WEAK_OVERLAP_THRESHOLD})."
            )
        elif batch_kind == "ood" and m["got"] == "good":
            # OOD query classified good — we'd need to TIGHTEN thresholds.
            suggestions.append(
                f"  - OOD probe {m['query']!r} classified 'good': "
                f"top_sim={m['top_sim']:.3f}, overlap={m['overlap']}. "
                f"To reject, raise WEAK_TOP_SIM_THRESHOLD above "
                f"{m['top_sim']:.3f} OR WEAK_OVERLAP_THRESHOLD above "
                f"{m['overlap']} (currently {WEAK_TOP_SIM_THRESHOLD} / "
                f"{WEAK_OVERLAP_THRESHOLD})."
            )
    return suggestions


def main() -> int:
    print("Retrieval-quality classification calibration audit")
    print(f"Thresholds: WEAK_TOP_SIM_THRESHOLD={WEAK_TOP_SIM_THRESHOLD}, "
          f"WEAK_OVERLAP_THRESHOLD={WEAK_OVERLAP_THRESHOLD}")

    retriever = HybridRetriever()

    golden_misses = run_batch(
        retriever, GOLDEN_QUERIES, "GOLDEN (in-domain)", expected={"good"},
    )
    ood_misses = run_batch(
        retriever, OOD_PROBES, "OOD (out-of-domain probes)", expected={"weak", "empty"},
    )

    print("\n=== SUMMARY ===")
    print(f"  Golden queries:  {len(GOLDEN_QUERIES) - len(golden_misses)}/{len(GOLDEN_QUERIES)} classified 'good'")
    print(f"  OOD probes:      {len(OOD_PROBES) - len(ood_misses)}/{len(OOD_PROBES)} classified 'weak'/'empty'")

    suggestions = (
        _suggest_thresholds(golden_misses, "golden")
        + _suggest_thresholds(ood_misses, "ood")
    )
    if suggestions:
        print("\nSuggested threshold adjustments (NOT applied — human review required):")
        for s in suggestions:
            print(s)
        print("\nResult: MISCATEGORIZATION DETECTED — see suggestions above.")
        return 1

    print("\nResult: classification is cleanly calibrated — no adjustments needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
