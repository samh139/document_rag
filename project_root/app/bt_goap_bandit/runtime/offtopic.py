# src/bt_goap_bandit/runtime/offtopic.py
from __future__ import annotations

def is_offtopic(hit_count: int, top_score: float, min_docs: int = 1, min_score: float = 2.0) -> bool:
    """
    Simple off-topic heuristic for BM25-style scores.
    Tune min_score using your corpus stats.
    """
    if hit_count < min_docs:
        return True
    if top_score < min_score:
        return True
    return False

def reason(hit_count: int, top_score: float, min_docs: int = 1, min_score: float = 2.0) -> str:
    if hit_count < min_docs:
        return f"No documents matched (need ≥{min_docs})."
    if top_score < min_score:
        return f"Top score {top_score:.2f} below threshold {min_score:.2f}."
    return ""
