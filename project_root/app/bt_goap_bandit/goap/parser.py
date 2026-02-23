"""
parser.py — v3.0 (Lightweight Lexical Extractor Only)
-----------------------------------------------------------------
This module performs only shallow parsing from Elasticsearch hits.

✓ No embeddings
✓ No SentenceTransformer
✓ No semantic similarity
✓ No corpus-driven anchor discovery
✓ No belief logic
✓ No duplication of belief.py functionality

Outputs:
  - lexical anchors (titles, tags, filenames)
  - lexical slots (detected basic entities)
  - keywords (fallback token list)
-----------------------------------------------------------------
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Utility: Normalize text safely
# ---------------------------------------------------------------------
def _safe(text: str | None) -> str:
    return text if isinstance(text, str) else ""


# ---------------------------------------------------------------------
# Extract lexical anchors from each hit
# ---------------------------------------------------------------------
def extract_dynamic_anchors(hits: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Extracts lexical anchors ONLY from metadata fields.
    These are NOT semantic anchors — purely raw strings.
    """
    anchors = {
        "titles": [],
        "chunk_tags": [],
        "file_tags": [],
        "filenames": [],
        "regions": [],
    }

    for h in hits:
        src = h.get("_source", {})

        fm = src.get("file_metadata", {}) or {}
        cm = src.get("chunk_metadata", {}) or {}

        # Title
        title = _safe(fm.get("title"))
        if title:
            anchors["titles"].append(title)

        # File-level tags
        for t in fm.get("tags", []) or []:
            anchors["file_tags"].append(_safe(t))

        # Chunk-level tags
        for t in cm.get("tags", []) or []:
            anchors["chunk_tags"].append(_safe(t))

        # Regions
        region = _safe(fm.get("region"))
        if region:
            anchors["regions"].append(region)

        # Filename
        fname = _safe(src.get("file_name"))
        if fname:
            anchors["filenames"].append(fname)

    logger.info(f"[Parser] Extracted lexical anchors: {anchors}")
    return anchors


# ---------------------------------------------------------------------
# Simple lexical slot extraction (no NLP)
# ---------------------------------------------------------------------
def extract_slots(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract basic lexical slots that may help GOAP.
    This does NOT guess scope / jurisdiction / benefit.
    """
    slots = {
        "mentioned_documents": [],
        "mentioned_steps": False,
        "mentioned_counts": [],
    }

    for h in hits:
        src = h.get("_source", {})
        content = _safe(src.get("content", "")).lower()

        # Detect step-like phrases
        if "step" in content or "process" in content:
            slots["mentioned_steps"] = True

        # Extract numeric references
        numbers = []
        for token in content.split():
            if token.isdigit():
                numbers.append(int(token))

        if numbers:
            slots["mentioned_counts"].extend(numbers)

        # Track referenced doc names
        fname = _safe(src.get("file_name"))
        if fname:
            slots["mentioned_documents"].append(fname)

    logger.info(f"[Parser] Extracted lexical slots: {slots}")
    return slots


# ---------------------------------------------------------------------
# Fallback keyword extraction
# ---------------------------------------------------------------------
def extract_keywords(hits: List[Dict[str, Any]]) -> List[str]:
    """
    Produces a simple list of frequent tokens from hit content.
    Not semantic — just lexical fallback.
    """
    words = []
    for h in hits:
        src = h.get("_source", {})
        content = _safe(src.get("content", "")).lower()

        for w in content.split():
            if 3 <= len(w) <= 25:
                words.append(w)

    top = list(dict.fromkeys(words))[:30]
    logger.info(f"[Parser] Extracted fallback keywords: {top}")
    return top


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def parse_slots_from_hits(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Public API returning a structured dict consumed by GOAP.
    """

    if not hits:
        logger.warning("[Parser] No hits received.")
        return {
            "anchors": {},
            "slots": {},
            "keywords": [],
        }

    anchors = extract_dynamic_anchors(hits)
    slots = extract_slots(hits)
    keywords = extract_keywords(hits)

    parsed = {
        "anchors": anchors,
        "slots": slots,
        "keywords": keywords,
    }

    logger.info(f"[Parser] Final parse: keys={list(parsed.keys())}")
    return parsed
