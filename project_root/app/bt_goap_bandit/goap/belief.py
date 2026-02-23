"""
goap/belief.py — v9.2 (Corpus-Aware, No Heuristics)
-----------------------------------------------------------------
Builds belief states directly from Elasticsearch corpus evidence.
✓ Merges file_metadata + chunk_metadata
✓ Splits comma-separated regions
✓ No regex or rule-based buckets
✓ Anchors are purely data-driven from corpus frequency
-----------------------------------------------------------------
"""

from __future__ import annotations
import logging, numpy as np
from typing import List, Dict, Any
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

ES_HOST = "http://10.3.0.5:9200"
ES_INDEX = "chunks_es"
MODEL_PATH = "all-MiniLM-L6-v2"
TOP_N = 10
CONF_THRESHOLD = 0.55

_model = None
_cached_anchors: Dict[str, List[tuple[str, str]]] | None = None


# ----------------------------------------------------------
# 🧠 Model loader
# ----------------------------------------------------------
def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_PATH)
    return _model


# ----------------------------------------------------------
# ⚙️ ES utilities
# ----------------------------------------------------------
def _connect_es() -> Elasticsearch | None:
    try:
        es = Elasticsearch(ES_HOST, request_timeout=15)
        if not es.ping():
            raise ConnectionError("Elasticsearch not reachable")
        return es
    except Exception as e:
        logger.warning(f"[Belief] ES connection failed: {e}")
        return None


def _fetch_terms(es: Elasticsearch, field: str, size: int = 1000) -> List[str]:
    """Return most frequent string values for given field."""
    try:
        body = {"size": 0, "aggs": {"vals": {"terms": {"field": f"{field}.keyword", "size": size}}}}
        res = es.search(index=ES_INDEX, body=body)
        return [b["key"] for b in res["aggregations"]["vals"]["buckets"]]
    except Exception:
        return []


# ----------------------------------------------------------
# 🧭 Corpus-driven anchor discovery
# ----------------------------------------------------------
def _discover_anchors() -> Dict[str, List[tuple[str, str]]]:
    es = _connect_es()
    if not es:
        return {
            "scope": [("general", "default scope")],
            "jurisdiction": [("global", "default jurisdiction")],
            "benefit_type": [("unspecified", "default benefit")],
        }

    titles = _fetch_terms(es, "file_metadata.title")
    tags1 = _fetch_terms(es, "file_metadata.tags")
    tags2 = _fetch_terms(es, "chunk_metadata.tags")
    regions_raw = _fetch_terms(es, "file_metadata.region")

    # --- Split comma-separated regions ---
    regions = []
    for r in regions_raw:
        parts = [p.strip() for p in r.split(",") if p.strip()]
        regions.extend(parts)
    regions = list(dict.fromkeys(regions))[:TOP_N] or ["Global"]

    # --- Build frequency-based tag pool ---
    tagpool = list(dict.fromkeys(titles + tags1 + tags2))
    counts = Counter(tagpool)
    most_common_tags = [t for t, _ in counts.most_common(TOP_N)]

    # --- Assign anchors purely from corpus data ---
    scope_anchors = [(t, f"scope anchor: {t}") for t in most_common_tags]
    jurisdiction_anchors = [(r, f"region anchor: {r}") for r in regions]
    benefit_anchors = [(t, f"benefit anchor: {t}") for t in most_common_tags]

    anchors = {
        "scope": scope_anchors,
        "jurisdiction": jurisdiction_anchors,
        "benefit_type": benefit_anchors,
    }

    logger.info(
        f"[Belief] 🔍 Anchors discovered — "
        f"{len(scope_anchors)} scope, {len(jurisdiction_anchors)} region, {len(benefit_anchors)} benefit."
    )
    return anchors


def _get_anchors() -> Dict[str, List[tuple[str, str]]]:
    global _cached_anchors
    if _cached_anchors is None:
        _cached_anchors = _discover_anchors()
    return _cached_anchors


# ----------------------------------------------------------
# 🔢 Similarity scoring
# ----------------------------------------------------------
def _compute_similarity_scores(hits: List[Dict[str, Any]], anchors: List[tuple]) -> Dict[str, float]:
    model = _get_model()
    docs = []
    for h in hits:
        fm = h.get("file_metadata", {})
        cm = h.get("chunk_metadata", {})
        docs.append(
            " ".join(
                [
                    h.get("content", ""),
                    str(fm.get("title", "")),
                    " ".join(fm.get("tags", []) or []),
                    " ".join(cm.get("tags", []) or []),
                    str(fm.get("region", "")),
                ]
            )
        )
    if not docs:
        return {label: 0.0 for label, _ in anchors}

    doc_embs = model.encode(docs, normalize_embeddings=True)
    scores = {}
    for label, desc in anchors:
        anchor_emb = model.encode(desc, normalize_embeddings=True)
        sim = util.cos_sim(anchor_emb, doc_embs).cpu().numpy()
        scores[label] = float(np.mean(sim))
    return scores


def _normalize(scores: Dict[str, float]) -> tuple[str | None, float]:
    if not scores:
        return None, 0.0
    label = max(scores, key=scores.get)
    val = float(np.clip((scores[label] + 1) / 2, 0, 1))
    return label, round(val, 2)


# ----------------------------------------------------------
# 🧩 Belief builder
# ----------------------------------------------------------
def build_belief_state(hits: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not hits:
        logger.warning("[Belief] No hits — returning empty belief.")
        return {
            "scope": {"value": None, "p": 0.0},
            "jurisdiction": {"value": None, "p": 0.0},
            "benefit_type": {"value": None, "p": 0.0},
        }

    anchors = _get_anchors()
    belief: Dict[str, Dict[str, Any]] = {}
    for slot, group in anchors.items():
        scores = _compute_similarity_scores(hits, group)
        label, prob = _normalize(scores)
        if prob < CONF_THRESHOLD:
            label, prob = None, 0.0
        belief[slot] = {"value": label, "p": prob}
        logger.debug(f"[Belief] {slot}={label} ({prob})")

    logger.info(f"[Belief] ✅ Built corpus-aware belief: {belief}")
    return belief
