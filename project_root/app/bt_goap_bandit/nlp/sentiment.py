# ==========================================================
# 🔹 sentiment.py — Pure Hugging Face Sentiment Analyzer
# ==========================================================
"""
Self-contained sentiment analysis module.
✅ Only uses Hugging Face Transformers.
🚀 No VADER, no fallback — consistent GPU/CPU inference.
"""

import logging
import threading
import os
from typing import Dict, Any
from model_classes.intent_agent_response import Sentiment,RawSentiment,Scores
logger = logging.getLogger(__name__)

_analyzer_lock = threading.Lock()
_analyzer_instance = None
_analyzer_mode = "hf"


# ----------------------------------------------------------
# Loader: Hugging Face Transformers
# ----------------------------------------------------------
def _load_hf_pipeline(model_name: str = None):
    """Load Hugging Face pipeline for sentiment analysis."""
    try:
        from transformers import pipeline

        model = model_name or os.getenv(
            "SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
        )
        device = 0 if os.getenv("USE_GPU", "1") in ("1", "true", "yes") else -1
        pipe = pipeline("sentiment-analysis", model=model, device=device)
        logger.info(f"[Sentiment] ✅ Using Hugging Face model: {model} (device={device})")
        return pipe
    except Exception as e:
        logger.error(f"[Sentiment] ❌ Failed to load Hugging Face model: {e}")
        raise RuntimeError("Sentiment model could not be initialized.")


# ----------------------------------------------------------
# Lazy singleton
# ----------------------------------------------------------
def _get_analyzer():
    """Thread-safe singleton loader for the HF sentiment model."""
    global _analyzer_instance
    if _analyzer_instance is None:
        with _analyzer_lock:
            if _analyzer_instance is None:
                _analyzer_instance = _load_hf_pipeline()
    return _analyzer_instance


# ----------------------------------------------------------
# Core Function
# ----------------------------------------------------------
def analyze_sentiment(text: str) ->Sentiment:
    """
    Returns normalized sentiment result:
    {
        "valence": "positive" | "neutral" | "negative",
        "intensity": float,
        "raw": { "mode": "hf", "scores": { ... } }
    }
    """
    if not text:
        return {"valence": "neutral", "intensity": 0.0, "raw": {"mode": "hf", "scores": {}}}

    analyzer = _get_analyzer()
    try:
        res = analyzer(text, truncation=True)[0]
        label = res["label"].lower()
        score = float(res.get("score", 0.0))
        intensity=float(res.get("intensity",0.0))
        if "pos" in label:
            valence = "positive"
        elif "neg" in label:
            valence = "negative"
        else:
            valence = "neutral"
        return Sentiment(valence= valence,
            intensity= intensity,
            raw= RawSentiment(mode= "hf", scores= Scores(label=label,score=score))
        )
    except Exception as e:
        logger.error(f"[Sentiment] ⚠️ HF inference failed: {e}")
        return Sentiment (valence= "neutral",intensity= 0.0, raw= RawSentiment(mode= "hf", scores= Scores(label="neutral",score=0)))


# ----------------------------------------------------------
# Export analyzer state
# ----------------------------------------------------------
__all__ = ["analyze_sentiment", "_analyzer_mode"]


# ----------------------------------------------------------
# Quick CLI Test
# ----------------------------------------------------------
if __name__ == "__main__":
    tests = [
        "I love working with this team!",
        "No, that didn’t help at all.",
        "I'm not sure if this will work.",
        "That’s terrible service.",
        "Okay, got it.",
    ]
    analyzer = _get_analyzer()
    print(f"\n[Sentiment] 🔍 Active Provider: {_analyzer_mode}")
    for t in tests:
        r = analyze_sentiment(t)
        print(f"{t:50} → {r['valence']:<8} ({r['intensity']:.2f}) [{r['raw']['mode']}]")
