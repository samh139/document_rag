# bt_goap_bandit/slot_resolver/hybrid_resolver.py
# =====================================================================
# HYBRID SLOT RESOLVER — Option D (Gemini + ES Evidence)
# =====================================================================
# Uses:
#   • Gemini → semantic classification of missing slot
#   • Elasticsearch → evidence verification
#   • SlotLex → existing chunk pipeline
#
# Zero hallucination:
#   Candidate values must be validated against ES retrieval evidence.
# =====================================================================

from __future__ import annotations
import json
import numpy as np
import aiohttp
from typing import Dict, List, Any


# =====================================================================
# 1. Gemini slot classifier — RETURNS CANDIDATE VALUES (NO GENERATION)
# =====================================================================

async def call_gemini_slot_classifier(
        query: str,
        missing_slot: str,
        entities: List[str],
        es_chunks: List[str]) -> Dict[str, Any]:

    """
    Calls Gemini to get candidate slot values.
    Strict classification — NOT generative.
    """

    prompt = f"""
You are a deterministic slot classifier.

User Query:
{query}

Missing Slot:
{missing_slot}

Entities:
{entities}

Evidence Chunks (the model must use these as grounding):
{es_chunks}

RULES:
1. Return ONLY JSON.
2. Only return normalized slot-value candidates.
3. Do NOT invent values not supported by evidence.
4. Candidates must be short verb/noun phrases.
5. Answer format:
{{
  "candidates": ["value1", "value2", ...]
}}
"""

    # -----------------------------
    # Gemini API CALL
    # -----------------------------
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    api_key = "<YOUR_GEMINI_API_KEY>"  # integrate later

    body = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "topP": 0.8,
            "maxOutputTokens": 128
        }
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params={"key": api_key}, json=body) as resp:
                data = await resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
    except Exception as e:
        print(f"[HybridResolver] Gemini error: {e}")
        return {"candidates": []}


# =====================================================================
# 2. ES evidence scoring — grounding classifier output to real data
# =====================================================================

def score_candidate_against_chunks(candidate: str, es_results: Dict[str, Any]) -> float:
    """
    Score a candidate slot value based on:
        • lexical similarity to ES chunk text
        • ES RRF/lex scores passed from SlotLex
        • simple embedding-free signal (fast)
    """

    chunks = es_results.get("top_chunks", [])
    if not chunks:
        return 0.0

    # Convert chunk text (already returned by SlotLex summary)
    text_blobs = [chunk for chunk in chunks]

    # Count substring presence across chunks
    count = sum(candidate.lower() in blob.lower() for blob in text_blobs)

    if count == 0:
        return 0.0

    # Normalize score
    return float(count) / len(text_blobs)


# =====================================================================
# 3. END-TO-END HYBRID RESOLUTION
# =====================================================================

async def resolve_missing_slot(
        query: str,
        entities: List[str],
        missing_slot: str,
        es_results: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Hybrid Slot Resolver:
        1. Gemini → classification candidates
        2. ES → evidence validation
        3. Select top validated candidate
    """

    # ------------------------------------------------------
    # Step 1: Run Gemini classification
    # ------------------------------------------------------
    gemini_resp = await call_gemini_slot_classifier(
        query=query,
        missing_slot=missing_slot,
        entities=entities,
        es_chunks=[c for c in es_results.get("top_chunks", [])]
    )

    candidates = gemini_resp.get("candidates", [])
    if not candidates:
        return None

    # ------------------------------------------------------
    # Step 2: Score against ES evidence
    # ------------------------------------------------------
    validated = []
    for c in candidates:
        score = score_candidate_against_chunks(c, es_results)
        if score > 0.05:
            validated.append((c, score))

    if not validated:
        return None

    # ------------------------------------------------------
    # Step 3: Pick highest-evidence candidate
    # ------------------------------------------------------
    validated.sort(key=lambda x: x[1], reverse=True)
    best_value, evidence = validated[0]

    return {
        "slot": missing_slot,
        "value": best_value,
        "confidence": float(evidence),
        "source": ["gemini", "es"]
    }
