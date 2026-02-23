# ==========================================================
# 🔹 slm_nlu_adapter.py — Gemini 2.5-Flash-powered NLU Adapter (manifest-aware)
# ==========================================================

import os
import re
import json
import logging
from typing import Dict, Any, List, Union
from  app_configs.llm_config import fire_fast_modal_request_chat_get_dict
from model_classes.intent_agent_response import SLMIdentifiedSubPlan

logger = logging.getLogger(__name__)


# ==========================================================
# 🔹 Load Manifest Dynamically (auto-fallback)
# ==========================================================
def _load_manifest() -> Dict[str, Any]:
    """Load meta_intent_manifest.json from config/ or bt/tree/ (auto-fallback)."""
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "..", "config", "meta_intent_manifest.json"),
        os.path.join(os.path.dirname(__file__), "..", "bt", "tree", "meta_intent_manifest.json"),
    ]

    for path in candidate_paths:
        try:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                with open(abs_path, "r") as f:
                    manifest = json.load(f)
                logger.info(f"[Gemini-NLU] Loaded {len(manifest)} manifest intents from {abs_path}")
                return manifest
        except Exception as e:
            logger.error(f"[Gemini-NLU] Failed to load manifest at {path}: {e}")

    logger.error("[Gemini-NLU] ❌ No valid meta_intent_manifest.json found.")
    return {}


# ==========================================================
# 🔹 Gemini-Backed Adapter (Manifest-Aware)
# ==========================================================
class SLMNLUAdapter:
    """Gemini-based NLU adapter used by runtime/cli (Manifest-driven)."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self.manifest = _load_manifest()
        self.valid_intents = list(self.manifest.keys())
        logger.info(f"[Gemini-NLU] Using {model} | Valid intents: {len(self.valid_intents)}")

    # ------------------------------------------------------
    def decompose(self, text: str) -> List[SLMIdentifiedSubPlan]:
        logger.info(f"[Gemini-NLU] 🧩 Decomposing multi-intent query via {self.model}")
        manifest_hint = json.dumps(self.valid_intents, indent=2)

        system_prompt = (
            "You decompose user text into smaller subqueries, one per intent. "
            "For each subquery, pick an intent ONLY from this list:\n"
            f"{manifest_hint}\n\n"
            "Return strictly JSON like:\n"
            '{"subplans":[{"text":"...","intent":"...","entities":["..."],"intent_confidence":<float> ,"sentiment_hint":"..."}]}'
        )
        user_prompt = f"User message:\n{text}\nRespond strictly in JSON."
        parsed={}
        models_dict = fire_fast_modal_request_chat_get_dict(system_prompt, user_prompt)
        subplans=models_dict.get("subplans")
        identitifed_subplans:List[SLMIdentifiedSubPlan]=[]
        print(f"subpalans found= {subplans}   ,length {(len(subplans))}")
        # Validate and normalize intents
        for sp in subplans:
            sp_text=sp["text"]
            intent = sp.get("intent", "generic_query")
            intent_confidence=sp.get("intent_confidence",0.5)
            subpan=SLMIdentifiedSubPlan(text=sp_text,intent=intent,entities=sp["entities"],sentiment_hint="",confidence=intent_confidence)
            identitifed_subplans.append(subpan)

        logger.info(f"[Gemini-NLU] ✅ Decomposition → {len(parsed.get('subplans', []))} subplans found")
        return identitifed_subplans

    # ------------------------------------------------------
    def parse(self, text: str) -> Dict[str, Any]:
        return self.analyze(text)
