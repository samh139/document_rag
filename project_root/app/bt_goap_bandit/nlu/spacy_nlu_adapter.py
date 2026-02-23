# ==========================================================
# 🔹 slm_nlu_adapter.py — Gemini 2.5-Flash-powered NLU Adapter (manifest-aware)
# ==========================================================

import os
import re
import json
import logging
from typing import Dict, Any, List, Union
from bt_goap_bandit.llm_config import fire_fast_modal_request_chat_get_dict

logger = logging.getLogger(__name__)


# ==========================================================
# 🔹 Safe JSON Parsing & Multi-Layer Unwrap
# ==========================================================
def _safe_json_parse(text: Union[str, dict], fallback: Dict = None) -> Dict:
    if fallback is None:
        fallback = {"intent": "generic_query", "confidence": 0.5, "entities": []}

    if not text:
        return fallback

    if isinstance(text, dict):
        return text

    cleaned = re.sub(r"```(?:json)?|```", "", str(text)).strip()
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    candidate = match.group(1) if match else cleaned

    def unwrap(obj):
        depth = 0
        while isinstance(obj, str) and obj.strip().startswith(("{", "[")) and depth < 6:
            try:
                obj = json.loads(obj)
            except Exception:
                break
            depth += 1
        return obj

    # try normal json
    try:
        parsed = json.loads(candidate)
        return unwrap(parsed)
    except Exception:
        fixed = re.sub(r"'", '"', candidate)
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        try:
            parsed = json.loads(fixed)
            return unwrap(parsed)
        except Exception:
            logger.warning("[Gemini-NLU] Fallback JSON parsing failed.")
            return fallback


# ==========================================================
# 🔹 Load Manifest (Flexible Schema)
# ==========================================================
def _load_manifest() -> Dict[str, Any]:
    """
    Load meta_intent_manifest.json:
    - Supports list or dict format
    - Always returns {intent_name: meta_dict}
    """
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "..", "config", "meta_intent_manifest.json"),
        os.path.join(os.path.dirname(__file__), "..", "bt", "tree", "meta_intent_manifest.json"),
    ]

    for path in candidate_paths:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            continue

        try:
            with open(abs_path, "r") as f:
                raw = json.load(f)

            # case A: dict manifest
            if isinstance(raw, dict):
                logger.info(f"[Gemini-NLU] Loaded {len(raw)} intents (dict) from {abs_path}")
                return raw

            # case B: list manifest → convert to dict
            if isinstance(raw, list):
                manifest = {}
                for item in raw:
                    name = (
                        item.get("intent")
                        or item.get("name")
                        or item.get("id")
                        or None
                    )
                    if name:
                        manifest[name] = item
                logger.info(f"[Gemini-NLU] Loaded {len(manifest)} intents (list→dict) from {abs_path}")
                return manifest

        except Exception as e:
            logger.error(f"[Gemini-NLU] Failed to load manifest at {path}: {e}")

    logger.error("[Gemini-NLU] ❌ meta_intent_manifest.json missing or invalid.")
    return {}


# ==========================================================
# 🔹 Gemini NLU Adapter
# ==========================================================
class SLMNLUAdapter:
    def __init__(self, model="gemini-2.5-flash"):
        self.model = model
        self.manifest = _load_manifest()
        self.valid_intents = list(self.manifest.keys())
        logger.info(f"[Gemini-NLU] Using {model} | Valid intents: {len(self.valid_intents)}")

    # ------------------------------------------------------
    def _build_manifest_prompt(self):
        prompt = [
            "You are an enterprise-grade intent classifier.",
            "Choose exactly ONE intent from this list:",
            json.dumps(self.valid_intents, indent=2),
            "",
            "Return JSON only:",
            '{"intent":"<intent>","confidence":0.0,"entities":[]}',
            "",
            "Examples for context:",
        ]

        for intent, meta in self.manifest.items():
            ex = meta.get("examples", [])
            prompt.append(f"- {intent}: examples={ex}")

        return "\n".join(prompt)

    # ------------------------------------------------------
    def analyze(self, text: str):
        logger.info(f"[Gemini-NLU] Analyzing: {text}")

        sys_prompt = self._build_manifest_prompt()
        try:
            raw = fire_fast_modal_request_chat_get_dict(sys_prompt, text)
        except Exception as e:
            logger.error(f"[Gemini-NLU] Request failed: {e}")
            return {"intent": "generic_query", "confidence": 0.3, "entities": []}

        parsed = _safe_json_parse(raw)

        intent = parsed.get("intent", "generic_query")
        if intent not in self.valid_intents:
            intent = "generic_query"

        return {
            "intent": intent,
            "confidence": float(parsed.get("confidence", 0.7)),
            "entities": parsed.get("entities", []),
        }

    # ------------------------------------------------------
    def decompose(self, text: str):
        logger.info(f"[Gemini-NLU] Decomposing: {text}")

        manifest_hint = json.dumps(self.valid_intents, indent=2)

        sys_prompt = (
            "Break user message into atomic intent subplans.\n"
            "Valid intents:\n" + manifest_hint +
            "\nReturn JSON only:\n"
            '{"subplans":[{"text":"...","intent":"...","entities":[]}]}'
        )

        try:
            raw = fire_fast_modal_request_chat_get_dict(sys_prompt, text)
        except Exception as e:
            logger.error(f"[Gemini-NLU] Decomposition failed: {e}")
            return {"subplans": []}

        parsed = _safe_json_parse(raw, fallback={"subplans": []})

        if "subplans" not in parsed:
            parsed = {"subplans": [parsed]}

        # validate intents
        for sp in parsed["subplans"]:
            if sp.get("intent") not in self.valid_intents:
                sp["intent"] = "generic_query"

        return parsed

    # ------------------------------------------------------
    def parse(self, text: str):
        return self.analyze(text)
