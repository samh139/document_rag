import json
import os

class IntentRouter:
    def __init__(self, manifest_path=None):
        # auto-locate manifest
        base = os.path.dirname(os.path.dirname(__file__))
        default_path = os.path.join(base, "config", "meta_intent_manifest.json")
        self.manifest_path = manifest_path or default_path

        self.manifest = self._load_manifest()
        self.intent_index = self._build_intent_index()
        self.bandit_map = self._build_behavior_to_bandit_map()

    # -----------------------------------------------------
    # Load Manifest
    # -----------------------------------------------------
    def _load_manifest(self):
        try:
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Router] ERROR loading manifest: {e}")
            return {}

    # -----------------------------------------------------
    # Build lookup tables for fast access
    # -----------------------------------------------------
    def _build_intent_index(self):
        index = {}
        for intent_key, meta in self.manifest.items():
            index[intent_key] = {
                "node_type": meta.get("node_type"),
                "intent_family": meta.get("intent_family"),
                "behavioral_family": meta.get("behavioral_family"),
                "default_scope": meta.get("default_scope"),
            }
        return index

    # -----------------------------------------------------
    # Behavioral → Bandit Arm (Option A Mapping)
    # -----------------------------------------------------
    def _build_behavior_to_bandit_map(self):
        return {
            "data_probe": "probe_missing",
            "analytical_reasoning": "open",
            "task_execution": "direct_execute",
            "clarification_request": "clarify_scope",
            "neutral_smalltalk": "open",
        }

    # -----------------------------------------------------
    # Public Routing API
    # -----------------------------------------------------
    def route(self, slm_output):
        """
        slm_output example:
        {
            "intent_key": "lookup_knowledge",
            "intent_family": "...",
            "slots": {...},
            "confidence": 0.92
        }
        """
        if not slm_output or "intent_key" not in slm_output:
            return self._fallback_route(slm_output)

        intent_key = slm_output["intent_key"]
        if intent_key not in self.intent_index:
            return self._fallback_route(slm_output)

        meta = self.intent_index[intent_key]
        behavioral_family = meta["behavioral_family"]
        bandit_arm = self.bandit_map.get(behavioral_family, "open")  # safe fallback

        return {
            "intent_key": intent_key,
            "node_type": meta["node_type"],
            "intent_family": meta["intent_family"],
            "behavioral_family": behavioral_family,
            "bandit_arm": bandit_arm,
            "default_scope": meta["default_scope"],
            "slots": slm_output.get("slots", {}),
            "confidence": slm_output.get("confidence", 0.0)
        }

    # -----------------------------------------------------
    # Fallback for unknown intent
    # -----------------------------------------------------
    def _fallback_route(self, slm_output):
        return {
            "intent_key": "generic_query",
            "node_type": "generic",
            "intent_family": "generic_query",
            "behavioral_family": "clarification_request",
            "bandit_arm": "clarify_scope",
            "default_scope": "contextual",
            "slots": slm_output.get("slots", {}) if slm_output else {},
            "confidence": slm_output.get("confidence", 0.0) if slm_output else 0.0
        }
