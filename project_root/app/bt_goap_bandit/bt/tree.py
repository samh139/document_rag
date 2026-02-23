"""
bt/tree.py — v7.3 (Clarify-aware, ES + GOAP integrated)
-------------------------------------------------------
Extends v6.5 by:
 • Auto-attaching ClarifyNode when GOAP signals blocking_slots
 • Logging clarification events
 • Maintaining backward compatibility with manifest-only routing
"""

from __future__ import annotations
import logging, os, json
from datetime import datetime
import py_trees as pt

from bt_goap_bandit.bt.nodes import (
    BB,
    PreprocessNode,
    SentimentNode,
    EntitiesNode,
    ResolveIntentNode,
    PublishNode,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Manifest loader
# ------------------------------------------------------------
MANIFEST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "meta_intent_manifest.json"
)

def load_manifest() -> dict:
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"[Router] Loaded meta_intent manifest with {len(data)} entries.")
        return data
    except Exception as e:
        logger.warning(f"[Router] Could not load manifest: {e}")
        return {}

META_INTENT_MAP = load_manifest()

# ------------------------------------------------------------
# Clarify node (stand-alone behaviour)
# ------------------------------------------------------------
class ClarifyNode(pt.behaviour.Behaviour):
    def __init__(self, bb: BB):
        super().__init__("ClarifyNode")
        self.bb = bb

    def update(self):
        meta_intent = getattr(self.bb, "meta_intent", "unknown")
        missing = getattr(self.bb, "missing_slots", [])
        conf = getattr(self.bb, "meta_intent_confidence", None)

        msg = self._build_prompt(meta_intent, missing, conf)
        plan = {
            "plan_id": f"goap-clarify-{int(datetime.now().timestamp())}",
            "meta_intent": meta_intent,
            "node_type": "clarify",
            "steps": ["detect_gap", "prompt_user", "capture_response"],
            "expected_outcome": "clarified_intent",
            "missing_slots": missing,
            "message": msg,
        }
        self.bb.subplans = [plan]
        self.bb.signal = {"emit": True, "directive": plan}
        logger.info(f"[ClarifyNode] Prompted clarification for {meta_intent}: {msg}")
        print(f"🤖 ClarifyNode: {msg}")
        return pt.common.Status.SUCCESS

    @staticmethod
    def _build_prompt(meta_intent, missing, conf):
        if missing:
            return f"I need clarification on {', '.join(missing)}."
        if conf and conf < 0.6:
            return f"I think you meant '{meta_intent}', could you confirm?"
        return "Could you provide more details so I can proceed accurately?"

# ------------------------------------------------------------
# Behaviour Tree builder
# ------------------------------------------------------------
def build_tree(user_input: str | dict, bb: BB, goap_plan: dict | None = None) -> pt.trees.BehaviourTree:
    """
    Build the runtime behaviour tree.  Optionally receives a GOAP plan
    so that blocking_slots can trigger ClarifyNode attachment.
    """
    logger.info("[BT] Building behaviour tree (v7.3)")
    root = pt.composites.Sequence(name="RootSequence", memory=False)

    # Standard pre-processing pipeline
    root.add_children([
        PreprocessNode(user_input, bb),
        SentimentNode(bb),
        EntitiesNode(bb),
        ResolveIntentNode(bb),
    ])

    # --- Router node (manifest-based) ---
    class RouteByMetaIntent(pt.behaviour.Behaviour):
        def __init__(self):
            super().__init__("RouteByMetaIntent")

        def update(self):
            meta_intent = getattr(bb, "meta_intent", None) or (
                user_input.get("meta_intent") if isinstance(user_input, dict) else "generic_query"
            )
            conf = getattr(bb, "meta_intent_confidence", None)
            meta_cfg = META_INTENT_MAP.get(meta_intent, {})
            bb.node_type = meta_cfg.get("node_type", "generic")
            bb.default_scope = meta_cfg.get("default_scope", "contextual")
            bb.required_slots = meta_cfg.get("required_slots", [])
            bb.meta_intent = meta_intent
            bb.meta_intent_confidence = conf

            plan = getattr(bb, "subplans", [{}])[0].get("plan", {})
            filled = list(plan.get("belief", {}).keys())
            bb.missing_slots = [s for s in bb.required_slots if s not in filled]

            logger.info(
                f"[Router] meta_intent={meta_intent}, node={bb.node_type}, "
                f"scope={bb.default_scope}, missing={bb.missing_slots}"
            )
            return pt.common.Status.SUCCESS

    root.add_child(RouteByMetaIntent())

    # --- ExecuteClarify: now aware of GOAP blocking slots ---
    class ExecuteClarify(pt.behaviour.Behaviour):
        def __init__(self):
            super().__init__("ExecuteClarify")

        def update(self):
            # Accept signals either from manifest inference or GOAP planner
            goap_missing = []
            if isinstance(goap_plan, dict):
                goap_missing = goap_plan.get("blocking_slots", [])
            manifest_missing = getattr(bb, "missing_slots", [])
            combined_missing = list(set(manifest_missing + goap_missing))

            if combined_missing:
                bb.missing_slots = combined_missing
                ClarifyNode(bb).update()
                logger.info(f"[ExecuteClarify] Triggered ClarifyNode for missing={combined_missing}")
            return pt.common.Status.SUCCESS

    root.add_child(ExecuteClarify())
    root.add_child(PublishNode(bb))

    tree = pt.trees.BehaviourTree(root)
    logger.info("[BT] Behaviour tree initialised successfully.")
    return tree

# ------------------------------------------------------------
# Local test
# ------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bb = BB()
    test_input = {"meta_intent": "view_process_steps", "meta_intent_confidence": 0.9}
    goap_plan = {"blocking_slots": ["process_name", "system_name"]}
    t = build_tree(test_input, bb, goap_plan)
    t.tick_once()
    print(getattr(bb, "signal", {}))
