"""
bt/nodes_clarify.py — Real Clarify Node
Builds a GOAP-style clarification plan and publishes a structured signal.
"""

import logging
import py_trees as pt
from datetime import datetime

logger = logging.getLogger(__name__)

class ClarifyNode(pt.behaviour.Behaviour):
    """
    Generates a clarification plan when meta_intent confidence is low
    or when Slot-Lex detects ambiguity.
    """

    def __init__(self, bb, name: str = "ClarifyNode"):
        super().__init__(name)
        self.bb = bb

    def initialise(self):
        logger.info("[ClarifyNode] Initialising clarification sequence...")

    def update(self):
        meta_intent = getattr(self.bb, "meta_intent", "unknown")
        confidence = getattr(self.bb, "meta_intent_confidence", None)
        missing_slots = getattr(self.bb, "missing_slots", [])

        # Construct a GOAP-compatible plan
        plan = {
            "plan_id": f"goap-clarify-{datetime.utcnow().timestamp():.0f}",
            "meta_intent": meta_intent,
            "node_type": "clarify",
            "steps": ["detect_gap", "prompt_user", "capture_response"],
            "expected_outcome": "clarified_intent",
            "confidence": confidence,
            "missing_slots": missing_slots,
            "engagement_hint": "interactive",
            "message": self._make_prompt(meta_intent, missing_slots, confidence)
        }

        # Store on blackboard so PublishNode can emit it
        self.bb.plan = plan
        self.bb.signal = {
            "emit": True,
            "directive": plan
        }

        logger.info(f"[ClarifyNode] Built clarify plan for {meta_intent}")
        return pt.common.Status.SUCCESS

    # -------------------------------------------------------------------------
    # Internal helper
    # -------------------------------------------------------------------------
    def _make_prompt(self, meta_intent: str, missing_slots, confidence) -> str:
        if missing_slots:
            slot_str = ", ".join(missing_slots)
            return f"I’m not sure about your {slot_str}. Could you clarify?"
        if confidence is not None and confidence < 0.6:
            return (
                f"I think you may want to '{meta_intent}', "
                "but I’m not fully certain. Could you confirm what you’d like me to do?"
            )
        return "Could you provide a bit more detail so I can assist accurately?"
