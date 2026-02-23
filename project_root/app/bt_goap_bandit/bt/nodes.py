"""
bt/nodes.py — Unified Behaviour Nodes (v7.3 Ready)
Restored + clean integration of ClarifyNode, PublishNode, Diagnostics.
"""

import logging
import py_trees as pt
from datetime import datetime

logger = logging.getLogger(__name__)

# ==========================================================
# Blackboard
# ==========================================================
class BB:
    """
    Simple shared blackboard object.
    Holds attributes shared across tree nodes and subplans.
    """
    pass


# ==========================================================
# Node: PreprocessNode
# ==========================================================
class PreprocessNode(pt.behaviour.Behaviour):
    """
    Normalizes user input and prepares text for downstream nodes.
    """

    def __init__(self, user_input, bb):
        super().__init__("PreprocessNode")
        self.user_input = user_input
        self.bb = bb

    def update(self):
        text = ""
        if isinstance(self.user_input, str):
            text = self.user_input
        elif isinstance(self.user_input, dict):
            text = self.user_input.get("text", "")

        self.bb.text = text.strip()
        logger.info(f"[PreprocessNode] Input normalized → '{self.bb.text}'")
        return pt.common.Status.SUCCESS


# ==========================================================
# Node: SentimentNode
# ==========================================================
class SentimentNode(pt.behaviour.Behaviour):
    """
    Logs and persists sentiment information.
    """

    def __init__(self, bb):
        super().__init__("SentimentNode")
        self.bb = bb

    def update(self):
        val = getattr(self.bb, "sentiment_hint", "neutral")
        self.bb.sentiment_hint = val
        logger.info(f"[SentimentNode] Sentiment hint → {val} (bypass)")
        return pt.common.Status.SUCCESS


# ==========================================================
# Node: EntitiesNode
# ==========================================================
class EntitiesNode(pt.behaviour.Behaviour):
    """
    Extracts or logs recognized entities.
    """

    def __init__(self, bb):
        super().__init__("EntitiesNode")
        self.bb = bb

    def update(self):
        entities = getattr(self.bb, "entities", [])
        self.bb.entities = entities
        logger.info(f"[EntitiesNode] Entities → {entities}")
        return pt.common.Status.SUCCESS


# ==========================================================
# Node: ResolveIntentNode
# ==========================================================
class ResolveIntentNode(pt.behaviour.Behaviour):
    """
    Resolves or validates the detected intent before planning.
    """

    def __init__(self, bb):
        super().__init__("ResolveIntentNode")
        self.bb = bb

    def update(self):
        intent = getattr(self.bb, "meta_intent", None)
        if not intent:
            logger.warning("[ResolveIntentNode] No intent found — cannot plan.")
            return pt.common.Status.FAILURE

        logger.info(f"[ResolveIntentNode] Intent → {intent}")
        self.bb.intent_resolved = True
        return pt.common.Status.SUCCESS


# ==========================================================
# Node: ClarifyNode (Unified)
# ==========================================================
class ClarifyNode(pt.behaviour.Behaviour):
    """
    Requests clarification when required slots are missing OR confidence is low.
    Designed for GOAP-compatible planning output.
    """

    def __init__(self, bb):
        super().__init__("ClarifyNode")
        self.bb = bb

    def initialise(self):
        logger.info("[ClarifyNode] Initialising clarification sequence...")

    def update(self):
        meta_intent = getattr(self.bb, "meta_intent", "unknown")
        confidence = getattr(self.bb, "meta_intent_confidence", None)
        missing = getattr(self.bb, "missing_slots", [])

        plan = {
            "plan_id": f"goap-clarify-{datetime.utcnow().timestamp():.0f}",
            "meta_intent": meta_intent,
            "node_type": "clarify",
            "steps": ["detect_gap", "prompt_user", "capture_response"],
            "expected_outcome": "clarified_intent",
            "confidence": confidence,
            "missing_slots": missing,
            "engagement_hint": "interactive",
            "message": self._make_prompt(meta_intent, missing, confidence)
        }

        # Store signal for PublishNode
        self.bb.plan = plan
        self.bb.signal = {"emit": True, "directive": plan}

        logger.info(f"[ClarifyNode] Built clarify plan for {meta_intent}")
        return pt.common.Status.SUCCESS

    @staticmethod
    def _make_prompt(meta_intent, missing, confidence):
        if missing:
            return f"I’m not sure about your {', '.join(missing)}. Could you clarify?"

        if confidence is not None and confidence < 0.6:
            return (
                f"I think you may want to '{meta_intent}', "
                "but I’m not fully certain. Could you confirm?"
            )

        return "Could you provide more details so I can proceed accurately?"


# ==========================================================
# Node: PublishNode
# ==========================================================
class PublishNode(pt.behaviour.Behaviour):
    """
    Final node — emits the behaviour tree’s output signal.
    """

    def __init__(self, bb):
        super().__init__("PublishNode")
        self.bb = bb

    def update(self):
        sig = getattr(self.bb, "signal", None)
        if sig:
            logger.info(f"[PublishNode] Output → {sig}")
        else:
            logger.info("[PublishNode] No signal emitted.")
        return pt.common.Status.SUCCESS


# ==========================================================
# Node: DiagnosticNode (Optional)
# ==========================================================
class DiagnosticNode(pt.behaviour.Behaviour):
    """
    Logs blackboard state for debugging.
    """

    def __init__(self, bb):
        super().__init__("DiagnosticNode")
        self.bb = bb

    def update(self):
        state = {k: v for k, v in vars(self.bb).items() if not k.startswith("_")}
        logger.info(f"[DiagnosticNode] Blackboard snapshot:\n{state}")
        return pt.common.Status.SUCCESS


# ==========================================================
# END OF FILE
# ==========================================================
