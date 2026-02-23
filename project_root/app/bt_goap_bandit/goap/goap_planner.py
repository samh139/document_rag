import json
from typing import Dict, Any

from bt_goap_bandit.goap.semantic_topic import detect_semantic_topic
from bt_goap_bandit.goap.hybrid_uncertainty_resolver import infer_missing_slots_hybrid
from bt_goap_bandit.goap.task_infer import infer_task_from_semantics
from bt_goap_bandit.slotlex import _slotlex
from model_classes.intent_agent_response import GoapPlan,GoapReasoning,Enriched
from app_logger import get_app_logger
log = get_app_logger(__name__)


# ============================================================
# SLOTLEX READINESS SIGNAL (unchanged)
# ============================================================
def compute_slotlex_readiness(text: str) -> Dict[str, float]:
    """
    Computes confidence based purely on SlotLex retrieval,
    used for Bandit → BT control.
    """
    enriched = _slotlex.enrich(text)

    num_chunks = enriched.summary.num_chunks
    scores = enriched.summary.scores

    if num_chunks == 0 or not scores:
        return {
            "missing_ratio": 1.0,
            "readiness_score": 0.0,
        }

    sc_vals = list(scores)
    max_sc, min_sc = max(sc_vals), min(sc_vals)
    rng = (max_sc - min_sc) or 1.0

    normalized = [(v - min_sc) / rng for v in sc_vals]
    readiness = sum(normalized) / len(normalized)

    return {
        "missing_ratio": round(1.0 - readiness, 4),
        "readiness_score": round(readiness, 4),
    }


# ============================================================
#               GOAP PLANNER v3 — CLEAN, SEMANTIC ONLY
# ============================================================
def build_goap_plan(intent: str, slotlex_output:Enriched) -> GoapPlan:
    """
    GOAP v3 — No keywords, No hardcoding.
    Uses:
      - intent (from SLM/Gemini)
      - semantic_topic() deterministic mapping
      - task_infer.py semantic task inference
      - SlotLex for semantic tags + grounding
      - HybridResolver only if all else fails
    """

    text = slotlex_output.text
    entities = slotlex_output.entities
    slots = {}
    semantic_tags = slotlex_output.semantic_tags

    # ============================================================
    # STEP 1 — Deterministic semantic topic (tool + required params)
    # ============================================================
    try:
        tool_name, required_params = detect_semantic_topic(
            intent=intent,
            text=text,
            entities=entities,
            slotlex_output=slotlex_output
        )
    except Exception as e:
        log.error(f"[GOAP] semantic_topic() failed: {e}", exc_info=True)
        return _fallback_no_tool(intent)

    if tool_name is None:
        return _fallback_no_tool(intent)

    # ============================================================
    # STEP 2 — Extract safe query text
    # ============================================================
    if isinstance(slots.get("query"), str) and slots["query"].strip():
        query_text = slots["query"].strip()
    else:
        query_text = text.strip()

    # ============================================================
    # STEP 3 — SEMANTIC PARAMETER INFERENCE
    # No keywords. No hardcoded names.
    # ============================================================
    filled_params = {}
    still_missing = []
    confidence = 1.0

    if tool_name == "task_executor":

        # 3A — If user explicitly gave a task → accept
        if slots.get("task"):
            filled_params["task"] = slots["task"]
            still_missing = []
            confidence = 1.0

        else:
            # 3B — SEMANTIC TASK INFERENCE (PRIMARY)
            task_guess = infer_task_from_semantics(
                user_text=text,
                semantic_tags=semantic_tags,
                slotlex_context=slotlex_output
            )

            task_value = task_guess.get("task")
            task_conf = float(task_guess.get("confidence", 0))

            if task_value:
                filled_params["task"] = task_value
                still_missing = []
                confidence = task_conf

            else:
                # 3C — Hybrid fallback (ES hints + LLM)
                hybrid_enriched = infer_missing_slots_hybrid(
                    tool=tool_name,
                    missing=["task"],
                    text=text,
                    entities=entities,
                    slotlex_output=slotlex_output,
                    semantic_tags=semantic_tags
                )
                filled_params = hybrid_enriched.get("filled", {})
                still_missing = hybrid_enriched.get("missing", [])
                confidence = hybrid_enriched.get("confidence", 0.4)

    else:
        # Non-task intents → only need query
        filled_params = {}
        still_missing = []
        confidence = 1.0

    # ============================================================
    # STEP 4 — Final Parameter Assembly
    # ============================================================
    final_params = {}

    if tool_name == "task_executor":
        if filled_params.get("task"):
            final_params["task"] = filled_params["task"]
    else:
        final_params["query"] = query_text

    # ============================================================
    # STEP 5 — SlotLex semantic readiness
    # ============================================================
    readiness = compute_slotlex_readiness(query_text)
    print(f"***readiness score {readiness}")

    # ============================================================
    # STEP 6 — FINAL GOAP PLAN STRUCTURE
    # ============================================================
    plan = GoapPlan(
        tool= tool_name,
        params= final_params,
        missing =still_missing,
       confidence= confidence,

        readiness_score= readiness["readiness_score"],
        missing_ratio=readiness["missing_ratio"],

        reasoning=GoapReasoning (
            intent=intent,
            semantic_topic= tool_name,
            semantic_tags= semantic_tags,
            entities= entities,
            required_params= required_params,
            auto_filled= filled_params
        )
    )


    return plan


# ============================================================
# SUPPORT — FALLBACK
# ============================================================
def _fallback_no_tool(intent: str) -> GoapPlan:
    return  GoapPlan(
        tool=None,
        params= {},
        missing= [],
        confidence= 0.0,
        readiness_score= 0.0,
        missing_ratio= 1.0,
    
        reasoning= GoapReasoning(
             intent=intent,
            semantic_topic=None,
            semantic_tags= [],
            entities= [],
            required_params= [],
            auto_filled= {}
        )
    )
