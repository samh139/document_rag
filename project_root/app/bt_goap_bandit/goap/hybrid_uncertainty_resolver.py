import logging
from typing import Dict, Any, List

from model_classes.intent_agent_response import Enriched

log = logging.getLogger(__name__)


# ============================================================
#  HYBRID RESOLVER (Drift-Proof, Schema-Safe)
# ============================================================
def infer_missing_slots_hybrid(
    tool: str,
    missing: List[str],
    text: str,
    entities: List[Any],           # kept for compatibility, but not used directly
    slotlex_output: Enriched,
    semantic_tags: List[str] = None
) -> Dict[str, Any]:
    """
    Hybrid uncertainty resolver:
    - NO keywords
    - NO intent guessing
    - Uses:
        • entities (if structured)
        • SlotLex retrieved contexts as LLM grounding
        • Gemini/LLM as fallback
    """

    # ============================================================
    # STEP 1 — ES-Based Hints from entities (optional)
    # ============================================================
    es_suggestions: Dict[str, str] = {}

    for param in missing:
        for ent in (slotlex_output.entities or []):
            # We don't know the exact schema → be defensive
            if not isinstance(ent, dict):
                continue

            # Try common patterns: slot_name / type / label
            slot_name = (
                ent.get("slot_name")
                or ent.get("type")
                or ent.get("label")
            )

            # Try common value carriers: value / text
            value = ent.get("value") or ent.get("text")

            if slot_name and slot_name.lower() == param.lower():
                if isinstance(value, str) and value.strip():
                    es_suggestions[param] = value.strip()
                    break

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"[Hybrid] ES suggestions = {es_suggestions}")

    # ============================================================
    # STEP 2 — Gemini fallback (LLM with enriched context)
    # ============================================================
    gemini_suggestions: Dict[str, Any] = {}

    try:
        from bt_goap_bandit.goap.slot_llm import infer_slots_with_llm

        # Build enriched prompt: user text + retrieved ES contexts
        llm_context = (
            f"User Input:\n{text}\n\n"
            f"Retrieved Contexts:\n"
            + "\n\n".join(slotlex_output.retrieved_contexts or [])
        )

        gemini_suggestions = infer_slots_with_llm(
            intent=tool,
            user_text=llm_context,
            slotlex_context=slotlex_output,
            semantic_topic=tool,
        ) or {}

    except Exception as e:
        log.error(f"[Hybrid] Gemini error: {e}", exc_info=True)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"[Hybrid] Gemini suggestions = {gemini_suggestions}")

    # ============================================================
    # STEP 3 — MERGE: ES first, then Gemini
    # ============================================================
    final_filled: Dict[str, Any] = {}
    still_missing: List[str] = []

    for param in missing:
        if param in es_suggestions:
            final_filled[param] = es_suggestions[param]
        elif param in gemini_suggestions:
            final_filled[param] = gemini_suggestions[param]
        else:
            still_missing.append(param)

    
    # ============================================================
    # STEP 4 — CONFIDENCE
    # ============================================================
    confidence = 1.0 if not still_missing else 0.4

    return {
        "filled": final_filled,
        "missing": still_missing,
        "confidence": confidence,
    }
