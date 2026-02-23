# =====================================================================
# slot_llm.py — LLM Slot Reasoning Layer
# Deterministic, slot-safe, zero-drift design
# =====================================================================

import json
from app_configs.llm_config import fire_fast_modal_request_chat_get_dict
from model_classes.intent_agent_response import Enriched
from dataclasses import asdict

def infer_slots_with_llm(intent: str, user_text: str, slotlex_context: Enriched, semantic_topic=None):
    
    """
    LLM-driven slot inference.
    
    ✔ No text keywords.
    ✔ No guessing or invented slot names.
    ✔ Intent semantics ONLY.
    ✔ SlotLex context used for grounding, not slot creation.
    ✔ Returns only:
        - required_slots   → list
        - user_provided    → dict
        - confidence       → float
    """

    system = """
You are a deterministic GOAP slot-inference engine.
You ONLY reason about slots required for the given tool/intent.
You MUST follow these rules:

STRICT RULES:
1. NEVER infer slot names from text.
2. NEVER invent domain-specific fields (e.g., policy_number, document_type).
3. Required slots MUST come ONLY from intent semantics.
4. If an intent requires NO slots → return "required_slots": [].
5. If user_text contains a value for a required slot, capture it in user_provided.
6. SlotLex semantic_tags and retrieved chunks are ONLY contextual signals.
7. If you are unsure → leave user_provided empty.
8. ALWAYS respond in strict JSON with keys: required_slots, user_provided, confidence.

Output example:
{
  "required_slots": ["query"],
  "user_provided": {"query": "how to file a claim"},
  "confidence": 0.90
}
"""

    user = {
        "intent": intent,
        "semantic_topic": semantic_topic,
        "user_text": user_text,
        "slotlex_summary": asdict(slotlex_context.summary),
        "slotlex_tags": slotlex_context.semantic_tags,
    }

    return fire_fast_modal_request_chat_get_dict(
        system_prompt=system,
        user_prompt=json.dumps(user, indent=2)
    )
