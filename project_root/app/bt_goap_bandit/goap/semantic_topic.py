# =====================================================================
# semantic_topic.py — Minimal deterministic + Gemini-Flash fallback
# =====================================================================

from typing import Tuple, List, Any
from app_logger import LoggerFactory
from app_configs.llm_config import fire_fast_modal_request_chat_get_dict
log = LoggerFactory.get_logger("semantic_topic")

# ---------------------------------------------------------------------
# 1) Deterministic rule (VERY small)
# Only high-confidence mappings. All smalltalk etc = LLM fallback.
# ---------------------------------------------------------------------
HARD_MAP = {
    "lookup_knowledge": ("rag_query", ["query"]),
    "lookup_knowledge_plus_reasoning": ("rag_reasoning", ["query"]),
    "task_execute": ("task_executor", ["task"]),
}

# These intents are *LLM-decided*, not hard-mapped:
LLM_ONLY_INTENTS = {
    "generic_query",
    "offtopic_query",
    "greeting",
    "farewell",
    "rapport_smalltalk",
    "gratitude_ack",
    "apology_ack",
    "get_file_template"
}

# ---------------------------------------------------------------------
# 2) Gemini Flash for semantic topic classification
# ---------------------------------------------------------------------
def _llm_semantic_tool(intent: str, text: str, semantic_tags: List[str]):
    """
    Ask Gemini Flash to choose the correct semantic tool based on:
    - Intent
    - User text
    - SlotLex semantic tags
    """


    system = """
You are a semantic dispatcher.

Your job:
- Map an intent to the correct GOAP tool.
- DO NOT infer unknown tool names.
- Choose ONLY from:
    rag_query, rag_reasoning, task_executor, generic_reasoner
- Use semantic tags as "soft evidence".

If unsure → generic_reasoner.

Return ONLY JSON:
{
  "tool": "...",
  "params": [...]
}
"""

    user = {
        "intent": intent,
        "text": text,
        "semantic_tags": semantic_tags[:5] if semantic_tags else []
    }

    result = fire_fast_modal_request_chat_get_dict(system, user)

    # Final guardrail
    tool = result.get("tool", "generic_reasoner")
    if tool not in ["rag_query", "rag_reasoning", "task_executor", "generic_reasoner"]:
        tool = "generic_reasoner"

    params = result.get("params", ["query"])

    return tool, params


# =====================================================================
# MAIN FUNCTION
# =====================================================================
def detect_semantic_topic(
    intent: str = None,
    text: str = None,
    user_text: str = None,
    query: str = None,
    entities: List[str] = None,
    semantic_tags: List[str] = None,
    slotlex_output: Any = None,
    **kwargs
) -> Tuple[str, List[str]]:

    final_text = text or user_text or query or ""

    
    # -----------------------------------------------------------------
    # STEP 1 — HARD deterministic mapping
    # -----------------------------------------------------------------
    if intent in HARD_MAP:
        tool, params = HARD_MAP[intent]

        return tool, params

    # -----------------------------------------------------------------
    # STEP 2 — LLM classification for soft intents
    # -----------------------------------------------------------------
    if intent in LLM_ONLY_INTENTS:
        log.debug(f"intent {intent}")
        tool, params = _llm_semantic_tool(intent, final_text, semantic_tags)
        return tool, params

    # -----------------------------------------------------------------
    # STEP 3 — UNKNOWN INTENT → safe fallback
    # -----------------------------------------------------------------
    
    log.debug("[semantic_topic] Unknown → DEFAULT rag_query")
    log.debug("--- semantic_topic END (FALLBACK) ---\n")

    return "rag_query", ["query"]
 