import json
import re

from app_configs.llm_config import fire_fast_modal_request_chat_get_dict
from model_classes.intent_agent_response import Enriched

def infer_task_from_semantics(user_text: str, semantic_tags: list, slotlex_context: Enriched ):
    """
    Pure semantic task inference.
    - No keywords
    - No hardcoded task names
    - LLM decides final task label
    """

    system = """
You are a semantic task-label generator for a GOAP planner.

RULES:
- NO keywords, NO pattern matching.
- NO hardcoded task names.
- Derive the task ONLY from semantics in:
  • user_text
  • SlotLex semantic_tags
  • Retrieved context summary
- Output SHORT machine-readable task labels, like:
    "apply_leave", "update_profile", "request_document", "book_meeting"
- If no task is needed → return null.
- Respond only with JSON:
  {"task": <string or null>, "confidence": <0..1>}
"""

    user = {
        "user_text": user_text,
        "semantic_tags": semantic_tags,
        "slotlex_summary": slotlex_context.retrieved_contexts,
    }

    return fire_fast_modal_request_chat_get_dict(system, json.dumps(user, indent=2))
