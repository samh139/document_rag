"""
runtime/slot_lex_reload.py — load slot lexicon dynamically
"""

from __future__ import annotations
import json, os
from typing import Dict, Any, List

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "slot_lex.json")

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _load_slotlex() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

SLOT_LEX: Dict[str, Any] = _load_slotlex()

# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------
def enrich_with_slotlex(plan: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Use SLOT_LEX definitions to auto-fill or validate plan slots."""
    if not plan or not SLOT_LEX:
        return plan
    txt = (text or "").lower()

    for slot, meta in SLOT_LEX.items():
        if slot not in plan.get("belief", {}):
            continue
        keywords_map = meta.get("keywords_map", {})
        for label, kws in keywords_map.items():
            if any(k in txt for k in kws):
                plan["belief"][slot]["value"] = label
                plan["belief"][slot]["p"] = 0.8
                break
    return plan

# ---------------------------------------------------------------------------
# Prompt helper
# ---------------------------------------------------------------------------
def suggest_for_slot(slot: str) -> List[str]:
    prompts = {
        "benefit_type": ["Is this about travel reimbursement, hardware, or general expense?"],
        "scope": ["Is this a policy question or a step-by-step SOP query?"],
    }
    return prompts.get(slot, [f"Could you clarify the {slot}?"])

__all__ = ["SLOT_LEX", "enrich_with_slotlex", "suggest_for_slot"]
