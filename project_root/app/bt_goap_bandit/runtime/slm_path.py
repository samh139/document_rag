# src/bt_goap_bandit/runtime/slm_path.py
from __future__ import annotations

import json, os, re
from typing import Any, Dict, List

__all__ = ["_postprocess_to_list", "slm_suggest_prompts"]

# Stronger default system prompt: concise + structured
_DEFAULT_SYSTEM = (
    "You suggest up to 3 short, high-signal follow-up questions or next steps "
    "to quickly resolve the user's task. Output ONLY bullets or a JSON array "
    "of strings. No preamble, no apologies, no explanations."
)

def _enabled() -> bool:
    # Read the flag at call time so tests or callers can flip BT_SLM via env
    return str(os.getenv("BT_SLM", "0")).strip() == "1"

def _build_prompt(primary: Dict[str, Any], plan: Dict[str, Any], sentiment: Dict[str, Any]) -> str:
    belief = (plan or {}).get("belief") or {}
    scope = ((belief.get("scope") or {}).get("value")) or (primary or {}).get("scope")
    benefit = ((belief.get("benefit_type") or {}).get("value"))
    jur = ((belief.get("jurisdiction") or {}).get("value")) or (primary or {}).get("jurisdiction")
    ready = (plan or {}).get("ready_to_retrieve")
    blocking = (plan or {}).get("blocking_slots") or []
    lines = []
    lines.append("You are helping optimize a retrieval assistant's next step.")
    lines.append(f"- scope: {scope!r}, benefit_type: {benefit!r}, jurisdiction: {jur!r}")
    lines.append(f"- readiness: {ready}, blocking_slots: {blocking}")
    s = (sentiment or {}).get("raw", {}).get("scores", {})
    if s:
        lines.append(f"- sentiment.compound: {s.get('compound', 0)}")
    lines.append("")
    lines.append("Suggest up to 3 short follow-ups or next steps (bullet form, no preamble).")
    return "\n".join(lines)


# ---- Post-processing helpers ----

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")

_STOP_PHRASES = (
    "sure", "here are", "suggestions", "to help", "as an ai",
    "i can", "you can", "consider", "you should", "we can",
)

_GENERIC_PATTERNS = (
    re.compile(r"^check (the )?documentation", re.I),
    re.compile(r"^have you checked", re.I),
    re.compile(r"^can you provide more context", re.I),
    re.compile(r"^refer to", re.I),
    re.compile(r"^have you tried searching", re.I),
    re.compile(r"^search (online|the web)", re.I),
    re.compile(r"^consult( a| the)? (.*)?(expert|advisor|team)", re.I),
)


def _looks_generic(s: str) -> bool:
    s = s.strip().lower()
    if any(p.match(s) for p in _GENERIC_PATTERNS):
        return True
    return len(s) < 6 or len(s) > 160

def _clean_line(s: str) -> str:
    s = s.strip(" -*\t:").strip()
    if not s:
        return ""

    low = s.lower()

    # Drop boilerplate lines with no actionable question
    if any(ph in low for ph in ("sure", "here are", "suggestions")) and "?" not in s:
        return ""

    # Trim known boilerplate prefixes.
    for ph in _STOP_PHRASES:
        if low.startswith(ph):
            if ":" in s:
                s = s.split(":", 1)[1].strip()
            elif " - " in s:
                s = s.split(" - ", 1)[1].strip()
            else:
                s = s[len(ph):].lstrip(" ,.-")
            break

    return s


def _parse_json_array(text: str) -> List[str]:
    if not text:
        return []
    m = _JSON_ARRAY_RE.search(text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x).strip() for x in arr if isinstance(x, str)]
    except Exception:
        return []

def _postprocess_to_list(text: str | None) -> List[str]:
    if not text:
        return []
    arr = _parse_json_array(text)
    lines: List[str] = []
    if arr:
        lines = arr
    else:
        for line in text.splitlines():
            line = _clean_line(line)
            if not line:
                continue
            lines.append(line)

    out: List[str] = []
    seen = set()
    for line in lines:
        line = _clean_line(line)
        if not line or _looks_generic(line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= 3:
            break
    out = out[:3]
    return out


def slm_suggest_prompts(primary: Dict[str, Any], plan: Dict[str, Any], sentiment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      {"slm_used": bool, "fallback": bool, "suggested_prompts": List[str]}
    """
    if not _enabled():
        return {"slm_used": False, "fallback": True, "suggested_prompts": []}

    try:
        from bt_goap_bandit.slm.ollama_client import OllamaClient
    except Exception:
        return {"slm_used": False, "fallback": True, "suggested_prompts": []}

    prompt = _build_prompt(primary or {}, plan or {}, sentiment or {})
    try:
        client = OllamaClient()
        text, _raw = client.generate(
            prompt, 
            system=os.getenv("BT_SLM_SYSTEM", _DEFAULT_SYSTEM),
            timeout=float(os.getenv("BT_SLM_TIMEOUT", "2.0"))
        )

        suggestions = _postprocess_to_list(text)
        return {"slm_used": True, "fallback": False, "suggested_prompts": suggestions}
    except Exception:
        return {"slm_used": False, "fallback": True, "suggested_prompts": []}