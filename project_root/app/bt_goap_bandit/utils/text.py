import re
from typing import List, Optional

# Very light keyword heuristics; replace with NER later if needed.
ENTITY_HINTS = [
    "FMLA","HRIS","policy","SOP","screenshots","Employee Relations CoE",
    "maternity leave","reimbursement","expense","claim","travel","assignment status",
]

JURIS_PATTERNS = {
    "Singapore": r"\bSingapore\b",
    "India": r"\bIndia\b",
    "US": r"\bUnited States\b|\bUSA\b|\bUS\b",
    "UK": r"\bUnited Kingdom\b|\bUK\b",
}

SENSITIVE_PATTERNS = [
    r"\bpassword\b",
    r"\baccount number\b",
    r"\bconfidential\b",
]

def extract_entities(text: str) -> List[str]:
    out: List[str] = []
    t = text or ""
    for h in ENTITY_HINTS:
        if re.search(re.escape(h), t, re.IGNORECASE):
            out.append(h)
    return out

def detect_jurisdiction(text: str) -> Optional[str]:
    t = text or ""
    for name, pat in JURIS_PATTERNS.items():
        if re.search(pat, t, re.IGNORECASE):
            return name
    return None

def guardrails_check(text: str) -> List[str]:
    flags = []
    if len(text or "") > 2000:
        flags.append("input_too_long")
    for pat in SENSITIVE_PATTERNS:
        if re.search(pat, text or "", re.IGNORECASE):
            flags.append("possible_sensitive_data")
            break
    return flags
