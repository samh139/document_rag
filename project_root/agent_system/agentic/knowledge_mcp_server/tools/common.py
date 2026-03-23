from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("KNOWLEDGE_MCP_MODEL", "gemma3:12b")
REQUEST_TIMEOUT = int(os.getenv("KNOWLEDGE_MCP_TIMEOUT", "90"))


def call_ollama(prompt: str, model: str | None = None, temperature: float = 0.2) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model or DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return (payload.get("response") or "").strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
