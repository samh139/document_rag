# agents/engagement_agent.py
import os
import requests
import json
from typing import Dict

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("ENGAGEMENT_MODEL", "gemma3:12b")  # change if needed

PROMPT = """
You are an intent classifier. Given the user's message, return strictly a JSON object ONLY (no explanation, no markdown)
with these fields:
{
  "intent": "<one-of: greeting | rag | smalltalk | unknown>",
  "confidence": float_between_0_and_1,
  "follow_up": "optional short question or empty string"
}

Rules:
- If the message is a greeting ("hi", "hello", "hey", "good morning", etc.) -> intent: greeting
- If the message is clearly about documents, policies, fees, banking, or asks a question (contains '?') -> intent: rag
- If it is small talk -> smalltalk
- Otherwise -> unknown
Be concise and always return valid JSON only.
User message:
"""  # we'll append user's message

def classify_with_ollama(text: str, timeout=15) -> Dict:
    payload = {"model": MODEL, "input": PROMPT + text}
    resp = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    out = resp.json()
    # Ollama output shapes can vary; try to extract 'text' key or full string.
    content = ""
    if isinstance(out, dict):
        if "text" in out:
            content = out["text"]
        else:
            # some versions return choices/outputs
            for k in ("choices","outputs"):
                if k in out and isinstance(out[k], list) and len(out[k])>0:
                    first = out[k][0]
                    if isinstance(first, dict) and "content" in first:
                        content = first["content"]
                    elif isinstance(first, dict) and "text" in first:
                        content = first["text"]
                    elif isinstance(first, str):
                        content = first
    if not content:
        content = json.dumps({"intent":"unknown","confidence":0.0,"follow_up":""})
    # Clean and parse JSON from model output: extract first { ... } substring
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = content[start:end+1]
    else:
        raw = content
    try:
        parsed = json.loads(raw)
    except Exception:
        # fallback: conservative classification
        parsed = {"intent":"unknown","confidence":0.0,"follow_up":""}
    # sanitize
    intent = parsed.get("intent","unknown")
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    follow_up = parsed.get("follow_up","") or ""
    return {"intent": intent, "confidence": confidence, "follow_up": follow_up}
