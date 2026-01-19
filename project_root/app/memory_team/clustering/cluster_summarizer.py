import os
import requests
from typing import List

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gemma3:12b")

SYSTEM_PROMPT = """
You are generating short semantic labels for document clusters.

Rules:
- Output ONE short summary (1-2 sentences)
- Describe what kind of user questions this cluster answers
- Do NOT use punctuation
- Do NOT mention documents or chunks
"""

def summarize_cluster(chunk_texts: List[str]) -> str:
    excerpts = "\n\n".join(
        f"- {text[:500]}" for text in chunk_texts
    )

    prompt = f"""
{SYSTEM_PROMPT}

Document excerpts:
{excerpts}

Cluster label:
"""

    payload = {
        "model": SUMMARY_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=90,
    )
    resp.raise_for_status()

    return resp.json().get("response", "").strip()
