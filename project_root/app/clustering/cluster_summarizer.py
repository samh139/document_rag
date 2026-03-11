import os
import requests
from typing import List

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gemma3:12b")

SYSTEM_PROMPT = """
You are generating business capability summaries for a banking knowledge system.

Your goal is to describe what customer questions this cluster can fully answer.

Rules:
- Output ONE short summary (1-2 sentences)
- Explicitly mention in summary banking services products fees charges limits or rules if present
- Use generic capability language not specific dates amounts or examples
- The summary should help decide whether this cluster alone can answer a question
- Do NOT mention documents chunks embeddings or clustering
- Do NOT use bullet points
- Do NOT use punctuation

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
