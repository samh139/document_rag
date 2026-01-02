# app/memory/stm_summarizer.py

import os
import json
import logging
import requests
from typing import List, Dict

logger = logging.getLogger("STMSummarizer")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
STM_MODEL = os.getenv("STM_MODEL", "gemma3:4b")


STM_SYSTEM_PROMPT = """
You are the Short-Term Memory Agent.

Your task is to UPDATE the conversation's short-term memory using ONLY the recent conversation window.

Rules (MANDATORY):
1. Merge all relevant information into ONE concise factual summary (≤100 tokens).
2. Deduplicate entities.
3. If this is the first turn, base STM only on that turn.
4. Always include last user and bot messages verbatim.
5. Do NOT hallucinate or infer new facts.
6. Output ONLY valid JSON. No extra text.

Output schema:
{
  "turn_number": <integer>,
  "topic": "<short topic>",
  "context_summary": "<≤100 tokens>",
  "entities": ["<entity1>", "<entity2>"],
  "last_user_message": "<verbatim>",
  "last_bot_message": "<verbatim>"
}
"""


def _build_conversation_context(
    sliding_turns: List[Dict],
    current_user_message: str,
    current_bot_message: str,
) -> str:
    context = ""
    for turn in sliding_turns:
        context += f"User: {turn['user']}\n"
        context += f"Bot: {turn['assistant']}\n"

    context += f"User: {current_user_message}\n"
    context += f"Bot: {current_bot_message}\n"

    return context


def generate_stm_summary(
    sliding_turns: List[Dict],
    current_user_message: str,
    current_bot_message: str,
) -> Dict:
    """
    Generate STM summary using Ollama.
    """

    conversation_context = _build_conversation_context(
        sliding_turns,
        current_user_message,
        current_bot_message,
    )

    prompt = f"""
{STM_SYSTEM_PROMPT}

Conversation Window:
{conversation_context}
"""

    payload = {
        "model": STM_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"  # 🔒 forces JSON
    }

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()

        raw = resp.json().get("response", "").strip()
        stm_summary = json.loads(raw)

        return stm_summary

    except Exception as e:
        logger.error(f"STM summarization failed: {e}")

        # 🛑 Hard fallback — NO hallucination
        return {
            "turn_number": len(sliding_turns) + 1,
            "topic": "",
            "context_summary": "",
            "entities": [],
            "last_user_message": current_user_message,
            "last_bot_message": current_bot_message,
        }
