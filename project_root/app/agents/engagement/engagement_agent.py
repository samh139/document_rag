# agents/engagement_agent.py
import os
import requests
import json
from typing import Dict

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
ENGAGEMENT_MODEL = os.getenv("ENGAGEMENT_MODEL", "gemma3:4b")  # change if needed

class EngagementAgent:
    """
    Classifies user intent using LLM.
    """

    INTENTS = ["GREETING", "BANK_QUERY", "OUT_OF_SCOPE"]

    @staticmethod
    def classify(text: str) -> str:
        prompt = f"""
You are an intent classifier for a BANKING ASSISTANT.

Your job is ONLY to classify user input into ONE of the following labels:

1. GREETING
2. BANK_QUERY
3. OUT_OF_SCOPE

Rules:
- BANK_QUERY includes ONLY questions related to banking, finance, loans, accounts, ATM charges, interest rates, RBI rules, KYC, cards, transactions.
- GREETING includes greetings, thanks, goodbyes, polite acknowledgements.
- OUT_OF_SCOPE includes everything else: weather, math, programming, opinions, chit-chat, general knowledge, personal questions.

DO NOT answer the question.
DO NOT explain.
Return ONLY the label.

User input:
"{text}"

"""

        payload = {
            "model": ENGAGEMENT_MODEL,
            "prompt": prompt,
            "stream": False
        }

        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60
        )
        resp.raise_for_status()

        result = resp.json().get("response", "").strip().upper()

        # Hard safety fallback
        if result not in EngagementAgent.INTENTS:
            return "OUT_OF_SCOPE"


        return result