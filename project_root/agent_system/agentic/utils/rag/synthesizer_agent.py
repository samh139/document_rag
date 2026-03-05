# agents/rag/synthesizer_agent.py

import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv("app/config/secrets.env")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:12b")

class SynthesizerAgent:
    """
    Takes retrieved chunks and synthesizes a grounded answer.
    """

    SYSTEM_PROMPT = """
You are a banking assistant.

You must answer the user's question using ONLY the information
provided in the CONTEXT below.

Rules:
- Do NOT use any external knowledge.
- Do NOT guess or assume.
- If the answer is not present in the context, say:
  "I don’t have this information in the available bank documents."
- Be concise and factual.
- If charges, rates, or dates are mentioned, include them clearly.
"""

    @staticmethod
    def _build_context(chunks: List[Dict], max_chars: int = 6000) -> str:
        """
        Build a context string from retrieved chunks.
        Truncates safely to avoid prompt overflow.
        """
        context_parts = []
        total_chars = 0

        for i, c in enumerate(chunks, start=1):
            text = c["content"].strip()
            header = f"\n[Chunk {i} | Source: {c.get('metadata', {}).get('file_name','')}]"
            block = f"{header}\n{text}\n"

            if total_chars + len(block) > max_chars:
                break

            context_parts.append(block)
            total_chars += len(block)

        return "\n".join(context_parts)

    @staticmethod
    def synthesize_answer(
        query: str,
        chunks: List[Dict],
    ) -> str:
        """
        Main entry point to generate the final answer.
        """

        if not chunks:
            return "I don’t have this information in the available bank documents."

        context = SynthesizerAgent._build_context(chunks)

        prompt = f"""
{SynthesizerAgent.SYSTEM_PROMPT}

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

        body = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,   # low temperature = factual
                "top_p": 0.9
            }
        }

        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=body,
                timeout=120
            )
            resp.raise_for_status()
            out = resp.json()
            return out.get("response", "").strip()

        except Exception as e:
            return f"Error generating answer: {str(e)}"
