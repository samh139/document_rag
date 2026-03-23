from __future__ import annotations

from typing import Dict

from knowledge_mcp_server.tools.common import call_ollama
from knowledge_mcp_server.prompts.prompt_registry import load_prompt


def generate_clarification_impl(
    refined_query: str,
    stm_context: str = "",
    ltm_context: str = "",
) -> Dict[str, str]:

    prompt_template = load_prompt("clarification", "v1")

    prompt = prompt_template.format(
        refined_query=refined_query,
        stm_context=stm_context,
        ltm_context=ltm_context,
    )

    response = call_ollama(prompt)

    question = response.strip()

    return {
        "question": question
    }


def register(mcp) -> None:
    @mcp.tool
    def generate_clarification(
        refined_query: str,
        stm_context: str = "",
        ltm_context: str = "",
    ) -> Dict[str, str]:
        """Generate a clarification question using query + conversation context."""
        return generate_clarification_impl(
            refined_query=refined_query,
            stm_context=stm_context,
            ltm_context=ltm_context,
        )