from __future__ import annotations

from typing import Dict, List

from agent_system.agentic.knowledge_mcp_server.tools.common import call_ollama
from agent_system.agentic.knowledge_mcp_server.prompts.prompt_registry import load_prompt


def _build_context(chunks: List[Dict]) -> str:
    #print(f"CHhunks : {chunks}")
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        file_name = chunk.get("file_name", "")
        chunk_id = chunk.get("chunk_id", "")
        chunk_content = chunk.get("content", "")

        context_parts.append(
            f"[{i}] (chunk_id={chunk_id}, file={file_name})\n{chunk_content}"
        )

    return "\n\n".join(context_parts)


def _build_citations(chunks: List[Dict]) -> List[Dict]:
    citations = []

    for chunk in chunks:
        citations.append({
            "chunk_id": chunk.get("chunk_id"),
            "file_name": chunk.get("file_name"),
            "chunk_content": chunk.get("content"),
        })

    return citations


def synthesize_answer_impl(
    query: str,
    chunks: List[Dict],
) -> Dict[str, any]:

    if not chunks:
        return {
            "answer": "I'm unable to find relevant information for your query. Could you please clarify or provide more details?",
            "citations": []
        }

    context = _build_context(chunks)
    #print(f"[synthesize_answer_impl] context = {context}")

    prompt_template = load_prompt("synthesis", "v1")

    prompt = prompt_template.format(
        query=query,
        context=context,
    )
    #rint(f"[synthesize_answer_impl] prompt = {prompt}")

    response = call_ollama(prompt)
    #print(f"[synthesize_answer_impl] response = {response}")

    answer = response.strip()

    citations = _build_citations(chunks)
    #print(f"[synthesize_answer_impl] citations = {citations}")

    return {
        "answer": answer,
        "citations": citations,
    }


def register(mcp) -> None:
    @mcp.tool
    def synthesize_answer(
        query: str,
        chunks: List[Dict],
    ) -> Dict[str, any]:
        """Generate a grounded answer from retrieved chunks with citations."""
        return synthesize_answer_impl(
            query=query,
            chunks=chunks,
        )