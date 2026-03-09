import os
import requests
import logging

from autogen_core import (
    RoutedAgent,
    MessageContext,
    TopicId,
    message_handler,
    type_subscription,
)

from agent_system.agentic.topics import AgenticTopic
from agent_system.agentic.messages import (
    RAGRetrievalResultMessage,
    FinalAnswerMessage,
)

from typing import List, Dict

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logger = logging.getLogger("BankRAGSynthesisAgent")
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------
# LLM config
# ------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", "gemma3:12b")

SYSTEM_PROMPT = """
You are a banking assistant.

Answer the user's question using ONLY the information
provided in the CONTEXT below.

You ARE allowed to:
- Extract charges, fees, limits, and rules from the context
- Rephrase and summarize information
- Combine information across multiple context chunks
- Answer even if the wording does not exactly match the question

Rules:
- Do NOT use external knowledge
- Do NOT invent values not present in the context
- If the context contains relevant information, you MUST answer using it
- IMPORTANT:
The context may contain tables, broken formatting, or fragmented sentences.
You MUST carefully scan the entire context for relevant financial values
such as interest rates, charges, limits, or account rules.

Even if the wording is not exact, extract the closest relevant information.

If a table contains the answer, interpret the table correctly.

Only say "I don’t have this information in the available bank documents"
if the context truly contains nothing about the topic.
- Be concise and factual
- Clearly list charges when applicable
"""

@type_subscription(topic_type=AgenticTopic.RAG_RETRIEVAL_OUTPUT.value)
class BankRAGSynthesisAgent(RoutedAgent):

    def __init__(self) -> None:
        super().__init__("BankRAGSynthesisAgent")

    def _build_context(self, chunks: List[Dict], max_chars: int = 6000) -> str:
        """
        Build a context string from retrieved chunks.
        Truncates safely to avoid prompt overflow.
        """
        logger.warning("🔥 _build_context CALLED 🔥")

        context_parts = []
        total_chars = 0

        logger.info(f"Building context from {len(chunks)} chunks")

        for i, c in enumerate(chunks, start=1):
            text = c.get("content", "").strip()

            header = (
                f"\n[Chunk {i} | Source: {c.get('metadata', {}).get('file_name', '')}]"
                f"\n(Relevance: May contain information related to the user question)"
            )

            block = f"{header}\n{text}\n"

            logger.info(f"[DEBUG] Chunk {i} preview: {text[:50]}")
            #logger.info(f"[DEBUG] Block length: {len(block)}")
            #logger.info(f"[DEBUG] Total chars so far: {total_chars}")

            if total_chars + len(block) > max_chars:
                logger.info("[DEBUG] Max context size reached, stopping")
                break

            context_parts.append(block)
            total_chars += len(block)

        logger.info(f"Final context size: {total_chars} chars")
        return "\n".join(context_parts)

    def synthesize_answer(self, query: str, chunks: List[Dict]) -> str:
        """
        Main entry point to generate the final answer.
        """
        logger.warning(f"🔥 synthesize_answer CALLED with {len(chunks)} chunks")

        if not chunks:
            logger.warning("No chunks provided to synthesizer")
            return "I don’t have this information in the available bank documents."

        context = self._build_context(chunks)

        prompt = f"""
    {SYSTEM_PROMPT}

    CONTEXT:
    {context}

    USER QUESTION:
    {query}

    ANSWER:
"""

        body = {
            "model": SYNTHESIS_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
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
            logger.exception("LLM generation failed")
            return f"Error generating answer: {str(e)}"


    @message_handler
    async def handle_retrieval_result(
        self,
        message: RAGRetrievalResultMessage,
        ctx: MessageContext,
    ) -> None:

        logger.info(
            f"[SYNTHESIS] Generating answer for session={message.session_id}"
        )

        answer = self.synthesize_answer(
            query=message.query,
            chunks=message.chunks,
        )

        output = FinalAnswerMessage(
            answer=answer,
            citations=[
            {   
                "chunk_id": c["chunk_id"],
                "file_name": c["metadata"].get("file_name"),
                "chunk_content": c["content"],
            }
                for c in message.chunks
            ],
            session_id=message.session_id,
            user_id=message.user_id,
            user_query=message.query
        )

        await self.publish_message(
            output,
            topic_id=TopicId(
                AgenticTopic.FINAL_RESPONSE.value,
                source=self.id.key,
            ),
        )

        logger.info(
            f"[SYNTHESIS] Answer published for session={message.session_id}"
        )
