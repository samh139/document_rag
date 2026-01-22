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

from app.agentic.topics import AgenticTopic
from app.agentic.messages import (
    RAGRetrievalResultMessage,
    FinalAnswerMessage,
)

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

Answer the user's question using ONLY the provided document excerpts.
If the answer is not found in the excerpts, say:
"I could not find this information in the provided documents."

Rules:
- Be concise and factual
- Do not hallucinate
- Do not use external knowledge
"""

@type_subscription(topic_type=AgenticTopic.RAG_RETRIEVAL_OUTPUT.value)
class BankRAGSynthesisAgent(RoutedAgent):

    def __init__(self) -> None:
        super().__init__("BankRAGSynthesisAgent")

    def _build_context(self, chunks):
        return "\n\n".join(
            f"[{i+1}] {c['content']}"
            for i, c in enumerate(chunks)
        )

    def _synthesize(self, question: str, chunks):
        try:
            context = self._build_context(chunks)

            prompt = f"""
            {SYSTEM_PROMPT}

            Question:
            {question}

            Document excerpts:
            {context}
            """

            payload = {
                "model": SYNTHESIS_MODEL,
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
        
        except Exception as e:
            logger.error(f"[SYNTHESIS ERROR] {e}")
            return "I could not generate an answer due to an internal error."

    @message_handler
    async def handle_retrieval_result(
        self,
        message: RAGRetrievalResultMessage,
        ctx: MessageContext,
    ) -> None:

        logger.info(
            f"[SYNTHESIS] Generating answer for session={message.session_id}"
        )
        print(f"Question: {message.query}")
        answer = self._synthesize(
            question=message.query,
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
