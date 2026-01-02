# app/agentic/rag/query_refiner_agent.py

import logging
import requests

from autogen_core import (
    RoutedAgent,
    MessageContext,
    TopicId,
    message_handler,
    type_subscription,
)

from app.agentic.topics import AgenticTopic
from app.agentic.messages import (
    EngagementOutputMessage,
    RAGRetrievalResultMessage,
)

from app.memory.memory_manager import MemoryManager
from app.configs.llm_config import (
    OLLAMA_URL,
    QUERY_REFINER_MODEL,
    OLLAMA_TIMEOUT,
)

logger = logging.getLogger("QueryRefinerAgent")
logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """
You are a Query Refinement Agent for a banking assistant.

Task:
- Convert the user query into a fully self-contained banking question.
- Use ONLY the provided conversation context.
- If the query is already complete, return it EXACTLY as-is.
- Do NOT answer the question.
- Do NOT add new information.

Return ONLY the refined query text.
"""

@type_subscription(topic_type=AgenticTopic.ENGAGEMENT_OUTPUT.value)
class QueryRefinerAgent(RoutedAgent):

    def __init__(self):
        super().__init__("QueryRefinerAgent")
        self.memory = MemoryManager()

    def _refine_query(
        self,
        original_query: str,
        stm_summary: dict | None,
    ) -> str:

        context_block = ""
        if stm_summary:
            context_block = f"""
Conversation context:
Topic: {stm_summary.get("topic", "")}
Summary: {stm_summary.get("context_summary", "")}
Entities: {", ".join(stm_summary.get("entities", []))}
"""

        payload = {
            "model": QUERY_REFINER_MODEL,
            "prompt": f"{SYSTEM_PROMPT}\n{context_block}\nUser query:\n{original_query}",
            "stream": False,
        }

        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()

        refined = resp.json().get("response", "").strip()
        return refined or original_query

    @message_handler
    async def handle_engagement_output(
        self,
        message: EngagementOutputMessage,
        ctx: MessageContext,
    ) -> None:

        logger.info("[QueryRefiner] Input query: %s", message.user_query)

        stm_summary = self.memory.get_stm_summary_cached(
            role="bank_user",
            user_id=message.user_id,
            session_id=message.session_id,
        )

        refined_query = self._refine_query(
            original_query=message.user_query,
            stm_summary=stm_summary,
        )

        logger.info("[QueryRefiner] Refined query: %s", refined_query)

        output = RAGRetrievalResultMessage(
            query=refined_query,
            chunks=[],
            session_id=message.session_id,
            user_id=message.user_id,
        )

        await self.publish_message(
            output,
            TopicId(
                AgenticTopic.RAG_RETRIEVAL_OUTPUT.value,
                source=self.id.key,
            ),
        )
