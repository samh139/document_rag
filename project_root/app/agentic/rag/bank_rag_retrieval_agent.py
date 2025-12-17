# app/agentic/rag/bank_rag_retrieval_agent.py
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
    EngagementOutputMessage,
    RAGRetrievalResultMessage,
)

# 🔁 Import your existing retriever
from app.agents.rag.retriever_agent import RetrieverAgent

logger = logging.getLogger("BankRAGRetrievalAgent")
logging.basicConfig(level=logging.INFO)


@type_subscription(topic_type=AgenticTopic.ENGAGEMENT_OUTPUT.value)
class BankRAGRetrievalAgent(RoutedAgent):

    def __init__(self) -> None:
        super().__init__("BankRAGRetrievalAgent")

    @message_handler
    async def handle_engagement_output(
        self,
        message: EngagementOutputMessage,
        ctx: MessageContext,
    ) -> None:

        # Guardrail — act only on BANK_QUERY
        if message.intent != "BANK_QUERY":
            logger.info(
                f"[RAG] Skipping intent={message.intent} for session={message.session_id}"
            )
            return

        logger.info(
            f"[RAG] Retrieving documents for session={message.session_id}"
        )
        print(f"User Query from RAGRetrievalAgent: {message.user_query}")
        retrieval = RetrieverAgent.retrieve(
            query=message.user_query,  # or original query if you store it later
            user_acl=[],
            top_k=5,
        )

        chunks = retrieval.get("chunks", [])

        logger.info(f"[RAG] Retrieved {len(chunks)} chunks")

        output = RAGRetrievalResultMessage(
            query=message.user_query,
            chunks=chunks,
            session_id=message.session_id,
            user_id=message.user_id,
        )

        await self.publish_message(
            output,
            topic_id=TopicId(
                AgenticTopic.RAG_RETRIEVAL_OUTPUT.value,
                source=self.id.key,
            ),
        )
        logger.info(f"[RAG] Published retrieval results for session={message.session_id}")