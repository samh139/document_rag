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
    RefinedQueryMessage,
    RAGRetrievalResultMessage,
)

# 🔁 Import your existing retriever
from app.agents.rag.retriever_agent import RetrieverAgent

logger = logging.getLogger("BankRAGRetrievalAgent")
logging.basicConfig(level=logging.INFO)


@type_subscription(topic_type=AgenticTopic.REFINED_QUERY_TOPIC.value)
class BankRAGRetrievalAgent(RoutedAgent):

    def __init__(self) -> None:
        super().__init__("BankRAGRetrievalAgent")

    @message_handler
    async def handle_engagement_output(
        self,
        message: RefinedQueryMessage,
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
        print(f"Refined User Query from RAGRetrievalAgent: {message.refined_query}")
        logger.info(f"Original: {message.original_query}")
        logger.info(f"Refined: {message.refined_query}")

        retrieval = RetrieverAgent.retrieve(
            query=message.refined_query,  # or original query if you store it later
            user_acl=[],
            top_k=5,
        )

        chunks = retrieval.get("chunks", [])

        logger.info(f"[RAG] Retrieved {len(chunks)} chunks")

        output = RAGRetrievalResultMessage(
            query=message.refined_query,
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