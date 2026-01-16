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
    ClusterRoutedQueryMessage, RAGRetrievalResultMessage
)
from app.memory_team.clustering.cluster_resolver import (
    resolve_cluster_ids_to_chunk_ids
)

# 🔁 Import your existing retriever
from app.agents.rag.retriever_agent import RetrieverAgent

logger = logging.getLogger("BankRAGRetrievalAgent")
logging.basicConfig(level=logging.INFO)


@type_subscription(topic_type=AgenticTopic.CLUSTER_ROUTED_QUERY_TOPIC.value)
class BankRAGRetrievalAgent(RoutedAgent):

    def __init__(self) -> None:
        super().__init__("BankRAGRetrievalAgent")

    @message_handler
    async def handle_cluster_routed_query(
        self,
        message: ClusterRoutedQueryMessage,
        ctx: MessageContext,
    ) -> None:

        if message.intent != "BANK_QUERY":
            return

        # 🚦 CASE B — ambiguous → clarification later
        if message.restrict_cluster_ids is None:
            logger.info("[RAG] No cluster restriction → clarification needed")
            return  # ClarificationAgent will handle this

        # 🚀 CASE A — resolve clusters → chunks
        restrict_ids = resolve_cluster_ids_to_chunk_ids(
            cluster_ids=message.restrict_cluster_ids,
            expand_level2=True,
        )

        retrieval = RetrieverAgent.retrieve(
            query=message.refined_query,
            user_acl=[],
            top_k=5,
            restrict_ids=restrict_ids,  # 🔥 NEW
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