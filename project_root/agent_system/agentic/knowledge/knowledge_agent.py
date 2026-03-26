from __future__ import annotations

from typing import Any, Dict, List, Optional

from autogen_core import MessageContext, RoutedAgent, message_handler ,type_subscription

from agent_system.agentic.messages import (
    UserMessage,
    FinalAnswerMessage,
    RefinedQueryMessage,
)
from agent_system.agentic.memory_team.stm.store import get_stm_summary
from agent_system.agentic.knowledge_mcp_server.client import KnowledgeMCPClient
from agent_system.agentic.utils.rag.embedding_client import embed_text
from agent_system.agentic.memory_team.ltm.ltm_service import retrieve_ltm_context
from agent_system.agentic.topics import AgenticTopic
from autogen_core import TopicId


@type_subscription(topic_type=AgenticTopic.REFINED_QUERY_TOPIC.value)
class KnowledgeAgent(RoutedAgent):
    def __init__(self, description: str = "Knowledge MCP orchestrator") -> None:
        super().__init__(description=description)
        self.mcp_client = KnowledgeMCPClient()

    def _get_stm_context(self, session_id: str) -> str:
        try:
            summary = get_stm_summary(session_id).get("context_summary", "")
            return summary or ""
        except Exception:
            return ""

    def _get_ltm_context(self, user_query:str, session_id: str) -> str:
        # Keep this minimal for now.
        try:
            context = retrieve_ltm_context(user_query, session_id)  
            return context or ""
        except Exception:
            return ""


    @message_handler
    async def handle_refined_query(
        self,
        message: RefinedQueryMessage,
        ctx: MessageContext,
    ) -> None:
        session_id = message.session_id
        user_query = message.original_query
        refined_query = message.refined_query

        # Step 1: embed refined query
        query_embedding = embed_text(refined_query)

        # Step 2: retrieve cluster
        cluster_results = self.mcp_client.retrieve_cluster(
            query_embedding=query_embedding,
            top_k=3,
        )

        # Step 3: detect ambiguity
        ambiguity_result = self.mcp_client.detect_ambiguity(
            cluster_results=cluster_results,
        )

        is_ambiguous = ambiguity_result.get("is_ambiguous")
        selected_cluster_id = ambiguity_result.get("selected_cluster_id")

        #print(f"[KnowledgeAgent] cluster_results = {cluster_results}")
        #print(f"[KnowledgeAgent] ambiguity_result = {ambiguity_result}")
        print(f"[KnowledgeAgent] selected_cluster_id = {selected_cluster_id}")

        if is_ambiguous:
            stm_context = self._get_stm_context(session_id)
            ltm_context = self._get_ltm_context(user_query, session_id)

            clarification_result = self.mcp_client.generate_clarification(
                refined_query=refined_query,
                stm_context=stm_context,
                ltm_context=ltm_context,
            )

            clarification_question = clarification_result.get(
                "question"
            )

            await self.publish_message(
            FinalAnswerMessage(
                session_id=session_id,
                user_id=message.user_id,
                user_query=user_query,
                answer=clarification_question,
                citations=[],
            ),
            topic_id=TopicId(
                AgenticTopic.FINAL_RESPONSE.value,
                source=self.id.key,
            ),
        )

            print(f"[KnowledgeAgent] Clarification question sent due to missing cluster: {clarification_question}")
            return 

        # Step 4: take selected cluster and its chunk_ids
        selected_cluster: Optional[Dict[str, Any]] = None
        for cluster in cluster_results:
            if cluster.get("cluster_id") == selected_cluster_id:
                selected_cluster = cluster
                break

        restrict_ids = selected_cluster.get("chunk_ids", []) or []

        # Step 5: retrieve documents within cluster scope
        chunks = self.mcp_client.retrieve_documents(
            query=refined_query,
            query_embedding=query_embedding,
            restrict_ids=restrict_ids,
            top_k=5,
        )
        #print(f"[KnowledgeAgent] Retrieved chunks: {chunks}")

        # Optional fallback: no chunks found even though cluster was valid
        if not chunks:
            stm_context = self._get_stm_context(session_id)
            ltm_context = self._get_ltm_context(user_query, session_id)

            clarification_result = self.mcp_client.generate_clarification(
                refined_query=refined_query,
                stm_context=stm_context,
                ltm_context=ltm_context,
            )

            clarification_question = clarification_result.get(
                "question",
            )

            await self.publish_message(
            FinalAnswerMessage(
                session_id=session_id,
                user_id=message.user_id,
                user_query=user_query,
                answer=clarification_question,
                citations=[],
            ),
            topic_id=TopicId(
                AgenticTopic.FINAL_RESPONSE.value,
                source=self.id.key,
            ),
        )

            print(f"[KnowledgeAgent] Clarification question sent due to no chunks retrieved: {clarification_question}")
            return

        # Step 6: synthesize final answer
        synthesis_result = self.mcp_client.synthesize_answer(
            query=refined_query,
            chunks=chunks,
        )

        answer = synthesis_result.get(
            "answer",
        )
        citations = synthesis_result.get("citations", [])
        print(f"[KnowledgeAgent] Synthesized answer: {answer} with citations: {citations}")

        await self.publish_message(
        FinalAnswerMessage(
            session_id=session_id,
            user_id=message.user_id,
            user_query=user_query,
            answer=answer,
            citations=citations,
        ),
        topic_id=TopicId(
            AgenticTopic.FINAL_RESPONSE.value,
            source=self.id.key,
        ),
    )