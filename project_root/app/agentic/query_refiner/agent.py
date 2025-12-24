from app.agentic.messages import (
    EngagementOutputMessage,
    RAGRetrievalResultMessage,
    FinalAnswerMessage,
)
from app.agentic.topics import AgenticTopic
from app.memory.memory_manager import MemoryManager
from app.agentic.query_refiner.decision import QueryDecision


class QueryRefinerAgent:

    def __init__(self, runtime):
        self.runtime = runtime
        self.memory = MemoryManager()

    async def handle(self, message: EngagementOutputMessage):
        """
        Entry point triggered by ENGAGEMENT_OUTPUT_TOPIC
        """

        role = "user"   # later from ACL
        user_id = message.user_id
        session_id = message.session_id
        query = message.user_query

        # 1️⃣ Fetch STM summary
        stm_summary = self.memory.get_stm_summary_cached(
            role=role,
            user_id=user_id,
            session_id=session_id,
        )

        # 2️⃣ Decide
        decision = self._decide(query, stm_summary)

        # 3️⃣ Act
        if decision.use_memory:
            return await self._answer_from_memory(
                message, stm_summary
            )

        if decision.use_rag:
            return await self._trigger_rag(message)

    # -------------------------
    # DECISION LOGIC (PHASE 2A)
    # -------------------------
    def _decide(self, query: str, stm_summary: dict | None) -> QueryDecision:

        if stm_summary and stm_summary.get("context_summary"):
            return QueryDecision(
                use_memory=True,
                use_rag=False,
                reason="Relevant STM summary exists",
            )

        return QueryDecision(
            use_memory=False,
            use_rag=True,
            reason="No sufficient memory context",
        )

    # -------------------------
    # ACTIONS
    # -------------------------
    async def _answer_from_memory(
        self,
        message: EngagementOutputMessage,
        stm_summary: dict,
    ):
        response = FinalAnswerMessage(
            answer=stm_summary["context_summary"],
            citations=[],
            session_id=message.session_id,
            user_id=message.user_id,
        )

        await self.runtime.publish(
            AgenticTopic.FINAL_RESPONSE,
            response,
        )

    async def _trigger_rag(
        self,
        message: EngagementOutputMessage,
    ):
        rag_msg = RAGRetrievalResultMessage(
            query=message.user_query,
            chunks=[],
            session_id=message.session_id,
            user_id=message.user_id,
        )

        await self.runtime.publish(
            AgenticTopic.RAG_RETRIEVAL_OUTPUT,
            rag_msg,
        )
