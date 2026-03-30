import asyncio
from autogen_core import (
    RoutedAgent,
    MessageContext,
    message_handler,
    type_subscription,
)

from agent_system.agentic.topics import AgenticTopic
from agent_system.agentic.messages import FinalAnswerMessage
from agent_system.agentic.memory_team.stm.agent import store_conversation_to_stm
from agent_system.agentic.memory_team.ltm.ltm_service import store_conversation_to_es
from agent_system.metrics_evaluator.rag_results_store_service import RAGResultsStoreService


@type_subscription(topic_type=AgenticTopic.FINAL_RESPONSE.value)
class FinalAnswerCollector(RoutedAgent):

    def __init__(self, pending_requests: dict):
        super().__init__("FinalAnswerCollector")
        self.pending_requests = pending_requests
        self.rag_results_store_service = RAGResultsStoreService()


    async def _persist_memory(self, message: FinalAnswerMessage):
        try:
            await store_conversation_to_stm(
                user_id=message.user_id,
                session_id=message.session_id,
                user_message=message.user_query,
                bot_response=message.answer,
            )

            await asyncio.to_thread(
                store_conversation_to_es,
                message.user_id,
                message.session_id,
                message.user_query,
                message.answer,
            )

            print(f"[Memory] STM + LTM stored for session {message.session_id}")

        except Exception as e:
            print(f"[Memory ERROR] session={message.session_id} err={e}")
    
    async def _persist_agent_response(self, message: FinalAnswerMessage):
        try:
            payload = {
                "session_id": message.session_id,
                "query_id": message.query_id,
                "user_id": message.user_id,
                "user_input": message.user_query,
                "bot_response": message.answer,
                "citations": [citation.model_dump() for citation in message.citations],
            }

            await self.rag_results_store_service.store_agent_response(payload)
            print(f"[AgentResponse] Stored agent response for query_id {message.query_id}")

        except Exception as e:
            print(f"[AgentResponse ERROR] query_id={message.query_id} err={e}")


    @message_handler
    async def handle_final_answer(
        self,
        message: FinalAnswerMessage,
        ctx: MessageContext,
    ) -> None:
        
        print(f"FinalAnswerCollector received message for session {message.session_id}")

        # Remove request from pending map
        future = self.pending_requests.pop(message.session_id, None)

        if future and not future.done():
            future.set_result(message)
        else:
            print(f"[Collector] No pending request for session {message.session_id}")

        asyncio.create_task(
            self._persist_memory(message),
            name=f"persist_memory:{message.session_id}"
        )
        asyncio.create_task(
            self._persist_agent_response(message),
            name=f"persist_agent_response:{message.query_id}"
        )

        print(f"FinalAnswerCollector resolved pending request for session {message.session_id}")