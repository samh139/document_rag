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
        


@type_subscription(topic_type=AgenticTopic.FINAL_RESPONSE.value)
class FinalAnswerCollector(RoutedAgent):

    def __init__(self, pending_requests: dict):
        super().__init__("FinalAnswerCollector")
        self.pending_requests = pending_requests

    async def _persist_memory(self, message: FinalAnswerMessage):
        try:
            # STM (assumed async)
            await store_conversation_to_stm(
                user_id=message.user_id,
                session_id=message.session_id,
                user_message=message.user_query,
                bot_response=message.answer,
            )

            # LTM (sync → thread)
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


    @message_handler
    async def handle_final_answer(
        self,
        message: FinalAnswerMessage,
        ctx: MessageContext,
    ) -> None:

        future = self.pending_requests.get(message.session_id)

        if future and not future.done():
            future.set_result(message)

        asyncio.create_task(
            self._persist_memory(message),
            name=f"persist_memory:{message.session_id}"
        )

        