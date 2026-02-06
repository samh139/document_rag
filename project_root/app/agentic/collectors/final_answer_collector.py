import asyncio
from autogen_core import (
    RoutedAgent,
    MessageContext,
    message_handler,
    type_subscription,
)

from app.agentic.topics import AgenticTopic
from app.agentic.messages import FinalAnswerMessage
from app.memory_team.stm.agent import store_conversation_to_stm
from app.memory_team.ltm.ltm_service import store_conversation_to_es

from app.runtime.runtime_result import RuntimeResult
from app.runtime.event_sink import RuntimeEventSink


@type_subscription(topic_type=AgenticTopic.FINAL_RESPONSE.value)
class FinalAnswerCollector(RoutedAgent):

    def __init__(self, sink: RuntimeEventSink):
        super().__init__("FinalAnswerCollector")
        self.sink = sink

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

    @message_handler
    async def handle_final_answer(
        self,
        message: FinalAnswerMessage,
        ctx: MessageContext,
    ) -> None:
        # 🔁 REPLACEMENT FOR response_queue.put()
        await self.sink.handle_runtime_result(
            RuntimeResult(
                status="COMPLETE",
                payload=message.dict()
            )
        )

        # 🧠 fire-and-forget persistence
        asyncio.create_task(
            self._persist_memory(message),
            name=f"persist_memory:{message.session_id}"
        )
