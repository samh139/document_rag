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


@type_subscription(topic_type=AgenticTopic.FINAL_RESPONSE.value)
class FinalAnswerCollector(RoutedAgent):

    def __init__(self, queue: asyncio.Queue):
        super().__init__("FinalAnswerCollector")
        self.queue = queue

    @message_handler
    async def handle_final_answer(
        self,
        message: FinalAnswerMessage,
        ctx: MessageContext,
    ) -> None:
        await self.queue.put(message)

    # ✅ Store STM asynchronously (NON-BLOCKING)
        asyncio.create_task(
            store_conversation_to_stm(
                user_id=message.user_id,
                session_id=message.session_id,
                user_message=message.user_query,
                bot_response=message.answer,
            )
        )
