from autogen_core import (
    RoutedAgent,
    MessageContext,
    message_handler,
    type_subscription,
)

from app.agentic.topics import AgenticTopic
from app.agentic.messages import ClarificationQuestionMessage

from app.runtime.runtime_result import RuntimeResult
from app.runtime.event_sink import RuntimeEventSink


@type_subscription(
    topic_type=AgenticTopic.CLARIFICATION_REQUIRED.value
)
class ClarificationCollector(RoutedAgent):

    def __init__(self, sink: RuntimeEventSink):
        super().__init__("ClarificationCollector")
        self.sink = sink

    @message_handler
    async def handle_clarification(
        self,
        message: ClarificationQuestionMessage,
        ctx: MessageContext,
    ) -> None:
        await self.sink.handle_runtime_result(
            RuntimeResult(
                status="WAIT",
                payload=message.model_dump()
            )
        )
