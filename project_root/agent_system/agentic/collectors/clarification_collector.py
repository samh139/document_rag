from autogen_core import (
    RoutedAgent,
    MessageContext,
    message_handler,
    type_subscription,
)

from agent_system.agentic.topics import AgenticTopic
from agent_system.agentic.messages import ClarificationQuestionMessage


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
        
        print("ClarificationCollector received message")
        
        await self.sink.handle_runtime_result(
            RuntimeResult(
                status="WAIT",
                payload=message.model_dump()
            )
        )
