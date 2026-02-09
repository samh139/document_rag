from autogen_core import TopicId

from app.runtime.runtime_factory import create_runtime
from app.runtime.event_sink import RuntimeEventSink
from app.websockets.session import ConversationSession
from app.agentic.messages import (
    BankUserMessage,
    ClarificationReplyMessage
)
from app.agentic.topics import AgenticTopic


class RuntimeContext:
    """
    Concrete runtime controller.
    Owns:
    - AutoGen runtime
    - Session binding
    - Start / resume control
    """

    def __init__(
        self,
        session: ConversationSession,
        sink: RuntimeEventSink,
    ):
        self.session = session
        self.sink = sink
        self.runtime = create_runtime(sink)

    async def start(self, message: BankUserMessage):
        """
        Start runtime with initial user input.
        """
        self.session.mark_active()

        await self.runtime.publish_message(
            message,
            topic_id=TopicId(
                AgenticTopic.USER_INPUT.value,
                source="websocket",
            )
        )

    async def resume(self, reply: ClarificationReplyMessage):
        """
        Resume runtime after clarification.
        """
        self.session.mark_active()

        await self.runtime.publish_message(
            reply,
            topic_id=TopicId(
                AgenticTopic.CLARIFICATION_REPLY.value,
                source="websocket",
            )
        )
