from autogen_core import TopicId , SingleThreadedAgentRuntime

from app.runtime.runtime_factory import create_runtime
from app.runtime.event_sink import RuntimeEventSink
#from app.websockets.session import ConversationSession

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        session: "ConversationSession",
        sink: RuntimeEventSink,
    ):
        self.session = session
        self.sink = sink
        self.runtime : SingleThreadedAgentRuntime | None = None

    async def initialize(self, session):
        self.runtime = await create_runtime(self.sink)
        await self.runtime.start()
        

    async def start(self, message: BankUserMessage):
        """
        Start runtime with initial user input.
        """
        self.session.mark_active()

        if not self.runtime:
            await self.initialize()

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

        if not self.runtime:
            await self.initialize()

        await self.runtime.publish_message(
            reply,
            topic_id=TopicId(
                AgenticTopic.CLARIFICATION_REPLY.value,
                source="websocket",
            )
        )
