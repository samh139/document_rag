import logging
from autogen_core import (
    RoutedAgent,
    MessageContext,
    message_handler,
    type_subscription,
)

from app.agentic.topics import AgenticTopic
from app.agentic.messages import EngagementOutputMessage

logger = logging.getLogger("EngagementOutputLogger")
logging.basicConfig(level=logging.INFO)


@type_subscription(topic_type=AgenticTopic.ENGAGEMENT_OUTPUT.value)
class EngagementOutputLogger(RoutedAgent):

    def __init__(self) -> None:
        super().__init__("EngagementOutputLogger")

    @message_handler
    async def handle_output(
        self,
        message: EngagementOutputMessage,
        ctx: MessageContext,
    ) -> None:
        logger.info(
            f"[ENGAGEMENT OUTPUT] "
            f"intent={message.intent} | "
            f"response={message.response_text} | "
            f"session={message.session_id}"
        )
