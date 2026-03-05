import logging
from autogen_core import (
    RoutedAgent,
    MessageContext,
    message_handler,
    type_subscription,
)

from agent_system.agentic.topics import AgenticTopic
from agent_system.agentic.messages import ClassifierOutputMessage

logger = logging.getLogger("ClassifierOutputLogger")
logging.basicConfig(level=logging.INFO)


@type_subscription(topic_type=AgenticTopic.CLASSIFIER_OUTPUT.value)
class ClassifierOutputLogger(RoutedAgent):

    def __init__(self) -> None:
        super().__init__("ClassifierOutputLogger")

    @message_handler
    async def handle_output(
        self,
        message: ClassifierOutputMessage,
        ctx: MessageContext,
    ) -> None:
        logger.info(
            f"[CLASSIFIER OUTPUT] "
            f"intent={message.intent} | "
            f"response={message.response_text} | "
            f"session={message.session_id}"
        )
