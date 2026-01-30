# app/agentic/ui_bridge/ui_bridge_agent.py
import asyncio
import logging
from autogen_core import RoutedAgent, MessageContext, message_handler, type_subscription, TopicId
from app.agentic.topics import AgenticTopic
from app.agentic.messages import ClarificationQuestionMessage, FinalAnswerMessage
from app.agentic.state import response_queue   # your existing queue

logger = logging.getLogger("UIBridgeAgent")

@type_subscription(topic_type=AgenticTopic.CLARIFICATION_QUESTION.value)
@type_subscription(topic_type=AgenticTopic.FINAL_RESPONSE.value)
class UIBridgeAgent(RoutedAgent):
    def __init__(self):
        super().__init__("UIBridgeAgent")

    @message_handler
    async def handle_ui_message(self, message, ctx: MessageContext):
        """
        Forward ClarificationQuestionMessage and FinalAnswerMessage to the response_queue.
        The API (and Streamlit) will read from this queue.
        """
        # Normalize a small JSON-friendly dict to put into the queue
        if isinstance(message, ClarificationQuestionMessage):
            payload = {
                "type": "clarification",
                "question": message.question,
                "session_id": message.session_id,
                "user_id": message.user_id,
            }
        elif isinstance(message, FinalAnswerMessage):
            payload = {
                "type": "final_answer",
                "answer": message.answer,
                "citations": getattr(message, "citations", []),
                "session_id": message.session_id,
                "user_id": message.user_id,
                "user_query": getattr(message, "user_query", None),
            }
        else:
            # unexpected type
            logger.warning("[UIBridge] Received unhandled type: %s", type(message))
            return

        # Non-blocking put, but if queue full will wait a short time
        try:
            await asyncio.wait_for(response_queue.put(payload), timeout=5.0)
            logger.debug("[UIBridge] forwarded message to response_queue: %s", payload["type"])
        except asyncio.TimeoutError:
            logger.error("[UIBridge] response_queue put timed out for session=%s", message.session_id)
