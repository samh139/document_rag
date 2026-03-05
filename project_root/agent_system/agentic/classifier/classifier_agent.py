import os
import requests
import logging

from autogen_core import (
    RoutedAgent,
    MessageContext,
    TopicId,
    message_handler,
    type_subscription,
)

from agent_system.agentic.topics import AgenticTopic
from agent_system.agentic.messages import (
    UserMessage,
    ClassifierOutputMessage,
    FinalAnswerMessage,
    ClarificationReplyMessage,   # ✅ NEW
)

from agent_system.agentic.memory_team.stm.store import get_clarification_context

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logger = logging.getLogger("ClassifierAgent")
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------
# LLM config
# ------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "gemma3:4b")

SYSTEM_PROMPT = """
You are a Classifer Agent.

Classify user input into ONE label:
GREETING | BANK_QUERY | OUT_OF_SCOPE

Return ONLY ONE WORD.
"""

# ------------------------------------------------------------------
@type_subscription(topic_type=AgenticTopic.USER_INPUT.value)
class ClassifierAgent(RoutedAgent):

    def __init__(self) -> None:
        super().__init__("Classifier Agent")

    # --------------------------------------------------------------
    def _classify(self, text: str) -> str:
        payload = {
            "model": CLASSIFIER_MODEL,
            "prompt": f"{SYSTEM_PROMPT}\nUser input:\n{text}",
            "stream": False,
        }

        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()

        intent = resp.json().get("response", "").strip().upper()
        return intent if intent in {"GREETING", "BANK_QUERY"} else "OUT_OF_SCOPE"

    # --------------------------------------------------------------
    @message_handler
    async def handle_user_message(
        self,
        message: UserMessage,
        ctx: MessageContext,
    ) -> None:

        print("ClassifierAgent received message")

        logger.info(f"[Classifier] User said: {message.content}")

        # 🔥 STEP 1: Check clarification context FIRST
        clarification = get_clarification_context(
            session_id=message.session_id,
            user_id=message.user_id,
        )

        if clarification and clarification.get("active", False):
            logger.info("[Classifier] Clarification reply detected")

            reply = ClarificationReplyMessage(
                content=message.content,
                session_id=message.session_id,
                user_id=message.user_id,
            )

            await self.publish_message(
                reply,
                topic_id=TopicId(
                    AgenticTopic.CLARIFICATION_REPLY.value,
                    source=self.id.key,
                ),
            )
            return

        # 🔹 STEP 2: Normal classification
        intent = self._classify(message.content)

        if intent in {"GREETING", "OUT_OF_SCOPE"}:
            final = FinalAnswerMessage(
                answer=(
                    "Hello! 👋 I can help with banking-related questions."
                    if intent == "GREETING"
                    else "I currently support banking-related queries only."
                ),
                citations=[],
                session_id=message.session_id,
                user_id=message.user_id,
                user_query=message.content,
            )

            await self.publish_message(
                final,
                topic_id=TopicId(
                    AgenticTopic.FINAL_RESPONSE.value,
                    source=self.id.key,
                ),
            )
            return

        # 🔹 STEP 3: BANK_QUERY → QueryRefiner
        output = ClassifierOutputMessage(
            intent=intent,
            user_query=message.content,
            response_text="Let me check that for you.",
            session_id=message.session_id,
            user_id=message.user_id,
        )

        await self.publish_message(
            output,
            topic_id=TopicId(
                AgenticTopic.ENGAGEMENT_OUTPUT.value,
                source=self.id.key,
            ),
        )
        logger.info("[Classifier] Published ClassifierOutputMessage")