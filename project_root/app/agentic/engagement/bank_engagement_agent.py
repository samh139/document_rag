# app/agentic/engagement/bank_engagement_agent.py

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

from app.agentic.topics import AgenticTopic
from app.agentic.messages import (
    BankUserMessage,
    EngagementOutputMessage,
)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logger = logging.getLogger("BankEngagementAgent")
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------
# LLM config
# ------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
ENGAGEMENT_MODEL = os.getenv("ENGAGEMENT_MODEL", "gemma3:4b")

SYSTEM_PROMPT = """
You are a Banking Engagement Agent.

Your task:
- Classify user input into ONE label:
  GREETING | BANK_QUERY | OUT_OF_SCOPE

Rules:
- Banking questions (ATM, interest, KYC, loans, RBI) → BANK_QUERY
- Greetings, thanks, goodbyes → GREETING
- Everything else → OUT_OF_SCOPE

Do NOT answer questions.
Do NOT explain.
Return ONLY ONE WORD.
"""

# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------
@type_subscription(topic_type=AgenticTopic.USER_INPUT.value)
class BankEngagementAgent(RoutedAgent):

    def __init__(self) -> None:
        super().__init__("BankEngagementAgent")

    def _classify(self, text: str) -> str:
        payload = {
            "model": ENGAGEMENT_MODEL,
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
        logger.info(f"[Classifier] Raw intent: {intent}")

        if intent not in {"GREETING", "BANK_QUERY", "OUT_OF_SCOPE"}:
            return "OUT_OF_SCOPE"

        return intent

    def _response_for_intent(self, intent: str) -> str:
        if intent == "GREETING":
            return "Hello! I can help you with banking-related questions."
        if intent == "BANK_QUERY":
            return "Got it — let me check that for you."
        return "I can only assist with banking-related questions."

    @message_handler
    async def handle_user_message(
        self,
        message: BankUserMessage,
        ctx: MessageContext,
    ) -> None:

            print(f"[BankEngagementAgent] Received:", message.content)
            #logger.info(f"[Input] {message}")

            intent = self._classify(message.content)
            response_text = self._response_for_intent(intent)

            output = EngagementOutputMessage(
                intent=intent,
                user_query=message.content,   # ✅ ORIGINAL QUESTION
                response_text=response_text,
                session_id=message.session_id,
                user_id=message.user_id,
            )
            #logger.info(f"[Output] {output}")
            print(f"[BankEngagementAgent] Response:", output)

            await self.publish_message(
                output,
                topic_id=TopicId(
                    AgenticTopic.ENGAGEMENT_OUTPUT.value,
                    source=self.id.key,
                ),
            )
