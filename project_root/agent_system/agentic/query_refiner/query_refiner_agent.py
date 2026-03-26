from autogen_core import (
    RoutedAgent,
    MessageContext,
    message_handler,
    type_subscription,
    TopicId,
)

from agent_system.agentic.topics import AgenticTopic
from agent_system.agentic.messages import ClassifierOutputMessage, RefinedQueryMessage
from agent_system.agentic.memory_team.stm.store import get_stm_summary
from agent_system.agentic.memory_team.ltm.ltm_service import retrieve_ltm_context
from agent_system.agentic.app.configs.llm_config import fire_fast_modal_request_chat

import logging

logger = logging.getLogger("QueryRefinerAgent")

system_prompt = """
You are a Query Refinement Agent for a regulated-domain RAG system
(Banking / KYC / AML).

Your task is to produce a SINGLE retrieval-ready query.

PRIMARY OBJECTIVE:
- Preserve the user's current intent exactly.
- Make the query explicit, unambiguous, and self-contained only when necessary.

IMPORTANT RULE:
- If the current user query is already explicit, complete, and retrieval-ready,
  return it unchanged.
- Do NOT inject prior conversation details unless the current query is clearly incomplete,
  referential, or depends on context.

Use memory ONLY in these cases:
- The query contains pronouns or vague references such as:
  "this", "that", "it", "they", "those", "what about", "and this", "current?", "charges?"
- The user is clearly continuing the immediately previous topic and the new query is incomplete by itself.

Do NOT use memory when:
- The current query already names the topic clearly
- Adding memory would narrow or expand the scope beyond the current wording
- Prior context is only loosely related

STRICT PROHIBITIONS:
- DO NOT introduce new entities, account types, products, or procedures unless directly required to resolve ambiguity
- DO NOT add historical details from STM/LTM unless the current query depends on them
- DO NOT answer the question
- DO NOT generalize or narrow the scope

OUTPUT RULES:
- Output exactly one query
- No explanation
- No bullets
- No metadata
- No formatting

CRITICAL SAFETY RULE:
- Remove or generalize sensitive identifiers such as account numbers, card numbers, customer IDs, and phone numbers.

If the current query is already explicit and complete, return it unchanged.
"""
# ------------------------------------------------------------------
@type_subscription(topic_type=AgenticTopic.CLASSIFIER_OUTPUT.value)
class QueryRefinerAgent(RoutedAgent):

    def __init__(self):
        super().__init__("QueryRefinerAgent")

    # --------------------------------------------------------------
    @message_handler
    async def handle_message(
        self,
        message: ClassifierOutputMessage,
        ctx: MessageContext
    ) -> None:

        # 🔹 Retrieve STM summary
        stm_summary = get_stm_summary(
            session_id=message.session_id,
            user_id=message.user_id
        )
        print(f"STM Summary from QueryRefinerAgent: {stm_summary}")

        # 🔹 Retrieve LTM context
        ltm_results = retrieve_ltm_context(
            query=message.user_query,
            user_id=message.user_id,
            session_id=message.session_id,
            top_n=3
        )

        user_prompt = f"""
Current User Query:
{message.user_query}

Short-Term Memory Summary:
{stm_summary.get("conversation_summary") or "None"}

Conversation Entities:
{stm_summary.get("conversation_entities") or "None"}

Relevant Long-Term Memory:
{[{"user": r["user_message"], "bot": r["bot_response"]} for r in ltm_results]}

Refined Retrieval Query:
"""

        refined_query = fire_fast_modal_request_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ).strip()

        print(f"Refined Query from QueryRefinerAgent: {refined_query}")

        output = RefinedQueryMessage(
            refined_query=refined_query,
            original_query=message.user_query,
            intent=message.intent,
            session_id=message.session_id,
            user_id=message.user_id,
        )

        await self.publish_message(
            output,
            topic_id=TopicId(
                AgenticTopic.REFINED_QUERY_TOPIC.value,
                source=self.id.key,
            ),
        )

        logger.info("[QueryRefiner] Published RefinedQueryMessage")