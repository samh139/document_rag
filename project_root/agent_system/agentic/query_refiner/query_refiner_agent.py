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

Your task is to produce exactly ONE retrieval-ready query.

PRIMARY OBJECTIVE:
- Preserve the user's current intent exactly.
- Make the query explicit, self-contained, and retrieval-ready.

CORE RULE:
- If the current user query is already standalone, explicit, and complete, return it unchanged.
- If the current user query is a follow-up, continuation, or reference to the previous topic, rewrite it into a complete standalone query using the recent conversation context.

YOU MUST USE RECENT CONTEXT WHEN THE QUERY IS A FOLLOW-UP.
Treat the query as a follow-up if it contains language like:
- "anything specific to ..."
- "what about ..."
- "how about ..."
- "and for ..."
- "for SBI?"
- "for current?"
- "charges?"
- "anything else?"
- "specific to ..."
- "in SBI?"
- "for savings?"
- short elliptical questions that depend on the previous turn

FOLLOW-UP RESOLUTION RULE:
- When the current query depends on the previous turn, combine the previous topic with the new qualifier.
- Preserve both:
  1. the previous topic
  2. the new qualifier / modifier from the current turn

EXAMPLES:
Current Query: "What are the instructions  digital lending"
If No Previous Query, return the clean refined query with same context
Output: "What are the instructions on digital lending ?"

Previous Query: "What are ATM charges for SBI?"
Current Query: "for salary accounts?"
Output: "What are the ATM charges for SBI salary accounts?"

DO NOT USE MEMORY WHEN:
- the current query is already complete by itself
- the previous context is unrelated
- adding past context would distort or over-expand the query

STRICT PROHIBITIONS:
- DO NOT answer the question
- DO NOT add recommendations or explanations
- DO NOT invent missing details beyond what is needed to resolve the follow-up
- DO NOT pull in unrelated history
- DO NOT over-expand the query with extra account types, dates, policies, or details unless the user explicitly referred to them

OUTPUT RULES:
- Output exactly one query
- No explanation
- No bullets
- No metadata
- No formatting

CRITICAL SAFETY RULE:
- Remove or generalize sensitive identifiers such as account numbers, card numbers, customer IDs, and phone numbers.

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
            query_id=message.query_id,
        )

        await self.publish_message(
            output,
            topic_id=TopicId(
                AgenticTopic.REFINED_QUERY_TOPIC.value,
                source=self.id.key,
            ),
        )

        logger.info("[QueryRefiner] Published RefinedQueryMessage")