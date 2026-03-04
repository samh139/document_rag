#app/agentic/query_refiner/query_refiner_agent.py

from autogen_core import (
    RoutedAgent,
    MessageContext,
    message_handler,
    type_subscription,
    TopicId,
)

from app.agentic.topics import AgenticTopic
from app.agentic.messages import EngagementOutputMessage, RefinedQueryMessage, ClarificationReplyMessage
from app.memory_team.stm.store import get_stm_summary , get_clarification_context, clear_clarification_context
from app.memory_team.ltm.ltm_service import retrieve_ltm_context
from app.configs.llm_config import fire_fast_modal_request_chat
from typing import Union

import logging

logger = logging.getLogger("QueryRefinerAgent")

system_prompt = """
You are a Query Refinement Agent for a regulated-domain RAG system
(Banking / KYC / AML).

Your task is to produce a SINGLE, retrieval-ready query that accurately
represents the user's current intent IN CONTEXT of the conversation.

PRIMARY OBJECTIVE:
- Preserve the user's intent exactly.
- Make the query explicit, unambiguous, and self-contained for document retrieval.

You MUST use conversation context when it is clearly relevant.

Allowed operations (ONLY when supported by context):
- Resolve pronouns or references using Short-Term Memory (e.g., "this", "that", "it")
- Carry forward entities, documents, or topics mentioned earlier
- Merge the current query with prior turns if the user is continuing the same topic
- Expand abbreviations already present (e.g., KYC → Know Your Customer)
- Remove filler words and conversational phrasing

STRICT PROHIBITIONS:
- DO NOT introduce new entities, tasks, procedures, or assumptions
- DO NOT change the question type (e.g., what → how, why → steps)
- DO NOT add recommendations, analysis, or answers
- DO NOT generalize or narrow the scope beyond what the user intended
- DO NOT invent missing details

UNCERTAINTY & EXCEPTIONS:
- If the user asks a conditional, negative, or exception-based question
  (e.g., "what if", "without", "missing", "not available"),
  you MUST preserve that condition explicitly in the refined query.

MEMORY USAGE RULES:
- Short-Term Memory (STM):
  Use to resolve references and continue the same discussion thread.
- Long-Term Memory (LTM):
  Use ONLY if it clearly refers to the same domain topic or recurring user goal.
  If unrelated, ignore it completely.

OUTPUT RULES (CRITICAL):
- Output ONE single query only
- No explanations
- No formatting
- No bullet points
- No metadata

CRITICAL SAFETY RULE:
- Remove or generalize sensitive identifiers such as:
  account numbers, card numbers, customer IDs, phone numbers
- Replace them with generic terms (e.g., "the account")

If the current query is already explicit and complete, return it unchanged.


"""

# ------------------------------------------------------------------
# SYSTEM PROMPT (B3 – strict disambiguation)
# ------------------------------------------------------------------
CLARIFICATION_PROMPT = """
You are resolving a clarification reply.

Original ambiguous question:
{original_query}

Possible interpretations:
{candidate_summaries}

User clarification reply:
{reply}

Task:
- Resolve the user's intended meaning
- Produce ONE explicit, retrieval-ready query
- Do NOT ask questions
- Do NOT explain
"""

# ------------------------------------------------------------------
@type_subscription(topic_type=AgenticTopic.ENGAGEMENT_OUTPUT.value)
@type_subscription(topic_type=AgenticTopic.CLARIFICATION_REPLY.value)
class QueryRefinerAgent(RoutedAgent):

    def __init__(self):
        super().__init__("QueryRefinerAgent")

    # --------------------------------------------------------------
    @message_handler
    async def handle_message(self, message :Union[EngagementOutputMessage, ClarificationReplyMessage], ctx: MessageContext) -> None:

        if isinstance(message, ClarificationReplyMessage):
            clarification = get_clarification_context(
                session_id=message.session_id,
                user_id=message.user_id,
            )

            if not clarification or not clarification.get("active", False):
                logger.warning("Clarification reply without active context — ignoring")
                return

            if not clarification.get("candidate_clusters"):
                logger.warning("Clarification context missing candidate_clusters — ignoring")
                clear_clarification_context(message.session_id, message.user_id)
                return

            candidate_summaries = [
                f"- {c['summary']}" for c in clarification["candidate_clusters"]
            ]

            prompt = CLARIFICATION_PROMPT.format(
                original_query=clarification["original_query"],
                candidate_summaries="\n".join(candidate_summaries),
                reply=message.content,
            )

            refined_query = fire_fast_modal_request_chat(
                system_prompt="",
                user_prompt=prompt,
            ).strip()

            clear_clarification_context(
                session_id=message.session_id,
                user_id=message.user_id,
            )

            output = RefinedQueryMessage(
                refined_query=refined_query,
                original_query=clarification["original_query"],
                intent="BANK_QUERY",
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
            return

        # 🔹 Normal engagement flow
        # Retrieve STM summary
        stm_summary = get_stm_summary(
            session_id=message.session_id,
            user_id=message.user_id
        )
        print(f"STM Summary from QueryRefinerAgent: {stm_summary}")

        # Retrieve LTM context
        ltm_results = retrieve_ltm_context(
            query=message.user_query,
            user_id=message.user_id,
            session_id=message.session_id,
            top_n=3
        )
        print(f"LTM Results from QueryRefinerAgent: {ltm_results}")

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
