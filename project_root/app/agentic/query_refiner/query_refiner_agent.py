#app/agentic/query_refiner/query_refiner_agent.py

from autogen_core import (
    RoutedAgent,
    MessageContext,
    message_handler,
    type_subscription,
    TopicId,
)

from app.agentic.topics import AgenticTopic
from app.agentic.messages import EngagementOutputMessage, RefinedQueryMessage
from app.memory_team.stm.store import get_stm_summary
from app.memory_team.ltm.ltm_service import retrieve_ltm_context
from app.configs.llm_config import fire_fast_modal_request_chat

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

If the current query is already explicit and complete, return it unchanged.


"""

@type_subscription(topic_type=AgenticTopic.ENGAGEMENT_OUTPUT.value)
class QueryRefinerAgent(RoutedAgent):

    def __init__(self):
        super().__init__("QueryRefinerAgent")

    @message_handler
    async def handle_user_query(
        self,
        message: EngagementOutputMessage,
        ctx: MessageContext,
    ) -> None:

        logger.info(f"[QueryRefiner] Refining query for session={message.session_id}")
        print(f"User Query recieved : {message.user_query}")

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
            user_prompt=user_prompt
        ).strip()

        print(f"Refined Query from QueryRefinerAgent: {refined_query}")

        output = RefinedQueryMessage(
            refined_query=refined_query,
            original_query=message.user_query,
            intent=message.intent,
            user_id=message.user_id,
            session_id=message.session_id,
        )

        await self.publish_message(
            output,
            topic_id=TopicId(
                AgenticTopic.REFINED_QUERY_TOPIC.value,
                source=self.id.key,
            ),
        )

        logger.info(f"[QueryRefiner] Refined query published")



