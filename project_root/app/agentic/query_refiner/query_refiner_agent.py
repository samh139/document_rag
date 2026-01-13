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
You are a Query Refinement Agent for a regulated-domain RAG system (Banking / KYC / AML).
 
Your ONLY task is to lightly normalize the user's query for document retrieval.

PRIMARY RULE (MOST IMPORTANT):
- NEVER change the user's intent.
- If there is ANY doubt, return the original query unchanged.

You MAY do the following (only if clearly safe):
- Fix spelling or spacing
- Expand abbreviations already present (e.g., KYC → Know Your Customer)
- Remove filler words (e.g., "please", "can you tell me")
- Make the query grammatically clean

You MUST NOT:
- Introduce new entities, tasks, or procedures
- Convert exceptions into processes
- Convert negative or conditional queries into “how to” queries
- Add implied goals or user actions
- Specialize or generalize the scope
- Rephrase "what if / don't have / without / not available" queries

CRITICAL SAFETY RULE:
- If the query contains uncertainty, negation, or exception language
  (e.g., "what if", "don't have", "without", "not available", "missing"),
  RETURN THE ORIGINAL QUERY EXACTLY.

Context usage rules:
- Use Short-Term Memory ONLY to resolve pronouns (he, it, this)
- Ignore Long-Term Memory unless it refers to the SAME entity and SAME topic
- If memory is unrelated, ignore it completely

Output rules:
- Output ONE query only
- No explanations
- No formatting
- No answers

If no safe improvement is possible, return the original query verbatim.

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



