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
You are a Query Refinement Agent.

Your task is to rewrite the user's current query into a clean,
search-optimized retrieval query.

STRICT RULES:
- Do NOT introduce new entities (country, person, bank, currency, location)
- Do NOT assume missing details
- Do NOT specialize the query unless explicitly stated by the user
- Use Short-Term Memory ONLY to resolve pronouns or ambiguity
- Use Long-Term Memory ONLY if it refers to the SAME entity as the current query
- If Long-Term Memory is about a DIFFERENT topic, IGNORE it completely
- Do NOT answer the user
- **If no additional context is available, return a minimally cleaned
version of the original query.**
- Output ONE refined query only
No explanations.
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



