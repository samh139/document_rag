# app/agentic/clarification/clarification_agent.py

import logging
import requests

from autogen_core import (
    RoutedAgent,
    MessageContext,
    message_handler,
    type_subscription,
    TopicId,
)

from app.agentic.topics import AgenticTopic
from app.agentic.messages import (
    ClusterRoutedQueryMessage,
    ClarificationQuestionMessage,
)

logger = logging.getLogger("ClarificationAgent")

OLLAMA_URL = "http://localhost:11434/api/generate"
CLARIFICATION_MODEL = "gemma3:12b"
TIMEOUT = 20


@type_subscription(topic_type=AgenticTopic.CLUSTER_ROUTED_QUERY_TOPIC.value)
class ClarificationAgent(RoutedAgent):

    def __init__(self) -> None:
        super().__init__("ClarificationAgent")

    @message_handler
    async def handle_ambiguous_query(
        self,
        message: ClusterRoutedQueryMessage,
        ctx: MessageContext,
    ) -> None:

        # Only handle ambiguous cases
        if message.restrict_cluster_ids is not None:
            return

        clusters = message.candidate_clusters or []
        if len(clusters) < 2:
            logger.warning("[Clarification] Not enough clusters to clarify")
            return

        summary_1 = clusters[0].get("summary", "")
        summary_2 = clusters[1].get("summary", "")

        prompt = f"""
You are a clarification question generator in a banking QA system.

Your job is to ask ONE precise question that will remove ambiguity
from the user's original query.

You must follow these rules strictly:

ALLOWED:
- Ask exactly ONE question
- The question must be short and specific
- The question must reference concrete options when possible

NOT ALLOWED:
- Do NOT answer the user's question
- Do NOT explain why clarification is needed
- Do NOT mention documents, clusters, summaries, or models
- Do NOT ask multiple questions
- Do NOT use generic phrases like "please clarify"

---

User's question:
"{message.refined_query}"

Possible meanings:
1) {summary_1}
2) {summary_2}

Ask ONE clarification question:
""".strip()

        logger.debug("[Clarification] Prompt:\n%s", prompt)

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": CLARIFICATION_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            question = resp.json().get("response", "").strip()
        except Exception:
            logger.exception("[Clarification] LLM failure")
            return


        output = ClarificationQuestionMessage(
            question=question,
            session_id=message.session_id,
            user_id=message.user_id,
        )

        await self.publish_message(
            output,
            topic_id=TopicId(
                AgenticTopic.CLARIFICATION_QUESTION.value,
                source=self.id.key,
            ),
        )

        logger.info(
            "[Clarification] Question published for session=%s",
            message.session_id,
        )
