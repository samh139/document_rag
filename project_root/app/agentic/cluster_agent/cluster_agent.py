#project_root/app/agentic/cluster_agent/cluster_agent.py

import logging

from autogen_core import (
    RoutedAgent,
    MessageContext,
    message_handler,
    type_subscription,
    TopicId,
)

from app.agentic.topics import AgenticTopic
from app.agentic.messages import (
    RefinedQueryMessage,
    ClusterRoutedQueryMessage,
)

logger = logging.getLogger("ClusterRouterAgent")


@type_subscription(topic_type=AgenticTopic.REFINED_QUERY_TOPIC.value)
class ClusterRouterAgent(RoutedAgent):

    def __init__(self) -> None:
        super().__init__("ClusterRouterAgent")

    @message_handler
    async def handle_refined_query(
        self,
        message: RefinedQueryMessage,
        ctx: MessageContext,
    ) -> None:

        logger.info(
            f"[ClusterRouter] Routing query for session={message.session_id}"
        )

        # 🚧 PHASE 1: pass-through (no clustering yet)
        routed = ClusterRoutedQueryMessage(
            original_query=message.original_query,
            refined_query=message.refined_query,
            intent=message.intent,
            restrict_ids=None,   # IMPORTANT: safe default
            session_id=message.session_id,
            user_id=message.user_id,
        )

        await self.publish_message(
            routed,
            topic_id=TopicId(
                AgenticTopic.CLUSTER_ROUTED_QUERY_TOPIC.value,
                source=self.id.key,
            ),
        )

        logger.info(
            f"[ClusterRouter] Published routed query (no cluster restriction)"
        )
