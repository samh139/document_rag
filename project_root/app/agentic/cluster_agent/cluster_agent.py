#project_root/app/agentic/cluster_agent/cluster_agent.py

import logging
from typing import List, Optional

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

from app.agents.rag.embedding_client import embed_text
from app.memory_team.clustering.vector_loader import get_es_connection


logger = logging.getLogger("ClusterRouterAgent")

# -----------------------------
# Routing thresholds (tunable)
# -----------------------------
MIN_TOP_SCORE = 0.70
MIN_SCORE_GAP = 0.10
MAX_CLUSTER_CHUNKS = 2000
TOP_K = 5


@type_subscription(topic_type=AgenticTopic.REFINED_QUERY_TOPIC.value)
class ClusterRouterAgent(RoutedAgent):

    def __init__(self) -> None:
        super().__init__("ClusterRouterAgent")
        self.es = get_es_connection()

    # -----------------------------
    @message_handler
    async def handle_refined_query(
        self,
        message: RefinedQueryMessage,
        ctx: MessageContext,
    ) -> None:

        logger.info(
            f"[ClusterRouter] Routing query for session={message.session_id}"
        )

        # 1️⃣ Embed query
        query_vector = embed_text(message.refined_query)

        # 2️⃣ Level-1 ANN search
        clusters = self._search_level1_clusters(query_vector)

        # 3️⃣ Decide routing
        routed_message = self._decide_routing(message, clusters)

        # 4️⃣ Publish
        await self.publish_message(
            routed_message,
            topic_id=TopicId(
                AgenticTopic.CLUSTER_ROUTED_QUERY_TOPIC.value,
                source=self.id.key,
            ),
        )

        logger.info(
            f"[ClusterRouter] Published routing "
            f"(restrict_ids={routed_message.restrict_cluster_ids}, "
            f"confidence={routed_message.routing_confidence:.2f})"
        )

    # -----------------------------
    def _search_level1_clusters(self, query_vector: List[float]) -> List[dict]:
        body = {
            "size": TOP_K,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"level": 1}}
                    ],
                    "must": [
                        {
                            "knn": {
                                "field": "vector",
                                "query_vector": query_vector,
                                "k": TOP_K,
                                "num_candidates": 20,
                            }
                        }
                    ]
                }
            }
        }

        res = self.es.search(
            index="clusters_v2",
            body=body,
        )

        hits = res.get("hits", {}).get("hits", [])
        results = []

        for h in hits:
            src = h["_source"]
            results.append({
                "cluster_id": src["cluster_id"],
                "score": float(h["_score"]),
                "chunk_count": src.get("meta_stats", {}).get("chunk_count", 0),
            })

        return results

    # -----------------------------
    def _decide_routing(
        self,
        message: RefinedQueryMessage,
        clusters: List[dict],
    ) -> ClusterRoutedQueryMessage:

        # ❌ No clusters → ambiguous
        if not clusters:
            return self._ambiguous(message, confidence=0.0)

        top = clusters[0]
        second = clusters[1] if len(clusters) > 1 else None

        score_gap = (
            top["score"] - second["score"]
            if second else top["score"]
        )

        confident = (
            top["score"] >= MIN_TOP_SCORE
            and score_gap >= MIN_SCORE_GAP
            and top["chunk_count"] <= MAX_CLUSTER_CHUNKS
        )

        if not confident:
            return self._ambiguous(message, confidence=top["score"])

        # ✅ Confident Level-1 routing
        return ClusterRoutedQueryMessage(
            original_query=message.original_query,
            refined_query=message.refined_query,
            intent=message.intent,
            restrict_cluster_ids=[top["cluster_id"]],
            routing_confidence=top["score"],
            routing_level=1,
            session_id=message.session_id,
            user_id=message.user_id,
        )

    # -----------------------------
    def _ambiguous(
        self,
        message: RefinedQueryMessage,
        confidence: float,
    ) -> ClusterRoutedQueryMessage:

        return ClusterRoutedQueryMessage(
            original_query=message.original_query,
            refined_query=message.refined_query,
            intent=message.intent,
            restrict_cluster_ids=None,
            routing_confidence=confidence,
            routing_level=None,
            session_id=message.session_id,
            user_id=message.user_id,
        )
