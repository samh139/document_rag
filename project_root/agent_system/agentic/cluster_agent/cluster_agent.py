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
import requests
from app.memory_team.stm.store import set_clarification_context



OLLAMA_URL = "http://localhost:11434/api/generate"
SUMMARY_ROUTING_MODEL = "gemma3:12b"
OLLAMA_TIMEOUT = 25  # seconds


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
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": TOP_K,
                "num_candidates": 50,
            },
            "query": {
                "term": {"level": 1}
            },
            "_source": ["cluster_id", "summary", "meta_stats"]
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
                "summary": src.get("summary", ""),
            })

        return results
    

    def _decide_routing(
    self,
    message: RefinedQueryMessage,
    clusters: List[dict],
) -> ClusterRoutedQueryMessage:

        # ❌ No clusters → ambiguous
        if not clusters:
            return self._ambiguous(message, confidence=0.0, candidate_clusters=[])

        top = clusters[0]
        second = clusters[1] if len(clusters) > 1 else None

        score_gap = (
            top["score"] - second["score"]
            if second else None
        )

        confident_ann = (
            top["score"] >= MIN_TOP_SCORE
            and top["chunk_count"] <= MAX_CLUSTER_CHUNKS
            and (
                score_gap is None
                or score_gap >= MIN_SCORE_GAP
            )
        )

        if not confident_ann:
            logger.info(
                "[ClusterRouter] ANN not confident → writing clarification context"
            )
            return self._ambiguous(
                message,
                confidence=top["score"],
                candidate_clusters=clusters[:2],
            )
        '''
        # 🔒 Summary-aware routing gate (CORRECT, KEEP THIS)
        summary_allows = self._summary_allows_routing(
            query=message.refined_query,
            top_cluster=top,
            second_cluster=second,
        )

        if not summary_allows:
            logger.info(
                "[ClusterRouter] Summary gate blocked routing → writing clarification context"
            )
            return self._ambiguous(
                message,
                confidence=top["score"],
                candidate_clusters=clusters[:2],
            )
        '''
        # ✅ Confident Level-1 routing
        return ClusterRoutedQueryMessage(
            original_query=message.original_query,
            refined_query=message.refined_query,
            intent=message.intent,
            restrict_cluster_ids=[top["cluster_id"]],
            routing_confidence=top["score"],
            routing_level=1,
            candidate_clusters=None,
            session_id=message.session_id,
            user_id=message.user_id,
        )

    # -----------------------------
    def _ambiguous(
    self,
    message: RefinedQueryMessage,
    confidence: float,
    candidate_clusters: List[dict],
) -> ClusterRoutedQueryMessage:

        return ClusterRoutedQueryMessage(
            original_query=message.original_query,
            refined_query=message.refined_query,
            intent=message.intent,
            restrict_cluster_ids=None,
            routing_confidence=confidence,
            routing_level=None,
            candidate_clusters=candidate_clusters,
            session_id=message.session_id,
            user_id=message.user_id,
        )

    

    def _summary_allows_routing(
    self,
    query: str,
    top_cluster: dict,
    second_cluster: Optional[dict],
) -> bool:
        """
        Uses LLM to decide whether routing to the top cluster
        is safe WITHOUT clarification.

        Safety rule:
        - On ANY error or uncertainty → return False (AMBIGUOUS)
        """

        top_summary = top_cluster.get("summary", "").strip()
        second_summary = (
            second_cluster.get("summary", "").strip()
            if second_cluster
            else "None"
        )

        prompt = f"""
    You are a routing safety judge in a question-answering system.

    Your job is to decide whether a user's question can be safely answered
    using ONLY the TOP cluster summary provided, WITHOUT asking a clarification question.

    You must follow these rules strictly:

    ROUTE (output: CLEAR) ONLY IF:
    - The user's question is specific
    - The TOP cluster summary clearly and completely covers the intent
    - No reasonable alternative interpretation exists

    DO NOT ROUTE (output: AMBIGUOUS) IF:
    - The question is broad or underspecified
    - Multiple interpretations are possible
    - The SECOND cluster summary could also reasonably answer the question
    - A clarification question would improve answer accuracy

    You are NOT allowed to:
    - Explain your reasoning
    - Suggest clarifications
    - Mention clusters or scores
    - Output anything except CLEAR or AMBIGUOUS

    ---

    Few-shot examples:

    Example 1:
    User question: "What documents are required for KYC?"
    Top cluster summary: "KYC requirements including identity proof and address proof"
    Second cluster summary: "Account opening procedures"

    Output: CLEAR

    ---

    Example 2:
    User question: "What are the charges?"
    Top cluster summary: "ATM fees, SMS charges, and account maintenance charges"
    Second cluster summary: "Debit and credit card fees"

    Output: AMBIGUOUS

    ---

    Example 3:
    User question: "How do I update my address?"
    Top cluster summary: "Procedures for updating customer address in bank records"
    Second cluster summary: "KYC verification requirements"

    Output: CLEAR

    ---

    Example 4:
    User question: "Tell me about bank rules"
    Top cluster summary: "General banking policies and procedures"
    Second cluster summary: "Regulatory compliance and audits"

    Output: AMBIGUOUS

    ---

    Now evaluate the following case:

    User question:
    "{query}"

    Top cluster summary:
    "{top_summary}"

    Second cluster summary:
    "{second_summary}"

    Output:
    """.strip()

        logger.debug("[SummaryGate] Prompt sent to LLM:\n%s", prompt)

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": SUMMARY_ROUTING_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 5,
                    },
                },
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
        except Exception as e:
            logger.exception(
                "[SummaryGate] LLM request failed — defaulting to AMBIGUOUS"
            )
            return False

        try:
            data = response.json()
            raw_output = data.get("response", "").strip().upper()
        except Exception:
            logger.exception(
                "[SummaryGate] Failed to parse LLM response — defaulting to AMBIGUOUS"
            )
            return False

        logger.info(f"[SummaryGate] Raw LLM output: '{raw_output}'")

        if raw_output == "CLEAR":
            logger.info("[SummaryGate] Verdict = CLEAR → routing allowed")
            return True

        if raw_output == "AMBIGUOUS":
            logger.info("[SummaryGate] Verdict = AMBIGUOUS → clarification required")
            return False

        logger.warning(
            "[SummaryGate] Unexpected LLM output '%s' — defaulting to AMBIGUOUS",
            raw_output,
        )
        return False