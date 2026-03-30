from fastapi import FastAPI, HTTPException

from agent_system.metrics_evaluator.schemas import MetricsEvaluationInputItem


from fastapi import APIRouter, HTTPException, Query
from typing import Any, Dict, List

from agent_system.agentic.utils.es.es_utils import get_es_connection


from agent_system.metrics_evaluator.rag_metrics_evaluator import (
    MetricsEvaluationInputItem,
    evaluate_feedback_on_metrics,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("metrics_api")

API_PREFIX = "/api/v1"
AGENT_RESPONSES_INDEX = "agent_responses"

metrics_router = APIRouter()


def _fetch_agent_response_by_query_id(query_id: str) -> Dict[str, Any]:
    es = get_es_connection()

    query = {
        "size": 1,
        "query": {
            "term": {
                "query_id": query_id
            }
        }
    }

    res = es.search(
        index=AGENT_RESPONSES_INDEX,
        body=query
    )

    hits = res.get("hits", {}).get("hits", [])
    if not hits:
        raise HTTPException(
            status_code=404,
            detail=f"No agent response found for query_id={query_id}"
        )

    return hits[0]["_source"]


def _build_metrics_input_item(doc: Dict[str, Any]) -> MetricsEvaluationInputItem:
    citations = doc.get("citations", []) or []

    retrieved_contexts: List[str] = [
        citation.get("chunk_content", "")
        for citation in citations
        if citation.get("chunk_content")
    ]

    return MetricsEvaluationInputItem(
        user_input=doc.get("user_input", ""),
        response=doc.get("bot_response", ""),
        retrieved_contexts=retrieved_contexts,
    )


@metrics_router.get(f"{API_PREFIX}/metrics/by_query_id")
def evaluate_metrics_by_query_id(query_id: str = Query(...)):
    try:
        print(f"[MetricsAPI] Fetching agent response for query_id: {query_id}")

        doc = _fetch_agent_response_by_query_id(query_id)
        print(f"[MetricsAPI] ES document fetched for query_id: {query_id}")

        metrics_input = _build_metrics_input_item(doc)
        print(f"[MetricsAPI] Metrics input built for query_id: {query_id}")

        result = evaluate_feedback_on_metrics(metrics_input)

        return {
            "query_id": query_id,
            "session_id": doc.get("session_id"),
            "user_input": doc.get("user_input", ""),
            "bot_response": doc.get("bot_response", ""),
            **result,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Error evaluating metrics for query_id={query_id}: {str(e)}"
        )