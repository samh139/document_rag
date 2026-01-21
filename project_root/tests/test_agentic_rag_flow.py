import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from app.agentic.messages import (
    EngagementOutputMessage,
    RefinedQueryMessage,
    ClusterRoutedQueryMessage,
    ClarificationQuestionMessage,
    RAGRetrievalResultMessage,
    FinalAnswerMessage,
)

from app.agentic.query_refiner.query_refiner_agent import QueryRefinerAgent
from app.agentic.clarification.clarification_agent import ClarificationAgent
from app.agentic.rag.bank_rag_retrieval_agent import BankRAGRetrievalAgent
from app.agentic.rag.bank_rag_synthesis_agent import BankRAGSynthesisAgent
from app.agentic.collectors.final_answer_collector import FinalAnswerCollector


@pytest.mark.asyncio
async def test_ambiguous_query_triggers_clarification():
    """
    GIVEN an ambiguous banking query
    WHEN it flows through ClusterRouter
    THEN ClarificationAgent should ask a question
    AND RAG should NOT be called
    """

    session_id = "s1"
    user_id = "u1"

    # Fake ambiguous routed message
    msg = ClusterRoutedQueryMessage(
        refined_query="What charges apply?",
        intent="BANK_QUERY",
        restrict_cluster_ids=None,
        candidate_clusters=[
            {"id": "fees", "summary": "ATM, SMS, account charges"},
            {"id": "card", "summary": "Debit and credit card fees"},
        ],
        session_id=session_id,
        user_id=user_id,
    )

    agent = ClarificationAgent()

    published = []

    async def fake_publish(message, topic_id):
        published.append(message)

    agent.publish_message = fake_publish

    await agent.handle_ambiguous_query(msg, None)

    assert len(published) == 1
    assert isinstance(published[0], ClarificationQuestionMessage)
    assert "ATM" in published[0].question or "card" in published[0].question


@pytest.mark.asyncio
async def test_clarified_query_goes_to_rag():
    """
    GIVEN a clarified query with resolved clusters
    WHEN routed confidently
    THEN RAG retrieval + synthesis should execute
    """

    session_id = "s2"
    user_id = "u2"

    msg = ClusterRoutedQueryMessage(
    original_query="What charges apply?",
    refined_query="What charges apply?",
    intent="BANK_QUERY",
    restrict_cluster_ids=None,
    candidate_clusters=[
        {"cluster_id": "fees", "summary": "ATM, SMS, account charges"},
        {"cluster_id": "card", "summary": "Debit and credit card fees"},
    ],
    routing_confidence=0.55,
    routing_level=None,
    session_id=session_id,
    user_id=user_id,
)

    routed_msg = ClusterRoutedQueryMessage(
    original_query="ATM charges",
    refined_query="ATM withdrawal charges",
    intent="BANK_QUERY",
    restrict_cluster_ids=["fees"],
    candidate_clusters=None,
    routing_confidence=0.92,
    routing_level=1,
    session_id=session_id,
    user_id=user_id,
)


    rag_agent = BankRAGRetrievalAgent()

    with patch(
        "app.agents.rag.retriever_agent.RetrieverAgent.retrieve"
    ) as mock_retrieve:
        mock_retrieve.return_value = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "content": "ATM withdrawals are charged at ₹20.",
                    "metadata": {"file_name": "fees.pdf"},
                }
            ]
        }

        published = []

        async def fake_publish(message, topic_id):
            published.append(message)

        rag_agent.publish_message = fake_publish

        await rag_agent.handle_cluster_routed_query(routed_msg, None)

        assert len(published) == 1
        assert isinstance(published[0], RAGRetrievalResultMessage)
        assert "ATM withdrawals" in published[0].chunks[0]["content"]
