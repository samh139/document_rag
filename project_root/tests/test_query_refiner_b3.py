import pytest
from unittest.mock import patch, AsyncMock

from app.agentic.query_refiner.query_refiner_agent import QueryRefinerAgent
from app.agentic.messages import (
    ClarificationReplyMessage,
    RefinedQueryMessage,
)
from app.agentic.topics import AgenticTopic


@pytest.mark.asyncio
async def test_query_refiner_resolves_clarification_b3():
    """
    GIVEN an active clarification context
    WHEN user replies with a clarification answer
    THEN QueryRefiner should:
      - resolve ambiguity
      - clear clarification context
      - emit RefinedQueryMessage
    """

    agent = QueryRefinerAgent()

    session_id = "sess-1"
    user_id = "user-123"

    clarification_ctx = {
        "active": True,
        "reason": "ambiguous_cluster",
        "candidate_clusters": [
            {
                "cluster_id": "fees",
                "summary": "ATM, SMS, account charges",
            },
            {
                "cluster_id": "card",
                "summary": "Debit and credit card charges",
            },
        ],
    }

    msg = ClarificationReplyMessage(
        content="ATM withdrawal charges",
        session_id=session_id,
        user_id=user_id,
    )

    ctx = AsyncMock()

    with patch(
        "app.agentic.query_refiner.query_refiner_agent.get_clarification_context",
        return_value=clarification_ctx,
    ), patch(
        "app.agentic.query_refiner.query_refiner_agent.clear_clarification_context"
    ) as clear_ctx, patch(
        "app.agentic.query_refiner.query_refiner_agent.fire_fast_modal_request_chat",
        return_value="ATM withdrawal charges applicable to the account",
    ), patch.object(
        agent,
        "publish_message",
        new_callable=AsyncMock,
    ) as publish:

        await agent.handle_clarification_reply(msg, ctx)

        # 🔥 Clarification must be cleared
        clear_ctx.assert_called_once_with(session_id, user_id)

        # 🔥 Refined query must be published
        publish.assert_called_once()

        published_msg = publish.call_args[0][0]
        topic = publish.call_args[1]["topic_id"].type

        assert isinstance(published_msg, RefinedQueryMessage)
        assert "ATM withdrawal charges" in published_msg.refined_query
        assert topic == AgenticTopic.REFINED_QUERY_TOPIC.value
