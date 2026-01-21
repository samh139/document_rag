import asyncio
from app.agentic.query_refiner.query_refiner_agent import QueryRefinerAgent
from app.agentic.messages import EngagementOutputMessage, ClarificationReplyMessage
from app.memory_team.stm.store import (
    set_clarification_context,
    get_clarification_context,
)

async def run():
    agent = QueryRefinerAgent()

    # -------- CASE 1: Normal refinement --------
    msg = EngagementOutputMessage(
        intent="BANK_QUERY",
        user_query="What are the charges?",
        response_text="Checking",
        session_id="s1",
        user_id="u1",
    )

    print("\n🔹 Running normal QueryRefiner path")
    await agent.handle_user_query(msg, None)

    # -------- CASE 2: Clarification resolution (B3) --------
    set_clarification_context(
        session_id="s2",
        user_id="u1",
        reason="ambiguous_cluster",
        candidate_clusters=[
            {"cluster_id": "fees", "summary": "ATM charges"},
            {"cluster_id": "card", "summary": "Card charges"},
        ],
    )

    reply = ClarificationReplyMessage(
        content="ATM withdrawal charges",
        session_id="s2",
        user_id="u1",
    )

    print("\n🔹 Running clarification resolution (B3)")
    await agent.handle_clarification_reply(reply, None)

    ctx = get_clarification_context("s2", "u1")
    assert ctx is None, "Clarification context should be cleared"

    print("✅ QueryRefiner tests PASSED")

if __name__ == "__main__":
    asyncio.run(run())
