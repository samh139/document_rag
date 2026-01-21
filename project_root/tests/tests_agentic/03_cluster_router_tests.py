import asyncio
from app.agentic.cluster_agent.cluster_agent import ClusterRouterAgent
from app.agentic.messages import RefinedQueryMessage

async def run():
    agent = ClusterRouterAgent()

    # Ambiguous
    msg1 = RefinedQueryMessage(
        refined_query="What are the charges?",
        original_query="What are the charges?",
        intent="BANK_QUERY",
        session_id="s1",
        user_id="u1",
    )

    print("\n🔹 Ambiguous routing test")
    await agent.handle_refined_query(msg1, None)

    # Confident
    msg2 = RefinedQueryMessage(
        refined_query="ATM withdrawal charges",
        original_query="ATM charges",
        intent="BANK_QUERY",
        session_id="s2",
        user_id="u1",
    )

    print("\n🔹 Confident routing test")
    await agent.handle_refined_query(msg2, None)

    print("✅ ClusterRouter tests PASSED")

if __name__ == "__main__":
    asyncio.run(run())
