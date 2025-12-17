import asyncio
import logging

from autogen_core import TopicId

from app.agentic.runtime import create_runtime
from app.agentic.topics import AgenticTopic
from app.agentic.messages import BankUserMessage

from app.agentic.engagement.bank_engagement_agent import BankEngagementAgent
from app.agentic.rag.bank_rag_retrieval_agent import BankRAGRetrievalAgent

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)


async def main():
    runtime = create_runtime()

    # ------------------------------------------------------------------
    # Register agents (Step-1 + Step-2 ONLY)
    # ------------------------------------------------------------------
    await BankEngagementAgent.register(
        runtime,
        "bank_engagement",
        lambda: BankEngagementAgent(),
    )

    await BankRAGRetrievalAgent.register(
        runtime,
        "bank_rag_retrieval",
        lambda: BankRAGRetrievalAgent(),
    )

    # ------------------------------------------------------------------
    # Start runtime
    # ------------------------------------------------------------------
    runtime.start()

    test_inputs = [
        "hi",
        "today weather is good",
        "What are SBI ATM charges?",
        "thanks",
    ]

    for text in test_inputs:
        print(f"\nUSER: {text}")

        await runtime.publish_message(
            BankUserMessage(
                content=text,
                session_id="sess-1",
                user_id="user-123",
            ),
            topic_id=TopicId(
                AgenticTopic.USER_INPUT.value,
                source="user",
            ),
        )

    # ------------------------------------------------------------------
    # Let all agents finish processing
    # ------------------------------------------------------------------
    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
