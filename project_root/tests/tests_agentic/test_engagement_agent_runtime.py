import asyncio

from autogen_core import TopicId

from app.agentic.runtime import create_runtime
from app.agentic.topics import AgenticTopic
from app.agentic.messages import BankUserMessage

from app.agentic.engagement.bank_engagement_agent import BankEngagementAgent
from app.agentic.engagement.engagement_output_logger import EngagementOutputLogger


async def main():
    # -------------------------------------------------
    # Create runtime
    # -------------------------------------------------
    runtime = create_runtime()

    # -------------------------------------------------
    # Register agents
    # IMPORTANT: Every published message type
    # MUST be handled by at least one agent
    # -------------------------------------------------
    await BankEngagementAgent.register(
        runtime,
        "bank_engagement",
        lambda: BankEngagementAgent(),
    )

    await EngagementOutputLogger.register(
        runtime,
        "engagement_output_logger",
        lambda: EngagementOutputLogger(),
    )

    # -------------------------------------------------
    # Start runtime
    # -------------------------------------------------
    runtime.start()

    # -------------------------------------------------
    # Test inputs
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Graceful shutdown
    # -------------------------------------------------
    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
