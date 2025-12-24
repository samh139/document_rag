import asyncio
import logging

from autogen_core import TopicId

from app.agentic.runtime import create_runtime
from app.agentic.topics import AgenticTopic
from app.agentic.messages import BankUserMessage

from app.agentic.engagement.bank_engagement_agent import BankEngagementAgent
from app.agentic.rag.query_refiner_agent import QueryRefinerAgent
from app.agentic.rag.bank_rag_retrieval_agent import BankRAGRetrievalAgent
from app.agentic.rag.bank_rag_synthesis_agent import BankRAGSynthesisAgent

logging.basicConfig(level=logging.INFO)

async def main():
    runtime = create_runtime()

    await BankEngagementAgent.register(
        runtime, "bank_engagement", BankEngagementAgent
    )

    await QueryRefinerAgent.register(
        runtime, "query_refiner", QueryRefinerAgent
    )

    await BankRAGRetrievalAgent.register(
        runtime, "bank_rag_retrieval", BankRAGRetrievalAgent
    )

    await BankRAGSynthesisAgent.register(
        runtime, "bank_rag_synthesis", BankRAGSynthesisAgent
    )

    runtime.start()

    test_inputs = [
        "What documents are needed?",
        "And for NR accounts?",
        "What about fees?",
    ]

    for text in test_inputs:
        print(f"\nUSER: {text}")
        await runtime.publish_message(
            BankUserMessage(
                content=text,
                session_id="sess-1",
                user_id="user-123",
            ),
            TopicId(
                AgenticTopic.USER_INPUT.value,
                source="user",
            ),
        )

    await runtime.stop_when_idle()

if __name__ == "__main__":
    asyncio.run(main())
