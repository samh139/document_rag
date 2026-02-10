import asyncio
import uuid
import pytest
import logging
from autogen_core import TopicId

from app.agentic.topics import AgenticTopic
from app.agentic.messages import (
    BankUserMessage,
    ClarificationReplyMessage,
)
from app.runtime.runtime_result import RuntimeResult
from app.memory_team.ltm.index_bootstrap import ensure_ltm_index
from tests.helpers.runtime_bootstrap import start_test_runtime

logging.basicConfig(level=logging.INFO)

@pytest.mark.asyncio
async def test_ambiguous_flow_end_to_end():
    """
    Ambiguous query
        → WAIT
        → clarification reply
        → COMPLETE
    """

    ensure_ltm_index()

    runtime, sink = await start_test_runtime()

    session_id = f"test-sess-{uuid.uuid4().hex[:8]}"
    user_id = "test-user-1"

    # STEP 1: ambiguous input
    await runtime.publish_message(
        BankUserMessage(
            content="what charges",
            session_id=session_id,
            user_id=user_id,
        ),
        topic_id=TopicId(
            AgenticTopic.USER_INPUT.value,
            source="test",
        ),
    )

    # STEP 2: WAIT
    wait_result: RuntimeResult = await asyncio.wait_for(
        sink.queue.get(),
        timeout=60,
    )

    assert wait_result.status == "WAIT"
    assert "question" in wait_result.payload

    print("\n[TEST] Clarification Question:")
    print(wait_result.payload["question"])

    # STEP 3: reply
    await runtime.publish_message(
        ClarificationReplyMessage(
            content="Savings account charges",
            session_id=session_id,
            user_id=user_id,
        ),
        topic_id=TopicId(
            AgenticTopic.CLARIFICATION_REPLY.value,
            source="test",
        ),
    )

    # STEP 4: COMPLETE
    final_result: RuntimeResult = await asyncio.wait_for(
        sink.queue.get(),
        timeout=60,
    )

    assert final_result.status == "COMPLETE"
    assert final_result.payload["session_id"] == session_id
    assert final_result.payload["user_id"] == user_id
    assert final_result.payload["answer"].strip()

    print("\n[TEST] Final Answer:")
    print(final_result.payload["answer"])


if __name__ == "__main__":
    asyncio.run(test_ambiguous_flow_end_to_end())
