# tests/test_flow_runtime/test_confident_flow.py

import asyncio
import uuid
import pytest
from autogen_core import TopicId

from app.agentic.runtime_instance import runtime
from app.agentic.state import response_queue
from app.agentic.topics import AgenticTopic
from app.agentic.messages import BankUserMessage, FinalAnswerMessage
from app.memory_team.ltm.index_bootstrap import ensure_ltm_index

from tests.helpers.runtime_bootstrap import start_test_runtime

@pytest.mark.asyncio
async def test_confident_flow():
    """Tests the confident routing flow end-to-end."""

    ensure_ltm_index()  # ✅ LTM index exists
    await start_test_runtime()  # ✅ agents + runtime started

    session_id = f"test-sess-{uuid.uuid4().hex[:8]}"
    user_id = "test-user-1"

    # 🔥 Start from the ONLY valid entry point
    await runtime.publish_message(
        BankUserMessage(
            content="What are the chequebook issuance charges for SBI savings accounts?",
            session_id=session_id,
            user_id=user_id,
        ),
        topic_id=TopicId(
            AgenticTopic.USER_INPUT.value,
            source="test",
        ),
    )

    # ⏳ Wait for final answer
    final_msg = await asyncio.wait_for(response_queue.get(), timeout=60)

    # ✅ Assertions
    assert isinstance(final_msg, FinalAnswerMessage)
    assert final_msg.session_id == session_id
    assert final_msg.user_id == user_id
    assert final_msg.answer.strip() != ""

    print("\n[TEST] Final Answer:")
    print(final_msg.answer)



if __name__ == "__main__":
    asyncio.run(test_confident_flow())