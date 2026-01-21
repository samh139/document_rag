import pytest
from app.agentic.messages import ClarificationReplyMessage


def test_clarification_reply_message_valid():
    msg = ClarificationReplyMessage(
        content="ATM withdrawal charges",
        session_id="sess-1",
        user_id="user-123",
    )

    assert msg.content == "ATM withdrawal charges"
    assert msg.session_id == "sess-1"
    assert msg.user_id == "user-123"
