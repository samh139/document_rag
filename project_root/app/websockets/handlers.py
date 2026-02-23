from app.websockets.session_event_sink import SessionEventSink
from app.runtime.runtime_context import RuntimeContext
from app.agentic.messages import (
    BankUserMessage,
    ClarificationReplyMessage,
)
from app.websockets.session import ConversationSession


async def handle_user_message(
    session: ConversationSession,
    payload: dict,
):
    # create runtime only once per session
    if not session.runtime_context:
        sink = SessionEventSink(session)
        session.runtime_context = RuntimeContext(session, sink)
        print("sink created :",sink)
        print("session.runtime_context created :",session.runtime_context)

    message = BankUserMessage(**payload)
    print("BankUserMessage created :",message)

    await session.runtime_context.start(message)


async def handle_clarification_reply(
    session: ConversationSession,
    payload: dict,
):
    if session.status != "WAITING_FOR_USER":
        raise RuntimeError("Session not waiting for clarification")

    if not session.runtime_context:
        raise RuntimeError("Runtime not initialized for session")

    message = ClarificationReplyMessage(**payload)

    await session.runtime_context.resume(message)


