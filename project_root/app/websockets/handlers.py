from app.websockets.session_event_sink import SessionEventSink
from app.runtime.runtime_context import RuntimeContext
from app.agentic.messages import (
    BankUserMessage,
    ClarificationReplyMessage,
)
from app.websockets.session import ConversationSession


async def handle_user_message(
    session: ConversationSession,
    msg: BankUserMessage,
):
    # create runtime only once per session
    if session.runtime is None:
        sink = SessionEventSink(session)
        session.runtime = RuntimeContext(session, sink)

    await session.runtime.start(msg)


async def handle_clarification_reply(
    session: ConversationSession,
    msg: ClarificationReplyMessage,
):
    if session.status != "WAITING_FOR_USER":
        raise RuntimeError("Session not waiting for clarification")

    await session.runtime.resume(msg)
