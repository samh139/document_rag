from app.websockets.session import ConversationSession
from app.runtime.runtime_factory import create_runtime


async def handle_user_message(
    session: ConversationSession,
    payload: dict,
):
    session.mark_active()

    if session.runtime is None:
        session.runtime = create_runtime(session)

    # Step-3 will publish message into runtime
    print(
        f"[SESSION {session.session_id}] "
        f"User message: {payload['content']}"
    )


async def handle_clarification_reply(
    session: ConversationSession,
    payload: dict,
):
    if session.status != "WAITING_FOR_USER":
        raise RuntimeError("Unexpected clarification reply")

    session.mark_active()

    # Step-3 will inject reply into runtime
    print(
        f"[SESSION {session.session_id}] "
        f"Clarification reply: {payload['content']}"
    )
