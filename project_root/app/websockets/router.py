#!project_root/app/websockets/router.py

from fastapi import APIRouter, WebSocket
from app.websockets.protocol import WSMessage
from app.websockets.session_manager import SessionManager
from app.websockets.handlers import (
    handle_user_message,
    handle_clarification_reply
)
from app.agentic.messages import (
    BankUserMessage,
    ClarificationReplyMessage
)

router = APIRouter()
session_manager = SessionManager()


@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_json()
            msg = WSMessage(**raw)

            if msg.type == "BANK_USER_MESSAGE":
                user_msg = BankUserMessage(**msg.payload)
                session = session_manager.get_or_create(
                    user_msg.session_id,
                    user_msg.user_id,
                    websocket
                )
                await handle_user_message(session, user_msg)

            elif msg.type == "CLARIFICATION_REPLY":
                reply = ClarificationReplyMessage(**msg.payload)
                session = session_manager.get(reply.session_id)
                await handle_clarification_reply(session, reply)

    except Exception:
        await websocket.close()
