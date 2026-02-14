from fastapi import APIRouter, WebSocket
from app.websockets.protocol import WSMessage
from app.websockets.session_manager import SessionManager
from app.websockets.handlers import (
    handle_user_message,
    handle_clarification_reply,
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
                session = session_manager.get_or_create(
                    session_id=msg.payload["session_id"],
                    user_id=msg.payload["user_id"],
                    websocket=websocket,
                )
                await handle_user_message(session, msg.payload)

            elif msg.type == "CLARIFICATION_REPLY":
                session = session_manager.get(msg.payload["session_id"])
                if not session:
                    raise RuntimeError("Session not found")

                await handle_clarification_reply(session, msg.payload)

    except Exception:
        await websocket.close()
