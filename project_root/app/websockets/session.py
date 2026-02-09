from datetime import datetime
from fastapi import WebSocket
from typing_extensions import Literal
from typing import Optional, Any

from app.runtime.runtime_result import RuntimeResult
from app.runtime.event_sink import RuntimeEventSink
from app.runtime.runtime_context import RuntimeContext



SessionStatus = Literal[
    "NEW",
    "ACTIVE",
    "WAITING_FOR_USER",
    "COMPLETED",
    "ERROR",
]


class ConversationSession(RuntimeEventSink):
    def __init__(
        self,
        session_id: str,
        user_id: str,
        websocket: WebSocket,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.websocket = websocket

        self.status: SessionStatus = "NEW"
        self.runtime = Optional[RuntimeContext] = None
        self.paused_payload: Optional[dict[str, Any]] = None

        self.created_at = datetime.utcnow()
        self.last_active_at = datetime.utcnow()

    def mark_active(self):
        self.status = "ACTIVE"
        self.last_active_at = datetime.utcnow()

    def mark_waiting(self, payload: dict):
        self.status = "WAITING_FOR_USER"
        self.paused_payload = payload
        self.last_active_at = datetime.utcnow()

    def mark_completed(self):
        self.status = "COMPLETED"
        self.last_active_at = datetime.utcnow()

    async def handle_runtime_result(
        self,
        result: RuntimeResult
    ):
        """
        This replaces response_queue.
        """
        if result.status == "WAIT":
            self.mark_waiting(result.payload)
            await self.websocket.send_json({
                "type": "CLARIFICATION_QUESTION",
                "payload": result.payload
            })

        elif result.status == "COMPLETE":
            self.mark_completed()
            await self.websocket.send_json({
                "type": "FINAL_ANSWER",
                "payload": result.payload
            })

        elif result.status == "ERROR":
            self.status = "ERROR"
            await self.websocket.send_json({
                "type": "ERROR",
                "payload": {
                    "reason": result.reason
                }
            })
