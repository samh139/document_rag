from typing import Dict
from fastapi import WebSocket
from app.websockets.session import ConversationSession


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, ConversationSession] = {}

    def get_or_create(
        self,
        session_id: str,
        user_id: str,
        websocket: WebSocket
    ) -> ConversationSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                websocket=websocket
            )
        return self._sessions[session_id]

    def get(self, session_id: str) -> ConversationSession:
        return self._sessions[session_id]

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions
