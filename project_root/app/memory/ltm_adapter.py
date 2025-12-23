# app/memory/ltm_adapter.py

from memory_team.ltm.ltm_service import LTMService
from .schemas import ConversationTurn


class LTMAdapter:
    def __init__(self):
        self.service = LTMService()

    def store_turn(self, turn: ConversationTurn):
        document = {
            "user_id": turn.user_id,
            "session_id": turn.session_id,
            "query": turn.user_query,
            "answer": turn.assistant_answer,
            "timestamp": turn.timestamp,
            "metadata": turn.metadata or {}
        }
        self.service.store(document)

    def search_user_history(self, user_id: str, query: str, limit: int = 5):
        return self.service.search(
            user_id=user_id,
            query=query,
            size=limit
        )
