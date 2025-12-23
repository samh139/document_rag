# app/memory/stm_adapter.py

import json
import redis
from typing import List

from app.memory.schemas import ConversationTurn
from app.configs.redis_config import REDIS_URL


class STMAdapter:
    def __init__(self):
        self.client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True
        )

    def _key(self, role: str, user_id: str, session_id: str) -> str:
        return f"stm:{role}:{user_id}:{session_id}"

    def get_turns(
        self,
        role: str,
        user_id: str,
        session_id: str,
        limit: int = 10,
    ) -> List[ConversationTurn]:

        raw = self.client.get(self._key(role, user_id, session_id))
        if not raw:
            return []

        data = json.loads(raw)
        return [ConversationTurn(**t) for t in data[-limit:]]

    def add_turn(
        self,
        role: str,
        user_id: str,
        session_id: str,
        turn: ConversationTurn,
        max_turns: int = 10,
    ) -> None:

        key = self._key(role, user_id, session_id)
        raw = self.client.get(key)

        turns = json.loads(raw) if raw else []

        # ✅ FIX: JSON-safe serialization
        turns.append(turn.model_dump(mode="json"))

        self.client.set(key, json.dumps(turns[-max_turns:]))

    def clear_session(
        self,
        role: str,
        user_id: str,
        session_id: str,
    ) -> None:
        self.client.delete(self._key(role, user_id, session_id))
