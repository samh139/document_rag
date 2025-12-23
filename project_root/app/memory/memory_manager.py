# app/memory/memory_manager.py

from datetime import datetime
from typing import List

from app.memory.schemas import ConversationTurn
from app.memory.stm_adapter import STMAdapter

class MemoryManager:
    def __init__(self):
        self.stm = STMAdapter()

    # 🔐 ACL gate (simple, extensible)
    def _validate_access(self, role: str):
        if role not in {"CUSTOMER", "RM", "OPS", "ADMIN"}:
            raise ValueError("Invalid role")

    def get_stm(
        self,
        role: str,
        user_id: str,
        session_id: str,
        limit: int = 10,
    ) -> List[dict]:

        self._validate_access(role)

        turns = self.stm.get_turns(
            role=role,
            user_id=user_id,
            session_id=session_id,
            limit=limit,
        )

        # Convert to LLM-friendly format
        return [
            {"user": t.user, "assistant": t.assistant}
            for t in turns
        ]

    def store_turn(
        self,
        role: str,
        user_id: str,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        metadata: dict | None = None,
    ) -> None:

        self._validate_access(role)

        turn = ConversationTurn(
            user=user_query,
            assistant=assistant_answer,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata or {},
        )

        self.stm.add_turn(
            role=role,
            user_id=user_id,
            session_id=session_id,
            turn=turn,
        )
