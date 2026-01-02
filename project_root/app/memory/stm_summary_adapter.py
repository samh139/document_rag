# app/memory/stm_summary_adapter.py

import json
import redis
from typing import Optional

from app.configs.redis_config import REDIS_URL


class STMSummaryAdapter:
    """
    Stores ONE rolling STM summary per (role, user_id, session_id).
    """

    def __init__(self):
        self.client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True
        )

    def _key(self, role: str, user_id: str, session_id: str) -> str:
        return f"stm_summary:{role}:{user_id}:{session_id}"

    def get_summary(
        self,
        role: str,
        user_id: str,
        session_id: str,
    ) -> Optional[dict]:
        raw = self.client.get(self._key(role, user_id, session_id))
        if not raw:
            return None
        return json.loads(raw)

    def set_summary(
        self,
        role: str,
        user_id: str,
        session_id: str,
        summary: dict,
    ) -> None:
        """
        Overwrites existing STM summary.
        """
        self.client.set(
            self._key(role, user_id, session_id),
            json.dumps(summary),
        )

    def clear_summary(
        self,
        role: str,
        user_id: str,
        session_id: str,
    ) -> None:
        self.client.delete(self._key(role, user_id, session_id))
