# app/memory/schemas.py

from pydantic import BaseModel
from typing import Dict, Optional


class ConversationTurn(BaseModel):
    user: str
    assistant: str
    timestamp: str
    metadata: Dict = {}
