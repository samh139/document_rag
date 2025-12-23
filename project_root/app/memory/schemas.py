# app/memory/schemas.py

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict

class ConversationTurn(BaseModel):
    user: str
    assistant: str
    timestamp: datetime
    metadata: Optional[Dict] = {}
