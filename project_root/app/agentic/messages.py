# app/agentic/messages.py

from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class BankUserMessage:
    content: str
    session_id :str
    user_id: str



class EngagementOutputMessage(BaseModel):
    intent: str
    response_text: str
    session_id: str
    user_id: str

