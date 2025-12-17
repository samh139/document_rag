# app/agentic/messages.py

from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Dict, Any


@dataclass
class BankUserMessage:
    content: str
    session_id :str
    user_id: str


class EngagementOutputMessage(BaseModel):
    intent: str
    user_query: str
    response_text: str
    session_id: str
    user_id: str


@dataclass
class RAGRetrievalResultMessage:
    query: str
    chunks: List[Dict[str, Any]]
    session_id: str
    user_id: str


class FinalAnswerMessage(BaseModel):
    answer: str
    citations: List[Dict]
    session_id: str
    user_id: str
