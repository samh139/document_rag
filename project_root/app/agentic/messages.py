# app/agentic/messages.py

from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


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


class Citation(BaseModel):
    chunk_id: str
    file_name: str
    chunk_content: str

class FinalAnswerMessage(BaseModel):
    answer: str
    citations: List[Citation]
    session_id: str
    user_id: str
    user_query: str


