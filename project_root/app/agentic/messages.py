# app/agentic/messages.py

from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal



class BankUserMessage(BaseModel):
    content: str
    session_id :str
    user_id: str


class EngagementOutputMessage(BaseModel):
    intent: str
    user_query: str
    response_text: str
    session_id: str
    user_id: str


class RAGRetrievalResultMessage(BaseModel):
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


class RefinedQueryMessage(BaseModel):
    original_query: str
    intent: str
    refined_query: str
    session_id: str
    user_id: str


class ClusterRoutedQueryMessage(BaseModel):
    original_query: str
    refined_query: str
    intent: str
    restrict_cluster_ids: Optional[List[str]]
    routing_confidence: float
    routing_level: Optional[int]
    candidate_clusters: Optional[List[dict]] = None  # 🔥 REQUIRED for clarification
    session_id: str
    user_id: str

class ClarificationQuestionMessage(BaseModel):
    question: str
    session_id: str
    user_id: str


class ClarificationReplyMessage(BaseModel):
    content: str
    session_id: str
    user_id: str
