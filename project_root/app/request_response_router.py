# request_response_router.py
import asyncio
from dataclasses import dataclass
from typing import Any, List
from agents.engagement_agent import classify_with_ollama
from agents.rag.embedding_client import embed_text
from agents.rag.retriever import hybrid_retrieve
from agents.rag.reranker_client import rerank
from agents.rag.synthesizer import synthesize_answer

@dataclass
class UserMessage:
    user_id: str
    session_id: str
    current_query: str
    roles: List[str] = None
    conversation_history: str = ""
    file_names: List[str] = None

async def generate_response(msg: UserMessage):
    # 1) classify with engagement agent (LLM)
    cls = classify_with_ollama(msg.current_query)
    intent = cls.get("intent","unknown")
    if intent == "greeting":
        return {"type":"engagement","text": cls.get("follow_up","Hi! How can I help?")}

    if intent != "rag":
        return {"type":"engagement","text": cls.get("follow_up","Could you rephrase?")}

    # 2) embed query
    qvec = embed_text(msg.current_query)

    # 3) retrieve
    hits = hybrid_retrieve(query_embedding=qvec, text_query=msg.current_query, top_k=10, acl_filter=msg.roles)

    # 4) optional rerank (if Jina available)
    try:
        reranked = rerank(msg.current_query, hits)
    except Exception:
        reranked = hits

    # 5) synthesize
    answer = synthesize_answer(reranked, msg.current_query)

    # 6) build response
    response = {
        "type":"rag",
        "answer": answer.get("answer"),
        "sources": answer.get("sources"),
        "confidence": answer.get("confidence", 0.5),
        "chunks": [h for h in reranked]
    }
    return response
