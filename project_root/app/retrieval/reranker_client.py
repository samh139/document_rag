# app/retrieval/reranker_client.py
import requests
from typing import List, Dict
from pydantic import BaseModel

RERANK_URL = "http://localhost:5100/rerank"

class Candidate(BaseModel):
    id: str
    text: str

def call_reranker(query: str, candidates: List[Dict]):
    body = {
        "query": query,
        "candidates": [{"id": c["chunk_id"], "text": c["content"]} for c in candidates]
    }
    resp = requests.post(RERANK_URL, json=body, timeout=10)
    resp.raise_for_status()
    return resp.json().get("rankings", [])
