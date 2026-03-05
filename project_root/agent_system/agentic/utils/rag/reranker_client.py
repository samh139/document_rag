# agents/rag/reranker_client.py
import os, requests
from typing import List, Dict
JINA_URL = os.getenv("JINA_RERANKER_URL","http://localhost:5100/rerank")

def rerank(query: str, candidates: List[Dict]):
    # candidates: list of {"id":..., "text":...}
    payload = {"query": query, "candidates": [{"id":c["chunk_id"], "text": c["source"]["content"]} for c in candidates]}
    resp = requests.post(JINA_URL, json=payload, timeout=30)
    resp.raise_for_status()
    out = resp.json()
    # expecting out["rankings"] as list of ids in order
    ranking_ids = out.get("rankings", [])
    # reorder
    id_map = {c["chunk_id"]: c for c in candidates}
    return [id_map[i] for i in ranking_ids if i in id_map]
