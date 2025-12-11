from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Candidate(BaseModel):
    id: str
    text: str

class RerankRequest(BaseModel):
    query: str
    candidates: List[Candidate]

class RerankResponse(BaseModel):
    rankings: List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rerank")
def rerank(req: RerankRequest):
    # TODO: wire in your cross-encoder / jina reranker model.
    # For now, a very simple heuristic: return candidates in same order
    ranked_ids = [c.id for c in req.candidates]
    return {"rankings": ranked_ids}
