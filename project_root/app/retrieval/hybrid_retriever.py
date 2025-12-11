# app/retrieval/hybrid_retriever.py
import os
from elasticsearch import Elasticsearch
from typing import List, Dict
from dotenv import load_dotenv
import numpy as np
import requests

load_dotenv("app/config/secrets.env")
ES_HOST = os.getenv("ES_HOST", "http://es:9200")
ES_INDEX = os.getenv("ES_INDEX", "dsprawl_documents")
es = Elasticsearch(ES_HOST)

def bm25_query(q: str, size=20):
    body = {
        "query": {
            "multi_match": {
                "query": q,
                "fields": ["content^3", "metadata.ai.summary", "metadata.ai.tags"]
            }
        },
        "size": size
    }
    r = es.search(index=ES_INDEX, body=body)
    return [(hit["_id"], hit["_score"], hit["_source"]) for hit in r["hits"]["hits"]]

def vector_query(vec: List[float], size=20):
    # script_score with cosine similarity
    body = {
        "size": size,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0",
                    "params": {"query_vector": vec}
                }
            }
        }
    }
    r = es.search(index=ES_INDEX, body=body)
    return [(hit["_id"], hit["_score"], hit["_source"]) for hit in r["hits"]["hits"]]

def reciprocal_rank_fusion(results_lists, k=20):
    # results_lists: list of lists of tuples (id, score, source)
    # simple RRF: score = sum(1/(k + rank))
    scores = {}
    for res in results_lists:
        for rank, (docid, score, src) in enumerate(res, start=1):
            scores.setdefault(docid, {"score": 0.0, "source": src})
            scores[docid]["score"] += 1.0 / (60 + rank)  # 60 is RRF k param
    ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return [(docid, vals["score"], vals["source"]) for docid, vals in ranked][:k]

def get_embeddings_for_query(query: str):
    OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
    resp = requests.post(f"{OLLAMA}/embeddings", json={"model": "nomic-embed-text", "input": query})
    resp.raise_for_status()
    return resp.json().get("embedding")

def retrieve(query: str, user_acl=None, topk=10):
    # ACL prefilter - build query with terms filter or pass in search body
    bm = bm25_query(query, size=topk*2)
    qvec = get_embeddings_for_query(query)
    vec = vector_query(qvec, size=topk*2)
    fused = reciprocal_rank_fusion([bm, vec], k=topk)
    # Optionally call reranker to score fused results:
    return fused[:topk]
