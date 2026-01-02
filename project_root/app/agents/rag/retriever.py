# agents/rag/retriever.py
import os
from elasticsearch import Elasticsearch
from typing import List, Dict
import math

ES_HOST = os.getenv("ES_HOST","http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX","dsprawl_documents")
es = Elasticsearch(ES_HOST)


def bm25_search(text: str, k=50):
    q = {
      "size": k,
      "query": {"match": {"content": {"query": text}}}
    }
    res = es.search(index=ES_INDEX, body=q)
    return [(h["_id"], h["_score"], h["_source"]) for h in res.get("hits",{}).get("hits",[])]
'''
def bm25_search(text: str, k=50):
    q = {
        "size": k,
        "query": {
            "match": {
                "content": {
                    "query": text,
                    "operator": "and",
                    "minimum_should_match": "75%"
                }
            }
        }
    }
    res = es.search(index=ES_INDEX, body=q)
    return [(h["_id"], h["_score"], h["_source"]) for h in res["hits"]["hits"]]

'''
def knn_search(vector: List[float], k=50):
    q = {
        "size": k,
        "knn": {
            "field": "embedding_vector",
            "query_vector": vector,
            "k": k,
            "num_candidates": max(200, k*5)
        }
    }
    res = es.search(index=ES_INDEX, body=q)
    return [(h["_id"], h["_score"], h["_source"]) for h in res["hits"]["hits"]]
'''
MIN_SEMANTIC_SCORE = 0.75

def knn_search(vector: List[float], k=50):
    q = {
        "size": k,
        "knn": {
            "field": "embedding_vector",
            "query_vector": vector,
            "k": k,
            "num_candidates": max(200, k*5)
        }
    }
    res = es.search(index=ES_INDEX, body=q)
    hits = res["hits"]["hits"]

    return [
        (h["_id"], h["_score"], h["_source"])
        for h in hits
        if h["_score"] >= MIN_SEMANTIC_SCORE
    ]

'''
def reciprocal_rank_fusion(results_lists, k=10, phi=60):
    """
    results_lists: list of lists where each inner list is [(id,score,source)...] ordered by rank
    RRF scoring: sum(1/(phi+rank)) per doc
    """
    scores = {}
    for res in results_lists:
        for rank, item in enumerate(res):
            docid = item[0]
            scores.setdefault(docid, 0.0)
            scores[docid] += 1.0 / (phi + rank + 1)
    # produce top-k sorted by score
    out = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    # fetch full docs
    docs = []
    for docid, sc in out:
        src = es.get(index=ES_INDEX, id=docid)["_source"]
        docs.append({"chunk_id": docid, "score": sc, "source": src})
    return docs
'''
def reciprocal_rank_fusion(results_lists, weights=None, k=10, phi=60):
    scores = {}
    weights = weights or [1.0] * len(results_lists)

    for res, w in zip(results_lists, weights):
        for rank, item in enumerate(res):
            docid = item[0]
            scores.setdefault(docid, 0.0)
            scores[docid] += w * (1.0 / (phi + rank + 1))

    out = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    docs = []
    for docid, sc in out:
        src = es.get(index=ES_INDEX, id=docid)["_source"]
        docs.append({"chunk_id": docid, "score": sc, "source": src})
    return docs
'''

def hybrid_retrieve(query_embedding, text_query=None, top_k=10, acl_filter: List[str]=None):
    bm = bm25_search(text_query, k=top_k) if text_query else []
    knn = knn_search(query_embedding, k=top_k)
    #fused = reciprocal_rank_fusion([bm, knn], k=top_k)
    fused = reciprocal_rank_fusion(
        [bm, knn],
        #weights=[1.0, 1.5],  # 🔥 favor semantic slightly
        k=top_k
    )
    # apply simple ACL prefilter if provided
    if acl_filter:
        fused = [d for d in fused if any(a in d["source"].get("acl",[]) for a in acl_filter)]
    return fused
