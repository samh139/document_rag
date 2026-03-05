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

def reciprocal_rank_fusion(results_lists, k=10, phi=60):
    scores = {}
    appearances = {}
    sources = {}

    for res in results_lists:
        norm = normalize_scores(res)

        for rank, (docid, es_score, src) in enumerate(res):
            # rank-based component
            rrf_score = 1.0 / (phi + rank + 1)

            # ES-score-based component
            score_component = norm.get(docid, 0.0)

            # combine (rank still matters, but score adds signal)
            combined = (0.7 * rrf_score) + (0.3 * score_component)

            scores[docid] = scores.get(docid, 0.0) + combined
            appearances[docid] = appearances.get(docid, 0) + 1
            sources[docid] = src

    # 🔥 boost docs that appear in BOTH bm25 & knn
    for docid in scores:
        if appearances[docid] > 1:
            scores[docid] *= 1.2

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    docs = []
    for docid, sc in ranked[:k]:
        docs.append({
            "chunk_id": docid,
            "score": sc,
            "source": sources[docid]
        })

    return docs

def dedupe_chunks(chunks):
    seen = set()
    deduped = []

    for c in chunks:
        src = c.get("source", {})
        content_hash = src.get("content_hash")

        # fallback safety (should rarely happen)
        if not content_hash:
            content_hash = hash(src.get("content", "")[:300])

        if content_hash not in seen:
            seen.add(content_hash)
            deduped.append(c)

    return deduped




def hybrid_retrieve(
    query_embedding,
    text_query=None,
    top_k=10,
    acl_filter=None,
    restrict_ids=None,  # now means chunk_ids
):
    bm = bm25_search(text_query, k=top_k * 2) if text_query else []
    knn = knn_search(query_embedding, k=top_k * 2)

    fused = reciprocal_rank_fusion([bm, knn], k=top_k * 3)

    # 🔥 dedupe AFTER fusion
    fused = dedupe_chunks(fused)

    # 🔒 ACL filter
    if acl_filter:
        fused = [
            d for d in fused
            if any(a in d["source"].get("acl", []) for a in acl_filter)
        ]

    # 🔒 CLUSTER FILTER (POST-FUSION, SAFE)
    if restrict_ids:
        allowed = set(restrict_ids)
        fused = [
            d for d in fused
            if d["chunk_id"] in allowed
        ]

    return fused[:top_k]




def normalize_scores(results):
    if not results:
        return {}
    scores = [r[1] for r in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return {r[0]: 1.0 for r in results}
    return {
        r[0]: (r[1] - min_s) / (max_s - min_s)
        for r in results
    }
