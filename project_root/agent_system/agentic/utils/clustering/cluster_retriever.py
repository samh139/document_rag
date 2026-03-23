# retrieval/cluster_retriever.py
# ---------------------------------------------------------
# Stage-1 Cluster Retriever (ANN-only)
# Matches retriever3 ANN behavior, minus BM25 and CE.
# ---------------------------------------------------------

from __future__ import annotations
from typing import List, Dict
from elasticsearch import Elasticsearch
from agent_system.agentic.utils.es.es_utils import get_es_connection
#from app_logger import get_app_logger
#from app_configs.app_env import app_env
#from app_configs.es_config import get_es_connection
#logger =get_app_logger("cluster_retriever")


class ClusterRetriever:

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.index = "dsprawl_documents"
        self.es = get_es_connection()
        print(f"[ClusterRetriever] Initialised | top_k={self.top_k}")

    # -----------------------------------------------------
    # ANN search (raw)
    # -----------------------------------------------------
    def search(self, query_vec) -> List[Dict]:
        print(f"[ClusterRetriever] Running ANN on {self.index} | k={self.top_k}")

        body = {
            "size": self.top_k,
            "query": {
                "knn": {
                    "field": "vector",
                    "query_vector": list(query_vec),
                    "k": self.top_k,
                    "num_candidates": max(50, self.top_k * 10)
                }
            }
        }

        try:
            res = self.es.search(index=self.index, body=body)
        except Exception as e:
            print(f"[ClusterRetriever] ANN cluster search failed: {e}")
            return []

        hits = res.get("hits", {}).get("hits", [])
        results = []

        for h in hits:
            src = h.get("_source", {})
            results.append({
                "cluster_id": src.get("cluster_id"),
                "chunk_ids": src.get("chunk_ids", []),
                "score": float(h.get("_score", 0)),
            })

        return results

    def retrieve(self, query_vec) -> List[Dict]:
        clusters = self.search(query_vec)
        if not clusters:
            print("[ClusterRetriever] No clusters returned")
        return clusters
