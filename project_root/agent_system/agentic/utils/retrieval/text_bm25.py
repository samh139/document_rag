# retrieval/text_bm25.py
# ---------------------------------------------------------
# BM25 Retriever — EXACT retriever3 behavior
# ---------------------------------------------------------

from __future__ import annotations
import logging
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
from agent_system.agentic.utils.es.es_utils import get_es_connection

logger = logging.getLogger("ChunkRetriever")
logging.basicConfig(level=logging.INFO)


class BM25Retriever:

    def __init__(self):
        self.es = get_es_connection()
        self.index = "dsprawl_documents"

    # -----------------------------------------------------
    # EXACT retriever3 BM25 query
    # -----------------------------------------------------
    def search(self, query: str, top_k: int, restrict_chunk_ids=None) -> List[Dict]:

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "content^4",
                                    "file_metadata.title^3",
                                    "file_metadata.summary^2",
                                    "file_metadata.tags^3",
                                    "chunk_metadata.tags^4"
                                ]
                            }
                        }
                    ]
                }
            }
        }

        if restrict_chunk_ids:
            body["query"]["bool"]["filter"] = [
                {"terms": {"chunk_id.keyword": restrict_chunk_ids}}
            ]

        try:
            print(f"[BM25Retriever] restrict_chunk_ids count = {len(restrict_chunk_ids) if restrict_chunk_ids else 0}")
            print(f"[BM25Retriever] first 5 restrict ids = {restrict_chunk_ids[:5] if restrict_chunk_ids else []}")
            res = self.es.search(index=self.index, body=body)
        except Exception as e:
            logger.error(f"[BM25Retriever] Search failed: {e}")
            return []

        hits = res.get("hits", {}).get("hits", [])
        results = []

        for h in hits:
            src = h.get("_source", {})
            results.append({
                "chunk_id": src.get("chunk_id"),
                "file_name": src.get("file_name"),
                "content": src.get("content"),
                "file_metadata": src.get("file_metadata"),
                "chunk_metadata": src.get("chunk_metadata"),
                "media_properties":src.get("media_properties"),
                "score": float(h.get("_score", 0)),
            })

        logger.info(f"[BM25Retriever] Retrieved {len(results)} BM25 hits")
        return results
