from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_system.agentic.utils.clustering.cluster_retriever import ClusterRetriever


def _normalize_cluster(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cluster_id": result.get("cluster_id"),
        "chunk_ids": result.get("chunk_ids", []) or [],
        "score": float(result.get("score", 0.0)),
    }


def retrieve_cluster_impl(
    query_embedding: List[float],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    retriever = ClusterRetriever(top_k=top_k)
    results = retriever.retrieve_clusters(query_vec=query_embedding)
    return [result for result in results] ## Add normalization later after testing 


def register(mcp) -> None:
    @mcp.tool
    def retrieve_cluster(
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieve top matching clusters using ANN over cluster vectors."""
        return retrieve_cluster_impl(
            query_embedding=query_embedding,
            top_k=top_k,
        )