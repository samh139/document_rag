from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_system.agentic.utils.clustering.chunk_retriever import ChunkRetriever


def _normalize_chunk(result: Dict[str, Any]) -> Dict[str, Any]:
    metadata = result.get("metadata", {}) or {}

    return {
        "chunk_id": result.get("chunk_id"),
        "score": float(result.get("score")),
        "content": result.get("content", ""),
        "file_name": result.get("file_name", ""),
        "cluster_id": result.get("cluster_id"),
        "metadata": metadata,
    }


def retrieve_documents_impl(
    query: str,
    query_embedding: List[float],
    restrict_ids: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    retriever = ChunkRetriever(top_k=top_k)
    results = retriever.retrieve(
        query=query,
        qvec=query_embedding,
        restrict_ids=restrict_ids,
    )
    return [_normalize_chunk(result) for result in results]


def register(mcp) -> None:
    @mcp.tool
    def retrieve_documents(
        query: str,
        query_embedding: List[float],
        restrict_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve top ranked chunks, optionally restricted to a cluster chunk scope."""
        return retrieve_documents_impl(
            query=query,
            query_embedding=query_embedding,
            restrict_ids=restrict_ids,
            top_k=top_k,
        )