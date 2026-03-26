from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_system.agentic.utils.clustering.chunk_retriever import ChunkRetriever

def _normalize_chunk(result) -> Dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "score": float(result.final_score),
        "content": result.content or "",
        "file_name": result.file or "",
        "cluster_id": None,
        "metadata": {
            "file_title": result.file_title,
            "chunk_tags": result.chunk_tags,
            "rank": result.final_rank,
            "bm_score": result.bm_score,
            "rrf": result.rrf,
            "relevant_score": result.relevant_score,
        },
    }

def retrieve_documents_impl(
    query: str,
    query_embedding: List[float],
    restrict_ids: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    retriever = ChunkRetriever(top_k=top_k)
    print(f"[ChunkRetriever] restrict_ids count = {len(restrict_ids) if restrict_ids else 0}")
    print(f"[ChunkRetriever] first 5 restrict_ids = {restrict_ids[:5] if restrict_ids else []}")
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