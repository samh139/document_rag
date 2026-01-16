# agents/rag/retriever_agent.py

from typing import List, Dict
from app.agents.rag.embedding_client import embed_text
from app.agents.rag.retriever import hybrid_retrieve
from typing import Optional

class RetrieverAgent:
    """
    Retrieves relevant document chunks for a banking query.
    No generation. No reasoning. Retrieval only.
    """

    @staticmethod
    def retrieve(
        query: str,
        user_acl: List[str],
        top_k: int = 5,
        restrict_ids: Optional[List[str]] = None,
    ) -> Dict:
        # 1. Embed query
        query_vector = embed_text(query)

        # 2. Hybrid retrieval (BM25 + Vector + RRF)
        results = hybrid_retrieve(
            query_embedding=query_vector,
            text_query=query,
            top_k=top_k,
            acl_filter=user_acl,
            restrict_ids=restrict_ids,  
        )

        # 3. Normalize output
        chunks = []
        for r in results:
            src = r["source"]
            chunks.append({
                "chunk_id": r["chunk_id"],
                "score": r["score"],
                "content": src.get("content", ""),
                "metadata": src.get("metadata", {}),
                "doc_id": src.get("doc_id")
            })

        return {
            "query": query,
            "chunks": chunks
        }
