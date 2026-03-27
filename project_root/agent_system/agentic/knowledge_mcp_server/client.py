from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_system.agentic.knowledge_mcp_server.tools.retrieve_cluster import (
    retrieve_cluster_impl,
)
from agent_system.agentic.knowledge_mcp_server.tools.retrieve_documents import (
    retrieve_documents_impl,
)
from agent_system.agentic.knowledge_mcp_server.tools.detect_ambiguity import (
    detect_ambiguity_impl,
)
from agent_system.agentic.knowledge_mcp_server.tools.generate_clarification import (
    generate_clarification_impl,
)
from agent_system.agentic.knowledge_mcp_server.tools.synthesize_answer import (
    synthesize_answer_impl,
)


class KnowledgeMCPClient:
    def retrieve_cluster(
        self,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        return retrieve_cluster_impl(
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def retrieve_documents(
        self,
        query: str,
        query_embedding: List[float],
        restrict_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        return retrieve_documents_impl(
            query=query,
            query_embedding=query_embedding,
            restrict_ids=restrict_ids,
            top_k=top_k,
        )

    def detect_ambiguity(
        self,
        cluster_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return detect_ambiguity_impl(cluster_results=cluster_results)

    def generate_clarification(
        self,
        refined_query: str,
        stm_context: str = "",
        ltm_context: str = "",
    ) -> Dict[str, str]:
        return generate_clarification_impl(
            refined_query=refined_query,
            stm_context=stm_context,
            ltm_context=ltm_context,
        )

    def synthesize_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return synthesize_answer_impl(
            query=query,
            chunks=chunks,
        )