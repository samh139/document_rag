from __future__ import annotations

from typing import Any, Dict, List, Optional


def detect_ambiguity_impl(
    cluster_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not cluster_results:
        return {
            "is_ambiguous": True,
            "selected_cluster_id": None,
            "candidate_cluster_ids": [],
        }

    candidate_cluster_ids = [
        cluster.get("cluster_id")
        for cluster in cluster_results
        if cluster.get("cluster_id")
    ]

    for cluster in cluster_results:
        chunk_ids = cluster.get("chunk_ids", []) or []
        cluster_id = cluster.get("cluster_id")

        if chunk_ids:
            return {
                "is_ambiguous": False,
                "selected_cluster_id": cluster_id,
                "candidate_cluster_ids": candidate_cluster_ids,
            }

    return {
        "is_ambiguous": True,
        "selected_cluster_id": None,
        "candidate_cluster_ids": candidate_cluster_ids,
    }


def register(mcp) -> None:
    @mcp.tool
    def detect_ambiguity(
        cluster_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Detect whether the query is ambiguous based on retrieved cluster results."""
        return detect_ambiguity_impl(cluster_results=cluster_results)