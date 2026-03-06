from agent_system.agentic.memory_team.clustering.config import CHUNKS_INDEX
from agent_system.agentic.memory_team.clustering.vector_loader import get_es_connection


def fetch_chunk_texts(chunk_ids, limit=8):
    if not chunk_ids:
        return []

    es = get_es_connection()

    res = es.search(
        index=CHUNKS_INDEX,
        body={
            "size": limit,
            "_source": ["content"],
            "query": {
                "terms": {
                    # IMPORTANT: keyword field
                    "chunk_id.keyword": chunk_ids
                }
            }
        }
    )

    hits = res.get("hits", {}).get("hits", [])
    return [
        h["_source"]["content"]
        for h in hits
        if h.get("_source", {}).get("content")
    ]
