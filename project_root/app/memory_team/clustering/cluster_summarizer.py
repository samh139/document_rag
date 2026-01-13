# ---------------------------------------------------------
# Cluster Summarizer
# ---------------------------------------------------------
# Converts a cluster's chunks into:
#   - summary_text
#   - summary_vector (used for retrieval)
# ---------------------------------------------------------

from app.embedding import embed_text  # reuse your embedder
from app.storage.chunk_store import get_chunks_by_ids
from app.llm import call_llm

def summarize_cluster(cluster_doc: dict) -> dict:
    """
    Enriches a cluster with semantic meaning.
    DOES NOT mutate storage directly.
    """

    chunk_ids = cluster_doc["chunk_ids"]

    # 1️⃣ Load raw chunk texts
    chunks = get_chunks_by_ids(chunk_ids)
    texts = [c["text"] for c in chunks]

    # 2️⃣ Build summarization prompt
    prompt = f"""
    You are summarizing a group of related banking/KYC documents.

    TASK:
    - Identify the common topic
    - Produce a concise, retrieval-optimized summary
    - Do NOT mention document structure
    - Do NOT mention "this cluster"

    DOCUMENTS:
    {' '.join(texts[:20])}   # cap for safety
    """

    # 3️⃣ Call LLM
    summary_text = call_llm(prompt).strip()

    # 4️⃣ Embed summary
    summary_vector = embed_text(summary_text)

    # 5️⃣ Attach to cluster doc
    cluster_doc["summary_text"] = summary_text
    cluster_doc["summary_vector"] = summary_vector
    cluster_doc["vector"] = summary_vector   # 🔥 KEY LINE

    return cluster_doc
