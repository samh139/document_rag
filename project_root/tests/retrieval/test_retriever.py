import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.agents.rag.embedding_client import embed_text
from app.agents.rag.retriever import hybrid_retrieve

query = "ATM withdrawal charges"
vec = embed_text(query)

results = hybrid_retrieve(vec, text_query=query, top_k=5)

for r in results:
    print("\n---")
    print("CHUNK:", r["chunk_id"])
    print("SCORE:", r["score"])
    print("DOC:", r["source"]["metadata"]["file_name"])
    print("TEXT:", r["source"]["content"][:200], "...")
