from app.agents.rag.retriever_agent import RetrieverAgent

query = "What are SBI ATM charges?"
user_acl = ["ROLE_USER"]

result = RetrieverAgent.retrieve(
    query=query,
    user_acl=user_acl,
    top_k=3
)

print("\nQuery:", result["query"])
for c in result["chunks"]:
    print("\n---")
    print("Chunk ID:", c["chunk_id"])
    print("Score:", c["score"])
    print("Doc:", c["metadata"].get("file_name"))
    print("Text:", c["content"][:200], "...")
