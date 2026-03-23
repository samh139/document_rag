from agent_system.agentic.knowledge_mcp_server.tools.retrieve_documents import retrieve_documents_impl

result = retrieve_documents_impl("What are ATM charges for SBI?", top_k=3)

print("\nRetrieved chunks:\n")
for i, chunk in enumerate(result, start=1):
    print(f"--- Chunk {i} ---")
    print("chunk_id :", chunk["chunk_id"])
    print("score    :", chunk["score"])
    print("file_name:", chunk["file_name"])
    print("content  :", chunk["content"][:300])
    print()
