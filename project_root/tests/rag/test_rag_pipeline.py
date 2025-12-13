from app.agents.rag.embedding_client import embed_text
from app.agents.rag.retriever_agent import RetrieverAgent
from app.agents.rag.synthesizer_agent import SynthesizerAgent

QUERY = "What are SBI ATM charges?"
USER_ACL = ["ROLE_USER"]

def run_test():
    print("\nUSER QUERY:")
    print(QUERY)

    # 1. Embed query
    print("\nEmbedding query...")
    query_vec = embed_text(QUERY)
    print(f"Embedding dim: {len(query_vec)}")

    # 2. Retrieve chunks
    retrieval_result = RetrieverAgent.retrieve(
    query=QUERY,
    user_acl=USER_ACL,
    top_k=5
)

    chunks = retrieval_result["chunks"]
    print(f"Retrieved {len(chunks)} chunks")

    for c in chunks:
        print("\n---")
        print("Doc:", c["metadata"].get("file_name"))
        print("Score:", c["score"])
        print("Text:", c["content"][:200], "...")

    # 3. Synthesize answer
    print("\nSynthesizing final answer...\n")
    answer = SynthesizerAgent.synthesize_answer(
        query=QUERY,
        chunks=chunks
    )

    print("FINAL ANSWER:")
    print(answer)


if __name__ == "__main__":
    run_test()
