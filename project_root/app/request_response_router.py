from app.agents.engagement.engagement_agent import EngagementAgent
from app.agents.rag.retriever_agent import RetrieverAgent
from app.agents.rag.synthesizer_agent import SynthesizerAgent

class RequestResponseRouter:

    @staticmethod
    def handle(
        user_input: str,
        user_acl: list[str] | None = None
    ) -> dict:
        """
        Entry point for the chatbot.
        Returns a structured response dict.
        """

        if not user_input or not user_input.strip():
            return {
                "type": "OUT_OF_SCOPE",
                "response": "Please enter a valid banking-related question."
            }

        intent = EngagementAgent.classify(user_input)
        print(f"[Router] Classified intent: {intent}")

        # 1. GREETING
        if intent == "GREETING":
            return {
                "type": "GREETING",
                "response": "Hello! I can help you with bank-related questions. Please ask."
            }


        # 2. BANK QUERY → RAG
        elif intent == "BANK_QUERY":
            retrieval = RetrieverAgent.retrieve(
                query=user_input,
                user_acl=user_acl or [],
                top_k=5
            )

            if not retrieval["chunks"]:
                return {
                    "type": "NO_DATA",
                    "response": (
                        "I could not find relevant information in the available documents."
                    )
                }

            answer = SynthesizerAgent.synthesize_answer(
                query=user_input,
                chunks=retrieval["chunks"]
            )

            return {
                "type": "BANK_QUERY",
                "response": answer,
                "sources": [
                    {
                        "chunk_id": c["chunk_id"],
                        "doc": c["metadata"].get("file_name"),
                        "score": c["score"]
                    }
                    for c in retrieval["chunks"]
                ]
            }

        # 3. Everything else → Out of scope
        else:
            return {
                "type": "OUT_OF_SCOPE",
                "response": "I can only assist with bank-related queries. Please ask a banking question."
            }