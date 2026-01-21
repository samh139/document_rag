from app.agentic.messages import (
    BankUserMessage,
    EngagementOutputMessage,
    RefinedQueryMessage,
    ClusterRoutedQueryMessage,
    ClarificationQuestionMessage,
    ClarificationReplyMessage,
    FinalAnswerMessage,
)

def test_message_contracts():
    BankUserMessage(
        content="What are ATM charges?",
        session_id="s1",
        user_id="u1",
    )

    EngagementOutputMessage(
        intent="BANK_QUERY",
        user_query="What are ATM charges?",
        response_text="Checking",
        session_id="s1",
        user_id="u1",
    )

    RefinedQueryMessage(
        refined_query="ATM withdrawal charges",
        original_query="What are ATM charges?",
        intent="BANK_QUERY",
        session_id="s1",
        user_id="u1",
    )

    ClusterRoutedQueryMessage(
        original_query="What are ATM charges?",
        refined_query="ATM withdrawal charges",
        intent="BANK_QUERY",
        restrict_cluster_ids=None,
        routing_confidence=0.72,
        routing_level="level1",
        candidate_clusters=[
            {
                "cluster_id": "fees",
                "score": 0.72,
                "chunk_count": 50,
                "summary": "ATM and account charges",
            }
        ],
        session_id="s1",
        user_id="u1",
    )

    ClarificationQuestionMessage(
        question="Are you asking about ATM charges or card charges?",
        session_id="s1",
        user_id="u1",
    )

    ClarificationReplyMessage(
        content="ATM charges",
        session_id="s1",
        user_id="u1",
    )

    FinalAnswerMessage(
        answer="ATM withdrawal charges are ₹20 per transaction.",
        citations=[],
        session_id="s1",
        user_id="u1",
        user_query="ATM charges",
    )

    print("✅ Message contract tests PASSED")

if __name__ == "__main__":
    test_message_contracts()
