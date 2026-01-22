# test_ambiguous_flow.py
import asyncio, uuid, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autogen_core import TopicId
from app.agentic.messages import ClusterRoutedQueryMessage
from app.agentic.topics import AgenticTopic
from app.agentic.runtime_instance import runtime
from app.agentic.state import response_queue

async def run_ambiguous():
    session_id = f"test-sess-{uuid.uuid4().hex[:8]}"
    user_id = "test-user-amb"

    candidate_clusters = [
        {"cluster_id": "c1", "summary": "How to update address for savings accounts"},
        {"cluster_id": "c2", "summary": "How to update address for credit cards"},
    ]

    ambiguous = ClusterRoutedQueryMessage(
        original_query="How do I update my address?",
        refined_query="how to update my address",
        intent="BANK_QUERY",
        restrict_cluster_ids=None,
        routing_confidence=0.45,
        routing_level=None,
        candidate_clusters=candidate_clusters,
        session_id=session_id,
        user_id=user_id,
    )

    print("[TEST] Publishing ambiguous ClusterRoutedQueryMessage")
    await runtime.publish_message(
        ambiguous,
        topic_id=TopicId(AgenticTopic.CLUSTER_ROUTED_QUERY_TOPIC.value, source="test")
    )

    print("[TEST] Waiting for clarification question on response_queue (timeout 30s)...")
    try:
        resp = await asyncio.wait_for(response_queue.get(), timeout=30)
        print("[TEST] Received response:", resp)
        # Now simulate user replying to the clarification question:
        # e.g., if resp is {"type":"clarification","question": "Do you mean savings or credit card?"}
        reply = "Savings account"
        print("[TEST] Simulating user reply:", reply)
        # Publish BankUserMessage similar to production flow:
        from app.agentic.messages import BankUserMessage
        await runtime.publish_message(
            BankUserMessage(content=reply, session_id=session_id, user_id=user_id),
            topic_id=TopicId(AgenticTopic.USER_INPUT.value, source="test")
        )
        # Wait for final answer after B3 and re-routing:
        resp2 = await asyncio.wait_for(response_queue.get(), timeout=30)
        print("[TEST] Final answer:", resp2)
    except asyncio.TimeoutError:
        print("[TEST] Timeout waiting for response — check logs")

if __name__ == "__main__":
    asyncio.run(run_ambiguous())
