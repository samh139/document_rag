# test_confident_flow.py
import asyncio
import uuid
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autogen_core import TopicId
from app.agentic.messages import RefinedQueryMessage, ClusterRoutedQueryMessage, BankUserMessage
from app.agentic.topics import AgenticTopic
from app.agentic.runtime_instance import runtime
from app.agentic.state import response_queue

async def run_confident():
    session_id = f"test-sess-{uuid.uuid4().hex[:8]}"
    user_id = "test-user-1"

    # Step 1: Simulate user message entering system (optional)
    await runtime.publish_message(
        BankUserMessage(content="How do I update my address?", session_id=session_id, user_id=user_id),
        topic_id=TopicId(AgenticTopic.USER_INPUT.value, source="test")
    )

    # Wait a little for pipeline if you want to let QueryRefiner + ClusterRouter run.
    # Or directly publish the confident ClusterRoutedQueryMessage to skip LLM/ES variability:
    routed = ClusterRoutedQueryMessage(
        original_query="How do I update my address?",
        refined_query="how to update address in bank records",
        intent="BANK_QUERY",
        restrict_cluster_ids=["cluster-123"],   # confident
        routing_confidence=0.92,
        routing_level=1,
        candidate_clusters=None,
        session_id=session_id,
        user_id=user_id,
    )

    print("[TEST] Publishing confident ClusterRoutedQueryMessage")
    await runtime.publish_message(
        routed,
        topic_id=TopicId(AgenticTopic.CLUSTER_ROUTED_QUERY_TOPIC.value, source="test")
    )

    # Now wait for final response from response_queue
    print("[TEST] Waiting for response_queue (timeout 30s)...")
    try:
        resp = await asyncio.wait_for(response_queue.get(), timeout=30)
        print("[TEST] Received response:", resp)
    except asyncio.TimeoutError:
        print("[TEST] Timeout waiting for response — check agent logs and retriever")
    
if __name__ == "__main__":
    asyncio.run(run_confident())
