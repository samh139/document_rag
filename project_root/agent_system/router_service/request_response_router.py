# agent_system/router_service/request_reponse_router.py

import asyncio
import json
from confluent_kafka import Consumer, Producer

from agent_system.agentic.messages import UserMessage
from agent_system.agentic.topics import AgenticTopic

from agent_system.router_service.state import pending_requests
from agent_system.router_service.agent_runtime import initialize_runtime, runtime
from autogen_core import TopicId


# ----------------------------
# Kafka Setup
# ----------------------------

consumer = Consumer({
    "bootstrap.servers": "127.0.0.1:9092",
    "group.id": "router-group",
    "auto.offset.reset": "earliest"
})

producer = Producer({
    "bootstrap.servers": "127.0.0.1:9092"
})

consumer.subscribe(["chat-requests"])


# ----------------------------
# Kafka → Agent Handler
# ----------------------------

async def process_message(data: dict):

    session_id = data["session_id"]
    user_message = data["message"]

    print(f"[Router] Processing session={session_id}")

    loop = asyncio.get_running_loop()
    future = loop.create_future()

    # Register this session
    pending_requests[session_id] = future

    # 🔥 Publish message into Agent Runtime
    await runtime.publish_message(
        UserMessage(
            content=user_message,
            session_id=session_id,
            user_id="default_user"
        ),
        topic_id=TopicId(
        AgenticTopic.USER_INPUT.value,
        source="router"
    )
    )

    try:
        # Wait for FinalAnswerCollector to resolve
        final_message = await asyncio.wait_for(future, timeout=180)

    except asyncio.TimeoutError:
        print(f"[Router] Timeout for session {session_id}")
        final_message = None

    # Cleanup
    pending_requests.pop(session_id, None)

    if final_message is None:
        return {
            "session_id": session_id,
            "message": "Sorry, request timed out."
        }

    return { ## FIx the issue
        "session_id": session_id,
        "message": final_message.answer,
        "citations": final_message.citations
    }


# ----------------------------
# Main Router Loop (Async)
# ----------------------------

async def router_loop():

    print("Router started...")

    while True:

        msg = consumer.poll(1.0)

        if msg is None:
            await asyncio.sleep(0.05)
            continue

        if msg.error():
            print("Kafka Error:", msg.error())
            continue

        data = json.loads(msg.value().decode())
        print("[Router] Received from Kafka:", data)

        response = await process_message(data)

        producer.produce(
            topic="chat-responses",
            key=response["session_id"],
            value=json.dumps(response)
        )

        producer.flush()

        print("[Router] Response sent to chat-responses")


# ----------------------------
# Startup
# ----------------------------

async def main():

    runtime = await initialize_runtime()
    await router_loop()


if __name__ == "__main__":
    asyncio.run(main())