# agent_system/router_service/request_reponse_router.py

import asyncio
import json
from confluent_kafka import Consumer, Producer

from agentic.messages import UserMessage
from agentic.topics import AgenticTopic

from state import pending_requests
from agent_runtime import initialize_runtime, runtime


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
            session_id=session_id,
            user_query=user_message,
            user_id="default_user"  # adjust if needed
        ),
        topic=AgenticTopic.USER_INPUT.value
    )

    try:
        # Wait for FinalAnswerCollector to resolve
        final_message = await asyncio.wait_for(future, timeout=60)

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

    return {
        "session_id": session_id,
        "message": final_message
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

    await initialize_runtime()
    await router_loop()


if __name__ == "__main__":
    asyncio.run(main())