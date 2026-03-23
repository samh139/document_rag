import asyncio
import json
import logging

from confluent_kafka import Consumer, Producer
from autogen_core import TopicId

from agent_system.agentic.messages import UserMessage
from agent_system.agentic.topics import AgenticTopic
from agent_system.router_service.agent_runtime import initialize_runtime, runtime
from agent_system.router_service.state import pending_requests

logging.getLogger("autogen_core").setLevel(logging.WARNING)

consumer = Consumer(
    {
        "bootstrap.servers": "127.0.0.1:9092",
        "group.id": "router-group",
        "auto.offset.reset": "earliest",
    }
)

producer = Producer({"bootstrap.servers": "127.0.0.1:9092"})
consumer.subscribe(["chat-requests"])


async def process_message(data: dict):
    session_id = data["session_id"]
    user_message = data["message"]
    user_id = data.get("user_id", "default_user")

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    pending_requests[session_id] = future

    outbound = UserMessage(
        content=user_message,
        session_id=session_id,
        user_id=user_id,
    )
    topic = AgenticTopic.USER_INPUT.value

    await runtime.publish_message(
        outbound,
        topic_id=TopicId(topic, source="router"),
    )

    try:
        result = await asyncio.wait_for(future, timeout=180)
    except asyncio.TimeoutError:
        pending_requests.pop(session_id, None)
        return {
            "session_id": session_id,
            "type": "error",
            "message": "Sorry, request timed out.",
        }

    pending_requests.pop(session_id, None)

    citations = []
    for citation in getattr(result, "citations", []) or []:
        citations.append(
            {
                "chunk_id": getattr(citation, "chunk_id", ""),
                "file_name": getattr(citation, "file_name", ""),
                "chunk_content": getattr(citation, "chunk_content", "")[:160],
            }
        )

    return {
        "session_id": session_id,
        "type": "answer",
        "answer": result.answer,
        "citations": citations,
    }


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

        raw = msg.value()
        if raw is None:
            continue

        text = raw.decode().strip()
        if not text:
            continue

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print("[Router] Skipping empty Kafka message")
            continue

        print("[Router] Received from Kafka:", data)
        response = await process_message(data)
        producer.produce(
            topic="chat-responses",
            key=response["session_id"],
            value=json.dumps(response),
        )
        producer.flush()

        print("[Router] Response sent to chat-responses")


async def main():
    await initialize_runtime()
    await router_loop()


if __name__ == "__main__":
    asyncio.run(main())