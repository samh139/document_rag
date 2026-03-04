#!agent_system/router_service/request_reponse_router.py

from confluent_kafka import Consumer, Producer
import json

consumer = Consumer({
    "bootstrap.servers": "127.0.0.1:9092",
    "group.id": "router-group",
    "auto.offset.reset": "earliest"
})

producer = Producer({
    "bootstrap.servers": "127.0.0.1:9092"
})

consumer.subscribe(["chat-requests"])

print("Router started...")

while True:
    msg = consumer.poll(1.0)

    if msg is None:
        continue

    if msg.error():
        print("Error:", msg.error())
        continue

    data = json.loads(msg.value().decode())
    print("Received from Kafka:", data)

    session_id = data["session_id"]
    user_message = data["message"]

    response = {
        "session_id": session_id,
        "message": f"Processed: {user_message}"
    }

    producer.produce(
        topic="chat-responses",
        key=session_id,
        value=json.dumps(response)
    )

    producer.flush()

    print("Response sent to chat-responses topic")