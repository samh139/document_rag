#!agent_system/websocket_service/kafka_producer.py

from confluent_kafka import Producer
import json

producer = Producer({
    "bootstrap.servers": "127.0.0.1:9092"
})

def send_message(topic, key, value):
    producer.produce(
        topic=topic,
        key=key,
        value=json.dumps(value)
    )
    producer.poll(0)