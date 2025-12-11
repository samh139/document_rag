# app/workers/embed_worker.py
import os
import json
import time
import requests
from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv

load_dotenv("app/config/secrets.env")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP") or os.getenv("KAFKA_BOOTSTRAP_LOCALHOST")
INPUT_TOPIC = os.getenv("KAFKA_CHUNKS_TOPIC", "dsprawl.chunks")
OUTPUT_TOPIC = os.getenv("KAFKA_EMBEDDED_TOPIC", "dsprawl.embedded_chunks")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")  # or local ollama socket

consumer = KafkaConsumer(
    INPUT_TOPIC,
    bootstrap_servers=[KAFKA_BOOTSTRAP],
    auto_offset_reset="earliest",
    group_id="embed-worker",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BOOTSTRAP],
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def get_embedding(text: str):
    print(f"Embedding request started for text length: {len(text)} characters") # Add this line
    body = {"model": "nomic-embed-text", "input": text}
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json=body, timeout=120) # Use a longer timeout
    resp.raise_for_status()
    out = resp.json()
    return out.get("embedding") or out


for msg in consumer:
    try:
        chunk = msg.value
        vec = get_embedding(chunk["content"])
        chunk["embedding_vector"] = vec
        producer.send(OUTPUT_TOPIC, chunk)
        producer.flush()
    except Exception as e:
        print("embed error:", e)
        time.sleep(1)
