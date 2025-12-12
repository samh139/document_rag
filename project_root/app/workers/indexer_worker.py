# app/workers/indexer_worker.py 
import os
import json
from kafka import KafkaConsumer
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
import time

load_dotenv("app/config/secrets.env")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP") or os.getenv("KAFKA_BOOTSTRAP_LOCALHOST")
EMBED_TOPIC = os.getenv("KAFKA_EMBEDDED_TOPIC", "dsprawl.embedded_chunks")
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
INDEX_NAME = os.getenv("ES_INDEX", "dsprawl_documents")

print(f"DEBUG: KAFKA_BOOTSTRAP set to: {KAFKA_BOOTSTRAP}")
print(f"DEBUG: Consuming from topic: {EMBED_TOPIC}")
print(f"DEBUG: Connecting to ES host: {ES_HOST}")

es = Elasticsearch(ES_HOST)

def index_chunk(doc: dict):
    # Map to ES document
    body = {
        "doc_id": doc["doc_id"],
        "chunk_id": doc["chunk_id"],
        "content": doc["content"],
        "metadata": doc.get("metadata", {}),
        "owner": doc.get("owner"),
        "acl": doc.get("acl", []),
        "embedding_vector": doc.get("embedding_vector"),
        "etag": doc.get("etag"),
    }
    es.index(index=INDEX_NAME, id=doc["chunk_id"], document=body)


try:
    consumer = KafkaConsumer(
        EMBED_TOPIC,
        bootstrap_servers=[KAFKA_BOOTSTRAP],
        auto_offset_reset="earliest",
        group_id="indexer-worker",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        # Add max_poll_interval_ms might help in debugging long connections
        # max_poll_interval_ms=300000 
    )
    print("DEBUG: Kafka Consumer initialized successfully.")
except Exception as e:
    print(f"ERROR: Could not connect to Kafka brokers {KAFKA_BOOTSTRAP}. Exception: {e}")
    exit(1) # Stop execution if connection fails

# ... es client initialization ...

print("DEBUG: Starting main consumer loop.")
for msg in consumer:
    try:
        #print("Message recieved in consumer :", msg)
        print("Indexing process is started..")
        index_chunk(msg.value)
        print("Indexing process is completed..")
    except Exception as e:
        print("index error:", e)
