# app/workers/indexer_worker.py
import os
import json
from kafka import KafkaConsumer
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

load_dotenv("app/config/secrets.env")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
EMBED_TOPIC = os.getenv("KAFKA_EMBEDDED_TOPIC", "dsprawl.embedded_chunks")
ES_HOST = os.getenv("ES_HOST", "http://es:9200")
INDEX_NAME = os.getenv("ES_INDEX", "dsprawl_documents")

consumer = KafkaConsumer(
    EMBED_TOPIC,
    bootstrap_servers=[KAFKA_BOOTSTRAP],
    auto_offset_reset="earliest",
    group_id="indexer-worker",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)

es = Elasticsearch(ES_HOST)  # add basic_auth if needed

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

for msg in consumer:
    try:
        index_chunk(msg.value)
    except Exception as e:
        print("index error:", e)
