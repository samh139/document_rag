from elasticsearch import Elasticsearch
from app_configs.app_env import app_env

es = Elasticsearch(hosts=[app_env.get_es_host()])

index_name = app_env.get_es_ltm_index()

mapping = {
    "mappings": {
        "properties": {
            "user_embedding": {"type": "dense_vector", "dims": 384},
            "bot_embedding": {"type": "dense_vector", "dims": 384},
            "combined_embedding": {"type": "dense_vector", "dims": 384},
            "user_id": {"type": "keyword"},
            "session_id": {"type": "keyword"},
            "user_message": {"type": "text"},
            "bot_response": {"type": "text"},
            "timestamp": {"type": "date"}
        }
    }
} 

# Create the index if it doesn't exist
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)
    print(f"Index '{index_name}' created")
else:
    print(f"Index '{index_name}' already exists")
