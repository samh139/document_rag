#app/memory_team/ltm/create_ltm_index.py

from elasticsearch import Elasticsearch

es_host="http://localhost:9200"
es_ltm_index="conversations"
es = Elasticsearch(es_host)

index_name = es_ltm_index

mapping = {
    "mappings": {
        "properties": {
            "user_embedding": {"type": "dense_vector", "dims": 768},
            "bot_embedding": {"type": "dense_vector", "dims": 768},
            "combined_embedding": {"type": "dense_vector", "dims": 768},
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
