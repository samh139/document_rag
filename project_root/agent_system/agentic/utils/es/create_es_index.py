#app/memory_team/ltm/create_ltm_index.py

from elasticsearch import Elasticsearch

es_ltm_index="conversations"

ltm_index_name = es_ltm_index

ltm_mapping = {
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

agent_responses_index_name = "agent_responses"

agent_responses_mapping = {
    "mappings": {
        "properties": {
            "session_id": {"type": "keyword"},
            "query_id": {"type": "keyword"},
            "user_id": {"type": "keyword"},
            "user_input": {"type": "text"},
            "bot_response": {"type": "text"},
            "citations": {
                "type": "nested",
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "chunk_content": {"type": "text"}
                }
            },
            "timestamp": {"type": "date"}
        }
    }
}