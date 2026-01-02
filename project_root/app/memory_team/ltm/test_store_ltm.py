#app/memory_team/ltm/test_store_ltm.py
import unittest
import uuid
from elasticsearch import Elasticsearch
from app.memory_team.ltm.ltm_service import store_conversation_to_es

ES_URL = "http://localhost:9200"
INDEX_NAME = "conversations"

class TestStoreConversation(unittest.TestCase):

    def setUp(self):
        self.es = Elasticsearch(ES_URL)
        assert self.es.ping(), f"Elasticsearch is not running at {ES_URL}"

    def test_store_conversation(self):
        session_id = f"test-sess-{uuid.uuid4()}"
        user_id = f"test-user-{uuid.uuid4()}"

        user_message = "documents required for opening a bank account"
        bot_response = (
            "To open an individual account, the following documents are required: "
            "OVD such as passport or driving license, Aadhaar, Voter ID, "
            "PAN or Form 60."
        )

        # Store
        doc_id = store_conversation_to_es(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            bot_response=bot_response,
        )

        self.assertIsNotNone(doc_id)

        # Retrieve
        stored_doc = self.es.get(index=INDEX_NAME, id=doc_id)["_source"]

        self.assertEqual(stored_doc["user_message"], user_message)
        self.assertEqual(stored_doc["bot_response"], bot_response)
        self.assertIn("combined_embedding", stored_doc)
        self.assertEqual(len(stored_doc["combined_embedding"]), 384)

        # Cleanup (optional)
        self.es.delete(index=INDEX_NAME, id=doc_id)

if __name__ == "__main__":
    unittest.main()