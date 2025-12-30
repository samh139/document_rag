import unittest
import uuid
from elasticsearch import Elasticsearch
from memory_team.ltm.ltm_service import store_conversation

ES_URL = "http://localhost:9200"
INDEX_NAME = "conversations"

class TestStoreConversation(unittest.TestCase):
    def setUp(self):
        self.es = Elasticsearch(ES_URL)
        assert self.es.ping(), "Elasticsearch is not running at http://localhost:9200"

    def test_store_conversation(self):
        session_id = "XYZ"
        user_id = "0987"
        # user_message = "How to download payslip in portal?"
        # bot_response = "To download payslip, login to the portal and go to the payslip section and click on download."

        user_message = "Which is the recommended browser for Oracle HCM?"
        bot_response = "Recommended browsers for Oracle HCM are Safari or Chrome."

        # Call the function to store
        doc_id = store_conversation(user_id,session_id, user_message, bot_response)
        self.assertIsNotNone(doc_id)

        # Retrieve and verify
        stored_doc = self.es.get(index=INDEX_NAME, id=doc_id)["_source"]
        self.assertEqual(stored_doc["user_message"], user_message)
        self.assertEqual(stored_doc["bot_response"], bot_response)
        self.assertIn("combined_embedding", stored_doc)
        self.assertEqual(len(stored_doc["combined_embedding"]), 384)

if __name__ == "__main__":
    unittest.main()
