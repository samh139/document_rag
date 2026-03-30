from datetime import datetime, timezone
from elasticsearch import Elasticsearch, helpers
import yaml, os
import uuid
import json
from agent_system.agentic.utils.es.es_utils import get_es_connection
import logging
from agent_system.agentic.utils.es.index_bootstrap import agent_responses_index_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_response_store_service")

class RAGResultsStoreService:
    def __init__(self, index_name=agent_responses_index_name):
        # ✅ Use centralized ES connection
        self.es = get_es_connection()
        self.index_name = index_name

    async def store_agent_response(self, response: dict):
        try:
            response["timestamp"] = datetime.now(timezone.utc).isoformat()

            res = self.es.index(
                index=self.index_name,
                document=response,
                refresh=True
            )
            logger.info(f"✅ Agent response indexed successfully: {res}")
            return res

        except Exception as e:
            logger.exception(f"❌ Exception during ES indexing: {e}")
            return None