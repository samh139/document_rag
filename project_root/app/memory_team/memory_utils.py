import sys
import os
import asyncio
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from memory_team.stm.store import get_stm
from memory_team.ltm.ltm_service import retrieve_ltm_context,store_conversation_to_es
from service.elastic_search_reranker import ElasticSearchReranker
from memory_team.stm.agent import store_conversation_to_stm

es = ElasticSearchReranker()

def fetch_conversation_memories(session_id: str, user_query: str, user_id: str = "1234"):
    """
    Fetch STM summary and LTM context for the given session and user query.
    """
    stm_summary = get_stm(session_id=session_id,user_id=user_id)
    ltm_context = retrieve_ltm_context(
        query=user_query,
        user_id=user_id,
        session_id=session_id
    )
    return stm_summary, ltm_context

async def store_conversation_background(user_id, session_id, user_message, bot_response):
    print("Storing conversation in background...")
    await asyncio.gather(
            asyncio.to_thread(store_conversation_to_es, user_id, session_id, user_message, bot_response),
            store_conversation_to_stm(user_id, session_id, user_message, bot_response)
        )
   
