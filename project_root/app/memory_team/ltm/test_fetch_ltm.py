#app/memory_team/ltm/test_fetch_ltm.py
from memory_utils import fetch_conversation_memories

stm_summary, ltm_convo = fetch_conversation_memories(session_id = "22fe73fb-a724-4677-9be3-3c4b64b23631",user_query = "How are you?",user_id = 1234)
print("stm_summary",stm_summary)
print("ltm_convo",ltm_convo)