import json
from typing import Dict, List
from app.memory_team.stm.redis_client import redis_client

SLIDING_WINDOW_SIZE = 10
def get_stm(session_id: str,user_id:str = "12345"):
    key = f"stm:{user_id}:{session_id}"
    data = redis_client.get(key)
    if data:
        return json.loads(data)
    return {
        "turn_number": 0,
        "topic": "",
        "context_summary": "",
        "entities": [],
        "last_user_message": "",
        "last_bot_message": ""
    }

def save_stm(session_id: str, stm: dict,user_id:str = "12345"):
    key = f"stm:{user_id}:{session_id}"
    redis_client.set(key, json.dumps(stm))

#Add Conversation turn to Redis 

def add_conversation(session_id: str, user_message: str, bot_message: str,user_id: str="12345"):
    """
    Push a new conversation turn to Redis and maintain a sliding window of last N turns,
    including turn_number for proper sequencing.
    """
    # Fetch last turn to get previous turn_number
    last_turns = get_last_n_conversations(session_id, user_id,n=1)
    previous_turn_number = last_turns[-1]["turn_number"] if last_turns else 0

    # Increment turn number
    new_turn_number = previous_turn_number + 1

    # Build turn object
    turn = {
        "turn_number": new_turn_number,
        "user_message": user_message,
        "bot_message": bot_message,
    }
    key = f"stm_history:{user_id}:{session_id}"
    # Push new turn to list
    redis_client.lpush(key, json.dumps(turn))
    
    # Trim list to maintain sliding window
    redis_client.ltrim(key, 0, SLIDING_WINDOW_SIZE - 1)




# Fetch last N conversation turns
def get_last_n_conversations(session_id: str,user_id:str = "12345", n: int = SLIDING_WINDOW_SIZE) -> List[Dict]:
    """
    Get last N conversation turns in chronological order (oldest first).
    """
    key = f"stm_history:{user_id}:{session_id}"

    raw_list = redis_client.lrange(key, 0, n - 1)
    
    # Reverse so oldest is first
    conversations = []
    for item in reversed(raw_list):
        try:
            conversations.append(json.loads(item))
        except json.JSONDecodeError:
            continue
    
    return conversations

def get_stm_summary(session_id: str, user_id: str= "12345"):
    stm = get_stm(session_id, user_id)
    return {
        "conversation_summary": stm.get("context_summary", ""),
        "conversation_entities": stm.get("entities", []),
        "last_user_message": stm.get("last_user_message", "")
    }

 

