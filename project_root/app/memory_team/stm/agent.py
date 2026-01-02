# stm_agent.py
import os
import sys

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Dict,List
from app.configs.llm_config import fire_fast_modal_request_chat_get_dict
from app.memory_team.stm.store import save_stm,add_conversation, get_last_n_conversations, save_stm
from app.agentic.rag.summarize_agent import get_summary_of_content



#Asynchronously update short-term memory in Redis.
async def store_conversation_to_stm(user_id:str,session_id: str, user_message: str, bot_response: str):
     
      try:
          bot_summary = await get_summary_of_content(context=bot_response)
          print("store_conversation_to_stm...")
          # Non-blocking call to STM processing
          updated_stm = process_user_query(
              user_id=user_id,
              session_id=session_id,
              user_query=user_message,
              bot_response=bot_summary
          )
          print("user-query,response: ",user_message,bot_response)
        #   print(f"STM updated for session {session_id}: {updated_stm}")
      except Exception as e:
          print("STM update failed:", e)


def process_user_query(session_id: str, user_query: str, bot_response: str,user_id:str = "12345") -> dict:
    """
    Process a user query and bot response using sliding-window STM:
    - Fetch last N conversation turns from Redis
    - Append current turn in memory
    - Generate updated STM summary using LLM
    - Save updated STM summary in Redis
    - Push current turn to Redis sliding window
    - Return updated STM
    """
    
    # Step 1: Fetch last N conversation turns (sliding window)
    sliding_conversations = get_last_n_conversations(session_id,user_id)

    # Step 2: Generate updated STM summary using sliding window + current turn
    updated_stm = update_stm_with_sliding_window(
        sliding_conversations=sliding_conversations,
        current_user_message=user_query,
        current_bot_message=bot_response
    )

    # Step 3: Save updated STM summary in Redis
    save_stm(session_id, updated_stm,user_id)

    # Step 4: Push current turn to Redis sliding window
    add_conversation(session_id, user_query, bot_response,user_id)

    # Step 5: Return updated STM
    return updated_stm



def update_stm_with_sliding_window(
    sliding_conversations: List[Dict],
    current_user_message: str,
    current_bot_message: str
) -> Dict:
    """
    Generate STM summary using LLM based solely on sliding window of last N conversations.
    """
    # Build sliding window context
    conversation_context = ""
    for turn in sliding_conversations:
        conversation_context += f"User: {turn['user_message']}\nBot: {turn['bot_message']}\n"
    
    # Append current turn
    conversation_context += f"User: {current_user_message}\nBot: {current_bot_message}\n"
    
    # System prompt with STM rules
    system_prompt = """
    You are the Short-Term Memory Agent.

    Your job is to MAINTAIN and UPDATE the conversation's short-term memory after each user-bot exchange.
    You MUST produce an UPDATED STM object based solely on the recent conversation context.

    Critical Rules:
    1. Merge all relevant context from the sliding window into one concise summary (≤100 tokens).
    2. Deduplicate entities and keep them short and relevant.
    3. If this is the first conversation turn, generate STM summary based on this turn alone.
    4. Always include the last user and bot messages verbatim.
    5. Output ONLY valid JSON — no explanations, text, or extra characters.

    Output schema:
    {
      "turn_number": <integer>,
      "topic": "<short topic capturing recent conversation>",
      "context_summary": "<≤100-token factual summary>",
      "entities": ["<entity1>", "<entity2>", ...],
      "last_user_message": "<verbatim user text>",
      "last_bot_message": "<verbatim bot text>"
    }
    """

    # User prompt for LLM
    user_prompt = f"""
    Conversation Window:
    {conversation_context}
    """

    # Call LLM to get updated STM
    updated_stm = fire_fast_modal_request_chat_get_dict(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    # Fallback in case LLM fails
    if not updated_stm or not isinstance(updated_stm, dict):
        updated_stm = {
            "turn_number": sliding_conversations[-1].get("turn_number", 0) + 1 if sliding_conversations else 1,
            "topic": "",
            "context_summary": "",
            "entities": [],
            "last_user_message": current_user_message,
            "last_bot_message": current_bot_message
        }

    return updated_stm





