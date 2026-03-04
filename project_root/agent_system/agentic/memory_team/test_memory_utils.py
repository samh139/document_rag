import asyncio
from memory_team.memory_utils import fetch_conversation_memories, store_conversation_background

# Example data
USER_ID = "test_user2"
SESSION_ID = "test_session2"
USER_MESSAGE = "Hi"
BOT_RESPONSE = "Hello! How can I assist you today?"
USER_QUERY = "project update"

def main():
    # --- Test fetch_conversation_memories ---
    print("Fetching conversation memories...")
    stm_summary, ltm_context = fetch_conversation_memories(
        session_id=SESSION_ID,
        user_query=USER_QUERY,
        user_id=USER_ID
    )
    print("STM Summary:", stm_summary)
    print("LTM Context:", ltm_context)

    # --- Test store_conversation_background ---
    # print("\nStoring conversation (in background)...")
    # store_conversation_background(
    #     user_id=USER_ID,
    #     session_id=SESSION_ID,
    #     user_message=USER_MESSAGE,
    #     bot_response=BOT_RESPONSE
    # )

    # # Give async background tasks time to complete
    # asyncio.run(asyncio.sleep(2))
    # print("Done! Check your storage systems (ES / STM) to confirm.")

if __name__ == "__main__":
    main()
