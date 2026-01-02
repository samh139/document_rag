# tests/test_memory_manager.py

from app.memory.memory_manager import MemoryManager


def test_stm_memory_flow():
    memory = MemoryManager()

    role = "customer"
    user_id = "user_123"
    session_id = "sess_abc"

    # Clear any previous state
    memory.clear_session(role, user_id, session_id)

    # Turn 1
    memory.store_turn(
        role=role,
        user_id=user_id,
        session_id=session_id,
        user_query="Hi",
        assistant_answer="Hello! How can I help?"
    )

    summary1 = memory.update_stm_summary(
        role=role,
        user_id=user_id,
        session_id=session_id,
        current_user_message="Hi",
        current_bot_message="Hello! How can I help?"
    )

    print("\nSTM SUMMARY AFTER TURN 1:")
    print(summary1)

    # Turn 2
    memory.store_turn(
        role=role,
        user_id=user_id,
        session_id=session_id,
        user_query="What documents are needed?",
        assistant_answer="PAN and Aadhaar are required."
    )

    summary2 = memory.update_stm_summary(
        role=role,
        user_id=user_id,
        session_id=session_id,
        current_user_message="What documents are needed?",
        current_bot_message="PAN and Aadhaar are required."
    )

    print("\nSTM SUMMARY AFTER TURN 2:")
    print(summary2)

    # Fetch cached summary
    cached = memory.get_stm_summary_cached(
        role=role,
        user_id=user_id,
        session_id=session_id,
    )

    print("\nCACHED STM SUMMARY:")
    print(cached)

    # Fetch raw turns
    turns = memory.get_stm_turns(
        role=role,
        user_id=user_id,
        session_id=session_id,
    )

    print("\nRAW STM TURNS:")
    for t in turns:
        print(f"User: {t.user} | Assistant: {t.assistant}")


if __name__ == "__main__":
    test_stm_memory_flow()
