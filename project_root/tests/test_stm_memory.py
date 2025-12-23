from app.memory.memory_manager import MemoryManager

def test_stm_basic_flow():
    memory = MemoryManager()

    role = "CUSTOMER"
    user_id = "user-123"
    session_id = "sess-1"

    # Clear old session
    memory.stm.clear_session(role, user_id, session_id)

    # Store turns
    memory.store_turn(
        role=role,
        user_id=user_id,
        session_id=session_id,
        user_query="Hi",
        assistant_answer="Hello! How can I help?",
    )

    memory.store_turn(
        role=role,
        user_id=user_id,
        session_id=session_id,
        user_query="What documents are needed?",
        assistant_answer="PAN and Aadhaar are required.",
    )

    # Fetch STM
    history = memory.get_stm(
        role=role,
        user_id=user_id,
        session_id=session_id,
    )

    print("\nSTM HISTORY:")
    for h in history:
        print(h)

    assert len(history) == 2


# 🔑 THIS IS THE IMPORTANT PART
if __name__ == "__main__":
    test_stm_basic_flow()
