from app.memory_team.stm.store import get_stm, get_last_n_conversations

def debug_read_stm(session_id, user_id):
    stm = get_stm(session_id, user_id)
    turns = get_last_n_conversations(session_id, user_id)

    print("STM SUMMARY:")
    print(stm)

    print("\nRECENT TURNS:")
    for t in turns:
        print(t)
