# demo.py
from memory_team.stm.agent import process_user_query, get_last_n_conversations
from memory_team.stm.store import get_stm
import json
from typing import Dict, List

if __name__ == "__main__":
    session_id = "22fe73fb-a724-4677-9be3-3c4b64b23631"
    user_id = 1234

    # Simulate multiple turns
    conversation_turns = [
        {
            "user": "How to download payslip in portal?",
            "bot": "To download payslip, login to the portal and go to the payslip section and click on download."
        },
        {
            "user": "Which is the recommended browser for Oracle HCM?",
            "bot": "Safari or Chrome are recommended browsers for Oracle HCM."
        },
        {
            "user": "How to apply for leave in Oracle HCM?",
            "bot": "Navigate to Time and Attendance > Request Leave, fill out the form, and submit."
        },
        {
            "user": "Can I see my benefits online?",
            "bot": "Yes, log in to the HR portal and check the Benefits section."
        }
    ]

    # Process each turn
    for idx, turn in enumerate(conversation_turns, start=1):
        stm = process_user_query(session_id, turn["user"], turn["bot"],user_id)
        print(f"\n=== STM after turn {idx} ===")
        print(json.dumps(stm, indent=2))

    # Optional: Print current sliding window in Redis
    sliding_window = get_last_n_conversations(session_id,user_id)
    print("\n=== Current sliding window (last N turns) ===")
    print(json.dumps(sliding_window, indent=2))

    # Get the final STM summary for the session
    final_stm = get_stm(session_id,user_id)
    print("\n=== Final STM Summary ===")
    print(json.dumps(final_stm, indent=2))
