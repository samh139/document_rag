# app/memory/memory_manager.py

from typing import List, Dict, Optional
from datetime import datetime

from app.memory.stm_adapter import STMAdapter
from app.memory.schemas import ConversationTurn
from app.memory.acl import validate_role
from app.memory.stm_summarizer import generate_stm_summary
from app.memory.stm_summary_adapter import STMSummaryAdapter



class MemoryManager:
    def __init__(self):
        self.stm = STMAdapter()
        self.stm_summary = STMSummaryAdapter()

    # -------------------------------
    # STORE TURN (USER + ASSISTANT)
    # -------------------------------
    def store_turn(
        self,
        role: str,
        user_id: str,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        metadata: Optional[dict] = None,
    ) -> None:

        validate_role(role)

        turn = ConversationTurn(
            user=user_query,
            assistant=assistant_answer,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata or {},
        )

        self.stm.add_turn(
            role=role,
            user_id=user_id,
            session_id=session_id,
            turn=turn,
        )

    # -------------------------------
    # GET RAW STM (LAST N TURNS)
    # -------------------------------
    def get_stm_turns(
        self,
        role: str,
        user_id: str,
        session_id: str,
        limit: int = 5,
    ) -> List[ConversationTurn]:

        validate_role(role)

        return self.stm.get_turns(
            role=role,
            user_id=user_id,
            session_id=session_id,
            limit=limit,
        )

    # -------------------------------
    # GET STM SUMMARY (LLM)
    # -------------------------------
    def get_stm_summary(
        self,
        role: str,
        user_id: str,
        session_id: str,
        current_user_message: str,
        current_bot_message: str,
        window: int = 5,
    ) -> Dict:

        validate_role(role)

        turns = self.get_stm_turns(
            role=role,
            user_id=user_id,
            session_id=session_id,
            limit=window,
        )
        print("TURNSSS:", turns)

        sliding_turns = [
            {
                "user": t.user,
                "assistant": t.assistant,
            }
            for t in turns
        ]
        print("SLIDING TURNS:", sliding_turns)

        return generate_stm_summary(
            sliding_turns=sliding_turns,
            current_user_message=current_user_message,
            current_bot_message=current_bot_message,
        )
    
    def update_stm_summary(
        self,
        role: str,
        user_id: str,
        session_id: str,
        current_user_message: str,
        current_bot_message: str,
        window: int = 5,
    ) -> dict:

        validate_role(role)

        turns = self.get_stm_turns(
            role=role,
            user_id=user_id,
            session_id=session_id,
            limit=window,
        )

        sliding_turns = [
            {"user": t.user, "assistant": t.assistant}
            for t in turns
        ]

        summary = generate_stm_summary(
            sliding_turns=sliding_turns,
            current_user_message=current_user_message,
            current_bot_message=current_bot_message,
        )

        self.stm_summary.set_summary(
            role=role,
            user_id=user_id,
            session_id=session_id,
            summary=summary,
        )

        return summary
    
    def get_stm_summary_cached(
        self,
        role: str,
        user_id: str,
        session_id: str,
    ) -> Optional[dict]:

        validate_role(role)

        return self.stm_summary.get_summary(
            role=role,
            user_id=user_id,
            session_id=session_id,
        )



    # -------------------------------
    # CLEAR SESSION
    # -------------------------------
    def clear_session(
        self,
        role: str,
        user_id: str,
        session_id: str,
    ) -> None:

        validate_role(role)
        self.stm.clear_session(role, user_id, session_id)
