from dataclasses import dataclass

@dataclass
class FeedbackInput:
    session_id: str
    query_id: str
    user_id: str
    user_question: str
    agent_answer: str
    feedback_type: str