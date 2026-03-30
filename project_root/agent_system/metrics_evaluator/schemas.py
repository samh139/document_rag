from typing import List
from pydantic import BaseModel

class MetricsEvaluationRequest(BaseModel):
    query_id: str


class MetricsEvaluationInputItem(BaseModel):
    user_input: str
    response: str
    retrieved_contexts: List[str]