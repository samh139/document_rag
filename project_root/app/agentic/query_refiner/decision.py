from pydantic import BaseModel


class QueryDecision(BaseModel):
    use_memory: bool
    use_rag: bool
    reason: str
