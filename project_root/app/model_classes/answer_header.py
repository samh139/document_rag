from dataclasses import dataclass
from typing import List

@dataclass
class AnswerHeader:
    query_id:str
    user_id:str
    session_id:str
    parts_count:int
    parts_queries:List[str]