from dataclasses import dataclass, field
from typing import List

@dataclass
class UserInput:
    message_text:str
    user_id:str
    session_id:str
    user_name:str
    user_role:str
    user_country:str
    file_names: List[str]= field(default_factory=list)
