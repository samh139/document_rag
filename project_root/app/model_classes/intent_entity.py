from dataclasses import dataclass
from typing import List

@dataclass
class IntentEntity:
    name:str
    node_type:str
    intent_family:str
    behavioral_family:str
    default_scope:str
    examples:List[str]
    tags:List[str]




