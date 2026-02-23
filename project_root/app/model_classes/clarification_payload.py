from model_classes.chunk_data import ChunkData
from dataclasses import dataclass
from typing import List

@dataclass
class ClarificationPayload:
    user_input:str
    chunks:List[ChunkData]
    payload:dict