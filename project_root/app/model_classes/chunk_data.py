from pydantic import BaseModel
from typing import Optional,List

from typing import List, Optional,Dict

class ChunkData(BaseModel):
    content:str
    chunk_id:str
    file:str
    file_title:str
    searched_for:str
    relevant_score:Optional[float] = None
    bm_score: Optional[float] = None
    semantic_score: Optional[float] = None
    rrf: Optional[float] = None
    final_rank: Optional[int] = None
    final_score:float
    chunk_tags:Optional[List[str]]
    source_warning: Optional[bool] = False
    available_in_files: Optional[List[str]] = None
    raw_score: Optional[float] = None
    parent_file: Optional[str] = None
    media_properties:Dict

