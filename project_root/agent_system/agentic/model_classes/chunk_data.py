from pydantic import BaseModel
from typing import Optional, List, Dict


class ChunkData(BaseModel):
    chunk_id: str
    file: str
    file_title: Optional[str] = ""
    searched_for: Optional[str] = ""
    content: str = ""

    # scoring
    relevant_score: Optional[float] = None
    bm_score: Optional[float] = None
    semantic_score: Optional[float] = None
    rrf: Optional[float] = None
    final_rank: Optional[int] = None
    final_score: float

    # metadata
    chunk_tags: Optional[List[str]] = None

    # optional / safe
    source_warning: Optional[bool] = False
    available_in_files: Optional[List[str]] = None
    raw_score: Optional[float] = None