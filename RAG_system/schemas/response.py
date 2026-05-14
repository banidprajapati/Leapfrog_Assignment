from typing import Optional

from pydantic import BaseModel


class ChunkResult(BaseModel):
    text: str

    chunk_id: Optional[str] = None
    job_id: Optional[str] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    category: Optional[str] = None
    level: Optional[str] = None
    location: Optional[str] = None
    tags: Optional[list[str]] = None

    # core scoring (final score after fusion/rerank)
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    chunks: list[ChunkResult]
