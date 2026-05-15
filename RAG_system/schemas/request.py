from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")
    filters: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional metadata filters, e.g. {\"level\": \"Senior Level\", \"company\": \"Walmart\"}",
    )
