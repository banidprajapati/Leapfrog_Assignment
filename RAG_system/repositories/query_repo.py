from typing import List, Optional

from qdrant_client.models import Filter
from RAG_system.schemas.response import ChunkResult
from RAG_system.services.retrieval_service import RetrievalService


class RetrievalRepository:
    def __init__(self):
        self.retrieval = RetrievalService()

    def search_chunks(self, query: str, top_k: int, query_filter: Optional[Filter] = None) -> List[ChunkResult]:
        results = self.retrieval.search(query, top_k=top_k, query_filter=query_filter)

        chunks = []

        # Convert vector results into response schema
        for r in results:
            payload = r.payload or {}
            metadata = payload.get("metadata", {})
            chunks.append(
                ChunkResult(
                    text=payload.get("text", ""),
                    chunk_id=metadata.get("chunk_id"),
                    job_id=metadata.get("job_id"),
                    job_title=metadata.get("job_title"),
                    company=metadata.get("company"),
                    category=metadata.get("category"),
                    level=metadata.get("level"),
                    location=metadata.get("location"),
                    tags=metadata.get("tags", []),
                    score=getattr(r, "score", None),
                )
            )

        return chunks

    def build_contexts(self, chunks: List[ChunkResult]) -> List[str]:
        contexts = []

        # Format chunks into prompt-ready context blocks
        for c in chunks[:10]:
            tags_str = ", ".join(c.tags) if c.tags else "N/A"

            formatted = f"""[Job ID: {c.job_id}]
Title: {c.job_title}
Company: {c.company}
Level: {c.level}
Location: {c.location}
Category: {c.category}
Tags: {tags_str}

{c.text}"""

            contexts.append(formatted)

        return contexts
