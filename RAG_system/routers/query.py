from fastapi import APIRouter, HTTPException

from RAG_system.schemas.request import QueryRequest
from RAG_system.schemas.response import ChunkResult, QueryResponse
from RAG_system.services.rag_service import RAGService
from RAG_system.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/query", tags=["query"])

retrieval = RetrievalService()
rag = RAGService()


@router.post("")
async def query(body: QueryRequest) -> QueryResponse:
    try:
        results = retrieval.search(body.query, top_k=body.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    if not results:
        return QueryResponse(
            query=body.query, answer="No relevant jobs found.", chunks=[]
        )

    chunks = []
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

    try:
        # Format each chunk with full metadata for the LLM
        contexts = []
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

        answer = rag.generate(body.query, contexts)

    except Exception as e:
        answer = f"LLM generation failed: {e}"

    return QueryResponse(query=body.query, answer=answer, chunks=chunks)
