from fastapi import APIRouter, HTTPException, Request
from qdrant_client.models import FieldCondition, Filter, MatchValue

from RAG_system.repositories.query_repo import RetrievalRepository
from RAG_system.schemas.request import QueryRequest
from RAG_system.schemas.response import QueryResponse
from RAG_system.services.rag_service import RAGService
from RAG_system.utils.query_parser import extract_filters

router = APIRouter(prefix="/query", tags=["query"])

retrieval_repository = RetrievalRepository()


def _build_filter(filters: dict[str, str]) -> Filter:
    """Convert a dict of metadata key-value pairs into a Qdrant Filter."""
    return Filter(
        must=[
            FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
            for k, v in filters.items()
        ]
    )


@router.post("")
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    rag_service = RAGService(client=request.app.state.openai_client)

    try:
        # Extract filters from query and merge with explicit filters
        extracted = extract_filters(body.query)
        merged = extracted.copy()
        if body.filters:
            merged.update(body.filters)
        query_filter = _build_filter(merged) if merged else None
        chunks = retrieval_repository.search_chunks(
            query=body.query,
            top_k=body.top_k,
            query_filter=query_filter,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {e}",
        )

    if not chunks:
        return QueryResponse(
            query=body.query,
            answer="No relevant jobs found.",
            chunks=[],
        )

    try:
        contexts = retrieval_repository.build_contexts(chunks)

        answer = rag_service.generate(
            body.query,
            contexts,
        )

    except Exception as e:
        answer = f"LLM generation failed: {e}"

    return QueryResponse(
        query=body.query,
        answer=answer,
        chunks=chunks,
    )
