from fastapi import APIRouter, HTTPException, Request

from RAG_system.repositories.query_repo import RetrievalRepository
from RAG_system.schemas.request import QueryRequest
from RAG_system.schemas.response import QueryResponse
from RAG_system.services.rag_service import RAGService

router = APIRouter(prefix="/query", tags=["query"])

retrieval_repository = RetrievalRepository()


@router.post("")
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    rag_service = RAGService(client=request.app.state.openai_client)

    try:
        chunks = retrieval_repository.search_chunks(
            query=body.query,
            top_k=body.top_k,
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
