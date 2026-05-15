from contextlib import asynccontextmanager

from fastapi import FastAPI

from RAG_system.core.logging_core import get_logger
from RAG_system.core.openai_client import close_openai_client, get_openai_client
from RAG_system.repositories.query_repo import RetrievalRepository
from RAG_system.routers.query_router import router as query_router
from RAG_system.services.reranker_service import RerankerHandler
from RAG_system.services.retrieval_service import RetrievalService
from RAG_system.services.vector_search_service import VectorSearchHandler

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load shared OpenAI client
    logger.info("Application Initialized")
    app.state.openai_client = get_openai_client()
    # Load and attach shared retrieval components
    logger.info("Loading shared retrieval components")
    app.state.reranker = RerankerHandler()
    app.state.vector_search = VectorSearchHandler()
    app.state.retrieval_service = RetrievalService(
        vector_search=app.state.vector_search, reranker_handler=app.state.reranker
    )
    app.state.retrieval_repository = RetrievalRepository(
        retrieval_service=app.state.retrieval_service
    )

    yield

    # Cleanup
    logger.info("Application Closing")
    # Close Qdrant client if present
    try:
        if hasattr(app.state, "vector_search") and getattr(
            app.state.vector_search, "client", None
        ):
            try:
                app.state.vector_search.client.close()
            except Exception:
                logger.warning("Failed to close Qdrant client cleanly")
    except Exception:
        logger.exception("Error during vector search cleanup")

    close_openai_client(app.state.openai_client)


app = FastAPI(
    title="LF RAG System",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(query_router)
