from contextlib import asynccontextmanager

from fastapi import FastAPI

from RAG_system.core.openai_client import close_openai_client, get_openai_client
from RAG_system.routers.query_router import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load shared OpenAI client
    app.state.openai_client = get_openai_client()

    yield

    # Cleanup
    close_openai_client(app.state.openai_client)


app = FastAPI(
    title="LF RAG System",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(query_router)
