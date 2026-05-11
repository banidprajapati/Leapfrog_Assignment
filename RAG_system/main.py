from fastapi import FastAPI

from RAG_system.routers.query import router as query_router

app = FastAPI(title="LF RAG System", version="0.1.0")
app.include_router(query_router)
