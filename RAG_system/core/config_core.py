from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration loaded from `.env` files.
    Uses Pydantic for validation and type safety across runtime environments.
    """

    # Core App Info
    PROJECT_NAME: str = "LF RAG System"
    ENVIRONMENT: str = "dev"

    # AWS configs
    QDRANT_APIKEY: str = Field(..., description="Qdrant API Key")
    QDRANT_CLUSTER_ENDPOINT: str = Field(..., description="Qdrant Cluster Key")
    COLLECTION_NAME: str = Field(..., description="Qdrant DB Collection Name")
    HF_TOKEN: str = Field(..., description="Hugging Face Token")
    OPENROUTER_API: str = Field(..., description="Openrouter API Key")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Creates and caches a single Settings instance (maxsize=1) to avoid repeated environment parsing.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
