# core/openai_client.py

from openai import OpenAI

from RAG_system.core.config_core import settings


def get_openai_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENROUTER_API,
    )


def close_openai_client(client: OpenAI) -> None:
    client.close()
