import json
from typing import List

from RAG_system.core.config_core import settings
from scripts.build_vector_db import VectorDBBuilder

QDRANT_URL = settings.QDRANT_CLUSTER_ENDPOINT
QDRANT_API_KEY = settings.QDRANT_APIKEY


def seed_vector_db(chunks: List[dict], saved_output_path: str | None = None):
    builder = VectorDBBuilder(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
    )

    saved_records = builder.upload_chunks(chunks)

    if saved_output_path:
        with open(saved_output_path, "w", encoding="utf-8") as f:
            json.dump(saved_records, f, indent=2, default=str)

    return saved_records
