import json
from typing import List

from scripts.build_vector_db import VectorDBBuilder

from RAG_system.core.config_core import settings

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


def seed_vector_db_from_file(chunk_path: str, saved_output_path: str | None = None):
    builder = VectorDBBuilder(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
    )

    chunks = builder.load_chunks(chunk_path)
    saved_records = builder.upload_chunks(chunks)

    if saved_output_path:
        with open(saved_output_path, "w", encoding="utf-8") as f:
            json.dump(saved_records, f, indent=2, default=str)

    return saved_records


def main():
    saved = seed_vector_db_from_file(
        chunk_path="RAG_system/data/processed/chunks.json",
        saved_output_path="RAG_system/data/processed/saved_vectors.json",
    )
    print(f"Saved metadata for {len(saved)} vectors.")


if __name__ == "__main__":
    main()
