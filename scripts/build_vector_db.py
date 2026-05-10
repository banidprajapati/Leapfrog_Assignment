import json
import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

COLLECTION_NAME = "LF_RAG_System"


class VectorDBBuilder:
    _embedders: dict = {}

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        if embedding_model not in VectorDBBuilder._embedders:
            VectorDBBuilder._embedders[embedding_model] = SentenceTransformer(embedding_model)
        self.embedder = VectorDBBuilder._embedders[embedding_model]

    def create_collection(self, vector_size: int):
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if COLLECTION_NAME in existing:
            print(f"Collection '{COLLECTION_NAME}' already exists.")
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        print(f"Created collection: {COLLECTION_NAME}")

    def embed_texts(self, texts: List[str]):
        return self.embedder.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def load_chunks(self, chunk_path: str):
        with open(chunk_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize_chunk(self, chunk: dict) -> tuple[str, dict, str]:
        """Normalize chunks from preprocess output or legacy format."""
        if "text" in chunk:
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {}) or {}
            point_id = str(uuid.uuid4())
            return text, metadata, point_id

        text = chunk.get("chunk_text", "")
        metadata = {
            "chunk_id": chunk.get("chunk_id"),
            "job_id": chunk.get("job_id"),
            "job_title": chunk.get("job_title"),
            "company": chunk.get("company"),
            "category": chunk.get("category"),
            "level": chunk.get("level"),
            "location": chunk.get("location"),
            "publication_date": chunk.get("publication_date"),
            "tags": chunk.get("tags", []),
        }
        point_id = str(uuid.uuid4())
        return text, metadata, point_id

    def upload_chunks(self, chunks: List[dict], batch_size: int = 64):
        if not chunks:
            return []

        vector_size = self.embedder.get_sentence_embedding_dimension()

        self.create_collection(vector_size)

        saved_records = []

        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i : i + batch_size]

            normalized_batch = [self._normalize_chunk(chunk) for chunk in batch]
            texts = [text for text, _, _ in normalized_batch]
            embeddings = self.embed_texts(texts)

            points = []

            for (text, metadata, point_id), embedding in zip(
                normalized_batch, embeddings
            ):
                payload = {
                    "text": text,
                    "metadata": metadata,
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                )

                saved_records.append(
                    {
                        "point_id": point_id,
                        "text": text,
                        "metadata": metadata,
                    }
                )

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )

        print("Finished uploading vectors.")
        return saved_records
