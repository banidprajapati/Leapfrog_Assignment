import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Document,
    Modifier,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from RAG_system.core.logging_core import get_logger

logger = get_logger(__name__)
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
            VectorDBBuilder._embedders[embedding_model] = SentenceTransformer(
                embedding_model
            )
        self.embedder = VectorDBBuilder._embedders[embedding_model]

    def create_collection(self, vector_size: int):
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if COLLECTION_NAME in existing:
            print(f"Collection '{COLLECTION_NAME}' already exists. Recreating...")
            self.client.delete_collection(COLLECTION_NAME)
            print("Deleted old collection.")

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(
                    modifier=Modifier.IDF,
                    index=SparseIndexParams(on_disk=False),
                )
            },
        )

        print(f"Created collection: {COLLECTION_NAME} (dense + BM25 sparse vectors)")

    def embed_texts(self, texts: List[str]):
        return self.embedder.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

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

        # Compute average document length for BM25
        all_texts = []
        normalized_all = []
        for chunk in chunks:
            text, metadata, point_id = self._normalize_chunk(chunk)
            all_texts.append(text)
            normalized_all.append((text, metadata, point_id))

        avg_document_length = (
            sum(len(text.split()) for text in all_texts) / len(all_texts)
            if all_texts
            else 0
        )
        logger.info(f"Average document length: {avg_document_length:.2f}")

        self.create_collection(vector_size)

        saved_records = []

        for i in tqdm(range(0, len(normalized_all), batch_size)):
            batch = normalized_all[i : i + batch_size]

            texts = [text for text, _, _ in batch]
            embeddings = self.embedder.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            )

            points = []

            for (text, metadata, point_id), embedding in zip(batch, embeddings):
                payload = {
                    "text": text,
                    "metadata": metadata,
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector={
                            "dense": embedding.tolist(),
                            "bm25": Document(
                                text=text,
                                model="Qdrant/bm25",
                                options={"avg_len": avg_document_length},
                            ),
                        },
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
