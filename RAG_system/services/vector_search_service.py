from qdrant_client import QdrantClient
from qdrant_client.models import Document, ScoredPoint
from sentence_transformers import SentenceTransformer

from RAG_system.core.config_core import settings
from RAG_system.core.logging_core import get_logger

logger = get_logger(__name__)


class VectorSearchHandler:
    _embedder: SentenceTransformer | None = None

    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_CLUSTER_ENDPOINT,
            api_key=settings.QDRANT_APIKEY,
        )
        if VectorSearchHandler._embedder is None:
            VectorSearchHandler._embedder = SentenceTransformer(
                "BAAI/bge-small-en-v1.5",
                token=settings.HF_TOKEN,
            )
        self.embedder = VectorSearchHandler._embedder

    def search(self, query: str, top_k: int = 5) -> list[ScoredPoint]:
        """Perform dense vector search only."""
        query_vector = self.embedder.encode(query, normalize_embeddings=True)
        result = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=top_k,
            using="dense",
        )
        return result.points

    def hybrid_search(
        self, query: str, top_k: int = 5, dense_weight: float = 0.7
    ) -> list[ScoredPoint]:
        """Hybrid search using dense embeddings and BM25 sparse vectors."""
        query_vector = self.embedder.encode(query, normalize_embeddings=True)

        # Search with dense vector (scores 0-1, higher is better)
        dense_response = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=top_k * 3,
            using="dense",
        )
        dense_results = dense_response.points

        # Search with BM25 sparse vector (higher is better)
        try:
            bm25_response = self.client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=Document(text=query, model="Qdrant/bm25"),
                limit=top_k * 3,
                using="bm25",
            )
            bm25_results = bm25_response.points
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}, using dense only")
            bm25_results = []

        # Normalize and fuse results
        scores: dict[str, float] = {}
        all_points: dict[str, ScoredPoint] = {}

        # Process dense results (score 0-1)
        for r in dense_results:
            scores[str(r.id)] = (r.score or 0) * dense_weight
            all_points[str(r.id)] = r

        # Process BM25 results - normalize to 0-1 range for fusion
        if bm25_results:
            max_bm25 = max((r.score or 0) for r in bm25_results)
            for r in bm25_results:
                pid = str(r.id)
                # Normalize BM25 score to 0-1 range
                normalized_bm25 = (r.score or 0) / max_bm25 if max_bm25 > 0 else 0
                bm25_score = normalized_bm25 * (1 - dense_weight)
                scores[pid] = scores.get(pid, 0) + bm25_score
                if pid not in all_points:
                    all_points[pid] = r

        # Sort by fused score
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top_k with fused scores
        fused_results = []
        for point_id, fused_score in sorted_ids[:top_k]:
            result = all_points[point_id]
            result.score = fused_score
            fused_results.append(result)

        return fused_results
