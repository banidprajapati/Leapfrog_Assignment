import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankerHandler:
    """Handles result reranking using cross-encoder models."""

    _reranker: CrossEncoder | None = None

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker handler with cross-encoder model.

        Args:
            model_name: HuggingFace model ID for cross-encoder
        """
        if RerankerHandler._reranker is None:
            self._load_reranker(model_name)

    def _load_reranker(self, model_name: str) -> None:
        """Load cross-encoder model."""
        logger.info(f"Loading reranker model: {model_name}...")
        try:
            RerankerHandler._reranker = CrossEncoder(model_name)
            logger.info("Reranker loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            raise

    def rerank(
        self, query: str, texts: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """
        Rerank texts based on relevance to query.

        Args:
            query: Query string
            texts: List of text snippets to rerank
            top_k: Number of top results to return (default: all)

        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        if not texts:
            logger.debug("No texts to rerank")
            return []

        if RerankerHandler._reranker is None:
            raise RuntimeError("Reranker not initialized")

        try:
            # Create pairs for cross-encoder
            pairs = [(query, text) for text in texts]
            scores = RerankerHandler._reranker.predict(pairs)

            # Create (index, score) tuples and sort by score descending
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

            top_k = top_k or len(ranked)
            result = ranked[:top_k]

            logger.debug(f"Reranked {len(texts)} texts → top {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise

    def rerank_with_texts(
        self, query: str, texts: list[str], top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """
        Rerank texts and return both text and score.

        Args:
            query: Query string
            texts: List of text snippets to rerank
            top_k: Number of top results to return (default: all)

        Returns:
            List of (text, score) tuples, sorted by score descending
        """
        ranked_indices = self.rerank(query, texts, top_k)
        return [(texts[idx], score) for idx, score in ranked_indices]
