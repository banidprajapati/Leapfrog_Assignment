from qdrant_client.models import ScoredPoint

from RAG_system.core.logging_core import get_logger
from RAG_system.services.reranker_service import RerankerHandler
from RAG_system.services.vector_search_service import VectorSearchHandler

logger = get_logger(__name__)


class RetrievalService:
    """Retrieval service combining hybrid vector search and cross-encoder reranking."""

    def __init__(
        self,
        vector_search: VectorSearchHandler,
        reranker_handler: RerankerHandler,
    ):
        """Initialize retrieval service with shared handlers from app.state.

        Both `vector_search` and `reranker_handler` must be provided by the
        application startup code (e.g., stored on `app.state`). This enforces
        a single shared instance for each resource.
        """
        if vector_search is None or reranker_handler is None:
            raise ValueError(
                "RetrievalService requires `vector_search` and `reranker_handler` to be provided from app.state"
            )

        logger.info("Initializing RetrievalService with injected handlers...")
        self.vector_search = vector_search
        self.reranker_handler = reranker_handler
        logger.info("RetrievalService initialized successfully")

    def _extract_text(self, payload) -> str:
        """Extract primary text from payload."""
        try:
            if isinstance(payload, dict):
                if "text" in payload:
                    return payload.get("text", "")
                if "metadata" in payload and isinstance(payload["metadata"], dict):
                    return payload["metadata"].get("text", "")
                for key in ["content", "body", "description", "summary"]:
                    if key in payload:
                        return payload.get(key, "")
                return ""

            if isinstance(payload, list):
                if len(payload) > 0:
                    if isinstance(payload[0], dict):
                        return self._extract_text(payload[0])
                    if isinstance(payload[0], str):
                        return " ".join(str(item) for item in payload)
                return ""

            if isinstance(payload, str):
                return payload

            logger.warning(f"Unexpected payload type: {type(payload)}")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from payload: {e}")
            return ""

    def _extract_metadata(self, payload) -> dict:
        """Extract metadata dict from payload safely."""
        if isinstance(payload, dict):
            meta = payload.get("metadata", {})
            if isinstance(meta, dict):
                return meta
        return {}

    def _compose_rerank_text(self, point: ScoredPoint) -> str:
        """Build reranker input text that includes metadata + content."""
        payload = point.payload or {}
        metadata = self._extract_metadata(payload)
        content = self._extract_text(payload)

        meta_parts = [
            f"Job ID: {metadata.get('job_id', '')}",
            f"Title: {metadata.get('job_title', '')}",
            f"Company: {metadata.get('company', '')}",
            f"Level: {metadata.get('level', '')}",
            f"Category: {metadata.get('category', '')}",
            f"Location: {metadata.get('location', '')}",
            f"Tags: {metadata.get('tags', '')}",
            f"Published_date: {metadata.get('publication_date', '')}",
        ]

        return "\n".join(meta_parts) + "\n\n" + content

    def rerank_results(
        self,
        query: str,
        results: list[ScoredPoint],
        top_k: int | None = None,
    ) -> list[ScoredPoint]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Original search query
            results: List of ScoredPoint results from hybrid search
            top_k: Number of top results to return (default: all)

        Returns:
            List of reranked ScoredPoint results
        """
        if not results:
            logger.debug("No results to rerank")
            return results

        try:
            texts = [self._compose_rerank_text(r) for r in results]

            valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
            if not valid_indices:
                logger.warning("No valid text found in results for reranking")
                return results

            valid_texts = [texts[i] for i in valid_indices]
            valid_results = [results[i] for i in valid_indices]

            ranked_indices = self.reranker_handler.rerank(query, valid_texts, top_k)

            final_results = []
            for orig_idx, score in ranked_indices:
                result = valid_results[orig_idx]
                result.score = float(score)
                final_results.append(result)

            logger.debug(f"Reranked {len(results)} -> {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            raise

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = True,
    ) -> list[ScoredPoint]:
        """
        End-to-end search pipeline:
        1) Hybrid dense + sparse vector retrieval
        2) Optional cross-encoder reranking

        Args:
            query: Search query
            top_k: Number of final results to return
            use_reranking: Whether to apply cross-encoder reranking

        Returns:
            List of final results
        """
        try:
            logger.info(f"Starting search for: {query[:100]}")

            # max 300 | min 40 chunks
            candidate_k = min(max(top_k * 20, 40), 300)
            hybrid_candidates = self.vector_search.hybrid_search(
                query, top_k=candidate_k
            )
            logger.debug(f"Hybrid search returned {len(hybrid_candidates)} candidates")

            if use_reranking:
                rerank_pool_size = min(len(hybrid_candidates), max(top_k * 3, top_k))
                rerank_pool = hybrid_candidates[:rerank_pool_size]
                final_results = self.rerank_results(query, rerank_pool, top_k=top_k)
            else:
                final_results = hybrid_candidates[:top_k]

            logger.info(f"Search completed: returned {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise
