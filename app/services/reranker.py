import logging
from typing import List, Tuple, Optional
from app.config import settings
from app.models.internal import SearchResult, RerankResult

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Production Reranking Engine using FlashRank (local ONNX ms-marco-TinyBERT-L-2-v2).
    Includes automatic graceful fallback to Reciprocal Rank Fusion (RRF) / Cosine Vector scores
    if the cross-encoder encounters any timeout, model error, or failure.
    """

    def __init__(self):
        self.model_name = settings.RERANKER_MODEL_NAME
        self.default_top_n = settings.RERANKER_TOP_N
        self.default_threshold = settings.SIMILARITY_THRESHOLD
        self._ranker = None
        self._initialize_ranker()

    def _initialize_ranker(self):
        """Initialize FlashRank Ranker instance."""
        try:
            from flashrank import Ranker
            logger.info("Initializing FlashRank reranker: %s", self.model_name)
            self._ranker = Ranker(model_name=self.model_name)
            logger.info("FlashRank reranker successfully loaded.")
        except Exception as e:
            logger.warning("Could not initialize FlashRank model '%s': %s. Will use RRF fallback.", self.model_name, str(e))
            self._ranker = None

    def _fallback_rerank(
        self,
        candidates: List[SearchResult],
        top_n: int,
        threshold: Optional[float] = None
    ) -> List[RerankResult]:
        """
        Graceful fallback ranking using Reciprocal Rank Fusion (RRF) & normalized vector cosine score.
        RRF formula: score = 1 / (60 + rank) + normalized_cosine_score
        """
        results: List[RerankResult] = []
        for rank, item in enumerate(candidates):
            # Combined score using RRF constant (60)
            rrf_score = (1.0 / (60.0 + item.original_rank)) + (max(0.0, item.score) * 0.5)
            results.append(
                RerankResult(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    content=item.content,
                    rerank_score=float(rrf_score),
                    original_score=float(item.score),
                    metadata=item.metadata,
                    rerank_rank=rank + 1
                )
            )

        # Sort by computed fallback score descending
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        if threshold is not None:
            filtered = [r for r in results if r.rerank_score >= threshold]
            if filtered:
                results = filtered
        return results[:top_n]

    async def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_n: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Tuple[List[RerankResult], bool]:
        """
        Rerank retrieved candidates against query.
        Returns: (ranked_results, fallback_triggered)
        """
        if not candidates:
            return [], False

        target_top_n = top_n or self.default_top_n
        target_threshold = threshold if threshold is not None else self.default_threshold

        # If ranker is not initialized or candidate list is trivial
        if not self._ranker:
            logger.info("FlashRank not available. Executing RRF fallback ranking.")
            return self._fallback_rerank(candidates, target_top_n, target_threshold), True

        try:
            from flashrank import RerankRequest

            passages = [
                {
                    "id": c.chunk_id,
                    "text": c.content,
                    "meta": {
                        "document_id": c.document_id,
                        "original_score": c.score,
                        "metadata": c.metadata
                    }
                }
                for c in candidates
            ]

            request = RerankRequest(query=query, passages=passages)
            reranked_raw = self._ranker.rerank(request)

            results: List[RerankResult] = []
            for rank_idx, item in enumerate(reranked_raw):
                score = float(item["score"])
                meta = item.get("meta", {})
                
                results.append(
                    RerankResult(
                        chunk_id=str(item["id"]),
                        document_id=meta.get("document_id", "unknown"),
                        content=item["text"],
                        rerank_score=score,
                        original_score=float(meta.get("original_score", 0.0)),
                        metadata=meta.get("metadata", {}),
                        rerank_rank=rank_idx + 1
                    )
                )

            # Sort by rerank score descending and slice to top_n
            results.sort(key=lambda x: x.rerank_score, reverse=True)
            return results[:target_top_n], False

        except Exception as e:
            logger.error("FlashRank execution failed (%s). Triggering graceful RRF fallback.", str(e))
            return self._fallback_rerank(candidates, target_top_n, target_threshold), True
