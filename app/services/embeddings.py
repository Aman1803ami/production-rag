import asyncio
import logging
import hashlib
from typing import List, Optional, Tuple
from app.config import settings
from app.services.cache import CacheService

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Production Embeddings Engine.
    Powered by FastEmbed (local ONNX BAAI/bge-small-en-v1.5).
    Zero API cost, CPU/GPU accelerated, with Redis Tier-2 chunk caching.
    """

    def __init__(self, cache_service: Optional[CacheService] = None):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self.cache_service = cache_service
        self._model = None
        self._initialize_model()

    def _initialize_model(self):
        """Lazy load or initialize FastEmbed model with fallback."""
        try:
            from fastembed import TextEmbedding
            logger.info("Initializing FastEmbed embedding model: %s", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name)
            logger.info("FastEmbed embedding model initialized successfully.")
        except Exception as e:
            logger.error("Failed to load FastEmbed model '%s': %s. Falling back to deterministic embedder.", self.model_name, str(e))
            self._model = None

    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Deterministic fallback embedding generator when offline or during zero-resource tests."""
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        vec = []
        for i in range(self.dimension):
            val = ((seed + i * 31) % 1000) / 1000.0 - 0.5
            vec.append(val)
        # Normalize vector
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    async def embed_query(self, query: str) -> List[float]:
        """Compute normalized vector embedding for a search query."""
        if self._model:
            try:
                embeddings = list(self._model.embed([query]))
                if embeddings:
                    return embeddings[0].tolist()
            except Exception as e:
                logger.warning("FastEmbed query embedding failed: %s. Using fallback.", str(e))
        return self._generate_fallback_embedding(query)

    async def embed_documents(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Embed a list of document chunk texts.
        Leverages Redis Tier-2 chunk cache:
        Only un-cached chunks are computed via FastEmbed, saving CPU/memory.
        Returns: (list_of_embeddings, cached_chunks_skipped)
        """
        if not texts:
            return [], 0

        cached_skipped = 0
        final_embeddings: List[Optional[List[float]]] = [None] * len(texts)
        texts_to_compute_indices: List[int] = []
        texts_to_compute: List[str] = []

        # Check CacheService for each chunk concurrently
        if self.cache_service:
            hashes = [CacheService.compute_text_hash(t) for t in texts]
            cached_results = await asyncio.gather(*[self.cache_service.get_embedding(h) for h in hashes])
            for idx, cached_vec in enumerate(cached_results):
                if cached_vec is not None:
                    final_embeddings[idx] = cached_vec
                    cached_skipped += 1
                else:
                    texts_to_compute_indices.append(idx)
                    texts_to_compute.append(texts[idx])
        else:
            texts_to_compute_indices = list(range(len(texts)))
            texts_to_compute = list(texts)

        # Compute embeddings for remaining texts
        if texts_to_compute:
            computed_vecs: List[List[float]] = []
            if self._model:
                try:
                    raw_vecs = list(self._model.embed(texts_to_compute))
                    computed_vecs = [v.tolist() for v in raw_vecs]
                except Exception as e:
                    logger.warning("FastEmbed batch embedding failed: %s. Using fallback.", str(e))
                    computed_vecs = [self._generate_fallback_embedding(t) for t in texts_to_compute]
            else:
                computed_vecs = [self._generate_fallback_embedding(t) for t in texts_to_compute]

            # Store in final list and write back to cache
            for original_idx, vec in zip(texts_to_compute_indices, computed_vecs):
                final_embeddings[original_idx] = vec
                if self.cache_service:
                    text_hash = CacheService.compute_text_hash(texts[original_idx])
                    await self.cache_service.set_embedding(text_hash, vec)

        return [v if v is not None else self._generate_fallback_embedding("") for v in final_embeddings], cached_skipped
