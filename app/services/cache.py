import hashlib
import json
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
import redis.asyncio as aioredis
from app.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """
    Production Two-Tier Redis Cache with resilient Soft-Fail architecture.
    Tier 1: Exact Query Response Cache (O(1) SHA-256 hash lookup)
    Tier 2: Text Chunk Embedding Cache (avoids duplicate embeddings during ingestion)
    """

    def __init__(self):
        self.redis_url = settings.REDIS_URL
        self.default_ttl = settings.REDIS_CACHE_TTL_SECONDS
        self.soft_fail = settings.REDIS_SOFT_FAIL
        self.client: Optional[aioredis.Redis] = None
        self._is_connected = False
        
        # In-memory fallback cache (LRU-like dict) if Redis is down
        self._memory_query_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_embedding_cache: Dict[str, List[float]] = {}
        self._memory_semantic_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._max_memory_entries = 1000

    async def connect(self):
        """Establish asynchronous Redis connection pool with timeout."""
        try:
            self.client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=0.5,
                socket_connect_timeout=0.5,
                retry_on_timeout=False
            )
            # Test connection
            await self.client.ping()
            self._is_connected = True
            logger.info("Connected to Redis at %s", self.redis_url)
        except Exception as e:
            self._is_connected = False
            if self.soft_fail:
                logger.warning("Redis is unavailable (%s). Operating in soft-fail mode with in-memory fallback cache.", str(e))
            else:
                logger.error("Failed to connect to Redis: %s", str(e))
                raise

    async def disconnect(self):
        """Cleanly close Redis connection pool."""
        if self.client:
            try:
                await self.client.aclose()
            except Exception as e:
                logger.warning("Error closing Redis connection: %s", str(e))
        self._is_connected = False

    async def ping(self) -> bool:
        """Check if Redis connection is currently healthy."""
        if not self.client:
            return False
        try:
            return bool(await self.client.ping())
        except Exception:
            return False

    async def clear_all(self):
        """Purge all cached responses across Redis and memory."""
        self._memory_query_cache.clear()
        self._memory_semantic_cache.clear()
        if self._is_connected and self.client:
            try:
                keys = await self.client.keys("rag:*")
                if keys:
                    await self.client.delete(*keys)
                logger.info("Purged %d keys from Redis cache.", len(keys))
            except Exception as e:
                logger.warning("Redis clear_all error: %s", str(e))

    @staticmethod
    def compute_query_hash(query: str, collection_name: str, top_k: int) -> str:
        """Normalize query and parameters into a deterministic SHA-256 hash."""
        normalized = f"{query.strip().lower()}|{collection_name}|{top_k}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Generate deterministic SHA-256 hash for raw text chunk."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    # --- Tier 1: Query Response Cache ---

    async def get_query_response(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached RAG response by query hash."""
        cache_key = f"rag:query:{query_hash}"
        try:
            if self._is_connected and self.client:
                cached_data = await self.client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
        except Exception as e:
            logger.warning("Redis query cache get failed: %s. Falling back to memory.", str(e))

        # Check in-memory fallback
        if query_hash in self._memory_query_cache:
            entry = self._memory_query_cache[query_hash]
            if time.time() < entry.get("expires_at", 0):
                return entry.get("data")
            else:
                del self._memory_query_cache[query_hash]
        return None

    async def set_query_response(
        self,
        query_hash: str,
        response_data: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ):
        """Save RAG response to query cache with TTL."""
        ttl = ttl_seconds or self.default_ttl
        cache_key = f"rag:query:{query_hash}"
        serialized = json.dumps(response_data)
        
        try:
            if self._is_connected and self.client:
                await self.client.set(cache_key, serialized, ex=ttl)
                return
        except Exception as e:
            logger.warning("Redis query cache set failed: %s. Storing in memory fallback.", str(e))

        # Store in memory fallback
        if len(self._memory_query_cache) >= self._max_memory_entries:
            # Evict oldest entry
            oldest_key = next(iter(self._memory_query_cache))
            del self._memory_query_cache[oldest_key]
        self._memory_query_cache[query_hash] = {
            "data": response_data,
            "expires_at": time.time() + ttl
        }

    # --- Tier 2: Chunk Embedding Cache ---

    async def get_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Retrieve precomputed embedding for identical text chunk."""
        cache_key = f"rag:emb:{text_hash}"
        try:
            if self._is_connected and self.client:
                raw_emb = await self.client.get(cache_key)
                if raw_emb:
                    return json.loads(raw_emb)
        except Exception as e:
            logger.warning("Redis embedding cache get failed: %s", str(e))

        return self._memory_embedding_cache.get(text_hash)

    async def set_embedding(self, text_hash: str, embedding: List[float]):
        """Cache computed embedding for a text chunk."""
        cache_key = f"rag:emb:{text_hash}"
        serialized = json.dumps(embedding)
        try:
            if self._is_connected and self.client:
                # Cache embeddings for 7 days
                await self.client.set(cache_key, serialized, ex=86400 * 7)
                return
        except Exception as e:
            logger.warning("Redis embedding cache set failed: %s", str(e))

        if len(self._memory_embedding_cache) >= self._max_memory_entries * 2:
            oldest = next(iter(self._memory_embedding_cache))
            del self._memory_embedding_cache[oldest]
        self._memory_embedding_cache[text_hash] = embedding

    # --- Tier 3: Semantic Vector Cache (> 0.90 Cosine Similarity) ---

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two float vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    async def get_semantic_query_response(
        self,
        query_vector: List[float],
        collection_name: str,
        threshold: Optional[float] = None
    ) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Check if an existing query in the collection has cosine similarity > threshold (default 0.90).
        Returns: (cached_response_dict, similarity_score) or None
        """
        target_threshold = threshold if threshold is not None else settings.SEMANTIC_CACHE_THRESHOLD
        now = time.time()
        best_match: Optional[Dict[str, Any]] = None
        highest_sim: float = 0.0

        def _evaluate_candidate(entry: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], float]:
            cached_vec = entry.get("vector")
            if cached_vec and len(cached_vec) == len(query_vector):
                return entry.get("data"), self._cosine_similarity(query_vector, cached_vec)
            return None, 0.0

        # 1. Check Redis if connected
        if self._is_connected and self.client:
            try:
                cache_key = f"rag:semantic:{collection_name}"
                records = await self.client.hgetall(cache_key)
                for entry_id, raw_json in records.items():
                    entry = json.loads(raw_json)
                    if entry.get("expires_at", 0) < now:
                        await self.client.hdel(cache_key, entry_id)
                        continue
                    data, sim = _evaluate_candidate(entry)
                    if sim > highest_sim:
                        highest_sim, best_match = sim, data
            except Exception as e:
                logger.warning("Redis semantic cache lookup error: %s. Falling back to memory.", str(e))

        # 2. Check In-Memory Semantic Cache
        mem_records = self._memory_semantic_cache.get(collection_name, [])
        valid_mem_records = []
        for entry in mem_records:
            if entry.get("expires_at", 0) >= now:
                valid_mem_records.append(entry)
                data, sim = _evaluate_candidate(entry)
                if sim > highest_sim:
                    highest_sim, best_match = sim, data
        self._memory_semantic_cache[collection_name] = valid_mem_records

        if best_match and highest_sim >= target_threshold:
            logger.info("Semantic cache match found! Similarity: %.4f (threshold: %.2f)", highest_sim, target_threshold)
            return best_match, highest_sim

        return None

    async def set_semantic_query_response(
        self,
        query: str,
        query_vector: List[float],
        response_data: Dict[str, Any],
        collection_name: str,
        ttl_seconds: Optional[int] = None
    ):
        """Store query embedding and answer in Semantic Cache with TTL."""
        ttl = ttl_seconds or self.default_ttl
        now = time.time()
        expires_at = now + ttl
        entry_id = hashlib.sha256(f"{query.strip().lower()}|{now}".encode("utf-8")).hexdigest()[:16]

        record = {
            "query": query,
            "vector": query_vector,
            "data": response_data,
            "expires_at": expires_at
        }
        serialized = json.dumps(record)

        # Store in Redis
        if self._is_connected and self.client:
            try:
                cache_key = f"rag:semantic:{collection_name}"
                await self.client.hset(cache_key, entry_id, serialized)
                # Ensure hash key itself has a TTL
                await self.client.expire(cache_key, ttl * 2)
            except Exception as e:
                logger.warning("Redis semantic cache set error: %s. Using memory fallback.", str(e))

        # Store in-memory
        if collection_name not in self._memory_semantic_cache:
            self._memory_semantic_cache[collection_name] = []

        mem_list = self._memory_semantic_cache[collection_name]
        mem_list.append(record)
        if len(mem_list) > settings.SEMANTIC_CACHE_MAX_ENTRIES:
            mem_list.pop(0)
