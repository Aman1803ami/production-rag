import pytest
from app.services.cache import CacheService


@pytest.mark.asyncio
async def test_cache_hashing_and_fallback():
    cache = CacheService()
    # Test deterministic hash
    h1 = CacheService.compute_query_hash("What is RAG?", "default", 10)
    h2 = CacheService.compute_query_hash("what is rag? ", "default", 10)
    assert h1 == h2

    # Test Tier 1 In-Memory Soft-Fail fallback
    response_payload = {
        "answer": "RAG stands for Retrieval-Augmented Generation.",
        "model_used": "test-model"
    }
    await cache.set_query_response(h1, response_payload, ttl_seconds=60)
    retrieved = await cache.get_query_response(h1)
    assert retrieved is not None
    assert retrieved["answer"] == response_payload["answer"]

    # Test Tier 2 Chunk Embedding Cache
    text_hash = CacheService.compute_text_hash("Sample chunk of text to embed")
    test_vec = [0.1, 0.2, 0.3, 0.4]
    await cache.set_embedding(text_hash, test_vec)
    cached_vec = await cache.get_embedding(text_hash)
    assert cached_vec == test_vec


@pytest.mark.asyncio
async def test_semantic_cache_threshold_behavior():
    cache = CacheService()
    coll = "test_semantic_coll"

    # Seed base query with vector [1.0, 0.0, 0.0]
    base_vec = [1.0, 0.0, 0.0]
    payload = {"answer": "Semantic cache answer", "model_used": "test"}
    await cache.set_semantic_query_response(
        query="What is Qdrant?",
        query_vector=base_vec,
        response_data=payload,
        collection_name=coll,
        ttl_seconds=120
    )

    # 1. Very similar query (cosine similarity ~ 0.98 > 0.90 threshold)
    # [0.98, 0.1989, 0.0] has norm 1.0, dot product with [1,0,0] = 0.98
    similar_vec = [0.98, 0.198997, 0.0]
    hit = await cache.get_semantic_query_response(similar_vec, coll, threshold=0.90)
    assert hit is not None
    matched_data, sim_score = hit
    assert matched_data["answer"] == payload["answer"]
    assert sim_score >= 0.90

    # 2. Distant query (cosine similarity ~ 0.50 < 0.90 threshold)
    # [0.5, 0.866, 0.0] has norm 1.0, dot product = 0.50
    distant_vec = [0.5, 0.866, 0.0]
    miss = await cache.get_semantic_query_response(distant_vec, coll, threshold=0.90)
    assert miss is None
