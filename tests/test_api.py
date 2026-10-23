import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_health_check_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "qdrant" in data["services"]
        assert "redis" in data["services"]


def test_ingest_and_query_flow():
    with TestClient(app) as client:
        # 1. Ingest sample document
        ingest_payload = {
            "content": (
                "# LiteLLM and Redis Architecture\n\n"
                "Our production RAG system uses Google Gemini as the primary LLM model. "
                "If Gemini hits a rate limit or service error, LiteLLM automatically fails over "
                "to a free small LLM like Groq Llama 3.1 or Ollama.\n\n"
                "Additionally, Redis caches exact queries to achieve sub-10ms response times."
            ),
            "document_id": "doc_arch_001"
        }

        ingest_res = client.post("/api/v1/ingest/text", json=ingest_payload)
        assert ingest_res.status_code == 200
        ingest_data = ingest_res.json()
        assert ingest_data["status"] == "success"
        assert ingest_data["total_chunks"] > 0

        # 2. Query the ingested knowledge base (Initial - Cache Miss)
        query_payload = {
            "query": "Which model does LiteLLM fail over to if Gemini fails?",
            "top_k": 5,
            "rerank_top_n": 3
        }
        query_res = client.post("/api/v1/query", json=query_payload)
        assert query_res.status_code == 200
        query_data = query_res.json()
        assert query_data["query"] == query_payload["query"]
        assert len(query_data["answer"]) > 0
        assert query_data["is_cached"] is False
        assert query_data["latency"]["total_ms"] > 0

        # 3. Repeat identical query - Should hit Redis Tier-1 Cache
        repeat_res = client.post("/api/v1/query", json=query_payload)
        assert repeat_res.status_code == 200
        cached_data = repeat_res.json()
        assert cached_data["is_cached"] is True
        assert cached_data["cache_type"] == "exact"
        assert cached_data["latency"]["cache_lookup_ms"] >= 0

        # 4. Semantically equivalent query - Should trigger Semantic Cache (> 0.90 similarity)
        sem_payload = {
            "query": "Which model does LiteLLM failover to when Gemini fails?",
            "top_k": 5,
            "rerank_top_n": 3
        }
        sem_res = client.post("/api/v1/query", json=sem_payload)
        assert sem_res.status_code == 200
        sem_data = sem_res.json()
        assert sem_data["is_cached"] is True
        assert sem_data["cache_type"] in ["exact", "semantic"]
        if sem_data["cache_type"] == "semantic":
            assert sem_data["semantic_similarity"] >= 0.90
