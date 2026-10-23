# ⚡ Production RAG (Retrieval-Augmented Generation) System

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-dc2626.svg?logo=qdrant)](https://qdrant.tech/)
[![Redis](https://img.shields.io/badge/Cache-Redis%20Two--Tier-d97706.svg?logo=redis&logoColor=white)](https://redis.io/)
[![FlashRank](https://img.shields.io/badge/Reranker-FlashRank%20ONNX-8b5cf6.svg)](https://github.com/PrithivirajDamodaran/FlashRank)
[![LiteLLM](https://img.shields.io/badge/LLM%20Router-LiteLLM-10b981.svg)](https://github.com/BerriAI/litellm)
[![Tests](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A high-performance, zero-downtime, production-grade **Retrieval-Augmented Generation (RAG)** backend and real-time interactive console. Built with **FastAPI**, **Qdrant Vector Database**, **Redis Two-Tier & Semantic Caching**, **FlashRank Local Reranking**, and **Resilient LiteLLM Multi-Provider Routing**.

---

## 🏛️ Architectural Overview

```
                      ┌──────────────────────────────────────────────┐
                      │             Client / Frontend UI             │
                      └──────────────────────┬───────────────────────┘
                                             │ HTTP / SSE Stream
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │          FastAPI Application Gateway         │
                      └──────────────────────┬───────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ Exact Query Cache ]                       [ Ingestion Pipeline ]
             SHA-256 (Hash Lookup)                        - Hierarchical Chunker
             (< 1ms Cache Hit Return)                     - FastEmbed BGE ONNX
                       │                                  - Redis Tier-2 Chunk Cache
                       │ (Cache Miss)                              │
                       ▼                                           ▼
          ┌─────────────────────────┐                    ┌──────────────────┐
          │ FastEmbed Vector Engine │                    │ Qdrant Vector DB │
          └────────────┬────────────┘                    └─────────▲────────┘
                       │ Query Vector                              │
                       ▼                                           │
          ┌─────────────────────────┐                              │
          │   Qdrant ANN Retrieval  │──────────────────────────────┘
          │    (Top-K Candidates)   │
          └────────────┬────────────┘
                       │ Candidate Chunks
                       ▼
          ┌─────────────────────────┐
          │ FlashRank Cross-Encoder │ ── (Fallback: Reciprocal Rank Fusion / RRF)
          │    (Top-N Passages)     │
          └────────────┬────────────┘
                       │ Ranked Context + Query
                       ▼
          ┌────────────────────────────────────────────────────────┐
          │             LiteLLM Resilient Router                   │
          │  Primary: Google Gemini (High Speed / Large Quota)     │
          │  Failover: Groq Llama 3.1 / Ollama / Local Synthesizer │
          └──────────────────────────┬─────────────────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │  Formatted Response + SSE    │
                      │  Latency Metrics Breakdown   │
                      └──────────────────────────────┘
```

---

## 🚀 Key Architectural Pillars

### 1. Resilient Multi-Provider LLM Routing (LiteLLM)
- **Primary Model**: Google Gemini (`gemini/gemini-2.0-flash` or `gemini-3.1-flash-lite`) for rapid latency and generous quota.
- **Failover / Fallback**: Automatic circuit-breaker failover to free small LLMs (e.g. `groq/llama-3.1-8b-instant` or local `ollama/llama3.2`).
- **Zero-Downtime Resilience**: Automatically handles 429 rate limits and 5xx outages without dropping requests.
- **Bootstrapping Mode**: Runs out of the box with an offline contextual synthesizer if API keys have not yet been configured.

### 2. High-Precision Local Reranking with RRF Fallback
- **Primary Reranker**: `FlashRank` (`ms-marco-TinyBERT-L-2-v2`), local ONNX cross-encoder running with zero API cost.
- **RRF Fallback**: If the cross-encoder is unavailable or encounters an error, it gracefully falls back to **Reciprocal Rank Fusion (RRF)** combined with normalized vector cosine similarity scores.

### 3. Multi-Tier Caching with Soft-Fail
- **Tier 1 (Exact Query Response Cache)**: Deterministic SHA-256 hash of `(normalized_query + collection + top_k)` returns cached answers in `< 1ms`.
- **Tier 2 (Chunk Embedding Cache)**: Hashes document chunk paragraphs to prevent re-computing embeddings for identical text during ingestion.
- **Tier 3 (Semantic Vector Similarity Cache)**: Detects semantically equivalent queries using cosine similarity (> 0.90 threshold).
- **Soft-Fail Architecture**: If Redis is offline, the service transparently degrades to an in-memory LRU cache without throwing 500 errors.

### 4. Zero-Cost Local Embeddings & Recursive Chunking
- **Embeddings**: `FastEmbed` (`BAAI/bge-small-en-v1.5`, 384 dimensions), local ONNX acceleration on CPU/GPU.
- **Recursive Chunking**: Hierarchical markdown and recursive character splitting respecting headers, paragraphs, and sentence boundaries.

### 5. Production Vector Database (Qdrant)
- **Primary**: Connects to remote Qdrant Docker or cloud cluster (`http://localhost:6333`).
- **Resilience Fallback**: If Docker is not running or Qdrant server is offline, automatically activates a local disk-backed or in-memory Qdrant engine.
- **Cached Collection Lookups**: In-memory verification cache prevents redundant database RPCs on hot search paths.

---

## 📁 Repository Structure

```
├── app/
│   ├── config.py                 # Pydantic BaseSettings & environment configuration
│   ├── main.py                   # FastAPI application factory, lifespan, CORS & middleware
│   ├── models/
│   │   ├── api.py                # Request/response schemas (QueryRequest, QueryResponse, LatencyMetrics)
│   │   └── internal.py           # Internal dataclasses (Chunk, SearchResult, RerankResult)
│   ├── services/
│   │   ├── cache.py              # Two-tier Redis cache with soft-fail & in-memory backup
│   │   ├── chunker.py            # Hierarchical markdown & recursive character chunker
│   │   ├── embeddings.py         # FastEmbed BGE ONNX local embedding engine (with batch cache lookups)
│   │   ├── vector_db.py          # Qdrant client manager with local disk fallback & collection caching
│   │   ├── reranker.py           # FlashRank cross-encoder with RRF rank fallback
│   │   ├── llm_router.py         # LiteLLM router (Gemini -> Groq/Ollama failover)
│   │   └── rag_pipeline.py       # Full end-to-end RAG orchestrator with latency breakdown
│   ├── api/
│   │   ├── dependencies.py       # Dependency injection singletons
│   │   └── routes.py             # REST API routes (/query, /query/stream, /ingest, /health)
│   └── static/
│       └── index.html            # Dark-mode Interactive Console & Visual Playground
├── tests/                        # Comprehensive unit & integration tests
│   ├── test_chunker.py
│   ├── test_cache.py
│   ├── test_reranker.py
│   └── test_api.py
├── scripts/
│   └── ingest_folder.py          # Bulk ingestion CLI utility for folders of PDF/MD/TXT files
├── docker-compose.yml            # Containerized deployment (Qdrant + Redis + RAG API)
├── Dockerfile                    # Production container build
├── requirements.txt              # Production Python dependencies
├── pytest.ini                    # Pytest configuration
├── .env.example                  # Documented environment configuration template
└── .gitignore                    # Environment & artifact ignore rules
```

---

## 🛠️ Quick Start

### 1. Local Python Setup

```powershell
# Clone repository
git clone https://github.com/<your-username>/production-rag-system.git
cd production-rag-system

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and supply GEMINI_API_KEY (optional; fallback synthesizer active if omitted)

# Start server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to:
- **Interactive RAG Console**: [http://localhost:8000/](http://localhost:8000/)
- **Swagger API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Probe**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

### 2. Docker Compose Deployment

Run the complete production stack (Qdrant vector database, Redis cache, and FastAPI):

```bash
docker-compose up -d --build
```

Services exposed:
- **RAG API & Web Playground**: `http://localhost:8000`
- **Qdrant Dashboard**: `http://localhost:6333/dashboard`
- **Redis Cache**: `localhost:6379`

---

## 🧪 Running Automated Tests

Run the test suite with `pytest`:

```powershell
.\.venv\Scripts\python -m pytest -v
```

All 7 test suites validate:
1. `test_health_check_endpoint`: Service readiness probes for Redis, Qdrant, and LiteLLM.
2. `test_ingest_and_query_flow`: Ingestion, chunking, retrieval, generation, and Tier-1 cache verification.
3. `test_cache_hashing_and_fallback`: Deterministic SHA-256 query hashing and soft-fail in-memory storage.
4. `test_semantic_cache_threshold_behavior`: Cosine threshold filtering (> 0.90) for semantic hits.
5. `test_chunker_basic_split`: Markdown hierarchy and boundary splitting.
6. `test_chunker_overlap`: Overlap preservation between sequential chunks.
7. `test_reranker_and_fallback`: FlashRank cross-encoder ranking and Reciprocal Rank Fusion fallback.

---

## 📡 API Reference

### 1. Execute RAG Query
```http
POST /api/v1/query
Content-Type: application/json

{
  "query": "How does the Redis caching tier handle network outages?",
  "top_k": 10,
  "rerank_top_n": 5,
  "bypass_cache": false
}
```

**Response**:
```json
{
  "request_id": "req_8a92f03b",
  "query": "How does the Redis caching tier handle network outages?",
  "answer": "### 🎯 Summary\nThe Redis caching tier implements a soft-fail architecture...",
  "sources": [
    {
      "chunk_id": "doc_arch_001_c0001",
      "document_id": "doc_arch_001",
      "content": "...",
      "score": 0.9412,
      "rank": 1,
      "metadata": {}
    }
  ],
  "model_used": "gemini-3.1-flash-lite",
  "is_cached": false,
  "cache_type": null,
  "semantic_similarity": null,
  "fallback_triggered": false,
  "reranker_fallback_triggered": false,
  "latency": {
    "cache_lookup_ms": 1.2,
    "embedding_ms": 12.4,
    "vector_search_ms": 4.8,
    "rerank_ms": 15.1,
    "llm_generation_ms": 280.5,
    "cache_write_ms": 0.8,
    "total_ms": 314.8
  }
}
```

### 2. Stream Response Tokens (SSE)
```http
POST /api/v1/query/stream
Content-Type: application/json

{
  "query": "Explain the chunking strategy.",
  "top_k": 5
}
```

### 3. Ingest Raw Text or Markdown
```http
POST /api/v1/ingest/text
Content-Type: application/json

{
  "content": "# System Architecture\nOur system uses Qdrant and Redis...",
  "document_id": "doc_manual_01",
  "metadata": { "department": "engineering" }
}
```

### 4. Upload and Ingest Documents (PDF, TXT, MD)
```http
POST /api/v1/ingest/file
Content-Type: multipart/form-data

file: [Select PDF/TXT/MD file]
document_id: optional_id
```

### 5. Health Check Probe
```http
GET /api/v1/health
```

---

## 🛡️ Resilience & Failover Matrix

| Component | Potential Event | Handled By | Behavior |
| :--- | :--- | :--- | :--- |
| **Redis Cache** | Timeout, disconnection, or failure | `CacheService` | Soft-fails automatically to in-memory TTL cache without raising 500 errors. |
| **Qdrant DB** | Docker stopped or cluster down | `VectorDBService` | Retries 3x with exponential backoff; activates local disk engine (`./data/qdrant`) or in-memory fallback. |
| **FlashRank** | Timeout, model error, or memory limit | `RerankerService` | Automatically falls back to **Reciprocal Rank Fusion (RRF)** + vector cosine similarity. |
| **LiteLLM** | 429 Rate limit or 5xx provider outage | `LLMRouterService` | Failover from Gemini -> Groq Llama 3.1 / Ollama -> Local contextual synthesizer. |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
