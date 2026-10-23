from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=4000, description="User search query / prompt")
    collection_name: Optional[str] = Field(default=None, description="Optional override collection name")
    top_k: int = Field(default=10, ge=1, le=50, description="Initial candidates retrieved from vector database")
    rerank_top_n: int = Field(default=5, ge=1, le=20, description="Top-N chunks to pass to LLM after reranking")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum rerank relevance threshold")
    bypass_cache: bool = Field(default=False, description="Whether to bypass the Redis query cache")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt instructions")


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LatencyMetrics(BaseModel):
    cache_lookup_ms: float = 0.0
    embedding_ms: float = 0.0
    vector_search_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_generation_ms: float = 0.0
    cache_write_ms: float = 0.0
    total_ms: float = 0.0


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    sources: List[SourceChunk]
    model_used: str
    is_cached: bool
    cache_type: Optional[str] = Field(default=None, description="Type of cache hit: 'exact', 'semantic', or None")
    semantic_similarity: Optional[float] = Field(default=None, description="Similarity score if semantic cache hit")
    fallback_triggered: bool
    reranker_fallback_triggered: bool
    latency: LatencyMetrics


class IngestTextRequest(BaseModel):
    content: str = Field(..., min_length=10, description="Raw text or markdown document to ingest")
    document_id: Optional[str] = Field(default=None, description="Unique identifier for the document")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom document metadata")
    collection_name: Optional[str] = Field(default=None, description="Target Qdrant collection")


class IngestResponse(BaseModel):
    document_id: str
    total_chunks: int
    collection_name: str
    cached_chunks_skipped: int
    time_taken_ms: float
    status: str = "success"


class ServiceStatus(BaseModel):
    healthy: bool
    details: str
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    environment: str
    services: Dict[str, ServiceStatus]
