import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server Configuration
    ENVIRONMENT: str = Field(default="development", description="Runtime environment (development, staging, production)")
    HOST: str = Field(default="0.0.0.0", description="API host bind address")
    PORT: int = Field(default=8000, description="API port")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # LiteLLM & Model Fallback Configuration
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API key")
    PRIMARY_MODEL: str = Field(default="gemini/gemini-3.1-flash-lite", description="Primary generation model")
    
    # Fallback Model configuration
    FALLBACK_MODEL: str = Field(default="groq/llama-3.1-8b-instant", description="Secondary failover model")
    GROQ_API_KEY: Optional[str] = Field(default=None, description="Groq API key for free Llama fallback")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Local Ollama URL for local fallback")

    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = Field(default="BAAI/bge-small-en-v1.5", description="Local BGE ONNX embedding model")
    EMBEDDING_DIMENSION: int = Field(default=384, description="Embedding vector dimensions")

    # Reranker Configuration
    RERANKER_MODEL_NAME: str = Field(default="ms-marco-TinyBERT-L-2-v2", description="FlashRank lightweight model")
    RERANKER_TOP_N: int = Field(default=5, description="Final top-N chunks passed to LLM")
    SIMILARITY_THRESHOLD: float = Field(default=0.25, description="Minimum relevance score threshold")

    # Vector DB (Qdrant) Configuration
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant service URL or cloud endpoint")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant cloud API key if used")
    QDRANT_COLLECTION: str = Field(default="production_rag_docs", description="Target collection name")
    QDRANT_PREFER_GRPC: bool = Field(default=False, description="Whether to prefer gRPC transport")
    QDRANT_FALLBACK_LOCAL_PATH: str = Field(default="./data/qdrant", description="Local fallback directory if Qdrant server is offline")

    # Redis Cache Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection string")
    REDIS_CACHE_TTL_SECONDS: int = Field(default=3600, description="Cache expiration in seconds")
    REDIS_SOFT_FAIL: bool = Field(default=True, description="Continue without failure if Redis is unavailable")
    SEMANTIC_CACHE_ENABLED: bool = Field(default=True, description="Enable semantic vector similarity cache")
    SEMANTIC_CACHE_THRESHOLD: float = Field(default=0.90, description="Cosine similarity threshold for semantic cache match (> 0.9)")
    SEMANTIC_CACHE_MAX_ENTRIES: int = Field(default=1000, description="Maximum semantic cache entries per collection")

    # Document Chunking Configuration
    CHUNK_SIZE: int = Field(default=500, description="Target chunk size in tokens/chars")
    CHUNK_OVERLAP: int = Field(default=50, description="Chunk overlap tokens/chars")


# Global cached settings instance
settings = Settings()
