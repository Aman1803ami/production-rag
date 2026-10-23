import logging
from typing import Optional

from app.services.cache import CacheService
from app.services.chunker import RecursiveChunker
from app.services.embeddings import EmbeddingService
from app.services.vector_db import VectorDBService
from app.services.reranker import RerankerService
from app.services.llm_router import LLMRouterService
from app.services.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Singletons
_cache_service: Optional[CacheService] = None
_chunker: Optional[RecursiveChunker] = None
_embedding_service: Optional[EmbeddingService] = None
_vector_db: Optional[VectorDBService] = None
_reranker: Optional[RerankerService] = None
_llm_router: Optional[LLMRouterService] = None
_pipeline: Optional[RAGPipeline] = None


async def initialize_services():
    """Application startup initialization hook."""
    global _cache_service, _chunker, _embedding_service, _vector_db, _reranker, _llm_router, _pipeline
    logger.info("Initializing RAG core services...")

    _cache_service = CacheService()
    await _cache_service.connect()

    _chunker = RecursiveChunker()
    _embedding_service = EmbeddingService(cache_service=_cache_service)
    _vector_db = VectorDBService()
    _reranker = RerankerService()
    _llm_router = LLMRouterService()

    _pipeline = RAGPipeline(
        cache_service=_cache_service,
        chunker=_chunker,
        embedding_service=_embedding_service,
        vector_db=_vector_db,
        reranker=_reranker,
        llm_router=_llm_router
    )
    logger.info("All RAG services successfully initialized.")


async def shutdown_services():
    """Application shutdown cleanup hook."""
    global _cache_service
    if _cache_service:
        await _cache_service.disconnect()


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline
    if not _pipeline:
        raise RuntimeError("RAGPipeline not initialized. Lifespan startup did not run.")
    return _pipeline


def get_cache_service() -> CacheService:
    global _cache_service
    if not _cache_service:
        raise RuntimeError("CacheService not initialized.")
    return _cache_service


def get_vector_db() -> VectorDBService:
    global _vector_db
    if not _vector_db:
        raise RuntimeError("VectorDBService not initialized.")
    return _vector_db


def get_llm_router() -> LLMRouterService:
    global _llm_router
    if not _llm_router:
        raise RuntimeError("LLMRouterService not initialized.")
    return _llm_router
