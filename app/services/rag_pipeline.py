import time
import logging
import uuid
from typing import Optional, List, Dict, Any, AsyncGenerator

from app.config import settings
from app.models.api import QueryRequest, QueryResponse, SourceChunk, LatencyMetrics, IngestResponse
from app.services.cache import CacheService
from app.services.chunker import RecursiveChunker
from app.services.embeddings import EmbeddingService
from app.services.vector_db import VectorDBService
from app.services.reranker import RerankerService
from app.services.llm_router import LLMRouterService

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Production RAG Pipeline Orchestrator.
    Coordinates:
    - Tier-1 Exact Query Caching (Redis)
    - Embedding generation (FastEmbed)
    - Vector similarity retrieval (Qdrant)
    - FlashRank Reranking with RRF Fallback
    - Resilient LiteLLM Routing (Gemini -> Free/Local Fallback)
    - Microsecond-accurate latency monitoring & graceful degradation
    """

    def __init__(
        self,
        cache_service: CacheService,
        chunker: RecursiveChunker,
        embedding_service: EmbeddingService,
        vector_db: VectorDBService,
        reranker: RerankerService,
        llm_router: LLMRouterService
    ):
        self.cache = cache_service
        self.chunker = chunker
        self.embedder = embedding_service
        self.vector_db = vector_db
        self.reranker = reranker
        self.llm_router = llm_router

    async def ingest_document(
        self,
        content: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> IngestResponse:
        """
        Ingest text content: chunks document, embeds un-cached chunks, and indexes into Qdrant.
        """
        start_time = time.perf_counter()
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:12]}"
        coll = collection_name or settings.QDRANT_COLLECTION

        # 1. Chunk document
        chunks = self.chunker.chunk_document(
            content=content,
            document_id=doc_id,
            metadata=metadata
        )

        if not chunks:
            return IngestResponse(
                document_id=doc_id,
                total_chunks=0,
                collection_name=coll,
                cached_chunks_skipped=0,
                time_taken_ms=0.0,
                status="empty_content"
            )

        # 2. Embed chunks (utilizing Redis Tier-2 cache)
        chunk_texts = [c.content for c in chunks]
        embeddings, cached_skipped = await self.embedder.embed_documents(chunk_texts)

        # 3. Upsert into Qdrant
        self.vector_db.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            collection_name=coll
        )

        total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Document '%s' ingested: %d chunks (%d from cache), %.2f ms",
            doc_id, len(chunks), cached_skipped, total_time_ms
        )

        return IngestResponse(
            document_id=doc_id,
            total_chunks=len(chunks),
            collection_name=coll,
            cached_chunks_skipped=cached_skipped,
            time_taken_ms=total_time_ms,
            status="success"
        )

    async def query(self, request: QueryRequest, request_id: Optional[str] = None) -> QueryResponse:
        """
        Execute full RAG pipeline with caching, hybrid retrieval, reranking, and resilient generation.
        """
        req_id = request_id or f"req_{uuid.uuid4().hex[:10]}"
        coll = request.collection_name or settings.QDRANT_COLLECTION
        start_total = time.perf_counter()
        latency = LatencyMetrics()

        # Step 1: Check Exact Query Cache (Tier 1)
        query_hash = CacheService.compute_query_hash(
            query=request.query,
            collection_name=coll,
            top_k=request.top_k
        )

        t_cache_start = time.perf_counter()
        if not request.bypass_cache:
            cached_result = await self.cache.get_query_response(query_hash)
            if cached_result:
                latency.cache_lookup_ms = round((time.perf_counter() - t_cache_start) * 1000, 2)
                latency.total_ms = round((time.perf_counter() - start_total) * 1000, 2)
                logger.info("[%s] Query cache hit for query: '%s'", req_id, request.query[:50])

                return QueryResponse(
                    request_id=req_id,
                    query=request.query,
                    answer=cached_result["answer"],
                    sources=[SourceChunk(**s) for s in cached_result.get("sources", [])],
                    model_used=cached_result.get("model_used", "cached"),
                    is_cached=True,
                    cache_type="exact",
                    semantic_similarity=1.0,
                    fallback_triggered=cached_result.get("fallback_triggered", False),
                    reranker_fallback_triggered=cached_result.get("reranker_fallback_triggered", False),
                    latency=latency
                )
        latency.cache_lookup_ms = round((time.perf_counter() - t_cache_start) * 1000, 2)

        # Step 2: Compute Query Embedding
        t_embed_start = time.perf_counter()
        query_vector = await self.embedder.embed_query(request.query)
        latency.embedding_ms = round((time.perf_counter() - t_embed_start) * 1000, 2)

        # Step 2b: Check Semantic Query Cache (> 0.90 cosine similarity)
        if settings.SEMANTIC_CACHE_ENABLED and not request.bypass_cache:
            t_sem_start = time.perf_counter()
            sem_match = await self.cache.get_semantic_query_response(
                query_vector=query_vector,
                collection_name=coll,
                threshold=settings.SEMANTIC_CACHE_THRESHOLD
            )
            if sem_match:
                cached_result, sim_score = sem_match
                sem_lookup_ms = round((time.perf_counter() - t_sem_start) * 1000, 2)
                latency.cache_lookup_ms += sem_lookup_ms
                latency.total_ms = round((time.perf_counter() - start_total) * 1000, 2)
                logger.info("[%s] Semantic cache hit (sim: %.4f >= %.2f) for query: '%s'", req_id, sim_score, settings.SEMANTIC_CACHE_THRESHOLD, request.query[:50])

                return QueryResponse(
                    request_id=req_id,
                    query=request.query,
                    answer=cached_result["answer"],
                    sources=[SourceChunk(**s) for s in cached_result.get("sources", [])],
                    model_used=cached_result.get("model_used", "semantic_cached"),
                    is_cached=True,
                    cache_type="semantic",
                    semantic_similarity=sim_score,
                    fallback_triggered=cached_result.get("fallback_triggered", False),
                    reranker_fallback_triggered=cached_result.get("reranker_fallback_triggered", False),
                    latency=latency
                )

        # Step 3: Vector Search in Qdrant
        t_search_start = time.perf_counter()
        raw_candidates = self.vector_db.search(
            query_vector=query_vector,
            top_k=request.top_k,
            collection_name=coll
        )
        latency.vector_search_ms = round((time.perf_counter() - t_search_start) * 1000, 2)

        # Step 4: FlashRank Reranking with RRF Fallback
        t_rerank_start = time.perf_counter()
        reranked_chunks, rerank_fallback = await self.reranker.rerank(
            query=request.query,
            candidates=raw_candidates,
            top_n=request.rerank_top_n,
            threshold=request.similarity_threshold
        )
        latency.rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000, 2)

        # Build context from reranked chunks
        context_parts = []
        source_chunks: List[SourceChunk] = []
        for idx, item in enumerate(reranked_chunks):
            context_parts.append(f"[Source {idx + 1}] (Score: {item.rerank_score:.4f}):\n{item.content}")
            source_chunks.append(
                SourceChunk(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    content=item.content,
                    score=item.rerank_score,
                    rank=idx + 1,
                    metadata=item.metadata
                )
            )

        context_str = "\n\n".join(context_parts) if context_parts else "No relevant documents found in index."

        # Step 5: LLM Generation (LiteLLM Router: Gemini -> Free/Local Fallback)
        t_gen_start = time.perf_counter()
        answer, model_used, llm_fallback = await self.llm_router.generate_response(
            query=request.query,
            context=context_str,
            custom_system_prompt=request.system_prompt
        )
        latency.llm_generation_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)

        # Step 6: Store in Redis Cache (Tier 1 Exact & Tier 3 Semantic)
        t_cache_w_start = time.perf_counter()
        cache_payload = {
            "answer": answer,
            "sources": [s.model_dump() for s in source_chunks],
            "model_used": model_used,
            "fallback_triggered": llm_fallback,
            "reranker_fallback_triggered": rerank_fallback
        }
        # Only cache when generated by a real model, never cache offline fallback
        if model_used not in ["offline-resilience-fallback", "local-bootstrapping-synthesizer"]:
            await self.cache.set_query_response(query_hash, cache_payload)
            if settings.SEMANTIC_CACHE_ENABLED:
                await self.cache.set_semantic_query_response(
                    query=request.query,
                    query_vector=query_vector,
                    response_data=cache_payload,
                    collection_name=coll
                )
        latency.cache_write_ms = round((time.perf_counter() - t_cache_w_start) * 1000, 2)

        latency.total_ms = round((time.perf_counter() - start_total) * 1000, 2)

        return QueryResponse(
            request_id=req_id,
            query=request.query,
            answer=answer,
            sources=source_chunks,
            model_used=model_used,
            is_cached=False,
            cache_type=None,
            semantic_similarity=None,
            fallback_triggered=llm_fallback,
            reranker_fallback_triggered=rerank_fallback,
            latency=latency
        )

    async def stream_query(
        self,
        request: QueryRequest,
        request_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream RAG response tokens via Server-Sent Events (SSE)."""
        coll = request.collection_name or settings.QDRANT_COLLECTION

        # Embed query
        query_vector = await self.embedder.embed_query(request.query)

        # Search candidates
        raw_candidates = self.vector_db.search(
            query_vector=query_vector,
            top_k=request.top_k,
            collection_name=coll
        )

        # Rerank
        reranked_chunks, _ = await self.reranker.rerank(
            query=request.query,
            candidates=raw_candidates,
            top_n=request.rerank_top_n,
            threshold=request.similarity_threshold
        )

        context_parts = [f"[Source {i+1}]: {c.content}" for i, c in enumerate(reranked_chunks)]
        context_str = "\n\n".join(context_parts) if context_parts else "No relevant documents found."

        # Stream tokens
        async for token in self.llm_router.stream_response(
            query=request.query,
            context=context_str,
            custom_system_prompt=request.system_prompt
        ):
            yield token
