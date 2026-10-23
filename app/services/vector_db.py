import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Set
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings
from app.models.internal import Chunk, SearchResult

logger = logging.getLogger(__name__)


class VectorDBService:
    """
    Production Vector Database Service powered by Qdrant.
    Supports remote Qdrant Docker/Cluster with automatic local disk/in-memory fallback,
    retry policies with Tenacity, and metadata payload indexing.
    """

    def __init__(self):
        self.url = settings.QDRANT_URL
        self.api_key = settings.QDRANT_API_KEY
        self.default_collection = settings.QDRANT_COLLECTION
        self.dimension = settings.EMBEDDING_DIMENSION
        self.fallback_path = settings.QDRANT_FALLBACK_LOCAL_PATH
        self.client: Optional[QdrantClient] = None
        self._is_local_fallback = False
        self._known_collections: Set[str] = set()
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qdrant client with automatic local persistence fallback."""
        try:
            if self.url and not self.url.startswith(":memory:"):
                logger.info("Attempting connection to Qdrant at %s", self.url)
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    prefer_grpc=settings.QDRANT_PREFER_GRPC,
                    timeout=3.0
                )
                # Verify connection with cluster/collections call
                self.client.get_collections()
                logger.info("Successfully connected to Qdrant server at %s", self.url)
                self._is_local_fallback = False
                return
        except Exception as e:
            logger.warning("Could not reach Qdrant server at %s (%s). Activating local disk fallback at '%s'.", self.url, str(e), self.fallback_path)

        # Fallback to local storage
        try:
            os.makedirs(self.fallback_path, exist_ok=True)
            self.client = QdrantClient(path=self.fallback_path)
            self._is_local_fallback = True
            logger.info("Local Qdrant engine initialized at %s", self.fallback_path)
        except Exception as e:
            logger.warning("Local directory storage failed (%s). Falling back to in-memory Qdrant.", str(e))
            self.client = QdrantClient(location=":memory:")
            self._is_local_fallback = True

    def ping(self) -> bool:
        """Check Qdrant client readiness."""
        if not self.client:
            return False
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=5),
        retry=retry_if_exception_type((UnexpectedResponse, Exception)),
        reraise=True
    )
    def ensure_collection(self, collection_name: Optional[str] = None):
        """Create target collection with Cosine distance and payload index if not present."""
        coll = collection_name or self.default_collection
        if coll in self._known_collections:
            return

        if not self.client:
            self._initialize_client()

        existing = [c.name for c in self.client.get_collections().collections]
        if coll not in existing:
            logger.info("Creating Qdrant collection '%s' with dimension %d", coll, self.dimension)
            self.client.create_collection(
                collection_name=coll,
                vectors_config=qmodels.VectorParams(
                    size=self.dimension,
                    distance=qmodels.Distance.COSINE
                )
            )
            # Create payload index for document_id filtering if on server
            if not self._is_local_fallback:
                try:
                    self.client.create_payload_index(
                        collection_name=coll,
                        field_name="document_id",
                        field_schema=qmodels.PayloadSchemaType.KEYWORD
                    )
                except Exception as e:
                    logger.debug("Payload index creation notice: %s", str(e))

        self._known_collections.add(coll)

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        collection_name: Optional[str] = None
    ) -> int:
        """Upsert a batch of document chunks and vectors into Qdrant."""
        if not chunks or not embeddings:
            return 0

        coll = collection_name or self.default_collection
        self.ensure_collection(coll)

        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            # Generate deterministic standard UUID from chunk_id
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(
                qmodels.PointStruct(
                    id=point_uuid,
                    vector=vec,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "chunk_index": chunk.index,
                        "metadata": chunk.metadata,
                        "token_count": chunk.token_count
                    }
                )
            )

        # Batch upsert in slices of 100
        batch_size = 100
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=coll,
                points=batch,
                wait=True
            )
            total_upserted += len(batch)

        logger.info("Upserted %d points to Qdrant collection '%s'", total_upserted, coll)
        return total_upserted

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        collection_name: Optional[str] = None,
        filter_document_id: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform vector similarity search with optional metadata payload filtering."""
        coll = collection_name or self.default_collection
        self.ensure_collection(coll)

        query_filter = None
        if filter_document_id:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=filter_document_id)
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=coll,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        )

        search_results: List[SearchResult] = []
        for rank, hit in enumerate(results.points):
            payload = hit.payload or {}
            search_results.append(
                SearchResult(
                    chunk_id=str(payload.get("chunk_id", hit.id)),
                    document_id=payload.get("document_id", "unknown"),
                    content=payload.get("content", ""),
                    score=float(hit.score),
                    metadata=payload.get("metadata", {}),
                    original_rank=rank + 1
                )
            )

        return search_results
