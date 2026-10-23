from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_rank: int = 0


@dataclass
class RerankResult:
    chunk_id: str
    document_id: str
    content: str
    rerank_score: float
    original_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    rerank_rank: int = 0
