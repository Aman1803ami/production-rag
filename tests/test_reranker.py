import pytest
from app.services.reranker import RerankerService
from app.models.internal import SearchResult


@pytest.mark.asyncio
async def test_reranker_and_fallback():
    reranker = RerankerService()
    candidates = [
        SearchResult(chunk_id="c1", document_id="d1", content="Apples and oranges are fruits.", score=0.8, original_rank=1),
        SearchResult(chunk_id="c2", document_id="d1", content="Python is a programming language used for machine learning.", score=0.75, original_rank=2),
        SearchResult(chunk_id="c3", document_id="d1", content="Quantum computing uses qubits and superposition.", score=0.5, original_rank=3)
    ]

    # Test reranking against a programming query
    results, fallback = await reranker.rerank(
        query="Tell me about Python and coding",
        candidates=candidates,
        top_n=2
    )

    assert len(results) <= 2
    # The python chunk should be ranked highest by cross-encoder or RRF
    top_chunk = results[0]
    assert "Python" in top_chunk.content or top_chunk.chunk_id == "c2"

    # Test explicit fallback method
    fallback_results = reranker._fallback_rerank(candidates, top_n=2, threshold=0.1)
    assert len(fallback_results) == 2
    assert fallback_results[0].rerank_score > fallback_results[1].rerank_score
