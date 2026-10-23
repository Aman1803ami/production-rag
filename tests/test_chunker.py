import pytest
from app.services.chunker import RecursiveChunker


def test_chunker_basic_split():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    text = (
        "# System Architecture\n\n"
        "This is paragraph one explaining the core design of our production RAG pipeline.\n\n"
        "## Subsystem Details\n\n"
        "This is paragraph two detailing the vector database and how Qdrant indexes vectors.\n\n"
        "### Ingestion Engine\n\n"
        "This is paragraph three explaining chunking, embeddings, and Redis caching."
    )
    chunks = chunker.chunk_document(text, document_id="test_doc_01")
    assert len(chunks) >= 3
    for c in chunks:
        assert c.document_id == "test_doc_01"
        assert len(c.content) > 0
        assert c.token_count > 0
        assert "chunk_index" in c.metadata


def test_chunker_overlap():
    chunker = RecursiveChunker(chunk_size=15, chunk_overlap=5)
    text = (
        "Sentence one is describing the initial configuration parameters. "
        "Sentence two provides deeper insight into the caching architecture. "
        "Sentence three covers vector databases and semantic indexing. "
        "Sentence four summarizes fallback mechanisms for zero downtime."
    )
    chunks = chunker.chunk_document(text, document_id="overlap_test")
    assert len(chunks) > 1
    # Check that adjacent chunks preserve context
    assert chunks[0].index == 0
    assert chunks[1].index == 1
