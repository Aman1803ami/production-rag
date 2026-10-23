import re
import uuid
from typing import List, Dict, Any, Optional
from app.models.internal import Chunk
from app.config import settings


class RecursiveChunker:
    """
    Production Recursive Character & Markdown Chunker.
    Splits text hierarchically using natural document boundaries (headers, paragraphs, sentences),
    guaranteeing maximum chunk length while preserving semantic integrity and overlap.
    """

    DEFAULT_SEPARATORS = [
        "\n\n# ",
        "\n\n## ",
        "\n\n### ",
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        " ",
        ""
    ]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separators = separators or self.DEFAULT_SEPARATORS

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")

    @staticmethod
    def estimate_token_count(text: str) -> int:
        """Heuristic token estimation (~4 characters per token in English)."""
        return max(1, len(text) // 4)

    def split_text(self, text: str) -> List[str]:
        """Split text recursively using hierarchical separators until chunks fit target size."""
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Core recursive splitting algorithm."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []

        # Find the highest-priority separator present in this text
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split text by chosen separator
        splits = text.split(separator) if separator else list(text)

        current_doc: List[str] = []
        total_len = 0

        for s in splits:
            s_len = len(s)
            sep_len = len(separator) if current_doc else 0

            if total_len + s_len + sep_len > (self.chunk_size * 4):  # Approx chars
                if current_doc:
                    joined = separator.join(current_doc).strip()
                    if joined:
                        final_chunks.append(joined)
                    
                    # Compute overlap chunk
                    overlap_doc: List[str] = []
                    overlap_len = 0
                    overlap_target_chars = self.chunk_overlap * 4
                    for prev_s in reversed(current_doc):
                        if overlap_len + len(prev_s) <= overlap_target_chars:
                            overlap_doc.insert(0, prev_s)
                            overlap_len += len(prev_s)
                        else:
                            break
                    current_doc = overlap_doc
                    total_len = overlap_len

                if s_len > (self.chunk_size * 4):
                    # Sub-segment is still larger than chunk_size, recurse with finer separators
                    if new_separators:
                        sub_chunks = self._recursive_split(s, new_separators)
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(s[:self.chunk_size * 4].strip())
                else:
                    current_doc.append(s)
                    total_len += s_len
            else:
                current_doc.append(s)
                total_len += s_len + sep_len

        if current_doc:
            joined = separator.join(current_doc).strip()
            if joined:
                final_chunks.append(joined)

        return [c for c in final_chunks if len(c.strip()) > 10]

    def chunk_document(
        self,
        content: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Convert raw document text into structured Chunk objects."""
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:12]}"
        meta = metadata or {}
        text_chunks = self.split_text(content)
        
        chunks: List[Chunk] = []
        for idx, text in enumerate(text_chunks):
            chunk_id = f"{doc_id}_c{idx:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    content=text,
                    index=idx,
                    metadata={**meta, "chunk_index": idx, "total_doc_chunks": len(text_chunks)},
                    token_count=self.estimate_token_count(text)
                )
            )
        return chunks
