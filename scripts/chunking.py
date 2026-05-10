from typing import Any, Dict, List

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class Chunker:
    """Split text into chunks with semantic awareness using HuggingFace."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=90,
            embed_model=embed_model,
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text semantically using HuggingFace embeddings."""
        if not text or len(text.strip()) < 50:
            return [text.strip()] if text.strip() else []

        doc = Document(text=text)
        nodes = self.splitter.get_nodes_from_documents([doc], show_progress=False)
        chunks = [node.get_content() for node in nodes]

        capped_chunks: List[str] = []
        for chunk in chunks:
            capped_chunks.extend(self.ensure_chunk_size(chunk, self.chunk_size))

        return capped_chunks

    def ensure_chunk_size(self, text: str, max_size: int) -> List[str]:
        """Ensure every chunk stays under the hard size limit."""
        if len(text) <= max_size:
            return [text]

        return self.split_to_fit(text, max_size)

    def split_to_fit(self, text: str, max_size: int) -> List[str]:
        """Split oversized text into smaller parts using sentence and word boundaries."""
        parts: List[str] = []
        current_part = ""

        sentences = [
            sentence.strip() for sentence in text.split(". ") if sentence.strip()
        ]
        if not sentences:
            sentences = [text.strip()]

        for sentence in sentences:
            candidate = (
                f"{current_part}. {sentence}".strip(". ") if current_part else sentence
            )

            if len(candidate) <= max_size:
                current_part = candidate
                continue

            if current_part:
                parts.append(current_part)
                current_part = ""

            if len(sentence) <= max_size:
                current_part = sentence
                continue

            words = sentence.split()
            current_word_part = ""
            for word in words:
                word_candidate = (
                    f"{current_word_part} {word}".strip() if current_word_part else word
                )
                if len(word_candidate) <= max_size:
                    current_word_part = word_candidate
                else:
                    if current_word_part:
                        parts.append(current_word_part)
                    current_word_part = word

            if current_word_part:
                current_part = current_word_part

        if current_part:
            parts.append(current_part)

        return parts

    def chunk_job(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a job semantically, preserving metadata."""
        description = job_data.get("Job Description", "")
        text_chunks = self.split_text(description)
        return [
            {
                "chunk_id": f"{job_data['ID']}_chunk_{idx}",
                "job_id": job_data["ID"],
                "job_title": job_data.get("Job Title"),
                "company": job_data.get("Company Name"),
                "category": job_data.get("Job Category"),
                "level": job_data.get("Job Level"),
                "location": job_data.get("Job Location"),
                "publication_date": str(job_data.get("Publication Date", "")),
                "tags": job_data.get("Tags", []),
                "chunk_text": chunk,
                "chunk_index": idx,
                "total_chunks": len(text_chunks),
            }
            for idx, chunk in enumerate(text_chunks)
        ]
