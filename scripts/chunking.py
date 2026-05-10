from typing import Any, Dict, List

from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class Chunker:
    _embed_models: dict = {}

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        model_name = "BAAI/bge-base-en-v1.5"
        if model_name not in Chunker._embed_models:
            Chunker._embed_models[model_name] = HuggingFaceEmbedding(
                model_name=model_name
            )
        embed_model = Chunker._embed_models[model_name]
        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=90,
            embed_model=embed_model,
        )
        self.size_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_text(self, text: str) -> List[str]:
        if not text or len(text.strip()) < 400:
            return [text.strip()] if text.strip() else []

        doc = Document(text=text)
        semantic_nodes = self.semantic_splitter.get_nodes_from_documents(
            [doc], show_progress=False
        )

        capped: List[str] = []
        chunk_size = self.size_splitter.chunk_size
        for node in semantic_nodes:
            content = node.get_content()
            if len(content) <= chunk_size:
                capped.append(content)
            else:
                sub_nodes = self.size_splitter.get_nodes_from_documents(
                    [Document(text=content)]
                )
                capped.extend(n.get_content() for n in sub_nodes)

        return capped

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
            }
            for idx, chunk in enumerate(text_chunks)
        ]
