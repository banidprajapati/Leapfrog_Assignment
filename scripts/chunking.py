import re
from html import unescape
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


HEADING_TAGS_PATTERN = r"b|h[1-6]|strong|u|em"


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
        self.chunk_size = chunk_size

    def _clean_html(self, html_text: str) -> str:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_sections(self, html_text: str) -> List[Tuple[str, str]]:
        pattern = re.compile(
            rf"<({HEADING_TAGS_PATTERN})(?:\s+[^>]*)?>(.*?)</\1>",
            re.IGNORECASE | re.DOTALL,
        )

        raw = []
        for m in pattern.finditer(html_text):
            tag_text = re.sub(r"<[^>]+>", "", m.group(2))
            tag_text = re.sub(r"\s+", " ", tag_text).strip()
            if tag_text:
                raw.append((m.start(), m.end(), tag_text))

        if not raw:
            return [("", self._clean_html(html_text))]

        merged = []
        for h in raw:
            if merged and (h[0] - merged[-1][1]) < 30:
                prev = merged[-1]
                merged[-1] = (prev[0], h[1], prev[2] + h[2])
            else:
                merged.append(h)

        sections = []
        prev_end = 0
        for i, (start, end, heading) in enumerate(merged):
            if i == 0 and start > 0:
                intro = self._clean_html(html_text[:start])
                if intro:
                    sections.append(("", intro))

            next_start = merged[i + 1][0] if i + 1 < len(merged) else len(html_text)
            content = self._clean_html(html_text[end:next_start])
            sections.append((heading, content))

        return sections

    def _chunk_section(self, text: str, heading: str) -> List[str]:
        if not text or len(text.strip()) < 10:
            return []
        prefix = f"[{heading}] " if heading else ""
        if len(text) <= self.chunk_size:
            return [prefix + text]
        sub_nodes = self.size_splitter.get_nodes_from_documents(
            [Document(text=text)]
        )
        return [prefix + n.get_content() for n in sub_nodes]

    def split_text(self, text: str) -> List[str]:
        if not text or len(text.strip()) < 400:
            return [text.strip()] if text.strip() else []

        doc = Document(text=text)
        semantic_nodes = self.semantic_splitter.get_nodes_from_documents(
            [doc], show_progress=False
        )

        capped: List[str] = []
        for node in semantic_nodes:
            content = node.get_content()
            if len(content) <= self.chunk_size:
                capped.append(content)
            else:
                sub_nodes = self.size_splitter.get_nodes_from_documents(
                    [Document(text=content)]
                )
                capped.extend(n.get_content() for n in sub_nodes)

        return capped

    def chunk_job(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_html = job_data.get("Raw Description", job_data.get("Job Description", ""))
        description = job_data.get("Job Description", "")

        sections = self._extract_sections(raw_html)

        if len(sections) == 1 and sections[0][0] == "":
            text_chunks = self.split_text(description)
        else:
            text_chunks = []
            for heading, content in sections:
                text_chunks.extend(self._chunk_section(content, heading))

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
