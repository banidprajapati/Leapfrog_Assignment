import json
import re
from html import unescape

from bs4 import BeautifulSoup
from scripts.chunking import Chunker
from scripts.load_data import load_csv


def clean_job_description(html_text: str) -> str:
    if not isinstance(html_text, str):
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = unescape(text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    return text


def preprocess_jobs(
    csv_path: str,
    output_path: str | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    cleaned_csv_path: str | None = None,
):
    df = load_csv(csv_path)
    df["Job Description"] = df["Job Description"].apply(clean_job_description)

    if cleaned_csv_path:
        df.to_csv(cleaned_csv_path, index=False)

    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = [c for _, row in df.iterrows() for c in chunker.chunk_job(row.to_dict())]

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, default=str)

    return chunks
