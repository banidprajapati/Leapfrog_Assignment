import json
import re
from html import unescape

from bs4 import BeautifulSoup
from chunking import Chunker
from load_data import load_csv


def clean_job_description(html_text: str) -> str:
    if not isinstance(html_text, str):
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ")  # Use space instead of \n
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
    return text.strip()


def preprocess_jobs(
    csv_path: str,
    output_path: str = "chunks.json",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
):
    df = load_csv(csv_path)
    df["Job Description"] = df["Job Description"].apply(clean_job_description)

    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = [c for _, row in df.iterrows() for c in chunker.chunk_job(row.to_dict())]

    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2, default=str)

    return chunks
