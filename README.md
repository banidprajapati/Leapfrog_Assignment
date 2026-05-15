# LF RAG System

Job listing RAG (Retrieval-Augmented Generation) system using FastAPI, Qdrant, and LLMs.

## Setup

```bash
uv venv && source .venv/bin/activate && uv sync
cp .env.example .env   # fill in your keys
python -m spacy download en_core_web_sm
```

## Create

Place your job listings CSV at `data/raw/LF_Jobs.csv` with these columns:

```
ID, Job Title, Job Description, Company Name, Job Category, Job Level, Job Location, Publication Date, Tags
```

Then run:

```bash
# All-in-one: preprocess, chunk, embed, seed Qdrant
python -m scripts.main
```

## Query

```bash
uvicorn RAG_system.main:app --port 8000

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "senior software engineer at Meta"}'
```

## Chunking

- HTML extracted by headings (h1-h6, b, strong, u, em)
- **Semantic splitting** via `llama-index` + `bge-base-en-v1.5` embeddings
- Falls back to `SentenceSplitter` for long sections
- Headings preserved as `[Heading]` prefix on chunks
- **Boilerplate filtered** — EEO, accommodations, legal disclaimers dropped
- Each chunk stores metadata: job_id, company, level, location, category, tags

## Evaluate

```bash
# Requires the server to be running on :8000
python -m evaluation.ragas_eval
```



## Structure

```
scripts/          — preprocessing, chunking, seeding Qdrant
RAG_system/
  core/           — config, logging, OpenAI client
  services/       — retrieval, reranking, vector search, generation
  routers/        — FastAPI endpoints
  schemas/        — request/response models
  repositories/   — data access layer
  utils/          — query parser (spaCy entity extraction)
data/             — raw CSV, processed chunks, test set
evaluation/       — RAGAS evaluation script
```
