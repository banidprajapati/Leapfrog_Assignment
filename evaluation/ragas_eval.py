import json
import time

import requests
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_entity_recall,
    context_precision,
    context_recall,
    faithfulness,
)

from RAG_system.core.config_core import settings

API_URL = "http://localhost:8000/query"


# -----------------------------
# SAFE HTTP CLIENT
# -----------------------------
def post_with_retry(url: str, payload: dict, retries: int = 3, timeout: int = 120):
    last_err = None

    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            last_err = e
            time.sleep(2**attempt)

    raise RuntimeError(f"Failed after retries: {last_err}")


# -----------------------------
# DATA LOADER
# -----------------------------
def load_test_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["test_queries"]


# -----------------------------
# RAG QUERY WRAPPER
# -----------------------------
def query_rag(question: str):
    data = post_with_retry(API_URL, {"query": question})

    chunks = data.get("chunks", [])

    # HARD GUARD: ensure valid structure
    contexts = []
    for c in chunks:
        if isinstance(c, dict) and "text" in c and c["text"]:
            contexts.append(str(c["text"]))

    return {
        "answer": data.get("answer", ""),
        "contexts": contexts,
    }


# -----------------------------
# DATASET BUILDER (HARDENED)
# -----------------------------
def build_ragas_dataset(test_samples):
    rows = []
    failed = 0

    for i, sample in enumerate(test_samples):
        question = sample["question"]
        ground_truth = sample.get("answer", "")

        print(f"[{i + 1}/{len(test_samples)}] Evaluating -> {question[:80]}")

        try:
            rag_result = query_rag(question)

            # skip broken outputs
            if not rag_result["answer"] or len(rag_result["contexts"]) == 0:
                failed += 1
                continue

            rows.append(
                {
                    "question": question,
                    "answer": rag_result["answer"],
                    "contexts": rag_result["contexts"],
                    "ground_truth": ground_truth,
                }
            )

        except Exception as e:
            failed += 1
            print(f"FAILED sample: {e}")
            continue

    print(f"\nFailed samples: {failed}")
    print(f"Valid samples: {len(rows)}")

    return Dataset.from_list(rows)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    test_samples = load_test_dataset("data/processed/test_dataset.json")

    ragas_dataset = build_ragas_dataset(test_samples)

    # -----------------------------
    # LLM WRAPPER (stable config)
    # -----------------------------
    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="qwen/qwen-2.5-7b-instruct",
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API,
            temperature=0,
            max_retries=3,
        )
    )

    # -----------------------------
    # EMBEDDINGS
    # -----------------------------
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    # -----------------------------
    # EVALUATION (ISOLATED RUN)
    # -----------------------------
    try:
        result = evaluate(
            ragas_dataset,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                context_entity_recall,
            ],
        )

    except Exception as e:
        raise RuntimeError(f"RAGAS evaluation crashed: {e}")

    # -----------------------------
    # OUTPUTS
    # -----------------------------
    print("\n===== RAGAS SCORES =====")
    print(result)

    df = result.to_pandas()

    print("\n===== DATAFRAME =====")
    print(df)

    df.to_csv("ragas_results.csv", index=False)

    print("\nSaved -> ragas_results.csv")


if __name__ == "__main__":
    main()
