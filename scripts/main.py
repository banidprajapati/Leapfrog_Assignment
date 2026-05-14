from scripts.preprocess_data import preprocess_jobs
from scripts.seed_db import seed_vector_db

INPUT_PATH = "data/raw/LF_Jobs.csv"
OUTPUT_CSV = "data/processed/cleaned_jobs.csv"
SAVED_JSON = "data/processed/saved_vectors.json"


def main():
    print("Creating chunks...")
    chunks = preprocess_jobs(
        csv_path=INPUT_PATH,
        cleaned_csv_path=OUTPUT_CSV,
    )

    print("Seeding vector database...")
    saved = seed_vector_db(chunks, SAVED_JSON)
    print(f"Saved vector record JSON: {SAVED_JSON} ({len(saved)} items)")

    print("Done.")


if __name__ == "__main__":
    main()
