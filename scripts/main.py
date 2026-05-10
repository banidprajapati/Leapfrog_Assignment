from scripts.load_data import load_csv
from scripts.preprocess_data import clean_job_description, preprocess_jobs

INPUT_PATH = "RAG_system/data/raw/LF_Jobs.csv"
OUTPUT_CSV = "RAG_system/data/processed/cleaned_jobs.csv"
OUTPUT_JSON = "chunks.json"


def main():
    print("Loading dataset...")
    df = load_csv(INPUT_PATH)

    print("Cleaning job descriptions...")
    df["Job Description"] = df["Job Description"].apply(clean_job_description)

    print("Saving cleaned CSV...")
    df.to_csv(OUTPUT_CSV, index=False)

    print("Creating chunks...")
    preprocess_jobs(INPUT_PATH, OUTPUT_JSON)

    print("Done.")


if __name__ == "__main__":
    main()
