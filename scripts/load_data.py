import pandas as pd

REQUIRED_COLS = [
    "ID",
    "Job Title",
    "Job Description",
    "Company Name",
    "Job Category",
    "Job Level",
    "Job Location",
    "Publication Date",
    "Tags",
]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[REQUIRED_COLS]

    for col in [
        "Job Title",
        "Company Name",
        "Job Category",
        "Job Level",
        "Job Location",
    ]:
        df[col] = df[col].str.strip().replace("", None)

    df["Tags"] = (
        df["Tags"]
        .fillna("")
        .str.lower()
        .str.split(", ")
        .apply(lambda x: [t.strip() for t in x if t.strip()])
    )
    df["Publication Date"] = pd.to_datetime(df["Publication Date"], errors="coerce")

    return df
