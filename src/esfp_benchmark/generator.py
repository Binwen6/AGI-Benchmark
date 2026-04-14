import pandas as pd
from typing import Optional
from esfp_benchmark.config import PHRASING_TEMPLATES, DEFAULT_CORPUS_PATH

def load_corpus(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the curated ESFP question corpus.
    The CSV has 6 columns (T1–T6) with variable row counts per type.
    """
    path = path or DEFAULT_CORPUS_PATH
    raw_df = pd.read_csv(path)

    # Convert categorised columns (T1-T6) into a row-based structure (type, question)
    corpus_df = raw_df.melt(var_name="type", value_name="question")

    # Remove empty rows, purely whitespace rows, and clean up text
    corpus_df = corpus_df.dropna(subset=["question"]).copy()
    corpus_df["question"] = corpus_df["question"].astype(str).str.strip()
    corpus_df = corpus_df[corpus_df["question"].str.len() > 0].reset_index(drop=True)

    # Automatically generate sequential IDs starting from 1
    corpus_df["id"] = corpus_df.index + 1

    assert {"id", "type", "question"}.issubset(corpus_df.columns), (
        "corpus CSV must contain columns: id, type, question"
    )

    print(f"Corpus loaded: {len(corpus_df)} questions across {corpus_df['type'].nunique()} types")
    return corpus_df

def generate_variants(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each corpus item into 5 phrasing variants (P0–P4).

    Returns
    -------
    pd.DataFrame with columns:
        item_id     : original corpus item ID
        type        : question category (T1–T4)
        question    : verbatim question text from corpus
        phrasing    : phrasing key (P0–P4)
        prompt_text : fully formatted prompt ready for inference
    """
    rows = []
    for _, row in corpus_df.iterrows():
        for phrasing, template in PHRASING_TEMPLATES.items():
            rows.append({
                "item_id":     row["id"],
                "type":        row["type"],
                "question":    row["question"],
                "phrasing":    phrasing,
                "prompt_text": template.format(question=row["question"]),
            })
    df = pd.DataFrame(rows)
    print(
        f"Generator produced {len(df)} variants "
        f"({corpus_df.shape[0]} questions x {len(PHRASING_TEMPLATES)} phrasings)"
    )
    return df
