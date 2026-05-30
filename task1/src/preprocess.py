"""
Task 1 — §B/§C: data loading, real-outcome labelling, and text preprocessing.

The target `y` is a REAL recorded outcome (the company's response category), not an
invented rule:  y=1 if the complaint was closed with monetary / non-monetary relief,
y=0 if closed with explanation.  Untimely / in-progress / administrative rows are dropped.

CFPB narratives contain redaction tokens (XXXX / XX) where PII was removed; these are
stripped before tokenisation so they do not dominate the LDA topics.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

RANDOM_STATE = 42

# Real filename of the export on disk (the guide's `cfpb_raw.csv` was a placeholder).
# Resolve relative to the project root (parent of src/) so it works no matter what the
# caller's working directory is — notebook in notebooks/, VS Code at the workspace root,
# or a script run from the repo root all resolve to the same file.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = str(_PROJECT_ROOT / "data" / "raw" / "complaints-2026-05-29_00_09.csv")

# CFPB column -> clean name. Issue/Sub-issue/Sub-product/Submitted via are known at complaint
# time (no leakage from the resolution) and are used as predictive categorical features.
COLUMN_MAP = {
    "Consumer complaint narrative": "narrative",
    "Product": "product",
    "Sub-product": "subproduct",
    "Issue": "issue",
    "Sub-issue": "subissue",
    "Submitted via": "submitted_via",
    "Company response to consumer": "company_response",
    "Timely response?": "timely",
    "Date received": "date",
}

# Real-outcome label definition (verified against the export's value counts).
RELIEF_RESPONSES = {"Closed with monetary relief", "Closed with non-monetary relief"}
EXPLAIN_RESPONSES = {"Closed with explanation", "Closed"}
# Everything else ("Untimely response", "In progress", ...) is dropped.

MIN_TOKENS = 10  # drop near-empty narratives (raw whitespace token count)

# ----------------------------------------------------------------------------- loading / labelling


def load_raw(path: str | Path = DEFAULT_RAW) -> pd.DataFrame:
    """Load the CFPB CSV and rename the columns we use to clean names."""
    usecols = [c for c in COLUMN_MAP]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, dtype=str)
    df = df.rename(columns=COLUMN_MAP)
    return df


def build_labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, deduplicate and attach the real-outcome target `y`."""
    df = df.copy()

    # Drop empty / near-empty narratives and duplicates.
    df = df[df["narrative"].notna()]
    df["narrative"] = df["narrative"].str.strip()
    df = df[df["narrative"] != ""]
    df = df.drop_duplicates(subset="narrative")
    df = df[df["narrative"].str.split().str.len() >= MIN_TOKENS]

    # Real-outcome target.
    def to_label(resp: str):
        if resp in RELIEF_RESPONSES:
            return 1
        if resp in EXPLAIN_RESPONSES:
            return 0
        return None  # untimely / in-progress / administrative -> dropped

    df["y"] = df["company_response"].map(to_label)
    df = df[df["y"].notna()].copy()
    df["y"] = df["y"].astype(int)
    return df.reset_index(drop=True)


def class_balance(df: pd.DataFrame) -> dict:
    n = len(df)
    pos = int(df["y"].sum())
    return {"n": n, "positive": pos, "negative": n - pos, "pct_positive": round(100 * pos / n, 2)}


def cleaning_attrition(df: pd.DataFrame) -> pd.DataFrame:
    """Row counts after each cleaning step (mirrors build_labeled) for an EDA funnel."""
    steps = []
    d = df.copy()
    steps.append(("Raw rows", len(d)))
    d = d[d["narrative"].notna()]
    d["narrative"] = d["narrative"].str.strip()
    d = d[d["narrative"] != ""]
    steps.append(("Non-empty narrative", len(d)))
    d = d.drop_duplicates(subset="narrative")
    steps.append(("After de-duplication", len(d)))
    d = d[d["narrative"].str.split().str.len() >= MIN_TOKENS]
    steps.append((f"Narrative >= {MIN_TOKENS} tokens", len(d)))
    labelled = d["company_response"].isin(RELIEF_RESPONSES | EXPLAIN_RESPONSES)
    steps.append(("Labelled (relief/explanation)", int(labelled.sum())))
    out = pd.DataFrame(steps, columns=["step", "rows"])
    out["dropped"] = out["rows"].shift(1).sub(out["rows"]).fillna(0).astype(int)
    return out


# ----------------------------------------------------------------------------- text preprocessing

# Domain-aware stopwords: standard English + CFPB redaction noise. Complaint-salient
# words (account, payment, credit, charge, refund, dispute, fee) are deliberately KEPT.
_REDACTION = {"xxxx", "xx", "xxxxxxxx", "xxx", "xxxxx", "xxxxxx"}
_EXTRA_STOP = {
    "would", "could", "also", "said", "told", "get", "got", "us", "im", "ive",
    "dont", "didnt", "company", "consumer", "complaint",
}

_TOKEN_RE = re.compile(r"[a-z]{3,}")          # alpha tokens, length >= 3 (drops numbers/punct)
_XSEQ_RE = re.compile(r"x{2,}", re.IGNORECASE)  # XXXX / XX redaction sequences


def _stopwords() -> set[str]:
    from nltk.corpus import stopwords
    return set(stopwords.words("english")) | _REDACTION | _EXTRA_STOP


def ensure_nltk() -> None:
    """Download the NLTK corpora the pipeline needs (idempotent)."""
    import nltk
    for pkg, path in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
        ("punkt", "tokenizers/punkt"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def clean_text(text: str) -> str:
    """Lowercase and strip the XXXX/XX redaction tokens before tokenisation."""
    text = text.lower()
    text = _XSEQ_RE.sub(" ", text)
    return text


def preprocess_corpus(narratives) -> list[list[str]]:
    """Return a list of cleaned, lemmatised token lists (one per narrative).

    Steps: lowercase -> strip XXXX -> alpha tokens (len>=3) -> drop stopwords ->
    WordNet lemmatise.  Token-frequency filtering (<10 docs / >50% docs) happens later
    at the gensim dictionary stage (see topics.build_dictionary_corpus).
    """
    ensure_nltk()
    from nltk.stem import WordNetLemmatizer

    stop = _stopwords()
    lemmatizer = WordNetLemmatizer()

    @lru_cache(maxsize=200_000)
    def lemma(tok: str) -> str:
        return lemmatizer.lemmatize(tok)

    docs: list[list[str]] = []
    for text in narratives:
        cleaned = clean_text(str(text))
        toks = [lemma(t) for t in _TOKEN_RE.findall(cleaned) if t not in stop]
        toks = [t for t in toks if len(t) >= 3 and t not in stop]
        docs.append(toks)
    return docs
