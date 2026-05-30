"""
Task 1 — §F: feature engineering for the GP / SVM models.

Feature vector per complaint (>= 4 inputs, single output `y`):
  - LDA topic distribution theta_0..theta_{K-1}   (K features)
  - sentiment           : VADER compound score of the narrative
  - doc_length          : number of cleaned tokens
  - complaint_lexicon   : count of negation / complaint cue words
  - urgency_markers     : count of escalation cue words (lawsuit, attorney, ...)
  - one-hot categoricals: product, sub-product, issue, sub-issue, submitted-via
    (known at complaint time -> no leakage from the resolution; high-cardinality fields are
     capped to their top-N categories with the rest folded into "__other__")

Numeric (non-theta) features are returned raw here; standardisation is done inside the model
split (fit on train only) to avoid leakage — see evaluate.fit_scaler.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Escalation / urgency cues (guide-specified, lightly extended).
URGENCY_WORDS = {
    "lawsuit", "attorney", "lawyer", "supervisor", "cancel", "escalate", "escalated",
    "fraud", "fraudulent", "legal", "sue", "court", "demand", "immediately", "urgent",
    "complaint", "report", "violation", "illegal", "harassment", "threat", "deceptive",
}

# Negation / complaint cues.
COMPLAINT_WORDS = {
    "not", "no", "never", "cannot", "wont", "didnt", "dont", "without", "refuse",
    "refused", "fail", "failed", "failure", "wrong", "error", "incorrect", "unable",
    "denied", "deny", "ignored", "unauthorized", "dispute", "disputed", "problem",
}

# Categorical columns to one-hot: (clean_name, top_n)  — top_n=None keeps all categories.
# Deliberately LOW-cardinality only: a 4K-scale experiment showed these lift GP AUC (0.63->0.64),
# whereas one-hotting high-cardinality Issue/Sub-issue (40+ sparse columns) makes the single
# length-scale GP degrade. Issue-level signal is left to future work (e.g. target/embedding encoding).
CATEGORICALS = [
    ("product", None),
    ("submitted_via", None),
    ("subproduct", 10),
]

_WORD_RE = re.compile(r"[a-z']+")


def _count(words: set[str], lowered: str) -> int:
    return sum(1 for w in _WORD_RE.findall(lowered) if w in words)


def _onehot(df: pd.DataFrame, col: str, top_n) -> pd.DataFrame:
    """Leakage-free one-hot: cap to the most frequent top_n categories (rest -> __other__)."""
    s = df[col].fillna("missing").astype(str)
    if top_n is not None:
        keep = s.value_counts().nlargest(top_n).index
        s = s.where(s.isin(keep), "__other__")
    return pd.get_dummies(s, prefix=col).astype(np.float32)


def build_features(theta: np.ndarray, df: pd.DataFrame, token_lists) -> pd.DataFrame:
    """Assemble the full feature matrix X (theta + text signals + one-hot categoricals)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    k = theta.shape[1]
    narratives = df["narrative"].tolist()

    sentiment, doc_len, complaint, urgency = [], [], [], []
    for text, toks in zip(narratives, token_lists):
        lowered = str(text).lower()
        sentiment.append(sia.polarity_scores(str(text))["compound"])
        doc_len.append(len(toks))
        complaint.append(_count(COMPLAINT_WORDS, lowered))
        urgency.append(_count(URGENCY_WORDS, lowered))

    X = pd.DataFrame(theta, columns=[f"theta_{i}" for i in range(k)])
    X["sentiment"] = sentiment
    X["doc_length"] = doc_len
    X["complaint_lexicon"] = complaint
    X["urgency_markers"] = urgency

    cat_frames = [_onehot(df, col, n) for col, n in CATEGORICALS if col in df.columns]
    if cat_frames:
        X = pd.concat([X.reset_index(drop=True)] + [c.reset_index(drop=True) for c in cat_frames],
                      axis=1)
    return X


# The continuous, human-interpretable features (used for the correlation/by-class EDA plot).
NUMERIC_FEATURES = ["sentiment", "doc_length", "complaint_lexicon", "urgency_markers"]
