"""
Task 1 — §D/§E: LDA topic modelling (common methodology) and the LSA comparison.

LDA is fitted on the FULL cleaned corpus for K in {5,10,15,20}; K is chosen by c_v
coherence (held-out perplexity reported alongside).  LSA = TF-IDF -> TruncatedSVD with
the same number of components, scored with the same c_v coherence for a fair comparison.
"""
from __future__ import annotations

import numpy as np

RANDOM_STATE = 42

# Coherence c_v uses a sliding window over `texts`; on tens of thousands of docs that is
# slow, so we estimate it on a representative random subset (a standard, disclosed
# approximation). 8K texts gives a stable c_v estimate at a fraction of the cost.
COHERENCE_SAMPLE = 8_000


# ----------------------------------------------------------------------------- LDA

def build_dictionary_corpus(token_lists, no_below=10, no_above=0.5):
    """gensim Dictionary + bag-of-words corpus, with extreme tokens filtered."""
    from gensim.corpora import Dictionary

    dictionary = Dictionary(token_lists)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(doc) for doc in token_lists]
    return dictionary, corpus


def _coherence(model, token_lists, dictionary, rng):
    from gensim.models import CoherenceModel

    if len(token_lists) > COHERENCE_SAMPLE:
        idx = rng.choice(len(token_lists), COHERENCE_SAMPLE, replace=False)
        texts = [token_lists[i] for i in idx]
    else:
        texts = token_lists
    # processes=1: c_v's parallel accumulator spawns workers that re-import the calling
    # module — on Windows (spawn) that raises a bootstrapping RuntimeError inside a script
    # or Jupyter kernel. The serial accumulator avoids it at a small speed cost.
    cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary,
                        coherence="c_v", processes=1)
    return float(cm.get_coherence())


def fit_lda(dictionary, corpus, k, passes=5, iterations=50):
    from gensim.models import LdaModel

    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=RANDOM_STATE,
        passes=passes,
        iterations=iterations,
        eval_every=None,
        alpha="auto",
        eta="auto",
    )


def select_k(dictionary, corpus, token_lists, k_values=(5, 10, 15, 20), passes=5):
    """Fit LDA for each K; return (results_df, models_dict).

    results_df columns: k, coherence_cv, log_perplexity, perplexity.
    """
    import pandas as pd

    rng = np.random.default_rng(RANDOM_STATE)
    import time
    rows, models = [], {}
    # held-out split for perplexity
    n = len(corpus)
    held = set(rng.choice(n, max(1, n // 10), replace=False).tolist())
    train_corpus = [c for i, c in enumerate(corpus) if i not in held]
    test_corpus = [c for i, c in enumerate(corpus) if i in held]

    for i, k in enumerate(k_values, 1):
        t = time.time()
        print(f"  [LDA {i}/{len(k_values)}] fitting K={k} on {len(train_corpus):,} docs ...",
              flush=True)
        model = fit_lda(dictionary, train_corpus, k, passes=passes)
        coh = _coherence(model, token_lists, dictionary, rng)
        log_perp = float(model.log_perplexity(test_corpus))  # per-word lower bound
        rows.append({
            "k": k,
            "coherence_cv": coh,
            "log_perplexity": log_perp,
            "perplexity": float(np.exp(-log_perp)),
        })
        models[k] = model
        print(f"  [LDA {i}/{len(k_values)}] K={k} done: c_v={coh:.3f}  ({time.time()-t:.0f}s)",
              flush=True)
    return pd.DataFrame(rows), models


def refit_full(dictionary, corpus, k, passes=8):
    """Refit the chosen-K model on the FULL corpus (used for the final θ matrix)."""
    return fit_lda(dictionary, corpus, k, passes=passes)


def topic_distribution_matrix(model, corpus, k) -> np.ndarray:
    """Dense per-document topic distribution θ, shape (n_docs, k)."""
    theta = np.zeros((len(corpus), k), dtype=np.float32)
    for i, bow in enumerate(corpus):
        for topic_id, prob in model.get_document_topics(bow, minimum_probability=0.0):
            theta[i, topic_id] = prob
    return theta


def top_words(model, k, topn=10) -> dict[int, list[str]]:
    return {t: [w for w, _ in model.show_topic(t, topn=topn)] for t in range(k)}


# ----------------------------------------------------------------------------- LSA

def fit_lsa(token_lists, n_components, topn=10):
    """TF-IDF -> TruncatedSVD. Returns (svd, vectorizer, top_words_per_component)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    joined = [" ".join(doc) for doc in token_lists]
    vec = TfidfVectorizer(min_df=10, max_df=0.5)
    dtm = vec.fit_transform(joined)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    svd.fit(dtm)

    terms = np.array(vec.get_feature_names_out())
    comp_words = {}
    for c in range(n_components):
        # top-|weight| terms; LSA components can be sign-mixed (a known interpretability cost)
        top_idx = np.argsort(np.abs(svd.components_[c]))[::-1][:topn]
        comp_words[c] = terms[top_idx].tolist()
    return svd, vec, comp_words


def coherence_of_topics(topic_words: dict[int, list[str]], token_lists, dictionary):
    """c_v coherence for an arbitrary set of top-word lists (used for both LDA & LSA)."""
    from gensim.models import CoherenceModel

    rng = np.random.default_rng(RANDOM_STATE)
    if len(token_lists) > COHERENCE_SAMPLE:
        idx = rng.choice(len(token_lists), COHERENCE_SAMPLE, replace=False)
        texts = [token_lists[i] for i in idx]
    else:
        texts = token_lists
    topics = list(topic_words.values())
    cm = CoherenceModel(
        topics=topics, texts=texts, dictionary=dictionary, coherence="c_v", processes=1
    )
    return float(cm.get_coherence())
