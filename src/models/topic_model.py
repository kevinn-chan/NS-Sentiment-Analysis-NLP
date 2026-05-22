"""
Stage 4 — BERTopic topic modelling on pre-computed chunk embeddings.

Runs locally on CPU (umap-learn + hdbscan).
On Kaggle with a T4 GPU, replace UMAP/HDBSCAN imports with cuML equivalents
(see notebooks/kaggle_topic_model.py).

Outputs
-------
data/processed/submissions_chunks.parquet   — adds topic_id_fine, topic_prob_fine, topic_id_coarse
data/processed/comments_chunks.parquet      — same
data/processed/chunk_topics.parquet         — lightweight assignments table (all chunks)
data/processed/topic_keywords_fine.csv      — top keywords per fine-grained topic
data/processed/topic_keywords_coarse.csv    — top keywords per coarse topic
data/processed/hierarchical_topics.parquet  — full merge-tree from BERTopic
models/bertopic_fine/                       — serialised fine-grained model
models/bertopic_coarse/                     — serialised coarse model (after reduce_topics)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    COMMENTS_CHUNKS,
    DATA_PROCESSED,
    SUBMISSIONS_CHUNKS,
)

log = logging.getLogger(__name__)

ROOT         = Path(__file__).parents[2]
MODELS_DIR   = ROOT / "models"
MODEL_FINE   = MODELS_DIR / "bertopic_fine"
MODEL_COARSE = MODELS_DIR / "bertopic_coarse"
TOPICS_OUT   = DATA_PROCESSED / "chunk_topics.parquet"
UMAP_CHECKPOINT = DATA_PROCESSED / "umap_embeddings.npy"  # saved after UMAP, skipped on retry

# ── Hyperparameters ───────────────────────────────────────────────────────────
UMAP_N_COMPONENTS = 5
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.0
UMAP_METRIC       = "cosine"
HDBSCAN_MIN_SIZE  = 50
RANDOM_STATE      = 42
NR_COARSE_TOPICS  = 20
OUTLIER_MIN_PROB  = 0.3

# Singlish particles appear uniformly across all topics — useless for
# distinguishing topic content, so excluded from keyword extraction.
_SINGLISH_STOP = ["lah", "lor", "leh", "sia", "meh", "hor", "wah", "ah"]


# ── Model construction ────────────────────────────────────────────────────────

def _build_vectorizer() -> CountVectorizer:
    """
    Bigrams capture NS-specific two-word phrases ("guard duty", "book out",
    "chao keng") that would be invisible to a unigram vectorizer.
    min_df=10 drops terms appearing in fewer than 10 chunks — removes
    spelling variants and rare slang that would pollute keyword lists.
    """
    base_stop = CountVectorizer(stop_words="english").get_stop_words()
    stop_words = list(base_stop) + _SINGLISH_STOP
    return CountVectorizer(
        ngram_range=(1, 2),
        stop_words=stop_words,
        min_df=10,
    )


def _build_representation_model() -> list:
    """
    Two-stage keyword pipeline applied after c-TF-IDF:

    1. KeyBERTInspired — re-ranks c-TF-IDF candidates by cosine similarity
       to the cluster's mean embedding. Picks words that are semantically
       central to the cluster, not just frequent.

    2. MaximalMarginalRelevance (diversity=0.3) — penalises redundant
       keywords. Without this, topics often output four synonyms for the same
       concept ("ns", "national service", "nsmen", "ns man"). MMR ensures each
       keyword in the final list adds new information.
    """
    return [
        KeyBERTInspired(),
        MaximalMarginalRelevance(diversity=0.3),
    ]


def _build_topic_model() -> BERTopic:
    umap_model = UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
        low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_SIZE,
        metric="euclidean",
        prediction_data=True,
        core_dist_n_jobs=1,   # single-threaded — parallel workers OOM on 809k points
    )
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=_build_vectorizer(),
        representation_model=_build_representation_model(),
        calculate_probabilities=True,
        verbose=True,
    )


# ── Outlier rescue ────────────────────────────────────────────────────────────

def _rescue_outliers(
    topic_model: BERTopic,
    docs: list[str],
    topics: np.ndarray,
    probs: np.ndarray,
    threshold: float = OUTLIER_MIN_PROB,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Soft-assigns outlier chunks (topic_id == -1) to their nearest topic.

    How it works: approximate_distribution() slides a fixed-width window
    across each document, scores each window against every topic's c-TF-IDF
    representation, and averages the scores into a probability distribution.
    If the peak probability clears `threshold`, the chunk is re-assigned to
    that topic and its stored probability is updated.

    Chunks that stay below the threshold remain -1 — they are genuinely
    ambiguous or too sparse to belong anywhere confidently.

    Returns updated (topics, probs) arrays — originals are not modified.
    """
    topics_out = topics.copy()
    probs_out  = probs.copy()

    outlier_idx = np.where(topics_out == -1)[0]
    if len(outlier_idx) == 0:
        log.info("  No outliers to rescue.")
        return topics_out, probs_out

    log.info(f"  Running approximate_distribution on {len(outlier_idx):,} outliers …")
    outlier_docs = [docs[i] for i in outlier_idx]

    # topic_distr shape: (n_outliers, n_topics)
    # Columns are in ascending topic-ID order, excluding -1.
    topic_distr, _ = topic_model.approximate_distribution(
        outlier_docs,
        use_embedding_model=False,  # bag-of-words approach — fast, no re-embedding
        batch_size=512,
    )

    # Map column index back to actual topic ID
    unique_topics = sorted(t for t in set(topics) if t != -1)
    best_col  = topic_distr.argmax(axis=1)
    best_prob = topic_distr.max(axis=1)

    rescued = 0
    for i, (col, prob) in enumerate(zip(best_col, best_prob)):
        if prob >= threshold:
            doc_idx = outlier_idx[i]
            topics_out[doc_idx] = unique_topics[col]
            probs_out[doc_idx]  = prob
            rescued += 1

    remaining = (topics_out == -1).sum()
    log.info(f"  Rescued {rescued:,} / {len(outlier_idx):,} outliers  |  "
             f"{remaining:,} remain as -1")
    return topics_out, probs_out


# ── Coarse mapping ────────────────────────────────────────────────────────────

def _build_fine_to_coarse(
    topics_fine_raw: np.ndarray,
    topics_coarse_raw: np.ndarray,
) -> dict[int, int]:
    """
    Build a {fine_topic_id → coarse_topic_id} lookup from non-outlier docs.

    reduce_topics() only sees the original unrescued assignments, so rescued
    outliers still appear as -1 inside the model. We derive their coarse topic
    by looking up their rescued fine topic in this mapping.
    """
    mapping: dict[int, int] = {}
    for fine, coarse in zip(topics_fine_raw, topics_coarse_raw):
        if fine != -1 and coarse != -1 and fine not in mapping:
            mapping[int(fine)] = int(coarse)
    return mapping


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_chunks(path: Path, cols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=cols)
    log.info(f"  Loaded {len(df):,} rows from {path.name}")
    return df


def _stack_embeddings(series: pd.Series) -> np.ndarray:
    """Convert a column of numpy arrays / lists to a 2-D float32 matrix."""
    first = series.iloc[0]
    if isinstance(first, np.ndarray):
        return np.stack(series.values).astype(np.float32)
    return np.array(series.tolist(), dtype=np.float32)


def _export_keywords(topic_model: BERTopic, label: str) -> None:
    topic_info = topic_model.get_topic_info()
    rows = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        kws = topic_model.get_topic(tid)
        rows.append({
            "topic_id": tid,
            "count":    row["Count"],
            "name":     row.get("Name", ""),
            "keywords": ", ".join(w for w, _ in kws[:10]),
            "scores":   ", ".join(f"{s:.4f}" for _, s in kws[:10]),
        })
    path = DATA_PROCESSED / f"topic_keywords_{label}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info(f"  Saved {path.name}  ({len(rows)} topics)")


def _write_assignments_back(path: Path, assignments: pd.DataFrame) -> None:
    """Add topic columns to an existing chunk parquet."""
    df = pd.read_parquet(path)
    merge_cols = ["chunk_id", "topic_id_fine", "topic_prob_fine", "topic_id_coarse"]
    df = df.merge(assignments[merge_cols], on="chunk_id", how="left", validate="1:1")
    df.to_parquet(path, index=False)
    log.info(f"  Updated {path.name}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_topic_modelling() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info("Loading chunks …")
    sub_cols = ["chunk_id", "doc_id", "text", "embedding", "subreddit",
                "created_utc", "score", "log_weight", "doc_type"]
    com_cols = sub_cols + ["depth", "post_id"]

    sub = _load_chunks(SUBMISSIONS_CHUNKS, sub_cols)
    com = _load_chunks(COMMENTS_CHUNKS, com_cols)
    com["created_utc"] = com["created_utc"].astype("datetime64[ms, UTC]")

    combined = pd.concat([sub, com], ignore_index=True)
    log.info(f"Total: {len(combined):,} chunks")

    # ── 2. Stack embeddings ───────────────────────────────────────────────────
    log.info("Stacking embeddings …")
    embeddings = _stack_embeddings(combined["embedding"])
    log.info(f"Matrix: {embeddings.shape}")
    docs = combined["text"].tolist()

    # ── 3. Fit BERTopic (fine-grained) ────────────────────────────────────────
    # UMAP checkpoint: if a previous run completed UMAP before crashing, load
    # the reduced embeddings directly and skip the ~3hr UMAP step.
    topic_model = _build_topic_model()

    if UMAP_CHECKPOINT.exists():
        # Previous run completed UMAP but crashed during HDBSCAN.
        # Replace umap_model.fit_transform with a function that returns the
        # cached 5-dim embeddings directly — BERTopic calls fit_transform
        # internally and will receive the pre-computed result, skipping UMAP.
        log.info(f"UMAP checkpoint found — loading {UMAP_CHECKPOINT.name}, skipping UMAP …")
        _cached = np.load(UMAP_CHECKPOINT)

        def _load_cached(X, y=None):
            log.info(f"  Returning cached UMAP embeddings ({_cached.shape})")
            return _cached

        topic_model.umap_model.fit_transform = _load_cached
    else:
        # First run — patch fit_transform to save output before HDBSCAN starts,
        # so a crash there doesn't require re-running UMAP.
        log.info("Fitting BERTopic (UMAP + HDBSCAN + representation) …")
        _orig_fit_transform = topic_model.umap_model.fit_transform

        def _save_and_return(X, y=None):
            result = _orig_fit_transform(X, y)
            np.save(UMAP_CHECKPOINT, result)
            log.info(f"  UMAP checkpoint saved → {UMAP_CHECKPOINT.name}")
            return result

        topic_model.umap_model.fit_transform = _save_and_return

    topics_raw, probs_raw = topic_model.fit_transform(docs, embeddings)
    topics_raw = np.array(topics_raw, dtype=np.int16)
    probs_raw  = probs_raw.max(axis=1).astype(np.float32)

    n_fine    = len(set(topics_raw)) - (1 if -1 in topics_raw else 0)
    n_outlier = (topics_raw == -1).sum()
    log.info(f"Fine topics: {n_fine}  |  Initial outliers: {n_outlier:,}")

    # ── 4. Save fine model ────────────────────────────────────────────────────
    # Must save BEFORE reduce_topics() modifies the model in-place.
    log.info(f"Saving fine model → {MODEL_FINE} …")
    topic_model.save(str(MODEL_FINE), serialization="safetensors", save_ctfidf=True)
    _export_keywords(topic_model, "fine")

    # ── 5. Hierarchical topic tree ────────────────────────────────────────────
    # Computes a full dendrogram showing which topics are most similar and
    # in what order they would merge. Saved for dashboard visualisation.
    log.info("Computing hierarchical topic tree …")
    hierarchical_df = topic_model.hierarchical_topics(docs)
    hierarchical_df.to_parquet(DATA_PROCESSED / "hierarchical_topics.parquet", index=False)
    log.info("  Saved hierarchical_topics.parquet")

    # ── 6. Rescue outliers (fine level) ──────────────────────────────────────
    log.info("Rescuing outliers …")
    topics_fine, probs_fine = _rescue_outliers(
        topic_model, docs, topics_raw, probs_raw
    )

    # ── 7. Coarse reduction ───────────────────────────────────────────────────
    # reduce_topics() cuts the dendrogram at the level that yields NR_COARSE_TOPICS
    # groups and updates the model in-place. The model is now the coarse model.
    # Rescued outliers are invisible to it (they're still -1 in topics_raw),
    # so we derive their coarse topic via the empirical fine→coarse mapping.
    log.info(f"Reducing to {NR_COARSE_TOPICS} coarse topics …")
    topics_coarse_raw, _ = topic_model.reduce_topics(docs, nr_topics=NR_COARSE_TOPICS)
    topics_coarse_raw = np.array(topics_coarse_raw, dtype=np.int16)

    fine_to_coarse = _build_fine_to_coarse(topics_raw, topics_coarse_raw)
    topics_coarse = np.array(
        [fine_to_coarse.get(int(t), t) if t != -1 else -1 for t in topics_fine],
        dtype=np.int16,
    )

    n_coarse_outlier = (topics_coarse == -1).sum()
    log.info(f"Coarse topics: {NR_COARSE_TOPICS}  |  Unmapped after coarse: {n_coarse_outlier:,}")

    log.info(f"Saving coarse model → {MODEL_COARSE} …")
    topic_model.save(str(MODEL_COARSE), serialization="safetensors", save_ctfidf=True)
    _export_keywords(topic_model, "coarse")

    # ── 8. Assignments table ──────────────────────────────────────────────────
    assignments = pd.DataFrame({
        "chunk_id":        combined["chunk_id"].values,
        "doc_id":          combined["doc_id"].values,
        "doc_type":        combined["doc_type"].values,
        "subreddit":       combined["subreddit"].values,
        "created_utc":     combined["created_utc"].values,
        "topic_id_fine":   topics_fine,
        "topic_prob_fine": probs_fine,
        "topic_id_coarse": topics_coarse,
    })
    assignments.to_parquet(TOPICS_OUT, index=False)
    log.info(f"Saved chunk_topics.parquet")

    # ── 9. Write back to chunk parquets ──────────────────────────────────────
    log.info("Writing topic columns back to chunk parquets …")
    _write_assignments_back(
        SUBMISSIONS_CHUNKS,
        assignments[assignments["doc_type"] == "submission"],
    )
    _write_assignments_back(
        COMMENTS_CHUNKS,
        assignments[assignments["doc_type"] == "comment"],
    )

    log.info("Stage 4 complete.")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    run_topic_modelling()
