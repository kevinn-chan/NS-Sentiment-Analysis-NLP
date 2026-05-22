"""
Kaggle GPU notebook — Stage 4 BERTopic (cuML-accelerated).

Upload submissions_chunks.parquet + comments_chunks.parquet as a Kaggle dataset,
update the two SUBMISSIONS_CHUNKS / COMMENTS_CHUNKS paths below, then run via
Save & Run All (committed mode — not interactive, so the kernel won't die on you).

Differences from src/models/topic_model.py
  - UMAP / HDBSCAN from cuml.manifold / cuml.cluster (GPU-accelerated)
  - Paths reference /kaggle/input and /kaggle/working
  - No local config import — paths are inlined
"""

# ── 0. Discover exact input paths ────────────────────────────────────────────
import os
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        print(os.path.join(root, f))

# ── 1. Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from sklearn.feature_extraction.text import CountVectorizer

# ── 2. Config ─────────────────────────────────────────────────────────────────
# Update these paths after running the os.walk cell above.
SUBMISSIONS_CHUNKS = "/kaggle/input/<your-dataset>/submissions_chunks.parquet"
COMMENTS_CHUNKS    = "/kaggle/input/<your-dataset>/comments_chunks.parquet"
OUT_DIR            = "/kaggle/working"

NR_COARSE_TOPICS = 20
OUTLIER_MIN_PROB = 0.3

_SINGLISH_STOP = ["lah", "lor", "leh", "sia", "meh", "hor", "wah", "ah"]

# ── 3. Load data ──────────────────────────────────────────────────────────────
sub_cols = ["chunk_id", "doc_id", "text", "embedding", "subreddit",
            "created_utc", "score", "log_weight", "doc_type"]
com_cols = sub_cols + ["depth", "post_id"]

sub = pd.read_parquet(SUBMISSIONS_CHUNKS, columns=sub_cols)
com = pd.read_parquet(COMMENTS_CHUNKS, columns=com_cols)
com["created_utc"] = com["created_utc"].astype("datetime64[ms, UTC]")

combined = pd.concat([sub, com], ignore_index=True)
print(f"Total chunks: {len(combined):,}")

# ── 4. Stack embeddings ───────────────────────────────────────────────────────
first = combined["embedding"].iloc[0]
embeddings = (
    np.stack(combined["embedding"].values).astype(np.float32)
    if isinstance(first, np.ndarray)
    else np.array(combined["embedding"].tolist(), dtype=np.float32)
)
print(f"Embedding matrix: {embeddings.shape}")
docs = combined["text"].tolist()

# ── 5. Build model ────────────────────────────────────────────────────────────
base_stop  = CountVectorizer(stop_words="english").get_stop_words()
vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    stop_words=list(base_stop) + _SINGLISH_STOP,
    min_df=10,
)
representation_model = [
    KeyBERTInspired(),
    MaximalMarginalRelevance(diversity=0.3),
]
umap_model = UMAP(
    n_components=5, n_neighbors=15, min_dist=0.0,
    metric="cosine", random_state=42,
)
hdbscan_model = HDBSCAN(
    min_cluster_size=50, metric="euclidean", prediction_data=True,
)
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    representation_model=representation_model,
    calculate_probabilities=True,
    verbose=True,
)

# ── 6. Fit ────────────────────────────────────────────────────────────────────
topics_raw, probs_raw = topic_model.fit_transform(docs, embeddings)
topics_raw = np.array(topics_raw, dtype=np.int16)
probs_raw  = probs_raw.max(axis=1).astype(np.float32)

n_fine    = len(set(topics_raw)) - (1 if -1 in topics_raw else 0)
n_outlier = (topics_raw == -1).sum()
print(f"Fine topics: {n_fine}  |  Initial outliers: {n_outlier:,}")

# ── 7. Save fine model (before reduce_topics modifies it) ─────────────────────
topic_model.save(f"{OUT_DIR}/bertopic_fine",
                 serialization="safetensors", save_ctfidf=True)

def export_keywords(model, label):
    rows = []
    for _, row in model.get_topic_info().iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        kws = model.get_topic(tid)
        rows.append({
            "topic_id": tid, "count": row["Count"], "name": row.get("Name", ""),
            "keywords": ", ".join(w for w, _ in kws[:10]),
            "scores":   ", ".join(f"{s:.4f}" for _, s in kws[:10]),
        })
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/topic_keywords_{label}.csv", index=False)
    print(f"Saved topic_keywords_{label}.csv  ({len(rows)} topics)")

export_keywords(topic_model, "fine")

# ── 8. Hierarchical topic tree ────────────────────────────────────────────────
hierarchical_df = topic_model.hierarchical_topics(docs)
hierarchical_df.to_parquet(f"{OUT_DIR}/hierarchical_topics.parquet", index=False)
print("Saved hierarchical_topics.parquet")

# ── 9. Rescue outliers ────────────────────────────────────────────────────────
topics_fine = topics_raw.copy()
probs_fine  = probs_raw.copy()
outlier_idx = np.where(topics_fine == -1)[0]

if len(outlier_idx) > 0:
    print(f"Rescuing {len(outlier_idx):,} outliers …")
    outlier_docs = [docs[i] for i in outlier_idx]
    topic_distr, _ = topic_model.approximate_distribution(
        outlier_docs, use_embedding_model=False, batch_size=512
    )
    unique_topics = sorted(t for t in set(topics_raw) if t != -1)
    best_col  = topic_distr.argmax(axis=1)
    best_prob = topic_distr.max(axis=1)
    rescued = 0
    for i, (col, prob) in enumerate(zip(best_col, best_prob)):
        if prob >= OUTLIER_MIN_PROB:
            topics_fine[outlier_idx[i]] = unique_topics[col]
            probs_fine[outlier_idx[i]]  = prob
            rescued += 1
    print(f"Rescued {rescued:,} / {len(outlier_idx):,}  |  "
          f"{(topics_fine == -1).sum():,} remain as -1")

# ── 10. Coarse reduction ──────────────────────────────────────────────────────
print(f"Reducing to {NR_COARSE_TOPICS} coarse topics …")
topics_coarse_raw, _ = topic_model.reduce_topics(docs, nr_topics=NR_COARSE_TOPICS)
topics_coarse_raw = np.array(topics_coarse_raw, dtype=np.int16)

# Build fine→coarse mapping from non-outlier docs, apply to rescued outliers
fine_to_coarse = {}
for fine, coarse in zip(topics_raw, topics_coarse_raw):
    if fine != -1 and coarse != -1 and fine not in fine_to_coarse:
        fine_to_coarse[int(fine)] = int(coarse)

topics_coarse = np.array(
    [fine_to_coarse.get(int(t), t) if t != -1 else -1 for t in topics_fine],
    dtype=np.int16,
)
print(f"Coarse unmapped: {(topics_coarse == -1).sum():,}")

topic_model.save(f"{OUT_DIR}/bertopic_coarse",
                 serialization="safetensors", save_ctfidf=True)
export_keywords(topic_model, "coarse")

# ── 11. Assignments table ─────────────────────────────────────────────────────
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
assignments.to_parquet(f"{OUT_DIR}/chunk_topics.parquet", index=False)
print("Saved chunk_topics.parquet")

# ── 12. Write topic columns back into chunk parquets ─────────────────────────
merge_cols = ["chunk_id", "topic_id_fine", "topic_prob_fine", "topic_id_coarse"]
for doc_type, src_path, out_name in [
    ("submission", SUBMISSIONS_CHUNKS, "submissions_chunks.parquet"),
    ("comment",    COMMENTS_CHUNKS,    "comments_chunks.parquet"),
]:
    df = pd.read_parquet(src_path)
    subset = assignments[assignments["doc_type"] == doc_type][merge_cols]
    df = df.merge(subset, on="chunk_id", how="left", validate="1:1")
    df.to_parquet(f"{OUT_DIR}/{out_name}", index=False)
    print(f"Saved {out_name}")

print("Stage 4 complete.")
