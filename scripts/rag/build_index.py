"""
RAG Phase 1 — Build FAISS index + chunk_metadata.parquet.

Loads embeddings directly from the chunk parquets (already saved by chunker.py).
Joins sentiment scores from chunk_sentiment.parquet.
Maps topic_id_fine → topic_macro using TOPIC_LABELS.
Normalises datetime to year + month integers.

Run once:
    python -m scripts.rag.build_index

Output:
    data/processed/new/chunk_faiss.index       (~2.1 GB, FAISS IndexFlatIP)
    data/processed/new/chunk_metadata.parquet  (lightweight, no embeddings)
"""

import logging
import sys
import time

import faiss
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))

from src.rag.config import (
    SUBMISSIONS_CHUNKS, COMMENTS_CHUNKS, CHUNK_SENTIMENT,
    FAISS_INDEX, CHUNK_METADATA, EMBEDDING_DIM,
)
from src.models.topic_labels import TOPIC_LABELS

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

TOPIC_MACRO_MAP = {tid: info["macro"] for tid, info in TOPIC_LABELS.items()}


def load_chunks() -> pd.DataFrame:
    """Load both chunk parquets, keep only columns needed."""
    KEEP = [
        "chunk_id", "doc_id", "doc_type", "subreddit",
        "created_utc", "score", "log_weight",
        "chunk_idx", "chunk_count",
        "topic_id_fine",
        "embedding", "text",
    ]

    log.info("Loading submissions_chunks …")
    subs = pd.read_parquet(SUBMISSIONS_CHUNKS, columns=KEEP)
    log.info(f"  {len(subs):,} submission chunks")

    log.info("Loading comments_chunks …")
    # comments have post_id instead of direct doc columns
    KEEP_COM = KEEP + ["post_id", "depth"]
    available = [c for c in KEEP_COM if c in pd.read_parquet(COMMENTS_CHUNKS, columns=["chunk_id"]).columns or True]
    coms = pd.read_parquet(COMMENTS_CHUNKS)
    coms = coms[[c for c in KEEP if c in coms.columns]]
    log.info(f"  {len(coms):,} comment chunks")

    df = pd.concat([subs, coms], ignore_index=True)
    log.info(f"  Combined: {len(df):,} chunks")
    return df


def load_sentiment() -> pd.DataFrame:
    """Load chunk_sentiment.parquet — sent_neg/neu/pos scores."""
    log.info("Loading chunk_sentiment …")
    sent = pd.read_parquet(CHUNK_SENTIMENT, columns=["chunk_id", "sent_neg", "sent_neu", "sent_pos"])
    log.info(f"  {len(sent):,} sentiment rows")
    return sent


def build_metadata(df: pd.DataFrame, sent: pd.DataFrame) -> pd.DataFrame:
    """
    Build the lightweight metadata table used for:
    - FAISS pre-filtering (year, month, subreddit, topic_macro)
    - Citation display in the chat UI (text_snippet, upvotes, etc.)

    Does NOT include embedding vectors — those stay in the FAISS index.
    """
    log.info("Building metadata …")

    # Join sentiment scores
    df = df.merge(sent, on="chunk_id", how="left")
    log.info(f"  After sentiment join: {len(df):,} rows ({df['sent_neg'].isna().sum():,} missing sentiment)")

    # Map topic_id_fine → topic_macro
    df["topic_macro"] = df["topic_id_fine"].map(TOPIC_MACRO_MAP).fillna("Unknown")

    # Normalise datetime → year + month (handle both ns and ms precision)
    ts = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
    df["year"]  = ts.dt.year.astype("Int16")
    df["month"] = ts.dt.month.astype("Int8")

    # Upvotes: score column, coerce to int, floor at 0
    df["upvotes"] = df["score"].fillna(0).clip(lower=0).astype(int)

    # Text snippet for citation display (first 300 chars)
    df["text_snippet"] = df["text"].fillna("").str[:300]

    # faiss_idx = row position in final sorted order (set after dedup below)
    # We'll assign it after sorting

    # Select final metadata columns
    meta = df[[
        "chunk_id", "doc_id", "doc_type", "subreddit",
        "year", "month", "topic_macro", "topic_id_fine",
        "sent_neg", "sent_neu", "sent_pos",
        "upvotes", "log_weight",
        "text_snippet",
    ]].copy()

    # Deduplicate on chunk_id (1,545 duplicates from batch boundary issue per HANDOFF)
    before = len(meta)
    meta = meta.drop_duplicates(subset="chunk_id", keep="first").reset_index(drop=True)
    log.info(f"  Deduped: {before - len(meta):,} duplicates removed → {len(meta):,} unique chunks")

    # faiss_idx = row index (used to map FAISS search results back to metadata)
    meta["faiss_idx"] = meta.index

    log.info(f"  Metadata shape: {meta.shape}")
    log.info(f"  Topic coverage: {meta['topic_macro'].nunique()} macros")
    log.info(f"  Year range: {meta['year'].min()} – {meta['year'].max()}")
    log.info(f"  Subreddits: {sorted(meta['subreddit'].unique().tolist())}")
    log.info(f"  Upvotes: mean={meta['upvotes'].mean():.1f}, max={meta['upvotes'].max()}")

    return meta


def extract_embeddings(df: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
    """
    Extract embeddings for unique chunks in metadata order.
    Returns float32 array shape (N, 768), L2-normalised.
    """
    log.info("Extracting embeddings …")

    # Build chunk_id → embedding mapping
    chunk_to_emb = {}
    for _, row in df.drop_duplicates(subset="chunk_id", keep="first").iterrows():
        emb = row["embedding"]
        if emb is None:
            continue
        if isinstance(emb, list):
            emb = np.array(emb, dtype=np.float32)
        elif not isinstance(emb, np.ndarray):
            emb = np.array(emb, dtype=np.float32)
        chunk_to_emb[row["chunk_id"]] = emb.astype(np.float32)

    # Stack in metadata order
    embeddings = []
    missing = 0
    for chunk_id in meta["chunk_id"]:
        if chunk_id in chunk_to_emb:
            embeddings.append(chunk_to_emb[chunk_id])
        else:
            embeddings.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
            missing += 1

    if missing:
        log.warning(f"  {missing:,} chunks had no embedding — filled with zeros")

    arr = np.stack(embeddings, axis=0).astype(np.float32)
    log.info(f"  Embedding array: {arr.shape}  dtype={arr.dtype}")

    # L2-normalise for inner-product = cosine similarity
    faiss.normalize_L2(arr)
    log.info("  L2-normalised ✓")

    return arr


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a flat inner-product index (exact cosine search on normalised vectors)."""
    log.info("Building FAISS IndexFlatIP …")
    t0 = time.time()
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    elapsed = time.time() - t0
    log.info(f"  Index built: {index.ntotal:,} vectors in {elapsed:.1f}s")
    return index


def main():
    t_start = time.time()

    # Load
    df = load_chunks()
    sent = load_sentiment()

    # Build metadata
    meta = build_metadata(df, sent)

    # Extract embeddings in metadata row order
    embeddings = extract_embeddings(df, meta)

    # Verify alignment
    assert len(meta) == len(embeddings), \
        f"Metadata ({len(meta)}) vs embeddings ({len(embeddings)}) mismatch!"

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Verify
    assert index.ntotal == len(meta), \
        f"FAISS ntotal ({index.ntotal}) != metadata rows ({len(meta)})"

    # Smoke test: search for first chunk embedding
    test_vec = embeddings[:1].copy()
    D, I = index.search(test_vec, 3)
    log.info(f"Smoke test — top-3 for chunk 0: indices={I[0].tolist()}  sims={D[0].tolist()}")
    assert I[0][0] == 0, "Smoke test FAILED: chunk 0 should be its own nearest neighbour"
    log.info("Smoke test PASSED ✓")

    # Save
    log.info(f"Saving FAISS index → {FAISS_INDEX}")
    FAISS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX))

    log.info(f"Saving chunk metadata → {CHUNK_METADATA}")
    meta.to_parquet(CHUNK_METADATA, index=False)

    elapsed = time.time() - t_start
    log.info(f"\n{'='*60}")
    log.info(f"Build complete in {elapsed/60:.1f} min")
    log.info(f"  FAISS index : {FAISS_INDEX}  ({FAISS_INDEX.stat().st_size / 1e9:.2f} GB)")
    log.info(f"  Metadata    : {CHUNK_METADATA}  ({CHUNK_METADATA.stat().st_size / 1e6:.1f} MB)")
    log.info(f"  Vectors     : {index.ntotal:,}")
    log.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
