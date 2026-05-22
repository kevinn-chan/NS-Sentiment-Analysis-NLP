# =============================================================================
# NS Sentiment — Comments Semantic Chunker with Context Prefixing (Kaggle)
# =============================================================================
# Setup instructions:
#   1. Ensure your Kaggle dataset contains BOTH:
#        - comments_clean.parquet
#        - submissions_clean.parquet   ← needed for title map
#      Upload both to the same dataset (ns-sentiment-comments) or separate ones.
#   2. Create a new Kaggle notebook (GPU T4 x2 accelerator, Internet ON)
#   3. Add the dataset(s) as data sources
#   4. Paste each cell below into separate notebook cells, run top-to-bottom
#   5. Download comments_chunks.parquet from /kaggle/working/ when done
#      and place it in data/processed/
# =============================================================================


# ── Cell 1: Install dependencies ──────────────────────────────────────────────
"""
!pip install sentence-transformers -q
!pip install spacy -q
"""


# ── Cell 2: Imports and config ────────────────────────────────────────────────

import numpy as np
import pandas as pd
import spacy
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Inlined from config.py
CHUNK_MIN_SENTENCES   = 2
CHUNK_MAX_SENTENCES   = 6
CHUNK_TOKEN_THRESHOLD = 400
CHARS_PER_TOKEN       = 4
SIMILARITY_THRESHOLD  = 0.5
EMBEDDING_MODEL       = "sentence-transformers/all-mpnet-base-v2"

# Update these paths based on os.listdir("/kaggle/input/") output
COMMENTS_PATH    = Path("/kaggle/input/datasets/kevinnchan/sentiment-chunking/comments_clean.parquet")
SUBMISSIONS_PATH = Path("/kaggle/input/datasets/kevinnchan/sentiment-chunking/submissions_clean.parquet")
OUTPUT_PATH      = Path("/kaggle/working/comments_chunks.parquet")


# ── Cell 3: Verify input files ────────────────────────────────────────────────

print("Checking input files...")
print("Comments exists:", COMMENTS_PATH.exists())
print("Submissions exists:", SUBMISSIONS_PATH.exists())


# ── Cell 4: Chunker functions ─────────────────────────────────────────────────

_nlp   = None
_model = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.blank("en")
        _nlp.add_pipe("sentencizer")
    return _nlp


def _get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)  # GPU auto-detected
    return _model


def _split_sentences(text: str) -> list:
    doc = _get_nlp()(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def _build_prefix(title: str) -> str:
    return f"Post: {title}\n\nComment: " if title else ""


def _group_by_similarity(sentences: list, embeddings: np.ndarray) -> list:
    """
    Groups sentences into semantically coherent chunks.
    Boundary fires when cosine similarity < SIMILARITY_THRESHOLD
    AND chunk has >= CHUNK_MIN_SENTENCES, or CHUNK_MAX_SENTENCES reached.
    Returns list of (chunk_text, mean_sentence_embedding) tuples.
    """
    chunks = []
    current_sents  = [sentences[0]]
    current_embeds = [embeddings[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            embeddings[i - 1].reshape(1, -1), embeddings[i].reshape(1, -1)
        )[0][0]
        over_max       = len(current_sents) >= CHUNK_MAX_SENTENCES
        semantic_break = sim < SIMILARITY_THRESHOLD and len(current_sents) >= CHUNK_MIN_SENTENCES

        if over_max or semantic_break:
            chunks.append((current_sents, current_embeds))
            current_sents  = [sentences[i]]
            current_embeds = [embeddings[i]]
        else:
            current_sents.append(sentences[i])
            current_embeds.append(embeddings[i])

    if current_sents:
        if len(current_sents) < CHUNK_MIN_SENTENCES and chunks:
            prev_sents, prev_embeds = chunks[-1]
            chunks[-1] = (prev_sents + current_sents, prev_embeds + current_embeds)
        else:
            chunks.append((current_sents, current_embeds))

    return [
        (" ".join(sents), np.mean(np.stack(embeds), axis=0))
        for sents, embeds in chunks
        if " ".join(sents).strip()
    ]


def _chunk_single_doc(doc_id, text, metadata, title_prefix=""):
    text = text.strip()
    if not text:
        return []

    sentences = _split_sentences(text)

    if len(sentences) <= 1:
        embed_text = f"{title_prefix}{text}" if title_prefix else text
        emb = _get_model().encode([embed_text], show_progress_bar=False)[0]
        return [{"chunk_id": f"{doc_id}_0", "doc_id": doc_id, "doc_type": "comment",
                 "chunk_idx": 0, "chunk_count": 1, "text": text,
                 "embedding": emb.tolist(), **metadata}]

    # Embed sentences WITHOUT prefix — grouping is about within-comment coherence
    sent_embeddings = _get_model().encode(sentences, show_progress_bar=False)
    grouped = _group_by_similarity(sentences, sent_embeddings)

    records = []
    for i, (chunk_text, _) in enumerate(grouped):
        embed_text = f"{title_prefix}{chunk_text}" if title_prefix else chunk_text
        chunk_emb  = _get_model().encode([embed_text], show_progress_bar=False)[0]
        records.append({
            "chunk_id": f"{doc_id}_{i}", "doc_id": doc_id, "doc_type": "comment",
            "chunk_idx": i, "chunk_count": len(grouped), "text": chunk_text,
            "embedding": chunk_emb.tolist(), **metadata,
        })
    return records


def chunk_comments(df: pd.DataFrame, title_map: dict) -> pd.DataFrame:
    metadata_cols = ["subreddit", "created_utc", "score", "log_weight",
                     "author", "post_id", "permalink", "parent_id", "depth"]

    char_threshold = CHUNK_TOKEN_THRESHOLD * CHARS_PER_TOKEN
    is_long  = df["body"].str.len() > char_threshold
    short_df = df[~is_long]
    long_df  = df[is_long]

    print(f"  Single-chunk: {len(short_df):,} | Needs splitting: {len(long_df):,}")

    # ── Fast path: short documents ────────────────────────────────────────────
    keep_cols    = ["id"] + [c for c in metadata_cols if c in df.columns] + ["body"]
    short_chunks = short_df[keep_cols].copy().rename(columns={"id": "doc_id", "body": "text"})
    short_chunks["doc_type"]    = "comment"
    short_chunks["chunk_idx"]   = 0
    short_chunks["chunk_count"] = 1
    short_chunks["chunk_id"]    = short_chunks["doc_id"].astype(str) + "_0"

    print("  Embedding short documents with title prefix...")
    titles = short_chunks["post_id"].astype(str).map(title_map).fillna("")
    embed_texts = [
        f"Post: {t}\n\nComment: {txt}" if t else txt
        for t, txt in zip(titles.tolist(), short_chunks["text"].tolist())
    ]

    embeddings = _get_model().encode(embed_texts, batch_size=128, show_progress_bar=True)
    short_chunks["embedding"] = [e.tolist() for e in embeddings]

    # ── Slow path: long documents ─────────────────────────────────────────────
    long_records = []
    for i, (_, row) in enumerate(long_df.iterrows()):
        text = str(row["body"]).strip()
        if not text:
            continue
        metadata = {c: row[c] for c in metadata_cols if c in row.index}
        title    = title_map.get(str(row.get("post_id", "")), "")
        prefix   = _build_prefix(title)
        long_records.extend(_chunk_single_doc(str(row["id"]), text, metadata, prefix))
        if (i + 1) % 500 == 0:
            print(f"    Split {i + 1:,} / {len(long_df):,} long documents")

    long_chunks = (
        pd.DataFrame(long_records) if long_records
        else pd.DataFrame(columns=short_chunks.columns)
    )

    return pd.concat([short_chunks, long_chunks], ignore_index=True)


# ── Cell 5: Run ───────────────────────────────────────────────────────────────

print("Loading submissions for title map...")
subs = pd.read_parquet(SUBMISSIONS_PATH, columns=["id", "title"])
title_map = dict(zip(subs["id"].astype(str), subs["title"].astype(str)))
print(f"Title map: {len(title_map):,} entries")
del subs

print("\nLoading comments_clean.parquet...")
coms = pd.read_parquet(COMMENTS_PATH)
print(f"Loaded {len(coms):,} comments")

com_chunks = chunk_comments(coms, title_map)

print(f"\nComments: {len(coms):,} docs → {len(com_chunks):,} chunks")
print(f"Saving to {OUTPUT_PATH}...")
com_chunks.to_parquet(OUTPUT_PATH, index=False)
print("Done. Download comments_chunks.parquet from /kaggle/working/")
