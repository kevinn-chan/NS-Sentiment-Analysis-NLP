"""
Stage 3 — Semantic chunking with integrated NS relevance filtering.

Changes from v1:
- All documents go through sentence splitting + similarity grouping regardless of length.
  The previous two-path shortcut (short docs → single embed) caused mixed-topic short
  documents to receive one topic label, losing the second topic's signal.
- NS relevance filter integrated at chunk level: only NS-relevant chunks are emitted,
  eliminating the need for the downstream ns_filter.py step.
- Two-pass batch embedding: sentences embedded for grouping only; final chunks
  re-embedded with context prefix for topic modelling.

Minimum chunk size: CHUNK_MIN_CHARS (200 chars). Trailing fragments below this are
merged into the previous chunk. Whole documents under 200 chars are kept as-is.
"""

import logging
import re

import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    SUBMISSIONS_CLEAN,
    COMMENTS_CLEAN,
    SUBMISSIONS_CHUNKS,
    COMMENTS_CHUNKS,
    DATA_PROCESSED,
    CHUNK_MIN_SENTENCES,
    CHUNK_MAX_SENTENCES,
    SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
)

log = logging.getLogger(__name__)

_nlp   = None
_model = None

# ── NS relevance filter (chunk-level) ─────────────────────────────────────────
# r/NationalServiceSG chunks always pass — entire subreddit is NS-contextual.
# For r/singapore / r/askSingapore: strong term OR ≥2 distinct weak term matches.

_STRONG_TERMS = {
    "enlistment", "enlist", "enlisted", "enlistee",
    "reservist", "reservists",
    "ord", "rod",
    "ippt",
    "ict",
    "bmt", "bmtc", "tekong",
    "ocs", "scs",
    "scdf", "spf", "ndu",
    "conscription", "conscript", "nsman", "nsmen", "ns man",
    "national service",
    "mindef",
    "pes status", "downpes", "medical board",
    "guard duty", "book in", "book out", "confined to camp",
    "route march", "outfield",
    "chao keng", "sign extra",
    "emart",
}

_WEAK_TERMS = {
    "ns", "nsf",
    "saf",
    "bmt",
    "pes",
    "db",
    "recruit",
    "sergeant", "encik",
    "vocation",
    "wayang",
    "ippt",
    "reservist",
    "ict",
    "rsi", "rso",
    "officer cadet",
    "in-camp", "ns training",
    "operationally ready",
    "ns deferment", "ns disruption",
    "defend singapore", "serve nation",
    "tekkan", "arrowed", "siao on",
    "off day", "leave pass",
}

_strong_pattern = re.compile(
    '|'.join(re.escape(t) for t in sorted(_STRONG_TERMS, key=len, reverse=True)),
    re.IGNORECASE,
)
_weak_pattern = re.compile(
    '|'.join(r'\b' + re.escape(t) + r'\b' for t in sorted(_WEAK_TERMS, key=len, reverse=True)),
    re.IGNORECASE,
)

_NS_SUBREDDIT = "nationalservicesg"


def _is_ns_chunk(text: str) -> bool:
    if _strong_pattern.search(text):
        return True
    matches = {m.lower() for m in _weak_pattern.findall(text)}
    return len(matches) >= 2


# ── NLP / model helpers ───────────────────────────────────────────────────────

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.blank("en")
        _nlp.add_pipe("sentencizer")
    return _nlp


def _get_model(device: str = "cpu"):
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {EMBEDDING_MODEL}  (device={device})")
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _model


# ── Core chunking logic ───────────────────────────────────────────────────────

def _split_sentences(texts: list[str]) -> list[list[str]]:
    """Batch sentence-split a list of texts using spaCy pipe."""
    nlp = _get_nlp()
    result = []
    for doc in nlp.pipe(texts, batch_size=512):
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        result.append(sents if sents else [""])
    return result


def _group_into_chunks(sentences: list[str], sent_embeddings: np.ndarray) -> list[str]:
    """
    Group sentences into semantically coherent chunks.

    A boundary fires when cosine similarity between adjacent sentences drops below
    SIMILARITY_THRESHOLD AND the current group has at least CHUNK_MIN_SENTENCES.
    CHUNK_MAX_SENTENCES is a hard cap regardless of similarity.

    Trailing fragments under CHUNK_MIN_CHARS are merged into the previous chunk.
    If the whole document produces only one chunk under CHUNK_MIN_CHARS, it is
    kept as-is — no document is discarded for being short.
    """
    if len(sentences) == 1:
        return [sentences[0]]

    groups: list[list[str]] = []
    current: list[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            sent_embeddings[i - 1].reshape(1, -1),
            sent_embeddings[i].reshape(1, -1),
        )[0][0]
        over_max       = len(current) >= CHUNK_MAX_SENTENCES
        semantic_break = sim < SIMILARITY_THRESHOLD and len(current) >= CHUNK_MIN_SENTENCES

        if over_max or semantic_break:
            groups.append(current)
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    # Merge trailing under-size fragment into previous chunk
    if len(current) < CHUNK_MIN_SENTENCES and groups:
        groups[-1].extend(current)
    else:
        groups.append(current)

    return [" ".join(g).strip() for g in groups if " ".join(g).strip()]


# ── DataFrame chunker ─────────────────────────────────────────────────────────

def _process_doc_batch(
    batch_rows: list,
    batch_sentences: list[list[str]],
    batch_sent_embs: list[np.ndarray],
    title_map: dict[str, str] | None,
    metadata_cols: list[str],
    doc_type: str,
) -> tuple[list[dict], int]:
    """Group + NS filter one batch of documents. Returns (records, n_filtered)."""
    records: list[dict] = []
    n_filtered = 0

    for row, sents, sent_embs in zip(batch_rows, batch_sentences, batch_sent_embs):
        subreddit = str(getattr(row, "subreddit", "")).lower()
        doc_id    = str(row.id)
        is_ns_sr  = subreddit == _NS_SUBREDDIT

        chunk_texts = _group_into_chunks(sents, sent_embs)

        prefix = ""
        if title_map is not None and hasattr(row, "post_id"):
            title  = title_map.get(str(row.post_id), "")
            prefix = f"Post: {title}\n\nComment: " if title else ""

        meta = {c: getattr(row, c) for c in metadata_cols if hasattr(row, c)}
        kept = 0
        for chunk_text in chunk_texts:
            if not is_ns_sr and not _is_ns_chunk(chunk_text):
                n_filtered += 1
                continue
            records.append({
                "doc_id":     doc_id,
                "doc_type":   doc_type,
                "text":       chunk_text,
                "_embed":     f"{prefix}{chunk_text}" if prefix else chunk_text,
                "_chunk_seq": kept,
                **meta,
            })
            kept += 1

        if kept > 0:
            for r in records[-kept:]:
                r["_total"] = kept

    return records, n_filtered


def _chunk_dataframe(
    df: pd.DataFrame,
    text_col: str,
    doc_type: str,
    metadata_cols: list[str],
    title_map: dict[str, str] | None = None,
    device: str = "cpu",
    doc_batch_size: int = 5000,
) -> pd.DataFrame:
    """
    Four-phase chunker with document-level batching to avoid GPU OOM.

    Phase 1 — Sentence split all documents (spaCy batch, CPU).
    Phase 2 — Embed sentences in doc-batches (GPU, batch_size=128).
    Phase 3 — Group into chunks + NS filter per batch.
    Phase 4 — Batch embed all surviving chunks with context prefix.
    """
    import torch

    model = _get_model(device)
    texts = df[text_col].fillna("").tolist()
    rows  = list(df.itertuples(index=False))

    # ── Phase 1: sentence splitting ───────────────────────────────────────────
    log.info(f"  Phase 1: splitting {len(texts):,} documents into sentences …")
    all_sentences = _split_sentences(texts)
    log.info(f"  Total sentences: {sum(len(s) for s in all_sentences):,}")

    # ── Phases 2 + 3: doc-batched sentence embedding + grouping ──────────────
    log.info(f"  Phases 2+3: embedding sentences + grouping (doc batches of {doc_batch_size:,}) …")
    all_records: list[dict] = []
    total_filtered = 0
    n_batches = (len(rows) + doc_batch_size - 1) // doc_batch_size

    for b in range(n_batches):
        lo = b * doc_batch_size
        hi = min(lo + doc_batch_size, len(rows))
        batch_rows  = rows[lo:hi]
        batch_sents = all_sentences[lo:hi]

        flat_sents = [s for sents in batch_sents for s in sents]
        flat_embs  = model.encode(
            flat_sents, batch_size=128, show_progress_bar=False, convert_to_numpy=True,
        )
        if device != "cpu":
            torch.cuda.empty_cache()

        batch_embs: list[np.ndarray] = []
        idx = 0
        for sents in batch_sents:
            n = len(sents)
            batch_embs.append(flat_embs[idx: idx + n])
            idx += n
        del flat_embs

        batch_records, n_filt = _process_doc_batch(
            batch_rows, batch_sents, batch_embs, title_map, metadata_cols, doc_type,
        )
        all_records.extend(batch_records)
        total_filtered += n_filt
        log.info(f"    Batch {b + 1}/{n_batches}  docs {lo:,}–{hi:,}  chunks so far: {len(all_records):,}")

    log.info(f"  NS filter removed {total_filtered:,} off-topic chunks")
    log.info(f"  Surviving chunks: {len(all_records):,}")

    if not all_records:
        return pd.DataFrame()

    for r in all_records:
        total       = r.pop("_total", 1)
        seq         = r.pop("_chunk_seq")
        r["chunk_idx"]   = seq
        r["chunk_count"] = total
        r["chunk_id"]    = f"{r['doc_id']}_{seq}"

    # ── Phase 4: batch embed chunks with context prefix ───────────────────────
    log.info("  Phase 4: batch embedding chunks with context prefix …")
    embed_texts = [r.pop("_embed") for r in all_records]
    chunk_embeddings = model.encode(
        embed_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True,
    )
    for r, emb in zip(all_records, chunk_embeddings):
        r["embedding"] = emb.tolist()

    return pd.DataFrame(all_records)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_chunking(device: str = "cpu"):
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # ── Submissions ───────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("CHUNKING SUBMISSIONS")
    log.info("=" * 60)
    subs = pd.read_parquet(SUBMISSIONS_CLEAN)
    title_map = dict(zip(subs["id"].astype(str), subs["title"].astype(str)))
    log.info(f"Built title map: {len(title_map):,} submissions")

    metadata_cols = [
        "subreddit", "created_utc", "score", "log_weight",
        "upvote_ratio", "num_comments", "text_source", "author", "permalink",
    ]
    sub_chunks = _chunk_dataframe(subs, "combined_text", "submission", metadata_cols, device=device)
    sub_chunks.to_parquet(SUBMISSIONS_CHUNKS, index=False)
    log.info(f"Submissions: {len(subs):,} docs → {len(sub_chunks):,} chunks → {SUBMISSIONS_CHUNKS.name}")
    del subs, sub_chunks

    # ── Comments ─────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("CHUNKING COMMENTS")
    log.info("=" * 60)
    coms = pd.read_parquet(COMMENTS_CLEAN)
    metadata_cols = [
        "subreddit", "created_utc", "score", "log_weight",
        "author", "post_id", "permalink", "parent_id", "depth",
    ]
    com_chunks = _chunk_dataframe(
        coms, "body", "comment", metadata_cols, title_map=title_map, device=device,
    )
    com_chunks.to_parquet(COMMENTS_CHUNKS, index=False)
    log.info(f"Comments: {len(coms):,} docs → {len(com_chunks):,} chunks → {COMMENTS_CHUNKS.name}")
    del coms, com_chunks

    log.info("Chunking complete.")


if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s  %(message)s")
    run_chunking()
