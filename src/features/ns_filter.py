"""
Stage 4b Step 1 — Stricter NS corpus filter for BERTopic rerun.

The first BERTopic run produced 1,255 fine topics and a 57.8% outlier rate.
Root cause: r/singapore and r/askSingapore chunks that passed the Stage 2
keyword filter are not all NS-relevant at the chunk level. A post only needs
one NS keyword to survive Stage 2, but its chunks about housing, food, or
relationships dilute the topic model.

This script re-filters at the chunk level, keeping:
- All r/NationalServiceSG chunks (entire subreddit is NS-contextual)
- r/singapore / r/askSingapore chunks that meet a stricter criterion:
    * ≥ 2 distinct NS keyword matches in chunk text, OR
    * strong-term match (enlistment, reservist, IPPT, ROD, ORD, etc.), OR
    * post-level engagement signal (log_weight > 1 AND num_comments > 10)
      — these are discussions that the community engaged with as NS topics

Output: data/processed/chunks_filtered.parquet
  Same schema as chunk_topics.parquet plus 'subreddit' column for inspection.
  Only chunk_id is strictly needed downstream; all other columns carried through
  so Stage 4b can load embeddings directly without a second merge.
"""

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    CHUNK_TOPICS,
    COMMENTS_CHUNKS,
    DATA_PROCESSED,
    SUBMISSIONS_CHUNKS,
)

log = logging.getLogger(__name__)

OUT_PATH = DATA_PROCESSED / "chunks_filtered.parquet"

# ── Keyword tiers ─────────────────────────────────────────────────────────────

# Tier 1: strong NS terms — a single match is sufficient evidence the chunk is
# NS-relevant regardless of surrounding context.
_STRONG_TERMS = {
    # Core milestones
    "enlistment", "enlist", "enlisted", "enlistee",
    "reservist", "reservists",
    "ord", "rod",                         # ORD = operationally ready date
    "ippt",                               # individual physical proficiency test
    "ict",                                # in-camp training
    # Service branches
    "bmt", "bmtc", "tekong",
    "ocs", "scs", "saf"
    "scdf", "spf", "ndu",
    # Policy / identity
    "conscription", "conscript", "nsman", "nsmen", "ns man",
    "national service",
    "mindef",
    "pes status", "downpes", "medical board",
    # Camp life strong signals
    "guard duty", "book in", "book out", "confined to camp",
    "route march", "outfield",
    "chao keng", "sign extra",
    "emart", "cookhouse"
}

# Tier 2: weaker NS terms — ambiguous on their own (e.g. "ns" can mean "not sure",
# "db" can mean decibels). Require ≥ 2 distinct matches for confidence.
_WEAK_TERMS = {
    "ns", "nsf",
    "saf",
    "bmt",       # also in strong but harmless to double-list
    "pes",
    "db",        # detention barracks — ambiguous
    "recruit",
    "sergeant", "encik",
    "vocation",
    "wayang",
    "ippt",      # also strong — safe double
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
    '|'.join(
        [re.escape(t) for t in sorted(_STRONG_TERMS, key=len, reverse=True)]
    ),
    re.IGNORECASE,
)

_weak_pattern = re.compile(
    '|'.join(
        [r'\b' + re.escape(t) + r'\b'
         for t in sorted(_WEAK_TERMS, key=len, reverse=True)]
    ),
    re.IGNORECASE,
)

# Engagement threshold for post-level signal (r/singapore / r/askSingapore only)
_LOG_WEIGHT_MIN  = 1.0
_NUM_COMMENTS_MIN = 10


def _is_ns_relevant(text: str) -> bool:
    """True if chunk text clears the strict NS relevance bar."""
    if _strong_pattern.search(text):
        return True
    weak_matches = _weak_pattern.findall(text)
    distinct = {m.lower() for m in weak_matches}
    return len(distinct) >= 2


def _load_chunks() -> pd.DataFrame:
    """
    Load both chunk parquets. Topic columns (topic_id_fine etc.) were written
    back into the chunk files after Stage 4, so no separate merge is needed.

    Submissions schema: doc_id, subreddit, created_utc, score, log_weight,
      upvote_ratio, num_comments, text_source, author, permalink, text,
      doc_type, chunk_idx, chunk_count, chunk_id, embedding, topic_id_fine,
      topic_prob_fine, topic_id_coarse
    Comments schema: same minus num_comments/upvote_ratio/text_source, plus
      post_id, parent_id, depth
    """
    sub_cols = [
        "chunk_id", "doc_id", "subreddit", "created_utc", "score",
        "log_weight", "num_comments", "text", "embedding", "doc_type",
        "chunk_idx", "chunk_count", "topic_id_fine", "topic_prob_fine",
        "topic_id_coarse",
    ]
    com_cols = [
        "chunk_id", "doc_id", "subreddit", "created_utc", "score",
        "log_weight", "text", "embedding", "doc_type", "depth", "post_id",
        "chunk_idx", "chunk_count", "topic_id_fine", "topic_prob_fine",
        "topic_id_coarse",
    ]

    sub = pd.read_parquet(SUBMISSIONS_CHUNKS, columns=sub_cols)
    com = pd.read_parquet(COMMENTS_CHUNKS,   columns=com_cols)

    log.info(f"  Submissions: {len(sub):,} chunks")
    log.info(f"  Comments:    {len(com):,} chunks")

    combined = pd.concat([sub, com], ignore_index=True)
    log.info(f"  Total:       {len(combined):,} chunks")
    return combined


def _build_post_engagement(combined: pd.DataFrame) -> pd.Series:
    """
    For non-NationalServiceSG chunks, mark posts that meet the engagement
    threshold (log_weight > 1 AND num_comments > 10).

    num_comments only lives on submission chunks. For comment chunks we join
    back via doc_id → post_id chain, but this is expensive. Simpler: build
    a set of doc_ids for high-engagement submissions, then mark comment chunks
    whose post_id is in that set.
    """
    # num_comments is only on submission rows; NaN for comment rows after concat
    num_comments = combined["num_comments"].fillna(0) if "num_comments" in combined.columns else pd.Series(0, index=combined.index)
    mask_sub_engaged = (
        (combined["doc_type"] == "submission")
        & (combined["log_weight"].fillna(0) > _LOG_WEIGHT_MIN)
        & (num_comments > _NUM_COMMENTS_MIN)
    )
    engaged_doc_ids = set(combined.loc[mask_sub_engaged, "doc_id"].unique())

    # post_id is only on comment rows; NaN for submission rows after concat
    post_id_col = combined["post_id"] if "post_id" in combined.columns else pd.Series(dtype=str, index=combined.index)
    is_engaged = (
        combined["doc_id"].isin(engaged_doc_ids)          # submission itself
        | post_id_col.isin(engaged_doc_ids)               # comment on engaged post
    )
    return is_engaged


def run_ns_filter() -> None:
    log.info("Loading chunks …")
    combined = _load_chunks()

    ns_sr = combined["subreddit"] == "NationalServiceSG"

    # For the two general subreddits apply strict chunk-level filter
    general_mask = ~ns_sr

    log.info("Applying chunk-level NS relevance filter to r/singapore + r/askSingapore …")
    text_relevant = combined["text"].apply(_is_ns_relevant)

    log.info("Building post-level engagement signal …")
    post_engaged = _build_post_engagement(combined)

    keep = ns_sr | (general_mask & (text_relevant | post_engaged))

    n_total  = len(combined)
    n_keep   = keep.sum()
    n_drop   = n_total - n_keep

    log.info(f"  Total chunks:  {n_total:,}")
    log.info(f"  Kept:          {n_keep:,}  ({n_keep / n_total * 100:.1f}%)")
    log.info(f"  Dropped:       {n_drop:,}  ({n_drop / n_total * 100:.1f}%)")

    by_sr = combined.groupby("subreddit").size().rename("total")
    kept_sr = combined[keep].groupby("subreddit").size().rename("kept")
    report = pd.concat([by_sr, kept_sr], axis=1)
    report["kept_%"] = (report["kept"] / report["total"] * 100).round(1)
    log.info(f"\n{report.to_string()}")

    filtered = combined[keep].reset_index(drop=True)
    filtered.to_parquet(OUT_PATH, index=False)
    log.info(f"\nSaved → {OUT_PATH.name}  ({len(filtered):,} rows)")


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run_ns_filter()
