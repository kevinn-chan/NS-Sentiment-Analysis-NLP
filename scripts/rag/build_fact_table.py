"""
RAG Phase 2 — Build rag_fact_table.parquet.

Sources:
  • temporal_sentiment.parquet     — pre-aggregated sentiment + commitment scores (Stage 8)
  • chunk_commitment_cascade.parquet   — chunk-level buyin/stance labels from Stage 5b LLM
                                     annotation (OPTIONAL — emits NaN columns if absent)
  • chunk_metadata.parquet         — chunk dimensions (year, month, subreddit, topic_macro)

New vs previous version:
  • Adds pct_committed / pct_critical / pct_neutral columns derived from chunk-level argmax
    classification (committed = max(support, critical, neutral) is support; etc.)
  • Adds upvote-weighted commitment columns from Stage 8 temporal output:
      wtd_buyin_committed, wtd_buyin_uncommitted, net_buyin
      wtd_stance_supportive, wtd_stance_critical, net_stance
      wtd_positive, wtd_negative, net_disposition
  • All other aggregation logic unchanged (6 granularity levels)
  • chunk_commitment_cascade.parquet is optional — build runs fine if absent (NaN columns + warning)

Run:
    python -m scripts.rag.build_fact_table
"""

import logging
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.rag.config import DATA_NEW, FACT_TABLE

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

TEMPORAL_SENTIMENT   = DATA_NEW / "temporal_sentiment.parquet"
CHUNK_COMMITMENT_LLM = DATA_NEW / "chunk_commitment_cascade.parquet"  # Stage 5b cascade labels (optional)
CHUNK_METADATA       = DATA_NEW / "chunk_metadata.parquet"

# Numeric columns carried from temporal_sentiment
AGG_COLS = [
    "doc_count",
    "mean_sent_neg", "mean_sent_neu", "mean_sent_pos",
    "wtd_sent_neg", "wtd_sent_neu", "wtd_sent_pos",
    "pct_neg", "pct_neu", "pct_pos",
    # Stage 5b SingBERT 4-stage cascade (buyin F1=0.714, stance F1=0.784)
    # Explicit labels (argmax ≥ 0.20 + lexicon): ~3% of corpus
    "pct_uncommitted", "pct_committed", "pct_critical", "pct_supportive",
    # Broad labels (explicit + sent_neg>0.70 neutrals → uncommitted/critical): ~25%/24%
    "pct_broad_uncommitted", "pct_broad_critical",
    # Upvote-weighted rates (Σ upvote_weight × label) / Σ upvote_weight
    "wtd_uncommitted", "wtd_committed", "wtd_critical", "wtd_supportive",
    "wtd_broad_uncommitted", "wtd_broad_critical",
    # Combined disposition (positive − negative, upvote-weighted)
    "net_disposition",
]


def weighted_mean(df: pd.DataFrame, cols: list[str], weight_col: str = "doc_count") -> dict:
    """Weighted mean of `cols` by `weight_col`."""
    w = df[weight_col].fillna(0)
    total = w.sum()
    result = {}
    for c in cols:
        if c == weight_col:
            result[c] = total
        elif c in df.columns:
            result[c] = float((df[c].fillna(0) * w).sum() / total) if total > 0 else float("nan")
    return result


def make_agg_rows(base: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate base by group_cols, summing doc_count and weighting everything else."""
    rows = []
    for keys, grp in base.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        agg = weighted_mean(grp, AGG_COLS)
        row.update(agg)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    # Fill missing dimension columns with "all"
    for dim in ["subreddit", "topic_macro"]:
        if dim not in result.columns:
            result[dim] = "all"
    return result


def build_commitment_pct(year_filter: int = 2018) -> pd.DataFrame | None:
    """
    Load Stage 5b SingBERT chunk labels (chunk_commitment_cascade.parquet) and aggregate
    explicit + broad buyin/stance rates to (year, month, subreddit, topic_macro).

    These columns are already baked into temporal_sentiment by Stage 8, so this
    function now serves as a verification join. Returns None if file is absent.
    """
    if not CHUNK_COMMITMENT_LLM.exists():
        log.warning(
            "  chunk_commitment_cascade.parquet not found — Stage 5b commitment columns "
            "will be sourced from temporal_sentiment only (baked in by Stage 8)."
        )
        return None

    log.info("Loading chunk_commitment_cascade and chunk_metadata …")
    cc = pd.read_parquet(CHUNK_COMMITMENT_LLM)
    meta = pd.read_parquet(CHUNK_METADATA,
                           columns=["chunk_id", "year", "month", "subreddit", "topic_macro"])

    log.info(f"  chunk_commitment_cascade: {len(cc):,} rows")
    log.info(f"  chunk_metadata:       {len(meta):,} rows")

    required = {"buyin_label", "stance_label"}
    missing = required - set(cc.columns)
    if missing:
        log.warning(f"  cascade parquet missing columns: {missing} — skipping chunk-level join.")
        return None

    df = cc[["chunk_id","buyin_label","stance_label"]].merge(meta, on="chunk_id", how="inner")
    df = df[df["year"] >= year_filter].copy()
    log.info(f"  After {year_filter}+ filter: {len(df):,} rows")

    df["_unc"]  = (df["buyin_label"]  == "uncommitted").astype(float)
    df["_com"]  = (df["buyin_label"]  == "committed").astype(float)
    df["_crit"] = (df["stance_label"] == "critical").astype(float)
    df["_sup"]  = (df["stance_label"] == "supportive").astype(float)
    df["_b_unc"]  = df["_unc"]
    df["_b_crit"] = df["_crit"]

    grp = df.groupby(["year","month","subreddit","topic_macro"], observed=True)
    pct = grp.agg(
        n_chunks=("chunk_id","count"),
        pct_uncommitted=("_unc","mean"),
        pct_committed=("_com","mean"),
        pct_critical=("_crit","mean"),
        pct_supportive=("_sup","mean"),
        pct_broad_uncommitted=("_b_unc","mean"),
        pct_broad_critical=("_b_crit","mean"),
    ).reset_index()

    log.info(f"  Commitment pct table (chunk-level): {len(pct):,} rows")
    return pct


def main():
    if not TEMPORAL_SENTIMENT.exists():
        log.error(f"temporal_sentiment.parquet not found at {TEMPORAL_SENTIMENT}")
        sys.exit(1)
    if not CHUNK_METADATA.exists():
        log.error(f"chunk_metadata.parquet not found at {CHUNK_METADATA}")
        sys.exit(1)

    # chunk_commitment_cascade.parquet is OPTIONAL — Stage 5b LLM labels.
    # If absent, the upvote-weighted commitment columns (wtd_buyin_*, wtd_stance_*,
    # net_buyin, net_stance, net_disposition) will be NaN in the fact table.
    # Run Stage 5b (commitment_llm_annotator.py) then re-run Stage 8 to populate them.
    if not CHUNK_COMMITMENT_LLM.exists():
        log.warning(
            f"chunk_commitment_cascade.parquet not found at {CHUNK_COMMITMENT_LLM}. "
            "Upvote-weighted commitment columns (wtd_buyin_*, wtd_stance_*, net_buyin, "
            "net_stance, wtd_positive, wtd_negative, net_disposition) will be NaN. "
            "Run Stage 5b and re-run Stage 8 to populate these columns."
        )

    # ── 1. Load and parse temporal_sentiment ─────────────────────────────────
    log.info("Loading temporal_sentiment …")
    ts = pd.read_parquet(TEMPORAL_SENTIMENT)
    log.info(f"  {len(ts):,} rows, {len(ts.columns)} cols")

    ts["year"]  = ts["year_month"].str[:4].astype(int)
    ts["month"] = ts["year_month"].str[5:7].astype(int)
    ts = ts[ts["year"] >= 2018].copy()
    log.info(f"  After 2018+ filter: {len(ts):,} rows")
    log.info(f"  Subreddits: {sorted(ts['subreddit'].unique().tolist())}")
    log.info(f"  Topic macros: {ts['topic_macro'].nunique()}")

    # ── 2. Build commitment class percentages from chunk level (optional) ────
    commit_pct = build_commitment_pct(year_filter=2018)

    # ── 3. Optional chunk-level join (Stage 8 already baked these in) ────────
    # Stage 8 writes pct_uncommitted, pct_committed, pct_broad_* etc. into
    # temporal_sentiment — build_commitment_pct is a verification cross-check.
    # If it returns data, log a delta summary. No structural join needed.
    if commit_pct is not None:
        _new_cols = ["pct_uncommitted","pct_committed","pct_critical","pct_supportive",
                     "pct_broad_uncommitted","pct_broad_critical"]
        _present  = [c for c in _new_cols if c in ts.columns]
        log.info(f"  Stage 5b columns already in temporal_sentiment: {_present}")
        log.info(f"  Chunk-level cross-check rows: {len(commit_pct):,} (not joined — baked by Stage 8)")
    else:
        log.info("  No chunk-level cross-check — Stage 8 baked commitment columns are used directly.")

    # ── 4. Base granularity: per (year, month, subreddit, topic_macro) ───────
    base_cols = ["year", "month", "subreddit", "topic_macro"] + [
        c for c in AGG_COLS if c in ts.columns
    ]
    base = ts[base_cols].copy()
    base["granularity"] = "month_sub_topic"
    log.info(f"  Base rows: {len(base):,}")

    # ── 5. Aggregate at 5 higher granularities ────────────────────────────────
    log.info("Aggregating …")

    month_sub = make_agg_rows(ts, ["year", "month", "subreddit"])
    month_sub["topic_macro"] = "all"
    month_sub["granularity"] = "month_sub"
    log.info(f"  month_sub rows:   {len(month_sub):,}")

    month_all = make_agg_rows(ts, ["year", "month"])
    month_all["subreddit"]   = "all"
    month_all["topic_macro"] = "all"
    month_all["granularity"] = "month"
    log.info(f"  month_all rows:   {len(month_all):,}")

    year_topic = make_agg_rows(ts, ["year", "topic_macro"])
    year_topic["subreddit"]  = "all"
    year_topic["month"]      = 0
    year_topic["granularity"]= "year_topic"
    log.info(f"  year_topic rows:  {len(year_topic):,}")

    year_sub = make_agg_rows(ts, ["year", "subreddit"])
    year_sub["topic_macro"]  = "all"
    year_sub["month"]        = 0
    year_sub["granularity"]  = "year_sub"
    log.info(f"  year_sub rows:    {len(year_sub):,}")

    year_all = make_agg_rows(ts, ["year"])
    year_all["subreddit"]    = "all"
    year_all["topic_macro"]  = "all"
    year_all["month"]        = 0
    year_all["granularity"]  = "year"
    log.info(f"  year_all rows:    {len(year_all):,}")

    # ── 6. Combine ────────────────────────────────────────────────────────────
    fact = pd.concat(
        [base, month_sub, month_all, year_topic, year_sub, year_all],
        ignore_index=True,
    )

    fact["year"]  = fact["year"].astype(int)
    fact["month"] = fact["month"].fillna(0).astype(int)
    fact["doc_count"] = fact["doc_count"].fillna(0).astype(int)
    for c in AGG_COLS:
        if c in fact.columns and c != "doc_count":
            fact[c] = fact[c].astype(float)

    log.info(f"\nFact table: {len(fact):,} rows × {len(fact.columns)} cols")
    log.info(f"  Granularities: {fact['granularity'].value_counts().to_dict()}")
    log.info(f"  Columns: {fact.columns.tolist()}")

    # Spot-check Stage 5b SingBERT commitment columns on a well-populated row
    sample_cols = [
        "year", "doc_count",
        "pct_uncommitted", "pct_committed", "pct_critical", "pct_supportive",
        "pct_broad_uncommitted", "pct_broad_critical",
        "wtd_uncommitted", "wtd_committed", "wtd_critical", "wtd_supportive",
        "net_disposition",
    ]
    sample = fact[
        (fact["granularity"] == "year") & (fact["year"] == 2023)
    ][[c for c in sample_cols if c in fact.columns]]
    log.info(f"\n  Sample (year=2023, all):\n{sample.to_string(index=False)}")

    FACT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    fact.to_parquet(FACT_TABLE, index=False)
    size_mb = FACT_TABLE.stat().st_size / 1e6
    log.info(f"\nSaved → {FACT_TABLE}  ({size_mb:.1f} MB)")
    log.info("Fact table build complete ✓")


if __name__ == "__main__":
    main()
