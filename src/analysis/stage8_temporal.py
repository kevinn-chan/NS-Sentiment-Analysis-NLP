"""
Stage 8 — Temporal Aggregation
================================
Rolls up doc-level sentiment + commitment scores into monthly time series.

Inputs:
    doc_sentiment.parquet           — one row per document (Stage 6 output)
    chunk_commitment_cascade.parquet    — chunk-level buyin/stance labels (Stage 5b, OPTIONAL)
    chunk_metadata.parquet          — chunk-level upvote scores (for weighting)

Outputs:
    temporal_sentiment.parquet          — monthly × subreddit × topic_macro
    temporal_sentiment_overall.parquet  — monthly × subreddit (no topic breakdown)

Key decisions:
    - Monthly granularity (year_month as string "YYYY-MM")
    - Simple mean aggregation (doc-level scores are already quality-controlled)
    - low_volume flag where doc_count < 30 in a (month, subreddit) bucket
    - topic_macro=None docs included in overall rollup, excluded from topic rollup
    - Data computed from 2010 onward; pre-2019 will mostly be low_volume
    - Commitment columns (chunk-level) joined optionally; skipped gracefully if absent

Upvote weighting formula (Stage 7 proven):
    wtd_committed = Σ(log(1+score_i) × is_committed_i) / Σ(log(1+score_i))
    where score = Reddit upvote count clamped to 0 if negative.

New temporal columns added when chunk_commitment_cascade.parquet is available:
    Buyin axis:
        pct_buyin_committed, pct_buyin_uncommitted, pct_buyin_neutral
        wtd_buyin_committed, wtd_buyin_uncommitted
        net_buyin
    Stance axis:
        pct_stance_supportive, pct_stance_critical, pct_stance_neutral
        wtd_stance_supportive, wtd_stance_critical
        net_stance
    Combined:
        pct_positive, pct_negative
        wtd_positive, wtd_negative
        net_disposition

Run:
    python -m src.analysis.stage8_temporal
"""

import time
import warnings
import numpy as np
import pandas as pd

DATA = 'data/processed/new'
OUT_TOPIC   = 'data/processed/new/temporal_sentiment.parquet'
OUT_OVERALL = 'data/processed/new/temporal_sentiment_overall.parquet'

# Chunk-level commitment labels from Stage 5b cascade inference
CHUNK_COMMITMENT_LLM = 'data/processed/new/chunk_commitment_cascade.parquet'
CHUNK_METADATA       = 'data/processed/new/chunk_metadata.parquet'

LOW_VOLUME_THRESHOLD = 30   # monthly subreddit buckets below this flagged


# ── Upvote weight helper ──────────────────────────────────────────────────────

def log1p_weight(score_series: pd.Series) -> pd.Series:
    """log(1 + max(score, 0)) — matches Stage 7 divergence formula."""
    return np.log1p(score_series.clip(lower=0).fillna(0))


# ── Commitment aggregation helpers ────────────────────────────────────────────

def _wtd_pct(df: pd.DataFrame, flag_col: str, weight_col: str) -> pd.Series:
    """
    Σ(weight × flag) / Σ(weight) per group — already grouped externally.
    Returns a scalar (used inside apply or agg lambdas).
    """
    w = df[weight_col]
    total_w = w.sum()
    if total_w == 0:
        return 0.0
    return float((df[flag_col] * w).sum() / total_w)


def rollup_commitment(
    commit_df: pd.DataFrame,
    group_cols: list,
) -> pd.DataFrame:
    """
    Aggregate chunk-level commitment labels into upvote-weighted monthly stats.

    Expects commit_df to have columns:
        chunk_id, buyin_label, stance_label, year_month, subreddit,
        topic_macro (optional), log_w (log(1+score))

    Returns a DataFrame indexed by group_cols with new commitment columns.
    """
    df = commit_df.copy()

    # Boolean flags
    df['is_buyin_committed']   = (df['buyin_label'] == 'committed').astype(float)
    df['is_buyin_uncommitted'] = (df['buyin_label'] == 'uncommitted').astype(float)
    df['is_buyin_neutral']     = (df['buyin_label'] == 'neutral').astype(float)

    df['is_stance_supportive'] = (df['stance_label'] == 'supportive').astype(float)
    df['is_stance_critical']   = (df['stance_label'] == 'critical').astype(float)
    df['is_stance_neutral']    = (df['stance_label'] == 'neutral').astype(float)

    # Combined positive/negative flags
    df['is_positive'] = (
        (df['buyin_label'] == 'committed') | (df['stance_label'] == 'supportive')
    ).astype(float)
    df['is_negative'] = (
        (df['buyin_label'] == 'uncommitted') | (df['stance_label'] == 'critical')
    ).astype(float)

    # Validate group_cols are present
    available_group_cols = [c for c in group_cols if c in df.columns]
    if len(available_group_cols) < len(group_cols):
        missing = set(group_cols) - set(available_group_cols)
        warnings.warn(f"Commitment rollup: missing group columns {missing} — skipping")
        return pd.DataFrame()

    grp = df.groupby(available_group_cols, observed=True)

    flag_cols = [
        'is_buyin_committed', 'is_buyin_uncommitted', 'is_buyin_neutral',
        'is_stance_supportive', 'is_stance_critical', 'is_stance_neutral',
        'is_positive', 'is_negative',
    ]

    # Raw percentage (unweighted)
    raw_pct = grp[flag_cols].mean().rename(columns={
        'is_buyin_committed':   'pct_buyin_committed',
        'is_buyin_uncommitted': 'pct_buyin_uncommitted',
        'is_buyin_neutral':     'pct_buyin_neutral',
        'is_stance_supportive': 'pct_stance_supportive',
        'is_stance_critical':   'pct_stance_critical',
        'is_stance_neutral':    'pct_stance_neutral',
        'is_positive':          'pct_positive',
        'is_negative':          'pct_negative',
    })

    # Upvote-weighted percentage: Σ(log_w × flag) / Σ(log_w)
    def wtd_mean(sub_df: pd.DataFrame, flag: str) -> float:
        w = sub_df['log_w']
        total = w.sum()
        return float((sub_df[flag] * w).sum() / total) if total > 0 else float('nan')

    wtd_rows = []
    for keys, sub in grp:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(available_group_cols, keys))
        row['wtd_buyin_committed']   = wtd_mean(sub, 'is_buyin_committed')
        row['wtd_buyin_uncommitted'] = wtd_mean(sub, 'is_buyin_uncommitted')
        row['wtd_stance_supportive'] = wtd_mean(sub, 'is_stance_supportive')
        row['wtd_stance_critical']   = wtd_mean(sub, 'is_stance_critical')
        row['wtd_positive']          = wtd_mean(sub, 'is_positive')
        row['wtd_negative']          = wtd_mean(sub, 'is_negative')
        wtd_rows.append(row)

    wtd_df = pd.DataFrame(wtd_rows).set_index(available_group_cols)

    result = raw_pct.join(wtd_df, how='left')

    # Net metrics
    result['net_buyin']        = result['pct_buyin_committed']   - result['pct_buyin_uncommitted']
    result['net_stance']       = result['pct_stance_supportive'] - result['pct_stance_critical']
    result['net_disposition']  = result['wtd_positive']          - result['wtd_negative']

    return result.reset_index()


# ── Main sentiment rollup ─────────────────────────────────────────────────────

def load():
    print("Loading doc_sentiment.parquet...")
    doc = pd.read_parquet(f'{DATA}/doc_sentiment.parquet')
    print(f"  {len(doc):,} docs")

    # Extract year_month as string (timezone-safe)
    doc['year_month'] = doc['created_utc'].dt.strftime('%Y-%m')

    print(f"  Date range: {doc['year_month'].min()} → {doc['year_month'].max()}")
    return doc


def load_commitment_chunks() -> pd.DataFrame | None:
    """
    Load chunk-level commitment labels + upvote scores.

    Returns merged DataFrame with columns:
        chunk_id, buyin_label, stance_label, year_month, subreddit,
        topic_macro, log_w
    Returns None if chunk_commitment_cascade.parquet does not exist.
    """
    import os
    if not os.path.exists(CHUNK_COMMITMENT_LLM):
        warnings.warn(
            f"chunk_commitment_cascade.parquet not found at {CHUNK_COMMITMENT_LLM}. "
            "Skipping upvote-weighted commitment columns in Stage 8 output. "
            "Run Stage 5b (commitment_llm_annotator.py) to generate this file."
        )
        return None

    print(f"Loading chunk_commitment_cascade.parquet...")
    commit = pd.read_parquet(CHUNK_COMMITMENT_LLM)
    print(f"  {len(commit):,} chunk commitment labels")

    # Validate required columns
    required = {'chunk_id', 'buyin_label', 'stance_label'}
    missing = required - set(commit.columns)
    if missing:
        warnings.warn(
            f"chunk_commitment_cascade.parquet missing columns: {missing}. "
            "Skipping commitment columns."
        )
        return None

    # Load metadata for upvote scores + temporal/spatial dims
    if not os.path.exists(CHUNK_METADATA):
        warnings.warn(
            f"chunk_metadata.parquet not found at {CHUNK_METADATA}. "
            "Skipping commitment columns."
        )
        return None

    print(f"Loading chunk_metadata.parquet for upvote scores...")
    meta_cols = ['chunk_id', 'year', 'month', 'subreddit', 'topic_macro', 'score', 'upvotes']
    meta = pd.read_parquet(
        CHUNK_METADATA,
        columns=[c for c in meta_cols if c in pd.read_parquet(CHUNK_METADATA, columns=[]).columns] or meta_cols
    )
    # Re-read with correct columns (avoid column existence check overhead)
    meta = pd.read_parquet(CHUNK_METADATA)
    # Use 'score' if available (Reddit upvote count), fall back to 'upvotes'
    if 'score' in meta.columns:
        meta['_score'] = meta['score']
    elif 'upvotes' in meta.columns:
        meta['_score'] = meta['upvotes']
    else:
        meta['_score'] = 0
        warnings.warn("chunk_metadata has no 'score' or 'upvotes' column — weights will be uniform.")

    # Build year_month from year + month if present
    if 'year' in meta.columns and 'month' in meta.columns:
        meta['year_month'] = (
            meta['year'].astype(str) + '-' +
            meta['month'].astype(str).str.zfill(2)
        )
    else:
        warnings.warn("chunk_metadata has no year/month columns. Cannot build year_month.")
        return None

    # Select relevant metadata columns
    keep_meta = ['chunk_id', 'year_month', 'subreddit', '_score']
    if 'topic_macro' in meta.columns:
        keep_meta.append('topic_macro')
    meta = meta[keep_meta].copy()

    # Merge
    merged = commit[['chunk_id', 'buyin_label', 'stance_label']].merge(
        meta, on='chunk_id', how='inner'
    )
    print(f"  After merge with metadata: {len(merged):,} rows")

    # Compute log weight
    merged['log_w'] = log1p_weight(merged['_score'])
    merged.drop(columns=['_score'], inplace=True)

    if 'topic_macro' not in merged.columns:
        merged['topic_macro'] = None

    return merged


def pct_label(group, label):
    """Fraction of docs in group where sent_label == label."""
    return (group['sent_label'] == label).mean()


def rollup(df: pd.DataFrame, group_cols: list, commit_chunks: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Generic monthly rollup for a given set of group columns.

    If commit_chunks is provided (chunk-level labels with log_w), appends
    upvote-weighted commitment columns to the output.
    """
    agg = df.groupby(group_cols, observed=True).agg(
        doc_count          = ('doc_id',          'count'),
        mean_sent_neg      = ('sent_neg',         'mean'),
        mean_sent_neu      = ('sent_neu',         'mean'),
        mean_sent_pos      = ('sent_pos',         'mean'),
        mean_commit_net    = ('commit_net',       'mean'),
        mean_commit_support= ('commit_support',   'mean'),
        mean_commit_critical=('commit_critical',  'mean'),
    ).reset_index()

    # Percentage by dominant label — compute via value_counts per group
    label_pcts = (
        df.groupby(group_cols + ['sent_label'], observed=True)['doc_id']
        .count()
        .reset_index(name='n')
    )
    totals = label_pcts.groupby(group_cols, observed=True)['n'].transform('sum')
    label_pcts['pct'] = label_pcts['n'] / totals

    for label in ['neg', 'neu', 'pos']:
        pivot = (
            label_pcts[label_pcts['sent_label'] == label]
            .set_index(group_cols)['pct']
            .rename(f'pct_{label}')
        )
        agg = agg.join(pivot, on=group_cols)

    agg[['pct_neg', 'pct_neu', 'pct_pos']] = (
        agg[['pct_neg', 'pct_neu', 'pct_pos']].fillna(0)
    )

    # Low volume flag: based on (year_month, subreddit) bucket size
    sub_monthly = df.groupby(['year_month', 'subreddit'], observed=True)['doc_id'].count().rename('_sub_count')
    agg = agg.join(sub_monthly, on=['year_month', 'subreddit'])
    agg['low_volume'] = agg['_sub_count'] < LOW_VOLUME_THRESHOLD
    agg.drop(columns='_sub_count', inplace=True)

    # ── Optional: join chunk-level commitment aggregates ──────────────────────
    if commit_chunks is not None:
        # Filter commit_chunks to group_cols that exist in it
        commit_group_cols = [c for c in group_cols if c in commit_chunks.columns]
        if len(commit_group_cols) == len(group_cols):
            commit_agg = rollup_commitment(commit_chunks, group_cols)
            if not commit_agg.empty:
                agg = agg.merge(commit_agg, on=group_cols, how='left')
                # New commitment columns list
                new_commit_cols = [
                    'pct_buyin_committed', 'pct_buyin_uncommitted', 'pct_buyin_neutral',
                    'wtd_buyin_committed', 'wtd_buyin_uncommitted', 'net_buyin',
                    'pct_stance_supportive', 'pct_stance_critical', 'pct_stance_neutral',
                    'wtd_stance_supportive', 'wtd_stance_critical', 'net_stance',
                    'pct_positive', 'pct_negative',
                    'wtd_positive', 'wtd_negative', 'net_disposition',
                ]
                for col in new_commit_cols:
                    if col not in agg.columns:
                        agg[col] = float('nan')
        else:
            missing = set(group_cols) - set(commit_chunks.columns)
            warnings.warn(
                f"Commitment chunks missing columns {missing} for rollup "
                f"at group_cols={group_cols}. Skipping commitment join for this rollup."
            )

    # Downcast floats
    float_cols = [c for c in agg.columns if agg[c].dtype == float]
    agg[float_cols] = agg[float_cols].astype('float32')

    return agg.sort_values(group_cols).reset_index(drop=True)


def print_summary(overall: pd.DataFrame, topic: pd.DataFrame):
    print("\n" + "="*50)
    print("OUTPUT SUMMARY")
    print("="*50)

    print(f"\nOverall (month × subreddit):")
    print(f"  Rows: {len(overall):,}")
    print(f"  Months covered: {overall['year_month'].nunique()}")
    print(f"  Low-volume buckets: {overall['low_volume'].sum()}")

    print(f"\nTopic-level (month × subreddit × topic_macro):")
    print(f"  Rows: {len(topic):,}")
    print(f"  Unique topic_macros: {topic['topic_macro'].nunique()}")
    print(f"  Low-volume buckets: {topic['low_volume'].sum()}")

    print(f"\nOverall sentiment trend (subreddit mean across all months, high-vol only):")
    hv = overall[~overall['low_volume']]
    trend_cols = ['mean_sent_neg', 'mean_sent_pos', 'mean_commit_net']
    trend = hv.groupby('subreddit')[trend_cols].mean().round(4)
    print(trend.to_string())

    # Report commitment columns if present
    commit_net_cols = [c for c in overall.columns if c.startswith(('pct_buyin', 'wtd_buyin', 'net_buyin',
                                                                     'pct_stance', 'wtd_stance', 'net_stance',
                                                                     'net_disposition'))]
    if commit_net_cols:
        print(f"\nCommitment columns present: {commit_net_cols}")
        sample_row = hv[hv['subreddit'] == 'singapore'].tail(1)
        if not sample_row.empty:
            for col in commit_net_cols[:6]:
                if col in sample_row.columns:
                    val = sample_row[col].iloc[0]
                    print(f"  {col} (sample r/singapore): {val:.4f}")
    else:
        print(f"\nNote: commitment columns not added (chunk_commitment_cascade.parquet not found).")

    print(f"\nDoc count by subreddit × year (overall, sample):")
    _ov = overall.copy()
    _ov['year'] = _ov['year_month'].str[:4]
    yr = _ov.groupby(['subreddit', 'year'])['doc_count'].sum().unstack(fill_value=0)
    print(yr[[c for c in yr.columns if c >= '2019']].to_string())


def main():
    t0 = time.time()

    doc = load()

    # Load optional chunk-level commitment labels
    print("\nLoading chunk-level commitment labels (optional)...")
    commit_chunks = load_commitment_chunks()
    if commit_chunks is not None:
        print(f"  Commitment chunks loaded: {len(commit_chunks):,} rows")
        new_cols = [
            'pct_buyin_committed', 'pct_buyin_uncommitted', 'pct_buyin_neutral',
            'wtd_buyin_committed', 'wtd_buyin_uncommitted', 'net_buyin',
            'pct_stance_supportive', 'pct_stance_critical', 'pct_stance_neutral',
            'wtd_stance_supportive', 'wtd_stance_critical', 'net_stance',
            'pct_positive', 'pct_negative',
            'wtd_positive', 'wtd_negative', 'net_disposition',
        ]
        print(f"  Will add {len(new_cols)} commitment columns to temporal output:")
        print(f"    {new_cols}")
    else:
        print("  Skipping commitment columns (file not found).")

    print("\nBuilding overall rollup (month × subreddit)...")
    # For overall rollup, topic_macro is not a group col — drop it from commit_chunks
    commit_overall = None
    if commit_chunks is not None:
        commit_overall = commit_chunks.drop(columns=['topic_macro'], errors='ignore')
    overall = rollup(doc, ['year_month', 'subreddit'], commit_chunks=commit_overall)
    print(f"  {len(overall):,} rows")

    print("\nBuilding topic rollup (month × subreddit × topic_macro)...")
    doc_with_topic = doc[doc['topic_macro'].notna()].copy()
    print(f"  Docs with valid topic: {len(doc_with_topic):,} / {len(doc):,}")
    commit_topic = None
    if commit_chunks is not None:
        commit_topic = commit_chunks[commit_chunks['topic_macro'].notna()].copy()
    topic = rollup(doc_with_topic, ['year_month', 'subreddit', 'topic_macro'], commit_chunks=commit_topic)
    print(f"  {len(topic):,} rows")

    print_summary(overall, topic)

    print(f"\nSaving outputs...")
    overall.to_parquet(OUT_OVERALL, index=False)
    print(f"  {OUT_OVERALL}")
    topic.to_parquet(OUT_TOPIC, index=False)
    print(f"  {OUT_TOPIC}")

    print(f"\nDone. Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
