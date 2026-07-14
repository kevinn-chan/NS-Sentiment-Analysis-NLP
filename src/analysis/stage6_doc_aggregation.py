"""
Stage 6 — Document-level aggregation
=====================================
Rolls up chunk-level sentiment + commitment + topic scores to one row per document.

Inputs (all from data/processed/new/):
    chunk_sentiment.parquet            — sent_neg, sent_neu, sent_pos per chunk (SingBERT v7)
    chunk_commitment.parquet           — commit_support, commit_critical, commit_neutral per chunk
                                         ⚠ BART C2D scores — Cohen's κ=0.127 vs human labels.
                                           Reliable for relative topic comparisons only.
                                           Use commitment_trend.csv (LLM, κ=0.752) for trend analysis.
    chunk_topics.parquet               — topic_id_fine, topic_id_coarse, topic_prob_fine per chunk
    comments_chunks.parquet            — log_weight, post_id, subreddit, created_utc, doc_type
    submissions_chunks.parquet         — log_weight, subreddit, created_utc, doc_type

Output:
    doc_sentiment.parquet              — one row per document, all scores aggregated

Key decisions:
    - log_weight floored at 1.0 so zero-weight chunks still contribute
    - Topic assigned by weighted mode: topic with highest Σ(topic_prob_fine × log_weight) per doc
    - Outlier chunks (topic_id_fine=-1) included in score aggregation, excluded from topic assignment
    - Submissions: post_id = doc_id (submission IS the post)
    - commit_* columns use BART NLI scores (unreliable at absolute level, kappa=0.127).
      For trend analysis, prefer commitment_llm_annual.csv (GPT-4.1-mini, kappa=0.752).

Future work (Stage 5b follow-up):
    - LLM commitment labels currently exist for ~10K stratified chunks (annual sample).
      When full-corpus LLM annotation is available, replace chunk_commitment.parquet input
      here and rerun to get reliable doc-level commit_* columns.

Run:
    python -m src.analysis.stage6_doc_aggregation
"""

import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
from src.models.topic_labels import TOPIC_LABELS

DATA = 'data/processed/new'
OUT  = 'data/processed/new/doc_sentiment.parquet'

SCORE_COLS = ['sent_neg', 'sent_neu', 'sent_pos',
              'prob_buyin_committed', 'prob_buyin_uncommitted', 'prob_buyin_neutral',
              'prob_stance_supportive', 'prob_stance_critical', 'prob_stance_neutral']

LOG_WEIGHT_FLOOR = 1.0   # floor so 0-weight chunks still contribute to averages


def load_inputs():
    t0 = time.time()
    print("Loading inputs...")

    sentiment = pd.read_parquet(f'{DATA}/chunk_sentiment.parquet')
    print(f"  sentiment:   {len(sentiment):,} rows")

    commitment = pd.read_parquet(f'{DATA}/chunk_commitment_cascade.parquet')
    print(f"  commitment:  {len(commitment):,} rows")
    print(f"  Using cascade output (buyin F1=0.714, stance F1=0.784).")

    # Deduplicate chunk_topics (309 duplicate chunk_ids from batch boundary)
    topics = pd.read_parquet(f'{DATA}/chunk_topics.parquet')
    topics = topics.drop_duplicates(subset='chunk_id', keep='first')
    print(f"  topics:      {len(topics):,} rows (after dedup)")

    comments = pd.read_parquet(
        f'{DATA}/comments_chunks.parquet',
        columns=['chunk_id', 'doc_id', 'post_id', 'subreddit', 'created_utc', 'log_weight', 'doc_type']
    )
    print(f"  comments:    {len(comments):,} rows")

    submissions = pd.read_parquet(
        f'{DATA}/submissions_chunks.parquet',
        columns=['chunk_id', 'doc_id', 'subreddit', 'created_utc', 'log_weight', 'doc_type']
    )
    # Submissions have no post_id — doc_id IS the post_id for submissions
    submissions['post_id'] = submissions['doc_id']
    print(f"  submissions: {len(submissions):,} rows")

    print(f"  Loaded in {time.time()-t0:.1f}s")
    return sentiment, commitment, topics, comments, submissions


def add_taxonomy(topics: pd.DataFrame) -> pd.DataFrame:
    """Map topic_id_fine → macro / sub / sub_sub taxonomy labels."""
    print("\nAdding taxonomy columns...")
    topics = topics.copy()
    topics['topic_macro']   = topics['topic_id_fine'].map(
        lambda t: TOPIC_LABELS.get(t, {}).get('macro'))
    topics['topic_sub']     = topics['topic_id_fine'].map(
        lambda t: TOPIC_LABELS.get(t, {}).get('sub'))
    topics['topic_sub_sub'] = topics['topic_id_fine'].map(
        lambda t: TOPIC_LABELS.get(t, {}).get('sub_sub'))
    outliers = (topics['topic_id_fine'] == -1).sum()
    print(f"  Outlier chunks (topic=-1, no taxonomy): {outliers:,}")
    return topics


def build_chunk_table(sentiment, commitment, topics, comments, submissions):
    """Join all chunk-level signals into one flat table."""
    print("\nBuilding chunk-level table...")
    t0 = time.time()

    chunks = pd.concat([comments, submissions], ignore_index=True)
    print(f"  Combined chunks: {len(chunks):,}")

    # Start from sentiment as the canonical chunk set (737,274 unique chunk_ids)
    merged = sentiment.merge(commitment, on='chunk_id', how='inner')

    merged = merged.merge(
        topics[['chunk_id', 'topic_id_fine', 'topic_id_coarse', 'topic_prob_fine',
                'topic_macro', 'topic_sub', 'topic_sub_sub']],
        on='chunk_id', how='left'
    )

    merged = merged.merge(
        chunks[['chunk_id', 'doc_id', 'post_id', 'subreddit', 'created_utc',
                'log_weight', 'doc_type']],
        on='chunk_id', how='left'
    )

    # Floor log_weight
    merged['log_weight_w'] = merged['log_weight'].clip(lower=LOG_WEIGHT_FLOOR)

    print(f"  Merged: {len(merged):,} rows | {merged['doc_id'].nunique():,} unique docs")
    print(f"  Built in {time.time()-t0:.1f}s")
    return merged


def aggregate_scores(merged: pd.DataFrame) -> pd.DataFrame:
    """Weighted mean of all score columns, grouped by doc_id."""
    print("\nAggregating scores...")
    t0 = time.time()

    # Weighted sum columns
    for col in SCORE_COLS:
        merged[f'_wsum_{col}'] = merged[col] * merged['log_weight_w']

    wsum_agg = {f'_wsum_{col}': (f'_wsum_{col}', 'sum') for col in SCORE_COLS}

    doc = merged.groupby('doc_id').agg(
        doc_type   =('doc_type',       'first'),
        post_id    =('post_id',        'first'),
        subreddit  =('subreddit',      'first'),
        created_utc=('created_utc',    'first'),
        chunk_count=('chunk_id',       'count'),
        weight_sum =('log_weight_w',   'sum'),
        **wsum_agg
    ).reset_index()

    # Normalise weighted sums → weighted means
    for col in SCORE_COLS:
        doc[col] = doc[f'_wsum_{col}'] / doc['weight_sum']
        doc.drop(columns=[f'_wsum_{col}'], inplace=True)
    doc.drop(columns=['weight_sum'], inplace=True)

    print(f"  Score aggregation: {len(doc):,} docs in {time.time()-t0:.1f}s")
    return doc


def assign_topics(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted mode topic per doc.
    For each doc, pick the topic_id_fine with the highest
    sum(topic_prob_fine × log_weight_w). Outlier chunks (topic=-1) excluded.
    """
    print("\nAssigning topics (weighted mode)...")
    t0 = time.time()

    valid = merged[merged['topic_id_fine'] != -1].copy()
    valid['topic_weight'] = valid['topic_prob_fine'] * valid['log_weight_w']

    # Sum topic weights per (doc_id, topic_id_fine)
    topic_scores = (
        valid
        .groupby(['doc_id', 'topic_id_fine', 'topic_id_coarse',
                  'topic_macro', 'topic_sub', 'topic_sub_sub'], observed=True)['topic_weight']
        .sum()
        .reset_index()
    )

    # Per doc: keep row with max topic_weight
    best_idx = topic_scores.groupby('doc_id')['topic_weight'].idxmax()
    best_topics = topic_scores.loc[best_idx].drop(columns='topic_weight').set_index('doc_id')

    # Docs with ALL outlier chunks get NaN topic (acceptable)
    outlier_docs = merged[merged['topic_id_fine'] == -1]['doc_id'].nunique()
    all_outlier = set(merged['doc_id'].unique()) - set(valid['doc_id'].unique())
    print(f"  Docs with valid topic: {len(best_topics):,}")
    print(f"  Docs with only outlier chunks (topic=None): {len(all_outlier):,}")
    print(f"  Topic assignment done in {time.time()-t0:.1f}s")
    return best_topics


def build_output(doc_scores: pd.DataFrame, best_topics: pd.DataFrame) -> pd.DataFrame:
    """Join scores + topics, add derived columns."""
    print("\nBuilding final output...")

    out = doc_scores.join(best_topics, on='doc_id', how='left')

    # Derived scores
    out['sent_label'] = out[['sent_neg', 'sent_neu', 'sent_pos']].idxmax(axis=1).str.replace('sent_', '')
    out['commit_net'] = out['prob_buyin_committed'] - out['prob_buyin_uncommitted']

    # Column order
    col_order = [
        'doc_id', 'doc_type', 'post_id', 'subreddit', 'created_utc', 'chunk_count',
        'sent_neg', 'sent_neu', 'sent_pos', 'sent_label',
        'prob_buyin_committed', 'prob_buyin_uncommitted', 'prob_buyin_neutral',
        'prob_stance_supportive', 'prob_stance_critical', 'prob_stance_neutral',
        'commit_net',
        'topic_id_fine', 'topic_id_coarse', 'topic_macro', 'topic_sub', 'topic_sub_sub',
    ]
    out = out[col_order]

    # Downcast floats to float32 to save disk
    float_cols = [c for c in out.columns if out[c].dtype == float]
    out[float_cols] = out[float_cols].astype('float32')

    return out


def print_summary(out: pd.DataFrame):
    print("\n" + "="*50)
    print("OUTPUT SUMMARY")
    print("="*50)
    print(f"Total docs:       {len(out):,}")
    print(f"  submissions:    {(out['doc_type']=='submission').sum():,}")
    print(f"  comments:       {(out['doc_type']=='comment').sum():,}")
    print(f"\nNulls:")
    print(out.isnull().sum()[out.isnull().sum() > 0])
    print(f"\nSentiment distribution:")
    print(out['sent_label'].value_counts().to_string())
    print(f"\nCommit net: mean={out['commit_net'].mean():.4f}, median={out['commit_net'].median():.4f}")
    print(f"\nTopic macro distribution (top 5):")
    print(out['topic_macro'].value_counts().head(5).to_string())
    print(f"\nChunk count stats:")
    print(out['chunk_count'].describe().to_string())
    print(f"\nDatetime range:")
    print(f"  earliest: {out['created_utc'].min()}")
    print(f"  latest:   {out['created_utc'].max()}")


def main():
    t_total = time.time()

    sentiment, commitment, topics, comments, submissions = load_inputs()
    topics = add_taxonomy(topics)
    merged = build_chunk_table(sentiment, commitment, topics, comments, submissions)

    doc_scores = aggregate_scores(merged)
    best_topics = assign_topics(merged)
    out = build_output(doc_scores, best_topics)

    print_summary(out)

    print(f"\nSaving to {OUT} ...")
    out.to_parquet(OUT, index=False)
    print(f"Done. Total time: {time.time()-t_total:.1f}s")


if __name__ == '__main__':
    main()
