"""
Stage 7 v2 — Divergence Score (Redesigned 2026-06-08)
=====================================================
Supersedes stage7_divergence.py / doc_divergence.parquet.

The naive metric (mean comment sentiment − mean submission sentiment) is
dominated by neutral chunks (89% of corpus), so divergence scores cluster
near zero and bury genuinely contentious threads. The redesign filters for
opinionated chunks and weights by community endorsement (upvotes).

Input:
    chunk_sentiment.parquet     — chunk_id, sent_neg/neu/pos (SingBERT v7)
    submissions_chunks.parquet  — doc_id, score, log_weight, topic_id_fine, ...
    comments_chunks.parquet     — doc_id, post_id, score, depth, ...
    chunk_topics.parquet        — chunk_id, topic_id_fine

Output:
    doc_divergence_v2.parquet           — one row per post with >=1 matched comment chunk
    topic_discourse_intensity.parquet   — one row per topic_macro

Metrics (all computed over opinionated chunks only: max(sent_neg, sent_pos) > 0.5):
    upvote_weighted_div    — Σ(|neg−pos|_i × log(1+upvotes_i)) / Σ(log(1+upvotes_i))
                             per post (primary ranking signal). Negative upvotes
                             clamped to 0 before log. Falls back to the unweighted
                             mean when every opinionated chunk has 0 upvotes.
    within_thread_variance — variance of sent_neg across comment chunks per thread
                             (disagreement signal; NaN if <2 opinionated comment chunks)
    discourse_intensity    — Σ(upvotes × |neg−pos|) per topic_macro (separate table)
    commitment_divergence  — filled from chunk_commitment_cascade.parquet (2026-07-06)
Plus per post:
    opinion_chunk_pct        — % of all chunks in thread that are opinionated
    total_upvotes            — sum of upvotes across post + comments (doc level)
    sentiment_divergence_raw — old naive metric, kept for comparison

Run:
    python -m src.analysis.stage7_divergence_v2
"""

import time
import numpy as np
import pandas as pd

from src.models.topic_labels import TOPIC_LABELS

DATA = 'data/processed/new'
OUT_POST  = f'{DATA}/doc_divergence_v2.parquet'
OUT_TOPIC = f'{DATA}/topic_discourse_intensity.parquet'

OPINION_THRESHOLD = 0.5   # max(sent_neg, sent_pos) must exceed this


def load_chunks() -> pd.DataFrame:
    """Build one chunk-level table (post + comment chunks) with sentiment, upvotes, topic."""
    print("Loading inputs...")
    sent = pd.read_parquet(f'{DATA}/chunk_sentiment.parquet')
    subs = pd.read_parquet(f'{DATA}/submissions_chunks.parquet',
                           columns=['chunk_id', 'doc_id', 'score', 'topic_id_fine'])
    coms = pd.read_parquet(f'{DATA}/comments_chunks.parquet',
                           columns=['chunk_id', 'doc_id', 'post_id', 'score', 'topic_id_fine'])
    print(f"  chunk_sentiment:    {len(sent):,}")
    print(f"  submission chunks:  {len(subs):,}")
    print(f"  comment chunks:     {len(coms):,}")

    subs['post_id']  = subs['doc_id']       # a submission belongs to its own thread
    subs['is_comment'] = False
    coms['is_comment'] = True

    # Keep only comments whose post is in the corpus
    known_posts = set(subs['doc_id'])
    n_before = len(coms)
    coms = coms[coms['post_id'].isin(known_posts)]
    print(f"  Orphan comment chunks skipped: {n_before - len(coms):,}")

    chunks = pd.concat([subs, coms], ignore_index=True)
    chunks = chunks.merge(sent, on='chunk_id', how='inner')
    print(f"  Chunk rows after sentiment join: {len(chunks):,}")

    # Map fine topic -> macro category
    chunks['topic_macro'] = chunks['topic_id_fine'].map(
        lambda t: TOPIC_LABELS.get(t, {}).get('macro'))

    # Opinion intensity filter mask + per-chunk signals
    chunks['opinionated'] = chunks[['sent_neg', 'sent_pos']].max(axis=1) > OPINION_THRESHOLD
    chunks['abs_polarity'] = (chunks['sent_neg'] - chunks['sent_pos']).abs()
    chunks['upvotes_clamped'] = chunks['score'].clip(lower=0)
    chunks['log_upvotes'] = np.log1p(chunks['upvotes_clamped'])

    print(f"  Opinionated chunks: {chunks['opinionated'].sum():,} "
          f"({chunks['opinionated'].mean()*100:.1f}%)")
    return chunks


def post_topic_macro(chunks: pd.DataFrame) -> pd.Series:
    """Modal topic_macro across the submission's own chunks, indexed by post_id."""
    sub_chunks = chunks[~chunks['is_comment']].dropna(subset=['topic_macro'])
    return sub_chunks.groupby('post_id')['topic_macro'].agg(
        lambda s: s.mode().iloc[0])


def compute_post_metrics(chunks: pd.DataFrame) -> pd.DataFrame:
    """All per-post metrics. Only posts with >=1 matched comment chunk are kept."""
    print("\nComputing per-post metrics...")
    t0 = time.time()

    # --- Metric 1: upvote-weighted sentiment divergence (opinionated chunks) ---
    op = chunks[chunks['opinionated']].copy()
    op['wx'] = op['abs_polarity'] * op['log_upvotes']
    m1 = op.groupby('post_id').agg(
        wx_sum   = ('wx',           'sum'),
        w_sum    = ('log_upvotes',  'sum'),
        polarity_mean = ('abs_polarity', 'mean'),
        n_opinion = ('chunk_id', 'count'),
    )
    m1['upvote_weighted_div'] = np.where(
        m1['w_sum'] > 0,
        m1['wx_sum'] / m1['w_sum'],
        m1['polarity_mean'],   # fallback: all opinionated chunks have 0 upvotes
    )

    # --- Metric 2: within-thread variance of sent_neg (opinionated comment chunks) ---
    m2 = (op[op['is_comment']]
          .groupby('post_id')['sent_neg']
          .var()                       # ddof=1; NaN when <2 chunks
          .rename('within_thread_variance'))

    # --- opinion_chunk_pct + comment presence (all chunks) ---
    base = chunks.groupby('post_id').agg(
        opinion_chunk_pct = ('opinionated', 'mean'),
        comment_chunks    = ('is_comment',  'sum'),
    )
    base['opinion_chunk_pct'] *= 100.0

    # --- total_upvotes: doc-level (chunks duplicate the doc's score) ---
    docs = chunks.drop_duplicates('doc_id')
    total_up = docs.groupby('post_id')['upvotes_clamped'].sum().rename('total_upvotes')

    # --- sentiment_divergence_raw: old naive metric (no opinion filter) ---
    com_neg = chunks[chunks['is_comment']].groupby('post_id')['sent_neg'].mean()
    sub_neg = chunks[~chunks['is_comment']].groupby('post_id')['sent_neg'].mean()
    raw = (com_neg - sub_neg).rename('sentiment_divergence_raw')

    df = base.join([m1[['upvote_weighted_div']], m2, total_up, raw])
    df['topic_macro'] = post_topic_macro(chunks)

    # commitment_divergence: filled post-hoc by scripts/rebuild_doc_commitment.py
    # (pct_uncommitted(comments) − pct_uncommitted(submissions)).
    df['commitment_divergence'] = np.nan

    # Keep only posts with at least one matched comment chunk
    df = df[df['comment_chunks'] > 0].drop(columns='comment_chunks')
    df = df.reset_index()

    col_order = [
        'post_id', 'topic_macro',
        'upvote_weighted_div', 'within_thread_variance',
        'opinion_chunk_pct', 'total_upvotes',
        'commitment_divergence', 'sentiment_divergence_raw',
    ]
    df = df[col_order]

    float_cols = [c for c in df.columns if df[c].dtype == float]
    df[float_cols] = df[float_cols].astype('float32')

    print(f"  Posts scored: {len(df):,} (done in {time.time()-t0:.1f}s)")
    return df


def compute_topic_intensity(chunks: pd.DataFrame) -> pd.DataFrame:
    """Metric 3 — discourse_intensity = Σ(upvotes × |sent_neg − sent_pos|) per topic_macro."""
    print("\nComputing topic discourse intensity...")
    op = chunks[chunks['opinionated'] & chunks['topic_macro'].notna()].copy()
    op['intensity'] = op['upvotes_clamped'] * op['abs_polarity']

    topic = op.groupby('topic_macro').agg(
        discourse_intensity = ('intensity',       'sum'),
        n_opinion_chunks    = ('chunk_id',        'count'),
        total_upvotes       = ('upvotes_clamped', 'sum'),
    ).reset_index().sort_values('discourse_intensity', ascending=False)

    topic['discourse_intensity'] = topic['discourse_intensity'].astype('float32')
    print(f"  Topics: {len(topic)}")
    return topic


def print_summary(posts: pd.DataFrame, topics: pd.DataFrame):
    print("\n" + "="*60)
    print("OUTPUT SUMMARY")
    print("="*60)
    print(f"Posts scored: {len(posts):,}")
    print(f"  upvote_weighted_div non-null:    {posts['upvote_weighted_div'].notna().sum():,}")
    print(f"  within_thread_variance non-null: {posts['within_thread_variance'].notna().sum():,}")
    print(f"  mean opinion_chunk_pct:          {posts['opinion_chunk_pct'].mean():.1f}%")

    print("\nTop 10 most divergent posts (upvote_weighted_div, total_upvotes >= 5):")
    top = (posts[posts['total_upvotes'] >= 5]
           .sort_values('upvote_weighted_div', ascending=False)
           .head(10))
    print(top[['post_id', 'topic_macro', 'upvote_weighted_div',
               'within_thread_variance', 'total_upvotes']].round(4).to_string(index=False))

    print("\nTop 5 topics by discourse intensity:")
    print(topics.head(5).round(1).to_string(index=False))

    # v2 vs naive ranking comparison
    cmp = posts[posts['total_upvotes'] >= 5].copy()
    cmp['rank_v2']  = cmp['upvote_weighted_div'].rank(ascending=False)
    cmp['rank_raw'] = cmp['sentiment_divergence_raw'].abs().rank(ascending=False)
    rho = cmp[['rank_v2', 'rank_raw']].corr(method='spearman').iloc[0, 1]
    top_v2  = set(cmp.nsmallest(100, 'rank_v2')['post_id'])
    top_raw = set(cmp.nsmallest(100, 'rank_raw')['post_id'])
    print(f"\nRanking comparison (v2 vs naive |raw|, posts with >=5 upvotes):")
    print(f"  Spearman rho:            {rho:.4f}")
    print(f"  Top-100 overlap:         {len(top_v2 & top_raw)} / 100")


def main():
    t_total = time.time()

    chunks = load_chunks()
    posts  = compute_post_metrics(chunks)
    topics = compute_topic_intensity(chunks)

    print_summary(posts, topics)

    print(f"\nSaving {OUT_POST} ...")
    posts.to_parquet(OUT_POST, index=False)
    print(f"Saving {OUT_TOPIC} ...")
    topics.to_parquet(OUT_TOPIC, index=False)
    print(f"Done. Total time: {time.time()-t_total:.1f}s")


if __name__ == '__main__':
    main()
