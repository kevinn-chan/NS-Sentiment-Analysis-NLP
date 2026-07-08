"""
Targeted commitment-class sampling for the full-corpus distillation route (Stage 5b).

The committed/critical classes are rare (1.7% / 4.0% of the corpus), so random
annotation/LLM-labelling wastes almost all effort on neutral. This script ranks the
full corpus by signals that correlate with the minority classes and builds two queues:

  1. TEST queue   (manual)  → commitment_testset_queue.csv
       ~200 chunks, minority-enriched, for the human to hand-label. Gives a real
       holdout to measure committed/critical recall on (today's 100-row set has only
       3 committed). These chunk_ids are RESERVED — never fed to LLM enrichment/training.

  2. ENRICH queue (LLM)     → commitment_llm_enrich_queue.csv
       ~15k chunks biased toward the minority classes, for gpt-4.1-mini labelling
       (~$6). Boosts critical training mass from ~480 to a few thousand.

Signals used (all precomputed, no model inference — fully free):
  - sent_neg  from chunk_sentiment_singbert.parquet   → critical correlates with negativity
  - lex_committed / lex_uncommitted from chunk_commitment_lexicon.parquet → weak class hints

Already-labelled chunks (commitment_llm_annual.csv, commitment_annotation.csv) are excluded.

Usage:
    python -m src.features.commitment_sampler
    python -m src.features.commitment_sampler --n-crit 13000 --n-commit 2000 --n-test 200
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "new"

CHUNK_COLS = ["chunk_id", "doc_type", "subreddit", "text"]

CORPUS_FILES = ["submissions_chunks.parquet", "comments_chunks.parquet"]
SENT_FILE    = "chunk_sentiment_singbert.parquet"
LEX_FILE     = "chunk_commitment_lexicon.parquet"

# chunk_ids that already carry a commitment label — never re-sample these
LABELLED_FILES = ["commitment_llm_annual.csv", "commitment_annotation.csv"]

TEST_QUEUE_PATH   = DATA_DIR / "commitment_testset_queue.csv"
ENRICH_QUEUE_PATH = DATA_DIR / "commitment_llm_enrich_queue.csv"

OUT_COLS = ["chunk_id", "doc_type", "subreddit", "queue_type",
            "sent_neg", "lex_committed", "lex_uncommitted", "text"]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text)))


def load_eligible() -> pd.DataFrame:
    """Full corpus, filtered, with sent_neg + lexicon signals joined; labelled chunks removed."""
    exclude = set()
    for fname in LABELLED_FILES:
        p = DATA_DIR / fname
        if p.exists():
            exclude.update(pd.read_csv(p)["chunk_id"].dropna())
    print(f"Excluding {len(exclude):,} already-labelled chunk_ids")

    parts = [pd.read_parquet(DATA_DIR / f, columns=CHUNK_COLS) for f in CORPUS_FILES]
    chunks = pd.concat(parts, ignore_index=True).drop_duplicates("chunk_id")
    print(f"Corpus chunks: {len(chunks):,}")

    chunks = chunks[~chunks["chunk_id"].isin(exclude)].copy()
    chunks["text"] = chunks["text"].fillna("").str.strip()
    chunks = chunks[chunks["text"].str.len() > 20]
    chunks = chunks[chunks["text"].apply(_word_count) >= 15]   # need context to judge commitment
    print(f"Eligible after filters + exclusions: {len(chunks):,}")

    sent = pd.read_parquet(DATA_DIR / SENT_FILE, columns=["chunk_id", "sent_neg"])
    lex  = pd.read_parquet(DATA_DIR / LEX_FILE,
                           columns=["chunk_id", "lex_committed", "lex_uncommitted"])
    chunks = chunks.merge(sent, on="chunk_id", how="left").merge(lex, on="chunk_id", how="left")
    chunks["sent_neg"]        = chunks["sent_neg"].fillna(0.0)
    chunks["lex_committed"]   = chunks["lex_committed"].fillna(0).astype(int)
    chunks["lex_uncommitted"] = chunks["lex_uncommitted"].fillna(0).astype(int)
    return chunks


def build(n_neg: int, n_unc_lex: int, n_commit: int, n_test: int,
          neg_thresh: float = 0.6, seed: int = 42):
    for p in (TEST_QUEUE_PATH, ENRICH_QUEUE_PATH):
        if p.exists():
            print(f"⚠️  {p.name} already exists. Delete it to rebuild. Aborting.")
            sys.exit(0)

    chunks = load_eligible()

    has_lex          = (chunks["lex_committed"] > 0) | (chunks["lex_uncommitted"] > 0)
    committed_only   = (chunks["lex_committed"] > 0) & (chunks["lex_uncommitted"] == 0)
    uncommitted_only = (chunks["lex_uncommitted"] > 0) & (chunks["lex_committed"] == 0)
    minority_pool    = chunks[(chunks["sent_neg"] > neg_thresh) | has_lex]

    # ------------------------------------------------------------------
    # 1. TEST queue (manual) — reserve first so it never leaks into training
    # ------------------------------------------------------------------
    n_test_minority = int(n_test * 0.7)
    n_test_random   = n_test - n_test_minority
    test_min  = minority_pool.sample(min(n_test_minority, len(minority_pool)), random_state=seed)
    test_min  = test_min.assign(queue_type="test_minority")
    remaining = chunks[~chunks["chunk_id"].isin(test_min["chunk_id"])]
    test_rand = remaining.sample(n_test_random, random_state=seed).assign(queue_type="test_random")
    test = pd.concat([test_min, test_rand], ignore_index=True)
    test = test.sample(frac=1, random_state=seed).reset_index(drop=True)
    test["human_label"] = ""

    reserved = set(test["chunk_id"])
    pool = chunks[~chunks["chunk_id"].isin(reserved)]

    # ------------------------------------------------------------------
    # 2. ENRICH queue (LLM) — committed vs UNCOMMITTED scheme.
    #    uncommitted candidates come from two complementary signals:
    #      a) top sent_neg          → the institutionally-critical / venting end
    #      b) lexicon-uncommitted   → keng / wayang / zao / "waste of time" disengagement
    #         (this end is mostly NOT negative-sentiment, so sent_neg misses it)
    #    committed candidates       → lexicon-committed hits (sign on / brotherhood)
    # ------------------------------------------------------------------
    neg = pool.nlargest(n_neg, "sent_neg").assign(queue_type="uncommitted_neg")

    unc_src = pool[uncommitted_only.reindex(pool.index, fill_value=False)
                   & ~pool["chunk_id"].isin(neg["chunk_id"])]
    unc = unc_src.sample(min(n_unc_lex, len(unc_src)),
                         random_state=seed).assign(queue_type="uncommitted_lex")

    taken = set(neg["chunk_id"]) | set(unc["chunk_id"])
    commit_src = pool[committed_only.reindex(pool.index, fill_value=False)
                      & ~pool["chunk_id"].isin(taken)]
    commit = commit_src.sample(min(n_commit, len(commit_src)),
                               random_state=seed).assign(queue_type="committed_candidate")

    enrich = pd.concat([neg, unc, commit], ignore_index=True)
    enrich = enrich.sample(frac=1, random_state=seed).reset_index(drop=True)
    enrich["llm_label"] = ""

    test[OUT_COLS + ["human_label"]].to_csv(TEST_QUEUE_PATH, index=False)
    enrich[OUT_COLS + ["llm_label"]].to_csv(ENRICH_QUEUE_PATH, index=False)

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*64}")
    print(f"  TEST queue (manual)  → {TEST_QUEUE_PATH.name}: {len(test)} chunks")
    print(f"     test_minority {len(test_min):>4}  (sent_neg>{neg_thresh} or lexicon hit)")
    print(f"     test_random   {len(test_rand):>4}  (neutral representation)")
    print(f"\n  ENRICH queue (LLM)   → {ENRICH_QUEUE_PATH.name}: {len(enrich):,} chunks")
    print(f"     uncommitted_neg     {len(neg):>6}  (top sent_neg; critical/venting end; "
          f"min={neg['sent_neg'].min():.3f})")
    print(f"     uncommitted_lex     {len(unc):>6}  (keng/wayang/zao lexicon hits)")
    print(f"     committed_candidate {len(commit):>6}  (committed-only lexicon hits)")
    est = len(enrich) * 0.00046
    print(f"\n  Est. LLM-labelling cost @ ~$0.00046/chunk: ~${est:.2f}")
    print(f"{'='*64}")
    print(f"  Next: hand-label {TEST_QUEUE_PATH.name}, then LLM-label {ENRICH_QUEUE_PATH.name}")
    print(f"{'='*64}\n")


def main():
    ap = argparse.ArgumentParser(description="Build commitment minority-enriched queues.")
    ap.add_argument("--n-neg",     type=int,   default=10000, help="uncommitted candidates via top sent_neg")
    ap.add_argument("--n-unc-lex", type=int,   default=3000,  help="uncommitted candidates via lexicon (keng/wayang)")
    ap.add_argument("--n-commit",  type=int,   default=2000,  help="committed candidates (lexicon)")
    ap.add_argument("--n-test",    type=int,   default=200,   help="manual test-set size")
    ap.add_argument("--neg-thresh", type=float, default=0.6,  help="sent_neg cutoff for minority pool")
    ap.add_argument("--seed",      type=int,   default=42)
    args = ap.parse_args()
    build(args.n_neg, args.n_unc_lex, args.n_commit, args.n_test, args.neg_thresh, args.seed)


if __name__ == "__main__":
    main()
