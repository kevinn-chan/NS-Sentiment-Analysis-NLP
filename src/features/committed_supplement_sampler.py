"""
Build a stratified committed-candidate annotation supplement.

Pulls from the existing committed_candidate pool in commitment_llm_enrich_queue.csv,
stratifies by which lexicon term fired (so "sign on" doesn't dominate 80%+ of the
sample), and writes a fresh annotation queue to:

    data/processed/new/committed_supplement_queue.csv

Usage:
    python -m src.features.committed_supplement_sampler
    python -m src.features.committed_supplement_sampler --n 150 --sign-on-cap 20
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "processed" / "new"
ENRICH_PATH = DATA / "commitment_llm_enrich_queue.csv"
TESTSET_PATH = DATA / "commitment_testset.parquet"
OUT_PATH = DATA / "committed_supplement_queue.csv"

COMMITTED_TERMS = [
    "proud to serve", "honour to serve", "honor to serve", "love serving", "love ns",
    "enjoyed ns", "enjoy ns", "worth serving", "worth the sacrifice", "meaningful experience",
    "gives me purpose", "gave me purpose", "made me a man", "built character",
    "builds character", "made me stronger", "makes me stronger", "shaped who i am",
    "grateful for ns", "grateful for national service", "defend singapore",
    "protect singapore", "protect our country", "national duty", "serve the nation",
    "serve our country", "duty to serve", "sign on", "signed on",
    "brotherhood", "ns is important", "national service is important",
    "ns is necessary", "national service is necessary", "important for singapore",
    "important for defence", "important for defense", "necessary sacrifice",
]


def first_hit(text: str) -> str:
    t = str(text).lower()
    for term in COMMITTED_TERMS:
        pat = re.compile(
            r"\b" + r"\s+".join(re.escape(w) for w in term.split()) + r"\b",
            re.IGNORECASE,
        )
        if pat.search(t):
            return term
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120,
                    help="Total supplement batch size (default 120)")
    ap.add_argument("--sign-on-cap", type=int, default=20,
                    help="Max 'sign on'/'signed on' chunks (default 20)")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    if OUT_PATH.exists():
        print(f"⚠️  {OUT_PATH.name} already exists.")
        ans = input("Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit(0)

    if not ENRICH_PATH.exists():
        print(f"❌  {ENRICH_PATH.name} not found. Run commitment_sampler.py first.")
        sys.exit(1)

    enrich = pd.read_csv(ENRICH_PATH)
    pool = enrich[enrich["queue_type"] == "committed_candidate"].copy()
    print(f"Committed candidate pool: {len(pool):,}")

    # Exclude chunk_ids already in testset
    existing_ids = set(pd.read_parquet(TESTSET_PATH)["chunk_id"])
    pool = pool[~pool["chunk_id"].isin(existing_ids)].copy()
    print(f"After removing testset overlap: {len(pool):,}")

    # Tag by first-firing term
    pool["trigger_term"] = pool["text"].apply(first_hit)
    print("\nTrigger term distribution:")
    print(pool["trigger_term"].value_counts().head(15).to_string())

    # Stratified sample: cap sign-on, spread remaining slots across other terms
    sign_on_pool = pool[pool["trigger_term"].isin(["sign on", "signed on"])]
    other_pool   = pool[~pool["trigger_term"].isin(["sign on", "signed on"])]

    n_sign_on = min(args.sign_on_cap, len(sign_on_pool))
    n_other   = min(args.n - n_sign_on, len(other_pool))

    sampled_sign_on = sign_on_pool.sample(n_sign_on, random_state=args.seed)

    # For other terms, stratify by term (proportional, min 1 per term that has candidates)
    other_grouped = other_pool.groupby("trigger_term", group_keys=False)
    sampled_other = other_grouped.apply(
        lambda g: g.sample(min(len(g), max(1, round(n_other * len(g) / len(other_pool)))),
                           random_state=args.seed)
    ).reset_index(drop=True)
    # Trim to exact n_other
    sampled_other = sampled_other.sample(min(n_other, len(sampled_other)), random_state=args.seed)

    supplement = pd.concat([sampled_sign_on, sampled_other], ignore_index=True)
    supplement = supplement.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    supplement["human_label"]  = ""
    supplement["human_stance"] = ""

    cols = ["chunk_id", "doc_type", "subreddit", "queue_type", "trigger_term",
            "sent_neg", "lex_committed", "lex_uncommitted", "text",
            "human_label", "human_stance"]
    supplement[cols].to_csv(OUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"  Supplement queue → {OUT_PATH.name}: {len(supplement)} chunks")
    print(f"  sign on / signed on: {n_sign_on}")
    print(f"  other terms:         {len(sampled_other)}")
    print(f"\n  Trigger breakdown in sample:")
    print(supplement["trigger_term"].value_counts().to_string())
    print(f"\n  Run annotator:")
    print(f"  python -m src.features.committed_supplement_annotator")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
