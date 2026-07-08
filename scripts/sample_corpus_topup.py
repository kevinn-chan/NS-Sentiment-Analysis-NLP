"""
Pull a top-up sample from the full corpus (submissions + comments chunks)
for additional LLM labelling, excluding chunk_ids already used in
stage2a/stage2b. Weighted toward keywords likely to signal committed/
supportive content, since the existing training set is skewed toward
critical/uncommitted.

Usage:
    python scripts/sample_corpus_topup.py --n 25000 --output transfer/corpus_topup.csv
"""
import argparse, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Keywords that skew toward committed/supportive content (thin classes)
BOOST_KEYWORDS = [
    "sign on", "signed on", "sign-on", "regular", "vocation",
    "worth it", "proud", "i support", "necessary for singapore",
    "defend singapore", "defend our country", "protect singapore",
    "brotherhood", "best time", "miss my army", "enjoyed my ns",
    "enjoyed ns", "loved ns", "loved my time",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25000, help="Total rows to sample")
    ap.add_argument("--boost-frac", type=float, default=0.4,
                     help="Fraction of sample drawn from keyword-boosted rows")
    ap.add_argument("--output", default=os.path.join(ROOT, "transfer", "corpus_topup.csv"))
    args = ap.parse_args()

    used_ids = set()
    for f in [
        "stage2a_silver_v2.csv", "stage2b_silver_v2.csv",
        "corpus_topup.csv", "corpus_topup2.csv", "corpus_topup3.csv",
        "corpus_topup_all_labelled.csv",
    ]:
        p = os.path.join(ROOT, "transfer", f)
        if os.path.exists(p):
            used_ids |= set(pd.read_csv(p, usecols=["chunk_id"])["chunk_id"])
    print(f"Excluding {len(used_ids)} already-used/already-labelled chunk_ids")

    frames = []
    for f in ["submissions_chunks.parquet", "comments_chunks.parquet"]:
        p = os.path.join(ROOT, "data", "processed", "new", f)
        df = pd.read_parquet(p, columns=["chunk_id", "text"])
        df = df[~df["chunk_id"].isin(used_ids)]
        frames.append(df)
    corpus = pd.concat(frames, ignore_index=True)
    corpus = corpus[corpus["text"].str.len() > 20]  # drop near-empty chunks
    print(f"Candidate pool: {len(corpus)} rows")

    pattern = "|".join(BOOST_KEYWORDS)
    is_boost = corpus["text"].str.lower().str.contains(pattern, regex=True, na=False)
    boosted = corpus[is_boost]
    rest = corpus[~is_boost]
    print(f"Keyword-matched rows available: {len(boosted)}")

    n_boost = min(int(args.n * args.boost_frac), len(boosted))
    n_rest = args.n - n_boost

    sample = pd.concat([
        boosted.sample(n_boost, random_state=0),
        rest.sample(min(n_rest, len(rest)), random_state=0),
    ], ignore_index=True).sample(frac=1, random_state=0)  # shuffle

    sample[["chunk_id", "text"]].to_csv(args.output, index=False)
    print(f"Saved {len(sample)} rows -> {args.output}")
    print(f"  Keyword-boosted: {n_boost} | Random: {len(sample) - n_boost}")


if __name__ == "__main__":
    main()
