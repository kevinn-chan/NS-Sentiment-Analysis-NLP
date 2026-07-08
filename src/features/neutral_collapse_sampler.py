"""
Sample buyin_label=neutral chunks that likely contain genuine committed or uncommitted
signal — the "neutral collapse" cases where the model missed the label.

Two batches:
  U — neutral chunks with uncommitted surface signals (keng, wayang, sian, zao, ...)
  C — neutral chunks with first-person committed signals (I signed, my NS, grateful, ...)

Output: data/processed/new/neutral_collapse_queue.csv
        columns: chunk_id, doc_type, subreddit, batch, trigger_phrase, text,
                 human_label, human_stance

Usage:
    python -m src.features.neutral_collapse_sampler
    python -m src.features.neutral_collapse_sampler --n-u 200 --n-c 150
"""
import argparse, re
from pathlib import Path
import pandas as pd

DATA         = Path(__file__).resolve().parents[2] / "data" / "processed" / "new"
LLM_PATH     = DATA / "chunk_commitment_cascade.parquet"
META_PATH    = DATA / "chunk_metadata.parquet"
TESTSET_PATH = DATA / "commitment_testset.parquet"
OUT_PATH     = DATA / "neutral_collapse_queue.csv"

# Uncommitted surface patterns (Singlish + explicit)
UNCOMMITTED_SIGNALS = [
    r"\bbo\s+ch[ua]p\b", r"\bboch[ua]p\b",
    r"\bchao\s+keng\b", r"\bkeng\b",
    r"\bwayang\b",
    r"\bsian(z|h)?\b",
    r"\bzao\b",
    r"\bord\s+(lo|mood|liao)\b",
    r"\bcounting\s+down\b",
    r"\bjust\s+want\s+to\s+ord\b",
    r"\bwaste\s+(of\s+)?(time|life|2|two)\b",
    r"\bns\s+(sucks|is\s+(terrible|horrible|trash|a\s+waste))\b",
    r"\bhate\s+ns\b",
    r"\bregret\s+(serving|enlisting)\b",
    r"\bnot\s+worth\s+(it|serving)\b",
    r"\bcannot\s+make\s+it\b",
    r"\bwasted\s+my\s+time\b",
    r"\bsmoke\b",
    r"\bpointless\b",
    r"\bwhy\s+bother\b",
]

# First-person committed surface patterns
COMMITTED_SIGNALS = [
    r"\bi\s+signed\s+on\b",
    r"\bi\s+(plan|want|intend)\s+to\s+sign\s+on\b",
    r"\bplanning\s+to\s+sign\s+on\b",
    r"\bgrateful\s+(for\s+)?(ns|national\s+service|my\s+service)\b",
    r"\bproud\s+to\s+serve\b",
    r"\bmade\s+me\s+(a\s+man|stronger|who\s+i\s+am)\b",
    r"\bbuilt\s+(my\s+)?character\b",
    r"\bmeaningful\s+experience\b",
    r"\bgave?\s+me\s+purpose\b",
    r"\bmy\s+ns\s+(was|is|taught|gave|made)\b",
    r"\blove\s+(ns|serving|national\s+service)\b",
    r"\bbrotherhood\b",
    r"\bworth\s+(it|serving|the\s+sacrifice)\b",
    r"\bns\s+(made|taught|shaped)\s+me\b",
    r"\bdefend\s+singapore\b",
    r"\bprotect\s+(singapore|our\s+country)\b",
    r"\bduty\s+to\s+serve\b",
    r"\bnational\s+duty\b",
]

_UNC_RE = re.compile("|".join(UNCOMMITTED_SIGNALS), re.I)
_COM_RE = re.compile("|".join(COMMITTED_SIGNALS),   re.I)


def first_hit(text: str, pattern: re.Pattern) -> str:
    m = pattern.search(str(text))
    return m.group(0).strip().lower() if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-u",   type=int, default=200, help="Uncommitted batch size")
    ap.add_argument("--n-c",   type=int, default=150, help="Committed batch size")
    ap.add_argument("--seed",  type=int, default=42)
    args = ap.parse_args()

    if OUT_PATH.exists():
        print(f"⚠️  {OUT_PATH.name} already exists.")
        ans = input("Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            return

    print("Loading LLM predictions …")
    llm  = pd.read_parquet(LLM_PATH, columns=["chunk_id", "buyin_label",
                                               "prob_buyin_uncommitted",
                                               "prob_buyin_committed"])
    meta = pd.read_parquet(META_PATH, columns=["chunk_id", "doc_id", "doc_type", "subreddit"])

    neutral = llm[llm["buyin_label"] == "neutral"].copy()
    print(f"Neutral-classified chunks: {len(neutral):,}")

    # Exclude chunks already in testset
    existing_ids = set(pd.read_parquet(TESTSET_PATH)["chunk_id"])
    neutral = neutral[~neutral["chunk_id"].isin(existing_ids)]
    print(f"After removing testset overlap: {len(neutral):,}")

    # Need text — join through metadata → doc chunks
    print("Loading chunk text …")
    neutral_meta = neutral.merge(meta, on="chunk_id", how="left")

    cc = pd.read_parquet(DATA / "comments_chunks.parquet",    columns=["doc_id", "text"])
    sc = pd.read_parquet(DATA / "submissions_chunks.parquet", columns=["doc_id", "text"])
    docs = pd.concat([cc, sc]).drop_duplicates("doc_id")
    del cc, sc

    df = neutral_meta.merge(docs, on="doc_id", how="left").dropna(subset=["text"])
    df["text"] = df["text"].str.strip()
    print(f"Neutral chunks with text: {len(df):,}")

    # ── Batch U: uncommitted signals ──────────────────────────────────────────
    df["_unc_hit"] = df["text"].apply(lambda t: first_hit(t, _UNC_RE))
    batch_u = df[df["_unc_hit"] != ""].copy()
    # Sort by prob_buyin_uncommitted descending so most-likely-real ones come first
    batch_u = batch_u.sort_values("prob_buyin_uncommitted", ascending=False)
    batch_u = batch_u.head(args.n_u * 3).sample(args.n_u, random_state=args.seed)
    batch_u["batch"]          = "U"
    batch_u["trigger_phrase"] = batch_u["_unc_hit"]
    print(f"\nBatch U (uncommitted signals): {len(batch_u):,}")
    print(batch_u["trigger_phrase"].value_counts().head(10).to_string())

    # ── Batch C: committed signals ────────────────────────────────────────────
    df["_com_hit"] = df["text"].apply(lambda t: first_hit(t, _COM_RE))
    batch_c = df[df["_com_hit"] != ""].copy()
    batch_c = batch_c.sort_values("prob_buyin_committed", ascending=False)
    batch_c = batch_c.head(args.n_c * 3).sample(args.n_c, random_state=args.seed)
    batch_c["batch"]          = "C"
    batch_c["trigger_phrase"] = batch_c["_com_hit"]
    print(f"\nBatch C (committed signals): {len(batch_c):,}")
    print(batch_c["trigger_phrase"].value_counts().head(10).to_string())

    # ── Combine and save ──────────────────────────────────────────────────────
    keep = ["chunk_id", "doc_type", "subreddit", "batch", "trigger_phrase", "text"]
    out  = pd.concat([batch_u[keep], batch_c[keep]], ignore_index=True)
    out  = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out["human_label"]  = ""
    out["human_stance"] = ""

    out.to_csv(OUT_PATH, index=False)
    print(f"\n{'='*60}")
    print(f"  Saved → {OUT_PATH.name}: {len(out)} chunks")
    print(f"  Batch U: {len(batch_u)}  |  Batch C: {len(batch_c)}")
    print(f"\n  Run annotator:")
    print(f"  python -m src.features.neutral_collapse_annotator")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
