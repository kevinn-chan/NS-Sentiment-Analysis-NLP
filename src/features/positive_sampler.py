"""
Targeted positive-class sampling for SingBERT v6 annotation campaign.

Scores a pool of unannotated corpus chunks using the existing SingBERT v5 model
(results-14/best_model), then builds a 600-chunk annotation queue deliberately
biased toward positive candidates — the class that has been starving for training data.

Queue composition (defaults):
  400 × positive_candidate  — top 400 chunks by SingBERT positive probability
  200 × random_balance      — random unannotated chunks (keeps neg/neu signal honest)

Running this twice (with --append) builds a larger queue without duplicates.

Usage:
    # Generate default 600-chunk queue
    python -m src.features.positive_sampler

    # Larger pool / more positives
    python -m src.features.positive_sampler --pool 4000 --n-pos 600 --n-balance 300

    # Append a second batch (for a second annotation session)
    python -m src.features.positive_sampler --append

    # Use a different model checkpoint
    python -m src.features.positive_sampler --model-path ~/Downloads/results-14/singbert_ns_sentiment/best_model
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent.parent.parent / "data" / "processed" / "new"
QUEUE_PATH = DATA_DIR / "annotation_queue.csv"

DEFAULT_MODEL_PATH = (
    Path.home() / "Downloads" / "results-14" / "singbert_ns_sentiment" / "best_model"
)

CHUNK_COLS = ["chunk_id", "doc_type", "subreddit", "text"]
LABEL_MAP  = {0: "negative", 1: "neutral", 2: "positive"}

# ---------------------------------------------------------------------------
# Text helpers (same vocab as llm_annotator.py)
# ---------------------------------------------------------------------------
import re as _re

_SG_VOCAB = {
    "lah","leh","lor","liao","sia","hor","mah","bah","nia",
    "sian","jialat","wayang","chao keng","saikang","siong","kena",
    "walao","walau","siao","aiyah","aiyoh","alamak","cheem","liddat",
    "shiok","song","swee","steady","lobang","slack","lepak",
    "bochap","bo chap","gg","tekan","suay","teruk","terok",
    "arrow","bo liao","tok kok","cmi","kns","tok gong","powderful",
    "gao gao","die die","wah","sibei",
}
_SG_PATTERN = _re.compile(
    r"\b(" + "|".join(_re.escape(w) for w in sorted(_SG_VOCAB, key=len, reverse=True)) + r")\b"
)

def _has_singlish(text: str) -> bool:
    return bool(_SG_PATTERN.search(str(text).lower()))

def _word_count(text: str) -> int:
    return len(_re.findall(r"\b\w+\b", str(text)))


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def load_model(model_path: Path):
    print(f"Loading SingBERT from {model_path} ...")
    if not model_path.exists():
        print(f"\n❌  Model not found at: {model_path}")
        print(f"    Pass --model-path to specify the correct path.")
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model     = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model  = model.to(device)
    print(f"  Model loaded on {device}")
    return tokenizer, model, device


def score_chunks(df: pd.DataFrame, tokenizer, model, device: str,
                 batch_size: int = 16) -> pd.DataFrame:
    """Add singbert_neg/neu/pos_prob + singbert_label columns to df."""
    texts     = df["text"].fillna("").tolist()
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch   = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            max_length=256,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.extend(probs.tolist())

        done = min(i + batch_size, len(texts))
        if done % 200 == 0 or done == len(texts):
            print(f"  {done:>5,}/{len(texts):,} scored ...", end="\r", flush=True)

    print(f"  {len(texts):,}/{len(texts):,} scored — done.        ")

    arr = np.array(all_probs)
    df  = df.copy()
    df["singbert_neg_prob"] = arr[:, 0].round(4)
    df["singbert_neu_prob"] = arr[:, 1].round(4)
    df["singbert_pos_prob"] = arr[:, 2].round(4)
    df["singbert_label"]    = arr.argmax(axis=1)
    df["singbert_label"]    = df["singbert_label"].map(LABEL_MAP)
    return df


# ---------------------------------------------------------------------------
# Main queue builder
# ---------------------------------------------------------------------------
def build_queue(pool_size: int = 2000, n_pos: int = 400, n_balance: int = 200,
                model_path: Path = DEFAULT_MODEL_PATH, append: bool = False):

    # ------------------------------------------------------------------
    # 1. Build exclusion set
    # ------------------------------------------------------------------
    exclude = set()
    for fname in ("blind_annotation.csv", "holdout_test.csv",
                  "llm_annotation.csv", "targeted_annotations.csv"):
        p = DATA_DIR / fname
        if p.exists():
            exclude.update(pd.read_csv(p)["chunk_id"].dropna())

    existing_queue = pd.DataFrame()
    if QUEUE_PATH.exists():
        existing_queue = pd.read_csv(QUEUE_PATH)
        if not append:
            print(f"⚠️  annotation_queue.csv already exists ({len(existing_queue)} rows).")
            print(f"    Use --append to add more chunks, or delete the file to start fresh.")
            sys.exit(0)
        exclude.update(existing_queue["chunk_id"].dropna())
        print(f"Append mode — {len(existing_queue)} existing queue rows excluded from re-sampling.")

    print(f"Exclusion set: {len(exclude):,} chunk_ids")

    # ------------------------------------------------------------------
    # 2. Sample pool from corpus
    # ------------------------------------------------------------------
    print(f"\nLoading corpus chunks ...")
    sub    = pd.read_parquet(DATA_DIR / "submissions_chunks.parquet", columns=CHUNK_COLS)
    com    = pd.read_parquet(DATA_DIR / "comments_chunks.parquet",    columns=CHUNK_COLS)
    chunks = pd.concat([sub, com], ignore_index=True).drop_duplicates("chunk_id")
    del sub, com

    chunks = chunks[~chunks["chunk_id"].isin(exclude)].copy()
    chunks["text"] = chunks["text"].fillna("").str.strip()
    chunks = chunks[chunks["text"].str.len() > 20]
    chunks["has_sg"] = chunks["text"].apply(_has_singlish)
    chunks["wc"]     = chunks["text"].apply(_word_count)
    chunks = chunks[chunks["wc"] >= 15]   # skip micro-chunks (< 15 words — not enough context to label)

    if len(chunks) < pool_size:
        print(f"⚠️  Only {len(chunks):,} eligible chunks available — using all of them.")
        pool_size = len(chunks)

    pool = chunks.sample(pool_size, random_state=42).reset_index(drop=True)
    print(f"Pool: {len(pool):,} chunks from {len(chunks):,} eligible")

    # ------------------------------------------------------------------
    # 3. Score with SingBERT
    # ------------------------------------------------------------------
    tokenizer, model, device = load_model(model_path)
    print(f"\nScoring {len(pool):,} chunks ...")
    pool = score_chunks(pool, tokenizer, model, device)

    # ------------------------------------------------------------------
    # 4. Build queue (top-N positive + random balance)
    # ------------------------------------------------------------------
    top_pos  = pool.nlargest(n_pos, "singbert_pos_prob").copy()
    top_pos["queue_type"] = "positive_candidate"

    remaining = pool[~pool["chunk_id"].isin(top_pos["chunk_id"])]
    rand_bal  = remaining.sample(min(n_balance, len(remaining)), random_state=7).copy()
    rand_bal["queue_type"] = "random_balance"

    new_queue = pd.concat([top_pos, rand_bal], ignore_index=True)
    new_queue = new_queue.sample(frac=1, random_state=99).reset_index(drop=True)
    new_queue["human_label"] = None

    # Merge with existing if appending
    if append and len(existing_queue) > 0:
        final_queue = pd.concat([existing_queue, new_queue], ignore_index=True)
    else:
        final_queue = new_queue

    final_queue.to_csv(QUEUE_PATH, index=False)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print(f"\n{'═'*60}")
    print(f"  Annotation queue {'updated' if append else 'built'}: {len(new_queue)} new chunks")
    print(f"  Total in queue: {len(final_queue)}")
    print(f"{'═'*60}")
    print(f"  positive_candidate  {len(top_pos):>4}  (top {n_pos} by SingBERT pos_prob)")
    print(f"  random_balance      {len(rand_bal):>4}  (random from remaining pool)")

    print(f"\n  SingBERT label distribution in new batch:")
    for lbl in ["positive", "neutral", "negative"]:
        n = (new_queue["singbert_label"] == lbl).sum()
        pct = n / len(new_queue) * 100
        bar = "█" * int(pct / 3)
        print(f"    {lbl:<10}  {n:>4}  ({pct:4.1f}%)  {bar}")

    pos_probs = top_pos["singbert_pos_prob"]
    print(f"\n  Positive candidate pos_prob range:")
    print(f"    min={pos_probs.min():.3f}  p25={pos_probs.quantile(.25):.3f}  "
          f"median={pos_probs.median():.3f}  max={pos_probs.max():.3f}")
    print(f"\n  ✅  Saved → {QUEUE_PATH}")
    print(f"\n  Next step:")
    print(f"    python -m src.features.targeted_annotator")
    print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build targeted positive-candidate annotation queue using SingBERT v5."
    )
    parser.add_argument("--pool",       type=int,  default=2000,
                        help="Number of unannotated chunks to score (default: 2000)")
    parser.add_argument("--n-pos",      type=int,  default=400,
                        help="Top-N positive candidates to include in queue (default: 400)")
    parser.add_argument("--n-balance",  type=int,  default=200,
                        help="Random balance chunks to add (default: 200)")
    parser.add_argument("--model-path", type=str,  default=str(DEFAULT_MODEL_PATH),
                        help=f"Path to SingBERT best_model directory")
    parser.add_argument("--append",     action="store_true",
                        help="Append new chunks to existing queue instead of overwriting")
    args = parser.parse_args()

    build_queue(
        pool_size=args.pool,
        n_pos=args.n_pos,
        n_balance=args.n_balance,
        model_path=Path(args.model_path),
        append=args.append,
    )


if __name__ == "__main__":
    main()
