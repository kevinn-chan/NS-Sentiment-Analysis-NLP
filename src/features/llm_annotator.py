"""
LLM-based auto-annotation pipeline for SingBERT fine-tuning golden dataset.

Uses Groq (free tier) with Llama 3.3 70B to auto-annotate NS Reddit chunks.
Human blind labels (blind_annotation.csv) serve as the golden validation set.

Free tier limits (Groq):
  - llama-3.3-70b-versatile: 14,400 requests/day, 30 req/min
  - At 30 req/min: 8,000 chunks ≈ 4.5 hours (within daily limit)

Setup:
    1. Get free API key at https://console.groq.com
    2. pip install openai   (Groq is OpenAI-compatible)
    3. export GROQ_API_KEY=gsk_...

Usage:
    # Step 1 — validate LLM against human labels, compute Cohen's Kappa
    python -m src.features.llm_annotator --validate

    # Step 2 — bulk annotate N chunks (only after kappa >= 0.70)
    python -m src.features.llm_annotator --annotate 8000

    # Step 3 — combine into final training dataset
    python -m src.features.llm_annotator --build-dataset
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR       = Path(__file__).parent.parent.parent / "data" / "processed" / "new"
BLIND_ANN_PATH = DATA_DIR / "blind_annotation.csv"
LLM_ANN_PATH   = DATA_DIR / "llm_annotation.csv"
TRAIN_PATH     = DATA_DIR / "singbert_train.csv"

CHUNK_COLS = ["chunk_id", "doc_type", "subreddit", "text"]

# Groq config
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = "llama-3.3-70b-versatile"  # 6,000 TPM free; trimmed prompt ~700 tok → ~8 RPM
# llama-3.1-8b-instant: 131k TPM but too small for nuanced sentiment — kappa degrades with complex prompts
RATE_LIMIT_SLEEP = 8.0   # seconds between calls → ~7.5 req/min (safely under 8 RPM at 700 tok/call)

# ---------------------------------------------------------------------------
# Few-shot examples — 3 targeted examples (one per class)
# Chosen to cover the hardest failure modes: question-as-complaint,
# factual-not-emotional, mild-positive-not-neutral
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    # NEGATIVE — complaint phrased as a question (not neutral just because it's a question)
    ("Are the sergeants still unreasonable, tekan them like hell for no reason? I've seen so many posts ranting about this, is it still happening?",
     "negative"),

    # NEUTRAL — describes hard NS things matter-of-factly (mentioning hardship ≠ complaining)
    ("During BMT the tekan sessions were intense. We would do pushups and leopard crawls. That's just how it works in the first few weeks.",
     "neutral"),

    # POSITIVE — mild satisfaction counts (not just enthusiasm)
    ("Managed to get a desk vocation, mostly 8 to 5, can book out most nights. Not what I expected from NS but I'll take it.",
     "positive"),
]

# ---------------------------------------------------------------------------
# System prompt — concise, ~300 tokens; 70B doesn't need hand-holding
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a sentiment classifier for Singapore National Service (NS) Reddit posts.

Classify the AUTHOR'S emotional state — not the topic — as negative, neutral, or positive.

NEGATIVE: author is frustrated, complaining, resentful, bitter, or distressed.
NEUTRAL:  author is informing, asking, advising, or describing without strong emotion.
POSITIVE: author feels satisfied, relieved, proud, grateful, or happy (mild counts).

Key rules:
1. Describing hard/tough NS experiences without frustration = NEUTRAL (not NEGATIVE).
2. A complaint phrased as a question is still NEGATIVE.
3. Mild satisfaction or relief = POSITIVE (not NEUTRAL).
4. Singlish slang (lah, leh, sian, tekan, ORD, encik) is normal NS vocabulary — read tone, not just words.

Reply with ONLY this JSON, nothing else:
{"label": "negative"}   or   {"label": "neutral"}   or   {"label": "positive"}"""


def build_messages(text: str) -> list:
    """Build few-shot message list for one chunk."""
    messages = []
    for example_text, example_label in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user",      "content": example_text[:800]})
        messages.append({"role": "assistant", "content": json.dumps({"label": example_label})})
    messages.append({"role": "user", "content": str(text)[:1500]})
    return messages


# ---------------------------------------------------------------------------
# Single API call
# ---------------------------------------------------------------------------
def call_llm(text: str, client, retries: int = 3) -> str | None:
    """Call Groq LLM with separate budgets for parse errors vs rate limits.

    Rate limit hits do NOT consume the retry budget — they back off and retry
    indefinitely with exponential backoff (60 → 120 → 240 → 300s cap).
    Parse errors (bad JSON, unexpected label) consume the retry budget.
    """
    parse_attempts = 0
    rate_wait      = 60   # seconds; doubles on each consecutive rate limit hit

    while parse_attempts < retries:
        try:
            resp  = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                         + build_messages(text),
                max_tokens=15,
                temperature=0.0,
            )
            raw   = resp.choices[0].message.content.strip()
            label = json.loads(raw).get("label", "").lower()
            if label in ("negative", "neutral", "positive"):
                rate_wait = 60   # reset backoff on success
                return label
            print(f"  ⚠️  Unexpected label (attempt {parse_attempts+1}): {raw!r}")
            parse_attempts += 1

        except json.JSONDecodeError:
            parse_attempts += 1
            print(f"  ⚠️  JSON error attempt {parse_attempts}: {raw!r}")

        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                # Rate limit — back off but don't burn retry budget
                print(f"  Rate limit hit — waiting {rate_wait}s ...")
                time.sleep(rate_wait)
                rate_wait = min(rate_wait * 2, 300)  # 60 → 120 → 240 → 300 cap
            else:
                parse_attempts += 1
                print(f"  ⚠️  API error (attempt {parse_attempts}): {e}")
                time.sleep(min(2 ** parse_attempts, 30))

    return None


def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set.")
        print("  1. Get a free key at https://console.groq.com")
        print("  2. export GROQ_API_KEY=gsk_...")
        sys.exit(1)

    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


# ---------------------------------------------------------------------------
# Singlish helpers
# ---------------------------------------------------------------------------
_VOCAB = {
    "lah","leh","lor","liao","sia","hor","mah","bah","nia",
    "sian","jialat","wayang","chao keng","saikang","siong","kena",
    "walao","walau","siao","aiyah","aiyoh","alamak","cheem","liddat",
    "shiok","song","swee","steady","lobang","slack","lepak",
    "bochap","bo chap","gg","tekan","suay","teruk","terok",
    "arrow","bo liao","tok kok","cmi","kns","tok gong","powderful",
    "gao gao","die die","wah","sibei",
}
_SG_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_VOCAB, key=len, reverse=True)) + r")\b"
)

def has_singlish(text: str) -> bool:
    return bool(_SG_PATTERN.search(str(text).lower()))

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text)))


# ---------------------------------------------------------------------------
# Step 1 — Validate against golden labels (Kappa gate)
# ---------------------------------------------------------------------------
def run_validate():
    try:
        from sklearn.metrics import cohen_kappa_score, classification_report
    except ImportError:
        print("Run: pip install scikit-learn")
        return

    client = get_client()

    golden = pd.read_csv(BLIND_ANN_PATH)
    golden = golden[
        golden["human_label"].notna() & (golden["human_label"] != "skip")
    ].copy().reset_index(drop=True)

    print(f"Validating {len(golden)} golden labels with {GROQ_MODEL} ...")
    print(f"Estimated time: ~{len(golden) * RATE_LIMIT_SLEEP / 60:.0f} min\n")

    llm_labels = []
    t0         = time.time()
    out        = DATA_DIR / "llm_validation.csv"   # save incrementally

    for i, (_, row) in enumerate(golden.iterrows()):
        label = call_llm(str(row["text"]), client)
        llm_labels.append(label)
        time.sleep(RATE_LIMIT_SLEEP)

        if (i + 1) % 25 == 0 or (i + 1) == len(golden):
            done    = [l for l in llm_labels if l]
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(golden) - i - 1) / rate
            print(f"  {i+1}/{len(golden)}  {rate:.1f}/s  ETA {eta/60:.0f}min")
            # Incremental save — safe to kill at any 25-chunk boundary
            partial = golden.iloc[: i + 1].copy()
            partial["llm_label"] = llm_labels
            partial.dropna(subset=["llm_label"]).to_csv(out, index=False)

    golden["llm_label"] = llm_labels
    valid = golden.dropna(subset=["llm_label"])
    failed = len(golden) - len(valid)

    kappa = cohen_kappa_score(valid["human_label"], valid["llm_label"])

    print(f"\n{'═'*58}")
    print(f"  Kappa Validation — {GROQ_MODEL}")
    print(f"{'═'*58}")
    print(f"  Chunks evaluated : {len(valid)}  ({failed} failed/skipped)")
    print(f"  Cohen's Kappa    : {kappa:.3f}")

    if kappa >= 0.80:
        verdict = "✅  EXCELLENT — proceed to --annotate"
    elif kappa >= 0.70:
        verdict = "✅  GOOD — proceed to --annotate"
    elif kappa >= 0.60:
        verdict = "⚠️  MODERATE — consider prompt tweaks before --annotate"
    else:
        verdict = "❌  POOR — revise prompt, do not proceed to --annotate"
    print(f"  Verdict          : {verdict}")

    print(f"\n  Per-label agreement (LLM vs human):")
    for lbl in ["negative", "neutral", "positive"]:
        sub   = valid[valid["human_label"] == lbl]
        agree = (sub["llm_label"] == lbl).mean() if len(sub) > 0 else 0
        print(f"    {lbl:<10} {agree:.1%}  (n={len(sub)})")

    print(f"\n  Full classification report:")
    print(classification_report(
        valid["human_label"], valid["llm_label"],
        labels=["negative", "neutral", "positive"], digits=3
    ))

    valid.to_csv(out, index=False)
    print(f"  Results saved → {out}")
    print(f"{'═'*58}\n")
    return kappa


# ---------------------------------------------------------------------------
# Step 2 — Bulk annotate N chunks from corpus
# ---------------------------------------------------------------------------
def run_annotate(n: int):
    client = get_client()

    # IDs to exclude (already annotated)
    exclude = set()
    if BLIND_ANN_PATH.exists():
        exclude.update(pd.read_csv(BLIND_ANN_PATH)["chunk_id"].dropna())
    if LLM_ANN_PATH.exists():
        existing = pd.read_csv(LLM_ANN_PATH)
        exclude.update(existing["chunk_id"].dropna())
        print(f"Resuming — {len(existing):,} LLM annotations already done")

    print("Loading chunks ...")
    sub    = pd.read_parquet(DATA_DIR / "submissions_chunks.parquet", columns=CHUNK_COLS)
    com    = pd.read_parquet(DATA_DIR / "comments_chunks.parquet",    columns=CHUNK_COLS)
    chunks = pd.concat([sub, com], ignore_index=True).drop_duplicates("chunk_id")
    del sub, com

    chunks           = chunks[~chunks["chunk_id"].isin(exclude)].copy()
    chunks["text"]   = chunks["text"].fillna("").str.strip()
    chunks           = chunks[chunks["text"].str.len() > 0]
    chunks["has_sg"] = chunks["text"].apply(has_singlish)
    chunks["wc"]     = chunks["text"].apply(word_count)

    # Stratified sample matching corpus proportions
    n_sg   = int(n * 0.047)
    n_each = int((n - n_sg) / 4)
    n_long = n - n_sg - 3 * n_each  # catch rounding remainder

    strata = [
        ("singlish",      chunks["has_sg"],                                                                           n_sg),
        ("submissions",   (chunks["doc_type"]=="submission") & ~chunks["has_sg"],                                    n_each),
        ("short_comment", (chunks["doc_type"]=="comment") & ~chunks["has_sg"] & (chunks["wc"]<80),                  n_each),
        ("medium_comment",(chunks["doc_type"]=="comment") & ~chunks["has_sg"] & chunks["wc"].between(80,249),        n_each),
        ("long_comment",  (chunks["doc_type"]=="comment") & ~chunks["has_sg"] & (chunks["wc"]>=250),                n_long),
    ]

    parts, used = [], set()
    for name, mask, target in strata:
        g = chunks[mask & ~chunks["chunk_id"].isin(used)]
        s = g.sample(min(target, len(g)), random_state=42)
        s = s.copy(); s["stratum"] = name
        used.update(s["chunk_id"])
        parts.append(s)
        print(f"  {name:<18} {len(s):,} chunks")

    sample = (pd.concat(parts, ignore_index=True)
                .sample(frac=1, random_state=77)
                .reset_index(drop=True))
    print(f"\nTotal: {len(sample):,} chunks")
    print(f"Estimated time at {60/RATE_LIMIT_SLEEP:.0f} req/min: "
          f"~{len(sample)*RATE_LIMIT_SLEEP/3600:.1f} hrs\n")

    records, t0 = [], time.time()

    for i, (_, row) in enumerate(sample.iterrows()):
        label = call_llm(str(row["text"]), client)
        records.append({
            "chunk_id":  row["chunk_id"],
            "stratum":   row["stratum"],
            "doc_type":  row["doc_type"],
            "subreddit": row["subreddit"],
            "has_sg":    row["has_sg"],
            "wc":        row["wc"],
            "text":      row["text"],
            "llm_label": label,
            "source":    "llm",
        })

        time.sleep(RATE_LIMIT_SLEEP)

        if (i + 1) % 250 == 0 or (i + 1) == len(sample):
            _save_llm(records)
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(sample) - i - 1) / rate
            pct     = (i + 1) / len(sample) * 100
            print(f"  [{pct:>5.1f}%] {i+1:>6,}/{len(sample):,}  "
                  f"{rate:.1f}/s  ETA {eta/3600:.1f}h  — saved")

    _save_llm(records)
    failed = sum(1 for r in records if r["llm_label"] is None)
    print(f"\nDone — {len(records):,} annotated, {failed} failed.")


def _save_llm(records: list):
    new_df = pd.DataFrame(records)
    if LLM_ANN_PATH.exists():
        existing = pd.read_csv(LLM_ANN_PATH)
        new_df   = pd.concat([existing, new_df], ignore_index=True).drop_duplicates("chunk_id")
    new_df.to_csv(LLM_ANN_PATH, index=False)


# ---------------------------------------------------------------------------
# Step 3 — Build training dataset
# ---------------------------------------------------------------------------
def run_build_dataset():
    missing = [p for p in (BLIND_ANN_PATH, LLM_ANN_PATH) if not p.exists()]
    if missing:
        print(f"Missing files: {[p.name for p in missing]}")
        return

    human = pd.read_csv(BLIND_ANN_PATH)
    human = human[human["human_label"].notna() & (human["human_label"] != "skip")].copy()
    human = human.rename(columns={"human_label": "label"})
    human["source"] = "human"
    human["weight"] = 3.0   # human labels 3× more trustworthy

    llm = pd.read_csv(LLM_ANN_PATH)
    llm = llm[llm["llm_label"].notna()].copy()
    llm = llm.rename(columns={"llm_label": "label"})
    llm["source"] = "llm"
    llm["weight"] = 1.0
    llm = llm[~llm["chunk_id"].isin(set(human["chunk_id"]))]

    cols = ["chunk_id","stratum","doc_type","subreddit","has_sg","wc","text","label","source","weight"]
    combined = pd.concat([human[cols], llm[cols]], ignore_index=True)

    print(f"{'═'*50}")
    print(f"  singbert_train.csv summary")
    print(f"{'═'*50}")
    print(f"  Human labels : {len(human):,}  (weight 3×)")
    print(f"  LLM labels   : {len(llm):,}  (weight 1×)")
    print(f"  Total rows   : {len(combined):,}")
    print(f"\n  Label distribution:")
    vc = combined["label"].value_counts()
    for lbl in ["negative", "neutral", "positive"]:
        n = vc.get(lbl, 0)
        print(f"    {lbl:<10} {n:>6,}  ({n/len(combined)*100:.1f}%)")
    combined.to_csv(TRAIN_PATH, index=False)
    print(f"\n  Saved → {TRAIN_PATH}")
    print(f"{'═'*50}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LLM auto-annotation pipeline using Groq (free tier)."
    )
    parser.add_argument("--validate",      action="store_true",
                        help="Validate LLM vs human golden labels, compute Cohen's Kappa")
    parser.add_argument("--annotate",      type=int, metavar="N",
                        help="Bulk annotate N chunks (run after --validate passes)")
    parser.add_argument("--build-dataset", action="store_true",
                        help="Combine human + LLM labels into singbert_train.csv")
    args = parser.parse_args()

    if args.validate:
        run_validate()
    elif args.annotate:
        run_annotate(args.annotate)
    elif args.build_dataset:
        run_build_dataset()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
