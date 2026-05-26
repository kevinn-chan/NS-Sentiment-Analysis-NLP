"""
LLM-based auto-annotation pipeline for SingBERT fine-tuning golden dataset.

Uses Cerebras llama-3.3-70b to auto-annotate NS Reddit chunks.
Human blind labels (blind_annotation.csv) serve as the golden validation set.

Provider history (why we landed on Cerebras):
  - Groq 8B:         6k TPM hard wall → rate limits at avg 442 tok/call
  - Groq 70B:        1k RPD → exhausted in one session
  - OpenAI 4.1-mini: 200 RPD at Tier 0; tier upgrade requires $5 *consumed* via API
                     (having $50 in credit balance does NOT count — must actually spend it)
                     Full project costs $1.52, which never clears the $5 threshold.
                     Manual tier requests blocked for new orgs. Dead end.
  - Cerebras:        5 RPM, 30k TPM, 1M TPD, FREE <- current
                     Validation ~40 min, bulk 8k ~27 hrs (one overnight run)

Cost: $0

Model: llama-3.3-70b  (OpenAI-compatible API, strong instruction-following at 70B)

Setup:
    1. Get free API key at cloud.cerebras.ai
    2. export CEREBRAS_API_KEY=...
    3. pip install cerebras-cloud-sdk  (if not already installed)

Usage:
    # Step 1 — validate LLM against human labels, compute Cohen's Kappa
    python -m src.features.llm_annotator --validate

    # Step 2 — bulk annotate all 8000 chunks in one overnight run
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
BLIND_ANN_PATH  = DATA_DIR / "blind_annotation.csv"
HOLDOUT_PATH    = DATA_DIR / "holdout_test.csv"
LLM_ANN_PATH    = DATA_DIR / "llm_annotation.csv"
TRAIN_PATH      = DATA_DIR / "singbert_train.csv"

CHUNK_COLS = ["chunk_id", "doc_type", "subreddit", "text"]

# ---------------------------------------------------------------------------
# Cost tracker — updated from real token usage on every API call
# ---------------------------------------------------------------------------
MODEL_PRICING = {
    # (input $/M tokens, output $/M tokens)
    "gpt-4.1-mini":   (0.40,   1.60),
    "gpt-4.1":        (2.00,   8.00),
    "gpt-4o-mini":    (0.15,   0.60),
    "gpt-4o":         (2.50,  10.00),
    "gpt-4.5-preview":(75.00, 150.00),
}
_cost = {"total": 0.0, "last_alert": 0.0}

def _track(usage):
    """Record cost from a response usage object; print alert every $0.10."""
    in_price, out_price = MODEL_PRICING.get(API_MODEL, (0.40, 1.60))
    cost = (usage.prompt_tokens * in_price + usage.completion_tokens * out_price) / 1_000_000
    _cost["total"] += cost
    while _cost["total"] >= _cost["last_alert"] + 0.10:
        _cost["last_alert"] += 0.10
        print(f"  💰  ${_cost['last_alert']:.2f} spent")

# OpenAI config — Tier 3 account (10,000 RPM, no RPD cap)
API_BASE_URL     = "https://api.openai.com/v1"
API_MODEL        = "gpt-4.1"  # $2.00/M in, $8.00/M out → $7.57 total for validation+8k bulk
RATE_LIMIT_SLEEP = 0.5        # 120 RPM effective — trivially under 10,000 RPM Tier 3 cap

# Tier 3 limits for gpt-4.1-mini:
#   RPM: 10,000  |  TPM: 50,000,000  |  RPD: no cap
#   At 0.5s sleep + ~2s call time ≈ ~24 RPM actual throughput
#   Validation (197 calls): ~14 min
#   Bulk 8k   (8000 calls): ~9.3 hrs  (can increase sleep to taste, or drop to 0.1s for ~1.5 hrs)

# ---------------------------------------------------------------------------
# Few-shot examples — 4 targeted examples covering the hardest failure modes
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    # NEGATIVE — resentful/bitter warning (indirect negativity, not an explicit rant)
    ("Just be careful for snitches lah, some people will report you for the smallest thing. Watch your back in camp.",
     "negative"),

    # NEUTRAL — pure factual description, no personal stake or emotion
    ("During BMT the tekan sessions were intense. We would do pushups and leopard crawls. That's just how it works in the first few weeks.",
     "neutral"),

    # POSITIVE — humor and lighthearted tone counts as positive
    ("Our sergeant told us to do 100 pushups then forgot to count halfway through lol. Easy day for us sia.",
     "positive"),

    # POSITIVE — personal progress and achievement
    ("I see my IPPT timings decreasing steadily. Cut 30 seconds off my 2.4km run this week, shiok.",
     "positive"),

    # POSITIVE — lax/relaxed posting; mentions rules but dominant tone is happy/satisfied
    ("It was the most chill thing, our Encik was super lax and we could take offs whenever we wanted. Not exactly allowed but yeah, good times.",
     "positive"),
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a sentiment classifier for Singapore National Service (NS) Reddit posts.

Classify the AUTHOR'S emotional state — not the topic — as negative, neutral, or positive.

NEGATIVE: author complains, criticises, resents, expresses frustration, bitterness, or sarcasm with clear personal emotional investment.
NEUTRAL:  author shares facts, gives advice, or asks questions with no personal emotional stake. Reporting difficulty without editorialising = NEUTRAL.
POSITIVE: author feels satisfied, proud, relieved, grateful, amused, or excited — humor, lighthearted remarks, and describing a chill/easy/relaxed experience all count.

Key rules:
1. Classify the AUTHOR'S ATTITUDE, not the topic. An author calmly reporting a rule or hardship without complaint = NEUTRAL. An author happily describing a lax or easy experience = POSITIVE.
2. Mixed tone: weigh the DOMINANT emotion. If the overall vibe is positive/relaxed with one negative aside, choose POSITIVE.
3. WHEN IN DOUBT between neutral and positive → choose POSITIVE. Jokes, laughter (LOL, lol, haha), Singlish positivity (shiok, lepak, chill) = positive.
4. Indirect resentment or bitter warnings (e.g. "watch out for snitches", sarcasm) = NEGATIVE.
5. Personal progress, achievement, or improvement = POSITIVE.
6. A complaint phrased as a question is still NEGATIVE.
7. Singlish slang (lah, leh, sian, tekan, shiok, ORD, encik) is normal NS vocabulary — read tone, not just words.

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
                model=API_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                         + build_messages(text),
                max_tokens=15,
                temperature=0.0,
            )
            raw   = resp.choices[0].message.content.strip()
            if resp.usage:
                _track(resp.usage)
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
                # Distinguish RPD (daily) from RPM (per-minute) — different recovery strategy
                if "per day" in err.lower() or "rpd" in err.lower():
                    import datetime
                    now_utc  = datetime.datetime.utcnow()
                    reset_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) \
                                + datetime.timedelta(days=1)
                    wait_hrs = (reset_utc - now_utc).seconds / 3600
                    print(f"\n  ❌  DAILY LIMIT (RPD) EXHAUSTED — cannot recover by sleeping.")
                    print(f"  └─ {err[:220]}")
                    print(f"\n  Quota resets at midnight UTC  ({wait_hrs:.1f} hrs from now).")
                    print(f"  Progress saved — re-run --validate after reset to resume.\n")
                    return None   # exit call_llm; outer loop will see None and stop
                else:
                    # RPM hit — back off and retry
                    print(f"  Rate limit hit — waiting {rate_wait}s ...")
                    print(f"  └─ {err[:200]}")
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

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set.")
        print("  export OPENAI_API_KEY=sk_...")
        sys.exit(1)

    return OpenAI(api_key=api_key)


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

    out = DATA_DIR / "llm_validation.csv"

    # Resume from prior partial run — skip already-labelled chunk_ids
    already_done = {}
    if out.exists():
        prev = pd.read_csv(out).dropna(subset=["llm_label"])
        already_done = dict(zip(prev["chunk_id"], prev["llm_label"]))
        if already_done:
            print(f"Resuming — {len(already_done)} already labelled, skipping.")

    remaining = golden[~golden["chunk_id"].isin(already_done)].copy().reset_index(drop=True)
    print(f"Validating {len(remaining)} golden labels with {API_MODEL} ...")
    print(f"Estimated time: ~{len(remaining) * RATE_LIMIT_SLEEP / 60:.0f} min\n")

    new_labels = []
    t0         = time.time()

    for i, (_, row) in enumerate(remaining.iterrows()):
        label = call_llm(str(row["text"]), client)
        if label is None and i == 0:
            # First call failed — likely RPD exhaustion; message already printed
            return None
        new_labels.append(label)
        time.sleep(RATE_LIMIT_SLEEP)

        if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed          # req/s
            eta     = (len(remaining) - i - 1) / rate
            rpm     = rate * 60
            print(f"  {i+1}/{len(remaining)}  {rpm:.1f} RPM  ETA {eta/60:.0f}min  [${_cost['total']:.3f} spent]")
            # Incremental save — safe to kill at any 25-chunk boundary
            partial = remaining.iloc[: i + 1].copy()
            partial["llm_label"] = new_labels
            # Merge with prior run and save
            combined = pd.concat(
                [pd.DataFrame([{"chunk_id": cid, "llm_label": lbl} for cid, lbl in already_done.items()]),
                 partial[["chunk_id", "llm_label"]].dropna(subset=["llm_label"])],
                ignore_index=True,
            )
            golden.merge(combined, on="chunk_id", how="left").dropna(
                subset=["llm_label"]
            ).to_csv(out, index=False)

    remaining["llm_label"] = new_labels

    # Rebuild full golden with all labels (prior + new)
    # Use object dtype explicitly to avoid pandas float64→string coercion error
    golden["llm_label"] = golden["chunk_id"].map(already_done).astype(object)
    for _, row in remaining.iterrows():
        golden.loc[golden["chunk_id"] == row["chunk_id"], "llm_label"] = row["llm_label"]

    valid = golden.dropna(subset=["llm_label"])
    failed = len(golden) - len(valid)

    kappa = cohen_kappa_score(valid["human_label"], valid["llm_label"])

    print(f"\n{'═'*58}")
    print(f"  Kappa Validation — {API_MODEL}")
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
                  f"{rate:.1f}/s  ETA {eta/3600:.1f}h  ${_cost['total']:.2f} spent  — saved")

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
    # Exclude human-labelled chunks (already in train at 3×) and holdout chunks (eval only)
    excluded_ids = set(human["chunk_id"])
    if HOLDOUT_PATH.exists():
        excluded_ids |= set(pd.read_csv(HOLDOUT_PATH)["chunk_id"])
    llm = llm[~llm["chunk_id"].isin(excluded_ids)]

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
