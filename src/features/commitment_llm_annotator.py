"""
LLM-based dual-axis commitment annotator for NS Reddit chunks.

Replaces unreliable BART C2D scores (kappa=0.127, see commitment_annotation.csv)
with GPT-4.1-mini few-shot classification on TWO independent axes:

Axis 1 — BUYIN (personal investment):
  committed   : author is personally invested, takes NS seriously, puts in effort
  uncommitted : author is disengaged — apathetic, just clearing time, minimal effort
  neutral     : no signal of the author's own buy-in level

Axis 2 — STANCE (institutional opinion):
  supportive : author endorses NS as worthwhile, necessary for defence, a positive institution
  critical   : author opposes NS — unfair, waste, exploitative, should be abolished, caused harm
  neutral    : discusses NS without expressing institutional opinion

These axes are INDEPENDENT:
  "NS made me who I am"                      → committed buyin + supportive stance
  "NS is necessary but I just want to ORD"   → uncommitted buyin + supportive stance
  "I gave NS my all but the system is broken" → committed buyin + critical stance
  "Only here to finish 2 years, zao liao"     → uncommitted buyin + neutral stance

Combined metric (computed downstream, not labelled):
  positive = committed OR supportive (either axis positive)
  negative = uncommitted OR critical (either axis negative)

KEY DISTINCTION from sentiment annotator (llm_annotator.py):
  Sentiment = author's emotional state (frustrated, happy, neutral)
  Buyin     = author's personal investment in their own service
  Stance    = author's institutional opinion on NS as a policy
  All three are INDEPENDENT: someone can be frustrated (negative sentiment) but still
  believe NS is necessary (supportive stance) and give it their all (committed buyin).

Budget:
  ~$5-7 for 10,500 annual samples (3 subreddits × 7 years × 500 chunks/cell)
  ~$0.05 for validation against 100 human annotations
  Hard cap enforced at $10.

Rate limits (Tier 3, 10,000 RPM):
  RATE_LIMIT_SLEEP=0.1s → ~200-300 RPM actual throughput (API latency bottleneck)
  10,500 chunks at 200 RPM → ~53 min

Usage:
    # Step 1 — validate GPT-4.1-mini against 100 human labels, check kappa
    python -m src.features.commitment_llm_annotator --validate

    # Step 2 — annotate annual stratified sample (can resume if interrupted)
    python -m src.features.commitment_llm_annotator --annotate

    # Step 3 — compute and print annual commitment trend with significance test
    python -m src.features.commitment_llm_annotator --trend

Corpus spans 2018–2025 (8 years). 2025 is a complete year (113k chunks, same scale as 2024).
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR        = Path(__file__).parent.parent.parent / "data" / "processed" / "new"
COMMIT_ANN_PATH = DATA_DIR / "commitment_annotation.csv"       # 100 human labels
COMMIT_LLM_VAL  = DATA_DIR / "commitment_llm_validation.csv"  # LLM vs human kappa
COMMIT_LLM_ANN  = DATA_DIR / "commitment_llm_annual.csv"       # annotated annual sample
COMMIT_TREND    = DATA_DIR / "commitment_trend.csv"             # aggregated per-year trend

CHUNK_COLS = ["chunk_id", "doc_type", "subreddit", "text", "created_utc"]

# Subreddits to include — must be tracked individually for trend analysis
TARGET_SUBS = ["NationalServiceSG", "singapore", "askSingapore"]

# Samples per (year × subreddit) cell
# 3 subs × 7 years × 500 = 10,500 max → ~$5.75
SAMPLES_PER_CELL = 500

# ---------------------------------------------------------------------------
# API config — OpenAI Tier 3
# ---------------------------------------------------------------------------
API_MODEL        = "gpt-4.1-mini"
RATE_LIMIT_SLEEP = 0.1        # s; Tier 3 can handle 10,000 RPM — actual throughput limited by latency
BUDGET_HARD_STOP = 5.0        # USD — hard cap; run aborts if exceeded

_cost      = {"total": 0.0, "last_warn": 0.0}
_cost_lock = threading.Lock()

MODEL_PRICING = {
    # (input $/M tokens, output $/M tokens)
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1":      (2.00, 8.00),
    "gpt-4o-mini":  (0.15, 0.60),
}


def _track(usage) -> bool:
    """Record cost from API usage object. Returns False if budget cap exceeded."""
    in_p, out_p = MODEL_PRICING.get(API_MODEL, (0.40, 1.60))
    cost = (usage.prompt_tokens * in_p + usage.completion_tokens * out_p) / 1_000_000
    with _cost_lock:
        _cost["total"] += cost
        if _cost["total"] >= _cost["last_warn"] + 1.0:
            _cost["last_warn"] = (_cost["total"] // 1.0) * 1.0
            print(f"  💰  ${_cost['total']:.2f} spent")
        over_budget = _cost["total"] >= BUDGET_HARD_STOP
    if over_budget:
        print(f"\n  ❌  Budget cap ${BUDGET_HARD_STOP:.0f} reached — stopping.")
        return False
    return True


# ---------------------------------------------------------------------------
# System prompt
# Explicitly distinguishes institutional stance from emotional sentiment.
# "When in doubt → NEUTRAL" pushes the model away from BART's false-positive bias.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are classifying Reddit comments about Singapore's National Service (NS) on TWO independent axes. This is NOT about emotional sentiment.

═══ AXIS 1: BUYIN — how invested is the author in their own service? ═══

COMMITTED: The author is personally invested. They take NS seriously, put in genuine effort, endorse it as worth doing, or see real value in it.
UNCOMMITTED: The author is NOT invested — apathetic / disengaged: just here to clear time, minimal effort, keng/slack mindset, no buy-in.
  e.g. "only here to finish 2 years, zao liao, don't care", "chao keng my way through", "just counting down to ORD"
NEUTRAL: The author discusses NS practically without revealing personal buy-in level.

UNCOMMITTED includes (no explicit "I don't care" needed):
  ✓ Author describes their OWN keng/slack approach as a strategy or attitude
  ✓ Explicitly clearing time without caring ("finish 2 years and go", "zao the moment ORD")
  ✓ Counting down or dismissing NS value ("just a number to clear", "doing the bare minimum")
  ✓ Signing on for money/perks rather than belief ("signed for the money") = UNCOMMITTED buyin

⚠️ CRITICAL RULE — narrating your NS experience in detail does NOT mean committed:
  - Someone who signed on for money and dropped out = UNCOMMITTED (financial motivation ≠ belief)
  - A veteran who says "NS should be abolished" = UNCOMMITTED (past service ≠ current buy-in)
  - Working hard during NS because you have to = NEUTRAL (compliance ≠ endorsement)
  - Being in distress or suffering during NS = NEUTRAL (hardship ≠ commitment)
  - Criticising NS conditions while still serving = NEUTRAL or UNCOMMITTED, NOT committed
  COMMITTED requires the author to EXPLICITLY endorse their own service commitment, express pride, or sign/stay on out of genuine belief.

BUYIN NEUTRAL — do NOT label these uncommitted or committed:
  - "Got one chao keng soldier in my section" → describing OTHERS' behaviour = buyin NEUTRAL
  - Personal frustrations or complaints during NS (bad food, tough training) without revealing buy-in = buyin NEUTRAL
  - Questions, advice, logistics, vocations, admin = buyin NEUTRAL
  - "Finally ORD!" or relief at finishing = buyin NEUTRAL (relief ≠ disengagement)
  - Discussing keng/wayang culture as an observation or joke without the author endorsing it = buyin NEUTRAL
  - Correcting factual claims about NS = buyin NEUTRAL
  - Describing suffering, distress, or mental health struggles = buyin NEUTRAL

If you cannot clearly identify the author's personal buy-in level, choose buyin NEUTRAL.

═══ AXIS 2: STANCE — does the author endorse or oppose NS as an institution? ═══

SUPPORTIVE: Author expresses belief NS is worthwhile, necessary for Singapore's defence, or a positive institution.
  e.g. "We need NS for defence", "NS made me who I am, glad I served", "necessary sacrifice"
CRITICAL: Author expresses NS is unfair, a waste, exploitative, should be changed/abolished, or caused lasting harm.
  e.g. "waste of 2 years", "slavery", "should be abolished", "doesn't translate to real life at all"
NEUTRAL: Discusses NS without expressing institutional opinion.

STANCE NEUTRAL — do NOT label these supportive or critical:
  - Describing OTHERS' views on NS = stance NEUTRAL
  - Personal frustrations (bad food, tough training) without institutional critique = stance NEUTRAL
  - Apathy without institutional opinion ("just want to ORD" with no view on NS itself) = stance NEUTRAL

If you cannot clearly identify the author's institutional opinion, choose stance NEUTRAL.

═══ KEY: THESE AXES ARE INDEPENDENT ═══

"NS made me who I am, glad I served" → buyin=committed, stance=supportive (both positive)
"NS is necessary for defence but I just want to clear my 2 years" → buyin=uncommitted, stance=supportive (mixed!)
"I gave NS my all but the system is broken and exploitative" → buyin=committed, stance=critical (mixed!)
"Only here to finish 2 years, zao liao" → buyin=uncommitted, stance=neutral (apathy, no institutional opinion)
"waste of 2 years, should be abolished" → buyin=uncommitted, stance=critical (both negative)
Describing OTHERS' views → buyin=neutral, stance=neutral

Reply with ONLY this JSON, nothing else:
{"buyin": "committed"|"uncommitted"|"neutral", "stance": "supportive"|"critical"|"neutral"}"""


# ---------------------------------------------------------------------------
# Few-shot examples — drawn from commitment_annotation.csv (human-labelled)
# Selected to demonstrate the hardest discrimination cases.
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    # COMMITTED buyin + SUPPORTIVE stance — geopolitical rationale; institutional endorsement
    (
        "They killed police and villagers. With such neighbours, I would want NS around.",
        "committed", "supportive"
    ),
    # COMMITTED buyin + SUPPORTIVE stance — defending NS/govt institutions against trust erosion
    (
        "This seems like a concerted effort to erode trust in govt institutions or SAF specifically…",
        "committed", "supportive"
    ),
    # UNCOMMITTED buyin + NEUTRAL stance — canonical apathy: clearing time, no institutional opinion
    (
        "We are only here finish 2 years, zao liao. Don't need to care whatsoever impression.",
        "uncommitted", "neutral"
    ),
    # UNCOMMITTED buyin + CRITICAL stance — keng/slack + economic critique of NS as institution
    (
        "lol he wayang also no use, salary still the same. do the bare minimum in ns doesn't "
        "really translate to the working world, cos in the working world we don't get paid in peanuts.",
        "uncommitted", "critical"
    ),
    # UNCOMMITTED buyin + CRITICAL stance — explicit anti-conscription institutional position
    (
        "If I were born in a country without conscription, my family and my life would be protected "
        "by a professional army. What is the point of living if the SAF sucks it out of you?",
        "uncommitted", "critical"
    ),
    # UNCOMMITTED buyin + CRITICAL stance — direct call to end conscription
    (
        "Seems like Societal cost of male enlistment is too high right now. No one wanna bear it so end conscription",
        "uncommitted", "critical"
    ),
    # UNCOMMITTED buyin + CRITICAL stance — economic critique: NS has no real-world value
    (
        'It\'s far from "not that much". If not for NS, I\'d expect to make at least $3k/month with my '
        "IT diploma. That's almost 5 times more than what I get for serving NS.",
        "uncommitted", "critical"
    ),
    # UNCOMMITTED buyin + CRITICAL stance — NS caused lasting personal harm
    (
        "Became dumb af... Took me at least 6months to recover from NS bullshit....after ORD",
        "uncommitted", "critical"
    ),
    # UNCOMMITTED buyin + NEUTRAL stance — signed on for money not belief; dropped out (OOC); no institutional view
    (
        "signed for the money, ooc cause the course is fkin hard, and the course commander are asshole. "
        "no welfare at all :( 5am to 9pm daily is not fun",
        "uncommitted", "neutral"
    ),
    # UNCOMMITTED buyin + CRITICAL stance — veteran opposing NS; past service ≠ current committed buyin
    (
        "Having served NS myself in a combat role, NS should be abolished lah. Most the time all the "
        "fuckups are from people that shouldn't be there in the first place.",
        "uncommitted", "critical"
    ),
    # NEUTRAL buyin + NEUTRAL stance — NS career advice; no signal on either axis
    (
        "Your best bet is to express interest during the recruitment talks and do well in bmt to get to OCS. "
        "From there it will be quite easy to sign on.",
        "neutral", "neutral"
    ),
    # NEUTRAL buyin + NEUTRAL stance — correcting a factual claim; no personal position on either axis
    (
        "Wtf don't listen to him, he's making it sound as if NS is this life or death ordeal. "
        "The chance of getting a permanent injury or dying while in NS is still very low",
        "neutral", "neutral"
    ),
]


def build_messages(text: str) -> list:
    """Build few-shot message list for one chunk (dual-axis: buyin + stance)."""
    msgs = []
    for ex_text, ex_buyin, ex_stance in FEW_SHOT_EXAMPLES:
        msgs.append({"role": "user",      "content": ex_text[:800]})
        msgs.append({"role": "assistant", "content": json.dumps({"buyin": ex_buyin, "stance": ex_stance})})
    msgs.append({"role": "user", "content": str(text)[:1500]})
    return msgs


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("Run: pip install openai")
        sys.exit(1)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set.\n  export OPENAI_API_KEY=sk_...")
        sys.exit(1)
    return OpenAI(api_key=key)


VALID_BUYIN  = {"committed", "uncommitted", "neutral"}
VALID_STANCE = {"supportive", "critical", "neutral"}


def call_llm(text: str, client, retries: int = 3) -> tuple[str, str] | None:
    """Call GPT-4.1-mini; return (buyin, stance) tuple or None on failure.

    Rate limit hits back off with exponential delay (separate from parse retry budget).
    Returns None on budget exhaustion (caller should stop the loop).
    """
    parse_attempts = 0
    rate_wait      = 30   # initial backoff seconds

    while parse_attempts < retries:
        try:
            resp  = client.chat.completions.create(
                model       = API_MODEL,
                messages    = [{"role": "system", "content": SYSTEM_PROMPT}]
                             + build_messages(text),
                max_tokens  = 40,
                temperature = 0.0,
            )
            raw   = resp.choices[0].message.content.strip()
            if resp.usage:
                if not _track(resp.usage):
                    return None   # budget exhausted
            parsed = json.loads(raw)
            buyin  = parsed.get("buyin", "").lower()
            stance = parsed.get("stance", "").lower()
            if buyin in VALID_BUYIN and stance in VALID_STANCE:
                rate_wait = 30   # reset backoff on success
                return (buyin, stance)
            print(f"  ⚠️  Unexpected labels: {raw!r}")
            parse_attempts += 1

        except json.JSONDecodeError:
            parse_attempts += 1
            print(f"  ⚠️  JSON error (attempt {parse_attempts}): {raw!r}")

        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                print(f"  Rate limit — waiting {rate_wait}s ...")
                time.sleep(rate_wait)
                rate_wait = min(rate_wait * 2, 300)
            else:
                parse_attempts += 1
                print(f"  ⚠️  API error (attempt {parse_attempts}): {e}")
                time.sleep(min(2 ** parse_attempts, 30))

    return None


# ---------------------------------------------------------------------------
# Step 1 — Validate against 100 human commitment annotations
# ---------------------------------------------------------------------------
def run_validate():
    """Validate dual-axis LLM labels against 100-row commitment_annotation.csv.

    NOTE: The old ground-truth has a single `human_label` column from the pre-pivot
    scheme (committed/critical/neutral). We map imperfectly:
      old "committed" → buyin=committed
      old "critical"  → stance=critical
      old "neutral"   → neutral on both axes
    This is a rough proxy — the real validation is the 200-row test set.
    """
    try:
        from sklearn.metrics import cohen_kappa_score, classification_report
    except ImportError:
        print("Run: pip install scikit-learn")
        return

    if not COMMIT_ANN_PATH.exists():
        print(f"Missing: {COMMIT_ANN_PATH.name}")
        print("  Run:  python -m src.features.commitment_annotator")
        return

    client = get_client()
    human  = pd.read_csv(COMMIT_ANN_PATH)
    human  = human[human["human_label"].notna()].copy().reset_index(drop=True)

    # Resume support — keyed on chunk_id, stores (buyin, stance) tuples
    already_done: dict[str, tuple[str, str]] = {}
    if COMMIT_LLM_VAL.exists():
        prev = pd.read_csv(COMMIT_LLM_VAL).dropna(subset=["llm_buyin"])
        for _, r in prev.iterrows():
            already_done[r["chunk_id"]] = (r["llm_buyin"], r["llm_stance"])
        if already_done:
            print(f"Resuming — {len(already_done)}/{len(human)} already labelled.")

    remaining = human[~human["chunk_id"].isin(already_done)].copy().reset_index(drop=True)
    est_cost  = len(remaining) * 0.00055
    print(f"Validating {len(remaining)} annotations with {API_MODEL} ...")
    print(f"Estimated cost: ~${est_cost:.2f}  |  "
          f"Estimated time: ~{len(remaining) * RATE_LIMIT_SLEEP / 60:.0f} min\n")

    new_results: list[tuple[str, str] | None] = []
    t0 = time.time()

    for i, (_, row) in enumerate(remaining.iterrows()):
        result = call_llm(str(row["text"]), client)
        new_results.append(result)
        time.sleep(RATE_LIMIT_SLEEP)

        if result is None and _cost["total"] >= BUDGET_HARD_STOP:
            break

        if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - t0
            rpm     = (i + 1) / elapsed * 60
            print(f"  {i+1}/{len(remaining)}  {rpm:.0f} RPM  ${_cost['total']:.3f}")
            # Incremental save
            partial = remaining.iloc[:i + 1].copy()
            partial["llm_buyin"]  = [r[0] if r else None for r in new_results]
            partial["llm_stance"] = [r[1] if r else None for r in new_results]
            rows_prev = [{"chunk_id": c, "llm_buyin": v[0], "llm_stance": v[1]}
                         for c, v in already_done.items()]
            combined = pd.concat([
                pd.DataFrame(rows_prev),
                partial[["chunk_id", "llm_buyin", "llm_stance"]].dropna(subset=["llm_buyin"])
            ], ignore_index=True)
            human.merge(combined, on="chunk_id", how="left").dropna(
                subset=["llm_buyin"]
            ).to_csv(COMMIT_LLM_VAL, index=False)

    remaining["llm_buyin"]  = [r[0] if r else None for r in new_results]
    remaining["llm_stance"] = [r[1] if r else None for r in new_results]

    # Merge into full human df
    buyin_map  = {c: v[0] for c, v in already_done.items()}
    stance_map = {c: v[1] for c, v in already_done.items()}
    human["llm_buyin"]  = human["chunk_id"].map(buyin_map).astype(object)
    human["llm_stance"] = human["chunk_id"].map(stance_map).astype(object)
    for _, row in remaining.iterrows():
        mask = human["chunk_id"] == row["chunk_id"]
        human.loc[mask, "llm_buyin"]  = row["llm_buyin"]
        human.loc[mask, "llm_stance"] = row["llm_stance"]

    valid  = human.dropna(subset=["llm_buyin"])
    failed = len(human) - len(valid)

    # --- Map old single-axis human_label to both axes for kappa ---
    # old "committed" → buyin=committed; old "critical" → stance=critical; old "neutral" → neutral
    valid["human_buyin_mapped"]  = valid["human_label"].map(
        {"committed": "committed", "critical": "neutral", "neutral": "neutral"}
    ).fillna("neutral")
    valid["human_stance_mapped"] = valid["human_label"].map(
        {"committed": "neutral", "critical": "critical", "neutral": "neutral"}
    ).fillna("neutral")

    kappa_buyin  = cohen_kappa_score(valid["human_buyin_mapped"],  valid["llm_buyin"])
    kappa_stance = cohen_kappa_score(valid["human_stance_mapped"], valid["llm_stance"])
    acc_buyin    = (valid["human_buyin_mapped"]  == valid["llm_buyin"]).mean()
    acc_stance   = (valid["human_stance_mapped"] == valid["llm_stance"]).mean()

    print(f"\n{'═'*68}")
    print(f"  Dual-Axis Commitment Validation — {API_MODEL}")
    print(f"{'═'*68}")
    print(f"  ⚠️  NOTE: Old ground-truth has single-axis labels (committed/critical/neutral).")
    print(f"  Mapping is imperfect — real validation is the 200-row test set.\n")
    print(f"  Chunks evaluated : {len(valid)}  ({failed} failed)")
    print(f"\n  BUYIN axis (committed/uncommitted/neutral):")
    print(f"    Accuracy      : {acc_buyin:.3f}  ({acc_buyin*100:.1f}%)")
    print(f"    Cohen's Kappa : {kappa_buyin:.3f}")
    print(f"\n  STANCE axis (supportive/critical/neutral):")
    print(f"    Accuracy      : {acc_stance:.3f}  ({acc_stance*100:.1f}%)")
    print(f"    Cohen's Kappa : {kappa_stance:.3f}")
    print(f"\n  BART baseline   : 0.127  (for reference)")

    if min(kappa_buyin, kappa_stance) >= 0.70:
        verdict = "✅  GOOD — proceed to --annotate"
    elif min(kappa_buyin, kappa_stance) >= 0.55:
        verdict = "⚠️  MODERATE — acceptable for trend analysis, proceed to --annotate"
    elif min(kappa_buyin, kappa_stance) >= 0.40:
        verdict = "⚠️  FAIR — better than BART; consider refining prompt before --annotate"
    else:
        verdict = "❌  POOR — revise prompt before proceeding (but mapping is imperfect)"
    print(f"  Verdict         : {verdict}")

    print(f"\n  Per-class accuracy — BUYIN (LLM vs mapped human):")
    for lbl in ["committed", "uncommitted", "neutral"]:
        sub   = valid[valid["human_buyin_mapped"] == lbl]
        agree = (sub["llm_buyin"] == lbl).mean() if len(sub) else 0.0
        print(f"    {lbl:<12}  {agree:.1%}  (n={len(sub)})")

    print(f"\n  Per-class accuracy — STANCE (LLM vs mapped human):")
    for lbl in ["supportive", "critical", "neutral"]:
        sub   = valid[valid["human_stance_mapped"] == lbl]
        agree = (sub["llm_stance"] == lbl).mean() if len(sub) else 0.0
        print(f"    {lbl:<12}  {agree:.1%}  (n={len(sub)})")

    report_buyin = classification_report(
        valid["human_buyin_mapped"], valid["llm_buyin"],
        labels=["committed", "uncommitted", "neutral"], digits=3, zero_division=0
    )
    report_stance = classification_report(
        valid["human_stance_mapped"], valid["llm_stance"],
        labels=["supportive", "critical", "neutral"], digits=3, zero_division=0
    )
    print(f"\n  BUYIN classification report:\n{report_buyin}")
    print(f"  STANCE classification report:\n{report_stance}")

    valid.to_csv(COMMIT_LLM_VAL, index=False)
    print(f"  Results → {COMMIT_LLM_VAL}")
    print(f"  Total cost: ${_cost['total']:.3f}")
    print(f"{'═'*68}\n")
    return (kappa_buyin, kappa_stance)


# ---------------------------------------------------------------------------
# Step 2 — Annotate annual stratified sample
# Stratification: 500 chunks per (year × subreddit) cell
# Covers 2018–2024 across NationalServiceSG, singapore, askSingapore
# ---------------------------------------------------------------------------
def run_annotate():
    client = get_client()

    # IDs already annotated — used to exclude from pool
    exclude = set()
    if COMMIT_ANN_PATH.exists():
        exclude.update(pd.read_csv(COMMIT_ANN_PATH)["chunk_id"].dropna())

    # Per-cell counts — used to decide how many MORE we need per (year × subreddit)
    cell_done: dict[tuple, int] = {}
    if COMMIT_LLM_ANN.exists():
        existing = pd.read_csv(COMMIT_LLM_ANN).dropna(subset=["llm_buyin"])
        exclude.update(existing["chunk_id"].dropna())
        for (yr, sr), grp in existing.groupby(["year", "subreddit"]):
            cell_done[(int(yr), str(sr))] = len(grp)
        n_cells_full = sum(1 for v in cell_done.values() if v >= SAMPLES_PER_CELL)
        print(f"Resuming — {len(existing):,} already annotated, "
              f"{n_cells_full} cells complete.")

    print("Loading chunk parquets ...")
    sub_df = pd.read_parquet(DATA_DIR / "submissions_chunks.parquet", columns=CHUNK_COLS)
    com_df = pd.read_parquet(DATA_DIR / "comments_chunks.parquet",   columns=CHUNK_COLS)
    pool   = pd.concat([sub_df, com_df], ignore_index=True).drop_duplicates("chunk_id")
    del sub_df, com_df

    pool["text"] = pool["text"].fillna("").str.strip()
    pool = pool[pool["text"].str.len() > 0].copy()
    pool["year"] = pd.to_datetime(
        pool["created_utc"], utc=True, errors="coerce"
    ).dt.year
    pool = pool[
        (pool["year"] >= 2018) & (pool["year"] <= 2025) &
        pool["subreddit"].isin(TARGET_SUBS) &
        ~pool["chunk_id"].isin(exclude)
    ].copy()

    # Build stratified sample — only top up cells that are below SAMPLES_PER_CELL
    parts = []
    for yr in sorted(pool["year"].unique()):
        for sr in TARGET_SUBS:
            already = cell_done.get((int(yr), str(sr)), 0)
            need    = max(0, SAMPLES_PER_CELL - already)
            if need == 0:
                print(f"  {yr}  {sr:<22}  ✓ complete ({already} done)")
                continue
            cell = pool[(pool["year"] == yr) & (pool["subreddit"] == sr)]
            n    = min(need, len(cell))
            if n > 0:
                s = cell.sample(n, random_state=42).copy()
                parts.append(s)
                status = f"top-up {already}→{already+n}" if already > 0 else "new"
                print(f"  {yr}  {sr:<22}  {n:>4,}  ({status})")
            elif already > 0:
                print(f"  {yr}  {sr:<22}  pool exhausted ({already} done, need {need} more)")

    sample = (pd.concat(parts, ignore_index=True)
                .sample(frac=1, random_state=77)
                .reset_index(drop=True))

    est_cost = len(sample) * 0.00055
    est_min  = len(sample) / 200
    print(f"\nTotal: {len(sample):,} chunks")
    print(f"Estimated cost : ~${est_cost:.2f}")
    print(f"Estimated time : ~{est_min:.0f} min at 200 RPM\n")

    records, t0 = [], time.time()

    for i, (_, row) in enumerate(sample.iterrows()):
        result = call_llm(str(row["text"]), client)
        records.append({
            "chunk_id":  row["chunk_id"],
            "year":      int(row["year"]),
            "subreddit": row["subreddit"],
            "doc_type":  row["doc_type"],
            "text":      row["text"],
            "llm_buyin":  result[0] if result else None,
            "llm_stance": result[1] if result else None,
        })
        time.sleep(RATE_LIMIT_SLEEP)

        # Budget exhausted
        if result is None and _cost["total"] >= BUDGET_HARD_STOP:
            _save_annual(records)
            print("Budget cap hit — progress saved. Re-run --annotate to resume.")
            return

        if (i + 1) % 500 == 0 or (i + 1) == len(sample):
            _save_annual(records)
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(sample) - i - 1) / rate
            pct     = (i + 1) / len(sample) * 100
            n_fail  = sum(1 for r in records if r["llm_buyin"] is None)
            print(f"  [{pct:>5.1f}%] {i+1:>6,}/{len(sample):,}  "
                  f"{rate:.1f}/s  ETA {eta/60:.0f}min  "
                  f"${_cost['total']:.2f}  ({n_fail} failed) — saved")

    failed = sum(1 for r in records if r["llm_buyin"] is None)
    print(f"\nDone — {len(records):,} annotated, {failed} failed.")
    print(f"Total cost: ${_cost['total']:.3f}")
    print(f"\nNext: python -m src.features.commitment_llm_annotator --trend")


# ---------------------------------------------------------------------------
# Step 2b — Annotate the minority-enriched enrich queue
# Reads commitment_llm_enrich_queue.csv, writes llm_label back in place.
# Resume-safe: skips rows that already have a label.
# ---------------------------------------------------------------------------
def run_annotate_enrich(max_workers: int = 20):
    enrich_path = DATA_DIR / "commitment_llm_enrich_queue.csv"
    if not enrich_path.exists():
        print(f"Missing: {enrich_path.name}")
        print("  Run:  python -m src.features.commitment_sampler")
        return

    client  = get_client()
    df      = pd.read_csv(enrich_path)
    df_lock = threading.Lock()

    for col in ("llm_buyin", "llm_stance"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    todo = df[df["llm_buyin"].str.len() == 0].copy()
    already_done = len(df) - len(todo)
    if already_done:
        print(f"Resuming — {already_done:,}/{len(df):,} already labelled.")

    est_cost = len(todo) * 0.00055
    est_min  = len(todo) / (max_workers / 2.5)   # ~2.5s avg latency per call
    print(f"Labelling {len(todo):,} chunks with {API_MODEL} ({max_workers} workers) ...")
    print(f"Estimated cost : ~${est_cost:.2f}")
    print(f"Estimated time : ~{est_min:.0f} min\n")

    t0        = time.time()
    completed = 0
    save_lock = threading.Lock()

    def _label(idx_row):
        idx, row = idx_row
        result = call_llm(str(row["text"]), client)
        return idx, result

    def _save():
        with df_lock:
            df.to_csv(enrich_path, index=False)

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for idx, row in todo.iterrows():
            futures[pool.submit(_label, (idx, row))] = idx

        for future in as_completed(futures):
            idx, result = future.result()

            with df_lock:
                df.at[idx, "llm_buyin"]  = result[0] if result else ""
                df.at[idx, "llm_stance"] = result[1] if result else ""
                completed += 1
                n_done = completed

            if _cost["total"] >= BUDGET_HARD_STOP:
                _save()
                print("Budget cap hit — progress saved. Re-run --annotate-enrich to resume.")
                pool.shutdown(wait=False, cancel_futures=True)
                return

            if n_done % 500 == 0 or n_done == len(todo):
                _save()
                elapsed = time.time() - t0
                rate    = n_done / elapsed
                eta     = (len(todo) - n_done) / rate
                pct     = n_done / len(todo) * 100
                with df_lock:
                    n_fail = (df["llm_buyin"] == "").sum()
                print(f"  [{pct:>5.1f}%] {n_done:>6,}/{len(todo):,}  "
                      f"{rate:.1f}/s  ETA {eta/60:.0f}min  "
                      f"${_cost['total']:.2f}  ({n_fail} unlabelled) — saved")

    done = df[df["llm_buyin"].str.len() > 0]
    print(f"\nDone — {len(done):,} labelled.")
    print(f"Buyin distribution:  {done['llm_buyin'].value_counts().to_dict()}")
    print(f"Stance distribution: {done['llm_stance'].value_counts().to_dict()}")
    print(f"Total cost: ${_cost['total']:.3f}")
    print(f"\nNext: merge into training data and run distillation.")


def _save_annual(records: list):
    new_df = pd.DataFrame(records)
    if COMMIT_LLM_ANN.exists():
        existing = pd.read_csv(COMMIT_LLM_ANN)
        new_df   = pd.concat([existing, new_df], ignore_index=True).drop_duplicates("chunk_id")
    # Drop legacy llm_label column if present (pre-dual-axis runs)
    if "llm_label" in new_df.columns and "llm_buyin" in new_df.columns:
        new_df = new_df.drop(columns=["llm_label"])
    new_df.to_csv(COMMIT_LLM_ANN, index=False)


# ---------------------------------------------------------------------------
# Step 3 — Compute and display annual commitment trend
# ---------------------------------------------------------------------------
def run_trend():
    try:
        from scipy.stats import linregress, spearmanr
    except ImportError:
        print("Run: pip install scipy")
        return

    if not COMMIT_LLM_ANN.exists():
        print("Missing commitment_llm_annual.csv — run --annotate first.")
        return

    df = pd.read_csv(COMMIT_LLM_ANN).dropna(subset=["llm_buyin"])
    print(f"Loaded {len(df):,} labelled chunks")
    print(f"Buyin distribution:  {df['llm_buyin'].value_counts().to_dict()}")
    print(f"Stance distribution: {df['llm_stance'].value_counts().to_dict()}\n")

    # --- BUYIN TREND ---
    buyin_rows = []
    for yr, grp in df.groupby("year"):
        n   = len(grp)
        p_c = (grp["llm_buyin"] == "committed").mean()
        p_u = (grp["llm_buyin"] == "uncommitted").mean()
        p_n = (grp["llm_buyin"] == "neutral").mean()
        ci_u = 1.96 * np.sqrt(p_u * (1 - p_u) / n) if n > 0 else 0.0
        buyin_rows.append({
            "year": yr, "n": n,
            "pct_committed": p_c, "pct_uncommitted": p_u, "pct_buyin_neutral": p_n,
            "uncommitted_ci95": ci_u, "net_buyin": p_c - p_u,
        })
    buyin_trend = pd.DataFrame(buyin_rows).sort_values("year")

    print(f"{'═'*72}")
    print(f"  BUYIN Trend — NS Reddit  ({API_MODEL})")
    print(f"{'═'*72}")
    print(f"  {'Year':>4}  {'N':>5}  {'Committed':>10}  {'Uncommitted':>12}  "
          f"{'Neutral':>8}  {'Net':>8}")
    print(f"  {'─'*68}")
    for _, r in buyin_trend.iterrows():
        net_sym = "▲" if r["net_buyin"] > 0 else "▼"
        print(f"  {int(r['year']):>4}  {int(r['n']):>5}  "
              f"{r['pct_committed']:>9.1%}  {r['pct_uncommitted']:>11.1%}  "
              f"{r['pct_buyin_neutral']:>7.1%}  "
              f"{net_sym}{abs(r['net_buyin']):.3f}")

    # --- STANCE TREND ---
    stance_rows = []
    for yr, grp in df.groupby("year"):
        n   = len(grp)
        p_s = (grp["llm_stance"] == "supportive").mean()
        p_k = (grp["llm_stance"] == "critical").mean()
        p_n = (grp["llm_stance"] == "neutral").mean()
        ci_k = 1.96 * np.sqrt(p_k * (1 - p_k) / n) if n > 0 else 0.0
        stance_rows.append({
            "year": yr, "n": n,
            "pct_supportive": p_s, "pct_critical": p_k, "pct_stance_neutral": p_n,
            "critical_ci95": ci_k, "net_stance": p_s - p_k,
        })
    stance_trend = pd.DataFrame(stance_rows).sort_values("year")

    print(f"\n{'═'*72}")
    print(f"  STANCE Trend — NS Reddit  ({API_MODEL})")
    print(f"{'═'*72}")
    print(f"  {'Year':>4}  {'N':>5}  {'Supportive':>11}  {'Critical':>9}  "
          f"{'Neutral':>8}  {'Net':>8}")
    print(f"  {'─'*68}")
    for _, r in stance_trend.iterrows():
        net_sym = "▲" if r["net_stance"] > 0 else "▼"
        print(f"  {int(r['year']):>4}  {int(r['n']):>5}  "
              f"{r['pct_supportive']:>10.1%}  {r['pct_critical']:>8.1%}  "
              f"{r['pct_stance_neutral']:>7.1%}  "
              f"{net_sym}{abs(r['net_stance']):.3f}")

    # --- COMBINED METRIC ---
    combined_rows = []
    for yr, grp in df.groupby("year"):
        n   = len(grp)
        pos = ((grp["llm_buyin"] == "committed") | (grp["llm_stance"] == "supportive")).mean()
        neg = ((grp["llm_buyin"] == "uncommitted") | (grp["llm_stance"] == "critical")).mean()
        combined_rows.append({"year": yr, "n": n, "pct_positive": pos, "pct_negative": neg})
    combined_trend = pd.DataFrame(combined_rows).sort_values("year")

    print(f"\n{'═'*72}")
    print(f"  COMBINED Trend (positive = committed OR supportive)")
    print(f"{'═'*72}")
    print(f"  {'Year':>4}  {'N':>5}  {'Positive':>9}  {'Negative':>9}  {'Net':>8}")
    print(f"  {'─'*50}")
    for _, r in combined_trend.iterrows():
        net = r["pct_positive"] - r["pct_negative"]
        net_sym = "▲" if net > 0 else "▼"
        print(f"  {int(r['year']):>4}  {int(r['n']):>5}  "
              f"{r['pct_positive']:>8.1%}  {r['pct_negative']:>8.1%}  "
              f"{net_sym}{abs(net):.3f}")

    # --- OLS on all three ---
    def _print_ols(label, yrs, vals):
        slope, intercept, r_val, p_val, _ = linregress(yrs, vals)
        rho, p_rho = spearmanr(yrs, vals)
        print(f"\n  {label} trend (OLS):")
        print(f"    Slope      : {slope*100:+.3f} pp/year")
        print(f"    r²         : {r_val**2:.3f}")
        print(f"    p-value    : {p_val:.4f}  "
              + ("✅ SIGNIFICANT" if p_val < 0.05 else "❌ not significant") + " at α=0.05")
        print(f"    Spearman ρ : {rho:+.3f}  (p={p_rho:.4f})")

    yrs_b = buyin_trend["year"].values.astype(float)
    _print_ols("Uncommitted (buyin)", yrs_b, buyin_trend["pct_uncommitted"].values)

    yrs_s = stance_trend["year"].values.astype(float)
    _print_ols("Critical (stance)", yrs_s, stance_trend["pct_critical"].values)

    yrs_c = combined_trend["year"].values.astype(float)
    _print_ols("Negative (combined)", yrs_c, combined_trend["pct_negative"].values)

    # Per-subreddit breakdown (uncommitted buyin)
    print(f"\n  Per-subreddit uncommitted (buyin) trend:")
    for sr in TARGET_SUBS:
        sub = df[df["subreddit"] == sr]
        if len(sub) < 50:
            print(f"    r/{sr:<22}  n={len(sub)} (too few — skipped)")
            continue
        st = (sub.groupby("year")["llm_buyin"]
                 .apply(lambda x: (x == "uncommitted").mean())
                 .reset_index())
        st.columns = ["year", "pct_uncommitted"]
        if len(st) >= 3:
            s, b, r2, p, _ = linregress(
                st["year"].values.astype(float), st["pct_uncommitted"].values
            )
            sig = "✅" if p < 0.05 else "  "
            print(f"    {sig} r/{sr:<22}  slope={s*100:+.3f}pp/yr  "
                  f"r²={r2:.3f}  p={p:.4f}  (n={len(sub):,})")

    # Save merged trend
    trend = buyin_trend.merge(stance_trend.drop(columns=["n"]), on="year")
    trend = trend.merge(combined_trend.drop(columns=["n"]), on="year")
    trend.to_csv(COMMIT_TREND, index=False)
    print(f"\n  Saved → {COMMIT_TREND}")
    print(f"{'═'*72}\n")
    return trend


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="LLM commitment annotator — replaces BART C2D scores (kappa=0.127)."
    )
    ap.add_argument(
        "--validate", action="store_true",
        help="Validate GPT-4.1-mini against 100 human commitment annotations"
    )
    ap.add_argument(
        "--annotate", action="store_true",
        help="Annotate stratified annual sample (3 subs × 8 years × 500 chunks)"
    )
    ap.add_argument(
        "--annotate-enrich", action="store_true",
        help="LLM-label commitment_llm_enrich_queue.csv (~15k minority-enriched chunks, ~$6.90)"
    )
    ap.add_argument(
        "--trend", action="store_true",
        help="Compute annual commitment trend from --annotate output"
    )
    args = ap.parse_args()

    if args.validate:
        run_validate()
    elif args.annotate:
        run_annotate()
    elif getattr(args, "annotate_enrich", False):
        run_annotate_enrich()
    elif args.trend:
        run_trend()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
