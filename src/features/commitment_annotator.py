"""
Blind commitment annotation CLI — accuracy test for Stage 5b BART-large-mnli scores.

Annotate chunks with one of 3 commitment labels:
    C  — COMMITTED  (author endorses NS as worth it / important for defence)
    X  — CRITICAL   (author opposes or dismisses NS as an institution)
    N  — NEUTRAL    (practical discussion, no stance on NS value)
    S  — skip       (genuinely unreadable / fragment too short to judge)
    B  — go back
    Q  — quit and save
    ?  — show the labelling guide again

Strata (100 chunks total):
    high_support   (20) — model's strong support predictions  → are these really committed?
    high_critical  (20) — model's strong critical predictions → are these really critical?
    short_chunks   (20) — <80 char fragments                  → fragment inflation check
    borderline     (20) — model uncertain (both scores 0.3-0.45) → hard cases
    singlish       (20) — Singlish terms present               → Singlish handling

Usage:
    python -m src.features.commitment_annotator           # annotate
    python -m src.features.commitment_annotator --report  # progress
    python -m src.features.commitment_annotator --compare # kappa vs BART scores
    python -m src.features.commitment_annotator --reset   # start fresh
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent.parent / "data" / "processed" / "new"
ANN_PATH    = DATA_DIR / "commitment_annotation.csv"
COMMIT_PATH = DATA_DIR / "chunk_commitment.parquet"

CHUNK_COLS  = ["chunk_id", "doc_type", "subreddit", "text"]

STRATA = [
    # (name,            target_n,  description)
    ("high_support",    20,  "model commit_support > 0.60"),
    ("high_critical",   20,  "model commit_critical > 0.50"),
    ("short_chunks",    20,  "text < 80 chars (fragment inflation zone)"),
    ("borderline",      20,  "model uncertain — neither score > 0.45"),
    ("singlish",        20,  "contains Singlish commitment vocabulary"),
]

# Singlish terms relevant to commitment (uncommitment vocabulary)
_SG_COMMIT = {
    "chao keng", "keng", "wayang", "bo chup", "bo chap", "bochup", "bochap",
    "sian", "slack", "chiong", "siong", "saikang", "arrow", "kena arrow",
    "lepak", "sign on", "signed on", "ownself", "lobang",
}
_SG_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_SG_COMMIT, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def has_singlish_commit(text: str) -> bool:
    return bool(_SG_PATTERN.search(str(text)))


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text)))


# ─────────────────────────────────────────────────────────────────────────────
GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║              COMMITMENT LABELLING GUIDE                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  You are labelling the AUTHOR'S STANCE ON NS AS AN INSTITUTION. ║
║  NOT their emotional state — their belief about NS value/worth.  ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  [C] COMMITTED                                                   ║
║      Author endorses NS. Believes it is worth it, important,    ║
║      meaningful, or necessary for Singapore's defence.           ║
║                                                                  ║
║      Examples:                                                   ║
║      · "NS made me a better person, I'm glad I served"          ║
║      · "We need to defend Singapore, NS is necessary"            ║
║      · "Even though it's tough, I understand why we do this"    ║
║      · "Proud to serve, signing on after NS"                     ║
║      · "NS is important for our country's security"              ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  [X] CRITICAL                                                    ║
║      Author opposes NS. Believes it is a waste of time,         ║
║      unfair, exploitative, pointless, or should be abolished.   ║
║                                                                  ║
║      Examples:                                                   ║
║      · "NS is a waste of 2 years, I resent being conscripted"   ║
║      · "The system is broken, chao keng all the way"            ║
║      · "Why should only men serve? This is slavery"              ║
║      · "NS won't defend against any real threat, it's wayang"   ║
║      · "Don't see the point, just going through the motions"    ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  [N] NEUTRAL                                                     ║
║      Author is discussing NS practically — asking questions,    ║
║      sharing information, or describing experience without       ║
║      expressing a view on whether NS is worth it.               ║
║                                                                  ║
║      Examples:                                                   ║
║      · "When is my enlistment date?"                             ║
║      · "BMT lasts 9 weeks, here's what to expect"               ║
║      · "Which vocation should I aim for?"                        ║
║      · "The food in camp was terrible" (complaint ≠ opposition)  ║
║      · "Finally ORD!" (relief, not an institutional stance)      ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  ⚠️  COMMITMENT IS NOT THE SAME AS SENTIMENT                     ║
║                                                                  ║
║  NEGATIVE sentiment ≠ CRITICAL commitment:                       ║
║    "I hate waking up at 5am but NS made me stronger"            ║
║    → NEGATIVE sentiment, COMMITTED (author endorses NS value)   ║
║                                                                  ║
║    "The food in camp sucks"                                       ║
║    → NEGATIVE sentiment, NEUTRAL (not a stance on NS worth)     ║
║                                                                  ║
║  POSITIVE sentiment ≠ COMMITTED:                                 ║
║    "Easy to chao keng here, quite shiok lah"                    ║
║    → POSITIVE sentiment, CRITICAL (dismissive of NS duty)       ║
║                                                                  ║
║    "Finally ORD! So happy!"                                       ║
║    → POSITIVE sentiment, NEUTRAL (relief, not an endorsement)   ║
║                                                                  ║
║  THE TEST: does the author think NS is WORTH IT for Singapore?  ║
║    YES clearly  → COMMITTED                                       ║
║    NO clearly   → CRITICAL                                        ║
║    No opinion   → NEUTRAL                                         ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
def build_sample(chunks: pd.DataFrame, commit: pd.DataFrame) -> pd.DataFrame:
    df = chunks.copy()
    df["text"] = df["text"].fillna("").str.strip()
    df = df[df["text"].str.len() > 0]
    df["wc"]     = df["text"].apply(word_count)
    df["has_sg"] = df["text"].apply(has_singlish_commit)
    df["char_len"] = df["text"].str.len()

    # Merge commit scores
    df = df.merge(commit[["chunk_id", "commit_support", "commit_critical", "commit_neutral"]],
                  on="chunk_id", how="inner")

    parts       = []
    used_ids    = set()

    # ── Stratum 1: high support ───────────────────────────────────────────────
    g = df[(df["commit_support"] > 0.60) & (~df["chunk_id"].isin(used_ids))]
    s = g.sample(min(20, len(g)), random_state=7).copy()
    s["stratum"] = "high_support"
    parts.append(s); used_ids.update(s["chunk_id"])

    # ── Stratum 2: high critical ──────────────────────────────────────────────
    g = df[(df["commit_critical"] > 0.50) & (~df["chunk_id"].isin(used_ids))]
    s = g.sample(min(20, len(g)), random_state=11).copy()
    s["stratum"] = "high_critical"
    parts.append(s); used_ids.update(s["chunk_id"])

    # ── Stratum 3: short chunks (fragment inflation zone) ────────────────────
    g = df[(df["char_len"] < 80) & (~df["chunk_id"].isin(used_ids))]
    s = g.sample(min(20, len(g)), random_state=17).copy()
    s["stratum"] = "short_chunks"
    parts.append(s); used_ids.update(s["chunk_id"])

    # ── Stratum 4: borderline (model uncertain) ───────────────────────────────
    g = df[
        (df["commit_support"] < 0.45) &
        (df["commit_critical"] < 0.45) &
        (df["commit_neutral"]  < 0.75) &   # not already clearly neutral
        (~df["chunk_id"].isin(used_ids))
    ]
    s = g.sample(min(20, len(g)), random_state=23).copy()
    s["stratum"] = "borderline"
    parts.append(s); used_ids.update(s["chunk_id"])

    # ── Stratum 5: Singlish commitment vocabulary ─────────────────────────────
    g = df[(df["has_sg"]) & (~df["chunk_id"].isin(used_ids))]
    s = g.sample(min(20, len(g)), random_state=31).copy()
    s["stratum"] = "singlish"
    parts.append(s); used_ids.update(s["chunk_id"])

    sample = (pd.concat(parts, ignore_index=True)
              .sample(frac=1, random_state=42)
              .reset_index(drop=True))

    return sample[["chunk_id", "stratum", "doc_type", "subreddit", "has_sg",
                   "wc", "char_len", "text",
                   "commit_support", "commit_critical", "commit_neutral"]]


# ─────────────────────────────────────────────────────────────────────────────
def load_results() -> pd.DataFrame:
    if ANN_PATH.exists():
        return pd.read_csv(ANN_PATH)
    return pd.DataFrame(columns=[
        "chunk_id", "stratum", "doc_type", "subreddit", "has_sg",
        "wc", "char_len", "text",
        "commit_support", "commit_critical", "commit_neutral",
        "human_label",
    ])


def save_results(df: pd.DataFrame):
    df.to_csv(ANN_PATH, index=False)


# ─────────────────────────────────────────────────────────────────────────────
def print_progress(results: pd.DataFrame):
    done     = results.dropna(subset=["human_label"])
    total    = sum(n for _, n, _ in STRATA)
    n        = len(done)
    skipped  = (done["human_label"] == "skip").sum()
    labelled = n - skipped

    print(f"\n{'═'*55}")
    print(f"  Commitment annotation  ({n}/{total} done)")
    print(f"{'═'*55}")
    if n == 0:
        print("  No annotations yet.\n")
        return

    print(f"\n  Labelled: {labelled}   Skipped: {skipped}")
    print(f"\n  Label distribution:")
    for lbl, key in [("committed","committed"),("critical","critical"),("neutral","neutral")]:
        cnt = int((done["human_label"] == key).sum())
        bar = "█" * int((cnt / max(labelled,1)) * 25)
        print(f"    {lbl:<12} {cnt:>3}  {bar}")

    print(f"\n  By stratum:")
    for name, target, desc in STRATA:
        g   = done[done["stratum"] == name]
        cnt = len(g[g["human_label"] != "skip"])
        print(f"    {name:<18} {cnt:>2}/{target}  — {desc}")

    print(f"\n{'═'*55}")
    if n >= total:
        print("  ✅  All done. Run --compare to see kappa vs BART scores.\n")


# ─────────────────────────────────────────────────────────────────────────────
def run_annotate():
    import tty, termios

    def getch() -> str:
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print("Loading data ...")
    sub    = pd.read_parquet(DATA_DIR / "submissions_chunks.parquet", columns=CHUNK_COLS)
    com    = pd.read_parquet(DATA_DIR / "comments_chunks.parquet",    columns=CHUNK_COLS)
    chunks = pd.concat([sub, com], ignore_index=True).drop_duplicates("chunk_id")
    commit = pd.read_parquet(COMMIT_PATH)
    del sub, com

    sample  = build_sample(chunks, commit)
    results = load_results()

    done_ids = set(results["chunk_id"].dropna())
    todo     = sample[~sample["chunk_id"].isin(done_ids)].reset_index(drop=True)
    reviewed = len(done_ids)
    total    = len(sample)

    if len(todo) == 0:
        print("All chunks annotated.")
        print_progress(results)
        print("Run --compare to see kappa vs BART scores.")
        return

    print(GUIDE)
    print(f"  {reviewed}/{total} done — resuming from #{reviewed + 1}\n")
    print("  Keys:  [C]=committed  [X]=critical  [N]=neutral  [S]=skip  [B]=back  [Q]=quit  [?]=guide\n")
    print("  ⚠️  No BART scores shown. Label on text alone.\n")

    rows = list(todo.iterrows())
    i    = 0

    while i < len(rows):
        _, row   = rows[i]
        position = reviewed + i + 1
        sg_flag  = "🇸🇬 " if row["has_sg"] else "   "
        text     = str(row["text"]).strip().replace("\n", " ")

        # Show stratum context (but not BART scores — this is blind)
        print(f"  ── [{position}/{total}] {sg_flag} {row['stratum']:<18} "
              f"({row['doc_type']}, r/{row['subreddit']}, {row['wc']}w) ──")
        print(f"\n  {text[:400]}\n")
        print("  > ", end="", flush=True)

        action = label = None
        while True:
            ch = getch().lower()
            if   ch == "c": label = "committed"; print("→ COMMITTED"); action = "label"; break
            elif ch == "x": label = "critical";  print("→ CRITICAL");  action = "label"; break
            elif ch == "n": label = "neutral";   print("→ NEUTRAL");   action = "label"; break
            elif ch == "s": label = "skip";      print("skip");        action = "label"; break
            elif ch == "q": print("quit");                              action = "quit";  break
            elif ch == "b": print("↩");                                 action = "back";  break
            elif ch == "?": print(); print(GUIDE); print("  > ", end="", flush=True)

        if action == "quit":
            print_progress(results)
            return

        if action == "back":
            if i > 0:
                i -= 1
                results = results.iloc[:-1].reset_index(drop=True)
                save_results(results)
                print("  ↩  Re-labelling previous chunk\n")
            else:
                print("  (already at first chunk this session)\n")
            continue

        new_row = row.to_dict()
        new_row["human_label"] = label
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        save_results(results)
        i += 1
        print()

    print("\nAll done!")
    print_progress(results)
    print("Run --compare to see kappa vs BART scores.")


# ─────────────────────────────────────────────────────────────────────────────
def run_compare():
    """Load annotations, compare against BART majority label, compute kappa."""
    try:
        from sklearn.metrics import cohen_kappa_score, classification_report
    except ImportError:
        print("pip install scikit-learn")
        return

    results  = load_results()
    labelled = results[
        results["human_label"].notna() & (results["human_label"] != "skip")
    ].copy().reset_index(drop=True)

    if len(labelled) < 10:
        print(f"Only {len(labelled)} labels so far — annotate more before comparing.")
        return

    # BART majority label: whichever of the 3 scores is highest
    def bart_label(row):
        scores = {
            "committed": row["commit_support"],
            "critical":  row["commit_critical"],
            "neutral":   row["commit_neutral"],
        }
        return max(scores, key=scores.get)

    labelled["bart_label"] = labelled.apply(bart_label, axis=1)

    n        = len(labelled)
    kappa    = cohen_kappa_score(labelled["human_label"], labelled["bart_label"])
    accuracy = (labelled["human_label"] == labelled["bart_label"]).mean()

    print(f"\n{'═'*60}")
    print(f"  BART C2D Accuracy vs Human Labels  (n={n})")
    print(f"{'═'*60}")
    print(f"\n  Overall accuracy    : {accuracy:.1%}")
    print(f"  Cohen's Kappa (κ)   : {kappa:.3f}")

    if   kappa >= 0.80: verdict = "✅  EXCELLENT — scores are reliable"
    elif kappa >= 0.60: verdict = "✅  GOOD — usable with documented caveats"
    elif kappa >= 0.40: verdict = "⚠️  MODERATE — recalibrate or re-run with better hypotheses"
    else:               verdict = "❌  POOR — scores are unreliable, re-run required"
    print(f"  Verdict             : {verdict}")

    # ── Per-stratum ──────────────────────────────────────────────────────────
    print(f"\n  ── Accuracy by stratum ──")
    for stratum, _, desc in STRATA:
        g = labelled[labelled["stratum"] == stratum]
        if len(g) == 0:
            continue
        acc = (g["human_label"] == g["bart_label"]).mean()
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"    {stratum:<18} [{bar}] {acc:.1%}  (n={len(g)})")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    print(f"\n  ── Confusion matrix (human → BART prediction) ──")
    for true_lbl in ["committed", "critical", "neutral"]:
        g    = labelled[labelled["human_label"] == true_lbl]
        if len(g) == 0:
            continue
        dist = g["bart_label"].value_counts()
        row  = []
        for pred_lbl in ["committed", "critical", "neutral"]:
            cnt  = int(dist.get(pred_lbl, 0))
            mark = "✓" if pred_lbl == true_lbl else " "
            row.append(f"{mark}{pred_lbl[:3].upper()}:{cnt:>2}")
        pct_right = int(dist.get(true_lbl, 0)) / len(g) * 100
        print(f"    human={true_lbl:<10} →  {'  '.join(row)}  (n={len(g)}, {pct_right:.0f}% right)")

    # ── Label distribution gap ────────────────────────────────────────────────
    print(f"\n  ── Label distribution: human vs BART ──")
    for col, lbl in [("human_label","Human"), ("bart_label","BART")]:
        vc    = labelled[col].value_counts(normalize=True)
        parts = [f"{k[:3].upper()}:{vc.get(k,0)*100:.0f}%"
                 for k in ["committed","critical","neutral"]]
        print(f"    {lbl:<8}  {' / '.join(parts)}")

    diff_sup = labelled["commit_support"].mean() - labelled["commit_critical"].mean()
    print(f"\n  ── BART score bias check ──")
    print(f"    Mean commit_support:  {labelled['commit_support'].mean():.4f}")
    print(f"    Mean commit_critical: {labelled['commit_critical'].mean():.4f}")
    print(f"    Support premium:      {diff_sup:+.4f}  (0.000 = unbiased; positive = pro-support prior)")

    # ── Full classification report ────────────────────────────────────────────
    print(f"\n  ── Full classification report ──")
    print(classification_report(
        labelled["human_label"], labelled["bart_label"],
        labels=["committed", "critical", "neutral"], digits=3
    ))

    print(f"{'═'*60}\n")
    results.to_csv(ANN_PATH, index=False)
    print(f"  Results saved → {ANN_PATH}")
    print(f"  To re-run Kaggle with better hypotheses, see HANDOFF.md Stage 5b section.\n")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Blind commitment annotation CLI — accuracy test for BART C2D scores."
    )
    parser.add_argument("--report",  action="store_true", help="Print progress only")
    parser.add_argument("--compare", action="store_true", help="Compute kappa vs BART scores")
    parser.add_argument("--reset",   action="store_true", help="Delete annotation file and start fresh")
    args = parser.parse_args()

    if args.reset:
        if ANN_PATH.exists():
            ANN_PATH.unlink()
            print(f"Deleted {ANN_PATH.name}")
        return

    if args.report:
        print_progress(load_results())
        return

    if args.compare:
        run_compare()
        return

    run_annotate()


if __name__ == "__main__":
    main()
