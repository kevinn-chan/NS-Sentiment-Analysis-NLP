"""
Manual accuracy spot-check CLI for chunk_sentiment.parquet.

Goes through 100 stratified chunks and records whether the XLM label
is correct. Resume-safe — saves after every keypress.

Usage:
    python -m src.features.spot_checker          # run / resume
    python -m src.features.spot_checker --report # print results only
    python -m src.features.spot_checker --reset  # start fresh

Keys:
    Y / Enter  — agree (label is correct)
    P          — should be POSITIVE
    N          — should be NEGATIVE
    U          — should be NEUTRAL
    S          — skip (unsure, don't count)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent.parent.parent / "data" / "processed" / "new"
SENT_PATH  = DATA_DIR / "chunk_sentiment.parquet"
CHECK_PATH = DATA_DIR / "spot_check.csv"

CHUNK_COLS = ["chunk_id", "doc_type", "subreddit", "text"]

# ---------------------------------------------------------------------------
# Singlish detector
# ---------------------------------------------------------------------------
_VOCAB   = {
    "lah","leh","lor","liao","sia","hor","mah","sian","jialat","wayang",
    "chao keng","saikang","siong","kena","walao","siao","aiyah","alamak",
    "shiok","bochap","bo chap","gg","tekan","suay","lepak","lobang","wah",
    "sibei","steady","swee","shiok leh","die die","gao gao",
}
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_VOCAB, key=len, reverse=True)) + r")\b"
)

def has_singlish(text: str) -> bool:
    return bool(_PATTERN.search(str(text).lower()))


# ---------------------------------------------------------------------------
# Sample selection — 100 chunks, stratified
# ---------------------------------------------------------------------------
STRATA = [
    ("high_neg",      {"label": "negative", "conf_min": 0.80},                    22),
    ("high_pos",      {"label": "positive", "conf_min": 0.80},                    22),
    ("high_neu",      {"label": "neutral",  "conf_min": 0.80},                    18),
    ("low_conf",      {"conf_min": 0.34, "conf_max": 0.50},                       18),
    ("singlish",      {"singlish": True},                                          20),
]

def build_sample(sent: pd.DataFrame, chunks: pd.DataFrame) -> pd.DataFrame:
    lmap = {"sent_neg": "negative", "sent_neu": "neutral", "sent_pos": "positive"}
    df = sent.copy()
    df["label"]     = df[["sent_neg", "sent_neu", "sent_pos"]].idxmax(axis=1).map(lmap)
    df["max_score"] = df[["sent_neg", "sent_neu", "sent_pos"]].max(axis=1)
    df = df.merge(chunks, on="chunk_id", how="left")
    df["has_sg"] = df["text"].fillna("").apply(has_singlish)

    parts = []
    for stratum, filters, n in STRATA:
        g = df.copy()
        if "label"    in filters: g = g[g["label"]     == filters["label"]]
        if "conf_min" in filters: g = g[g["max_score"] >= filters["conf_min"]]
        if "conf_max" in filters: g = g[g["max_score"] <= filters["conf_max"]]
        if "singlish" in filters: g = g[g["has_sg"]    == filters["singlish"]]
        # Exclude already sampled chunk_ids
        if parts:
            used = pd.concat(parts)["chunk_id"].values
            g = g[~g["chunk_id"].isin(used)]
        sampled = g.sample(min(n, len(g)), random_state=42)
        sampled = sampled.copy()
        sampled["stratum"] = stratum
        parts.append(sampled)

    sample = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=99)
    return sample[["chunk_id", "stratum", "label", "max_score",
                   "sent_neg", "sent_neu", "sent_pos",
                   "doc_type", "subreddit", "has_sg", "text"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def load_results() -> pd.DataFrame:
    if CHECK_PATH.exists():
        return pd.read_csv(CHECK_PATH)
    return pd.DataFrame(columns=["chunk_id", "stratum", "label", "max_score",
                                  "sent_neg", "sent_neu", "sent_pos",
                                  "doc_type", "subreddit", "has_sg", "text",
                                  "human_verdict", "corrected_label"])

def save_results(df: pd.DataFrame):
    df.to_csv(CHECK_PATH, index=False)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(results: pd.DataFrame):
    done = results.dropna(subset=["human_verdict"])
    n    = len(done)
    if n == 0:
        print("No results yet.")
        return

    agreed    = (done["human_verdict"] == "agree").sum()
    disagreed = (done["human_verdict"] == "disagree").sum()
    skipped   = (done["human_verdict"] == "skip").sum()
    accuracy  = agreed / (agreed + disagreed) if (agreed + disagreed) > 0 else 0

    print(f"\n{'═'*62}")
    print(f"  Spot-check report  ({n} reviewed)")
    print(f"{'═'*62}")
    print(f"\n  Overall accuracy (agree / non-skip)")
    print(f"    Correct   : {agreed:>4}  ({agreed/(agreed+disagreed)*100:.1f}%)")
    print(f"    Wrong     : {disagreed:>4}  ({disagreed/(agreed+disagreed)*100:.1f}%)")
    print(f"    Skipped   : {skipped:>4}")
    print(f"    Accuracy  : {accuracy:.1%}")

    print(f"\n  Accuracy by stratum")
    for stratum, g in done.groupby("stratum"):
        g2  = g[g["human_verdict"] != "skip"]
        if len(g2) == 0: continue
        acc = (g2["human_verdict"] == "agree").mean()
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        flag = "  ← LOW" if acc < 0.70 else ""
        print(f"    {stratum:<14} [{bar}] {acc:.1%}  (n={len(g2)}){flag}")

    print(f"\n  Accuracy by predicted label")
    for lbl, g in done.groupby("label"):
        g2  = g[g["human_verdict"] != "skip"]
        if len(g2) == 0: continue
        acc = (g2["human_verdict"] == "agree").mean()
        print(f"    {lbl:<10} {acc:.1%}  (n={len(g2)})")

    wrong = done[done["human_verdict"] == "disagree"]
    if len(wrong) > 0:
        print(f"\n  Mislabelled chunks")
        for _, r in wrong.iterrows():
            fix = r.get("corrected_label", "?")
            text = str(r["text"])[:90].replace("\n", " ")
            sg = "🇸🇬 " if r["has_sg"] else "   "
            print(f"    {sg}[{r['label']} → {fix}]  conf={r['max_score']:.2f}  {text}")

    print(f"\n{'─'*62}")
    if accuracy >= 0.80:
        print(f"  ✅  {accuracy:.1%} accuracy — strong.")
    elif accuracy >= 0.70:
        print(f"  ✅  {accuracy:.1%} accuracy — acceptable.")
    else:
        print(f"  ⚠️   {accuracy:.1%} accuracy — review model.")
    print(f"{'═'*62}\n")


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------
def run():
    import tty, termios

    def getch():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print("Loading data...")
    sent   = pd.read_parquet(SENT_PATH)
    sub    = pd.read_parquet(DATA_DIR / "../../.." / "data/processed/new/submissions_chunks.parquet",
                             columns=CHUNK_COLS)
    com    = pd.read_parquet(DATA_DIR / "../../.." / "data/processed/new/comments_chunks.parquet",
                             columns=CHUNK_COLS)
    sub    = pd.read_parquet(DATA_DIR / "submissions_chunks.parquet", columns=CHUNK_COLS)
    com    = pd.read_parquet(DATA_DIR / "comments_chunks.parquet",    columns=CHUNK_COLS)
    chunks = pd.concat([sub, com], ignore_index=True).drop_duplicates("chunk_id")
    del sub, com

    sample  = build_sample(sent, chunks)
    results = load_results()

    # Resume: find chunks not yet reviewed
    done_ids = set(results["chunk_id"].dropna()) if len(results) > 0 else set()
    todo     = sample[~sample["chunk_id"].isin(done_ids)].reset_index(drop=True)

    total    = len(sample)
    reviewed = len(done_ids)

    if len(todo) == 0:
        print("All chunks reviewed.")
        print_report(results)
        return

    print(f"  {reviewed}/{total} done — resuming from #{reviewed+1}\n")
    print("  Keys: [Y/Enter]=correct  [P]=positive  [N]=negative  [U]=neutral  [S]=skip  [Q]=quit\n")

    for _, row in todo.iterrows():
        reviewed += 1
        sg_flag   = "🇸🇬 " if row["has_sg"] else "   "
        text      = str(row["text"]).strip().replace("\n", " ")

        print(f"  ── [{reviewed}/{total}] {sg_flag} {row['stratum']:<14} "
              f"({row['doc_type']}, {row['subreddit']}) ──────────────────")
        print(f"  Predicted : {row['label'].upper():<10} "
              f"conf={row['max_score']:.2f}  "
              f"neg={row['sent_neg']:.2f} neu={row['sent_neu']:.2f} pos={row['sent_pos']:.2f}")
        print(f"\n  {text[:300]}\n")
        print("  > ", end="", flush=True)

        while True:
            ch = getch().lower()
            if ch in ("y", "\r", "\n"):
                verdict, corrected = "agree", row["label"]
                print("agree ✓")
                break
            elif ch == "p":
                verdict, corrected = "disagree", "positive"
                print("→ POSITIVE")
                break
            elif ch == "n":
                verdict, corrected = "disagree", "negative"
                print("→ NEGATIVE")
                break
            elif ch == "u":
                verdict, corrected = "disagree", "neutral"
                print("→ NEUTRAL")
                break
            elif ch == "s":
                verdict, corrected = "skip", ""
                print("skip")
                break
            elif ch == "q":
                print("quit")
                print_report(results)
                return

        new_row = row.to_dict()
        new_row["human_verdict"]   = verdict
        new_row["corrected_label"] = corrected
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        save_results(results)
        print()

    print("\nAll done!")
    print_report(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--reset",  action="store_true")
    args = parser.parse_args()

    if args.reset:
        if CHECK_PATH.exists():
            CHECK_PATH.unlink()
            print("Reset — spot_check.csv deleted.")
        return

    if args.report:
        print_report(load_results())
        return

    run()


if __name__ == "__main__":
    main()
