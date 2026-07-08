"""
Interactive annotation CLI for the targeted positive-candidate queue.

Reads annotation_queue.csv (built by positive_sampler.py), walks you through
each chunk for human labelling, and saves results to targeted_annotations.csv.

When done, run --merge to append your labels into blind_annotation.csv and
rebuild singbert_train.csv — that feeds directly into the v6 Kaggle notebook.

Usage:
    python -m src.features.targeted_annotator              # annotate / resume
    python -m src.features.targeted_annotator --report     # progress report only
    python -m src.features.targeted_annotator --merge      # merge into training data

Keys during annotation:
    n — negative    u — neutral    p — positive
    s — skip        q — quit & save

Progress is saved after every label — safe to quit and resume anytime.
"""

import argparse
import sys
import termios
import textwrap
import tty
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR         = Path(__file__).parent.parent.parent / "data" / "processed" / "new"
QUEUE_PATH       = DATA_DIR / "annotation_queue.csv"
ANNOTATIONS_PATH = DATA_DIR / "targeted_annotations.csv"
BLIND_ANN_PATH   = DATA_DIR / "blind_annotation.csv"
TRAIN_PATH       = DATA_DIR / "singbert_train.csv"

LABEL_KEYS = {"n": "negative", "u": "neutral", "p": "positive"}


# ---------------------------------------------------------------------------
# Terminal helpers (same as annotator.py)
# ---------------------------------------------------------------------------
def getch() -> str:
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def clear():
    print("\033[2J\033[H", end="")


def color(text: str, code: str) -> str:
    codes = {"red": "31", "yellow": "33", "green": "32", "cyan": "36",
             "blue": "34", "bold": "1", "dim": "2", "reset": "0"}
    return f"\033[{codes.get(code, '0')}m{text}\033[0m"


def fmt_label(label: str | None) -> str:
    if label == "negative": return color("negative", "red")
    if label == "positive": return color("positive", "green")
    if label == "neutral":  return color("neutral",  "yellow")
    return color(str(label or "?"), "dim")


def fmt_prob_bar(prob: float, label_color: str = "green") -> str:
    """Visual confidence bar for a probability in [0, 1]."""
    filled = int(prob * 20)
    bar    = color("█" * filled, label_color) + color("░" * (20 - filled), "dim")
    return f"[{bar}] {prob:.3f}"


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def _save(new_rows: list):
    new_df = pd.DataFrame(new_rows)
    if ANNOTATIONS_PATH.exists():
        existing = pd.read_csv(ANNOTATIONS_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates("chunk_id")
    else:
        combined = new_df
    combined.to_csv(ANNOTATIONS_PATH, index=False)


def _load_done() -> set:
    if ANNOTATIONS_PATH.exists():
        done = pd.read_csv(ANNOTATIONS_PATH)
        done = done[done["human_label"].notna()]
        return set(done["chunk_id"])
    return set()


# ---------------------------------------------------------------------------
# Annotation loop
# ---------------------------------------------------------------------------
def annotate():
    if not QUEUE_PATH.exists():
        print(f"❌  Queue not found: {QUEUE_PATH}")
        print(f"    Run first: python -m src.features.positive_sampler")
        sys.exit(1)

    queue    = pd.read_csv(QUEUE_PATH)
    done_ids = _load_done()

    todo = queue[~queue["chunk_id"].isin(done_ids)].reset_index(drop=True)

    total_all  = len(queue)
    total_done = len(done_ids)

    if len(todo) == 0:
        print(f"✅  All {total_all} chunks already labelled.")
        print(f"    Run --report to see results, or --merge to push into training data.")
        return

    print(f"\n{total_done}/{total_all} labelled. {len(todo)} remaining.")
    print(f"Queue composition: {queue['queue_type'].value_counts().to_dict()}")
    print(f"\nKeys:  n=negative  u=neutral  p=positive  s=skip  q=quit")
    input("\nPress Enter to start …")

    new_rows      = []
    session_count = 0   # tracks labels given this session (new_rows resets each save)

    for _, row in todo.iterrows():
        clear()

        # Progress bar
        n_done = total_done + session_count
        pct    = n_done / total_all * 100
        bar    = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(color(f" [{bar}] {n_done}/{total_all} ({pct:.0f}%)", "cyan"))

        # Queue type indicator
        qt = str(row.get("queue_type", ""))
        if qt == "positive_candidate":
            qt_str = color("⊕ positive candidate", "green")
        else:
            qt_str = color("◌ random balance", "dim")

        pos_prob = float(row.get("singbert_pos_prob", 0))
        sb_label = str(row.get("singbert_label", "?"))

        print(color(f" {qt_str}  |  r/{row['subreddit']}  |  {row['doc_type']}  |  wc={int(row['wc'])}",
                    "dim"))
        print()

        # Text
        wrapped = textwrap.fill(str(row["text"]), width=82)
        print(color("┌─ Text " + "─" * 75, "dim"))
        for line in wrapped.split("\n"):
            print(f"│ {line}")
        print(color("└" + "─" * 81, "dim"))
        print()

        # SingBERT signal — shown as reference only (you are the ground truth)
        neg_prob = float(row.get("singbert_neg_prob", 0))
        neu_prob = float(row.get("singbert_neu_prob", 0))
        print(f"  SingBERT →  {fmt_label(sb_label)}")
        print(f"    pos  {fmt_prob_bar(pos_prob, 'green')}")
        print(f"    neu  {fmt_prob_bar(neu_prob, 'yellow')}")
        print(f"    neg  {fmt_prob_bar(neg_prob, 'red')}")
        print()
        print(color("  n=negative  u=neutral  p=positive  s=skip  q=quit", "bold"))
        print("  Your label: ", end="", flush=True)

        while True:
            key = getch()
            if key in LABEL_KEYS:
                label = LABEL_KEYS[key]
                print(fmt_label(label))
                new_rows.append({
                    "chunk_id":    row["chunk_id"],
                    "human_label": label,
                    "stratum":     qt,                  # positive_candidate or random_balance
                    "doc_type":    row["doc_type"],
                    "subreddit":   row["subreddit"],
                    "has_sg":      row.get("has_sg", False),
                    "wc":          row["wc"],
                    "text":        row["text"],
                })
                _save(new_rows)
                new_rows = []       # flushed; reset buffer (saves per-label)
                session_count += 1  # keep progress bar accurate
                break
            elif key == "s":
                print(color("skipped", "dim"))
                break
            elif key in ("q", "\x03"):
                print(color("quit", "dim"))
                _save(new_rows)
                print(f"\nSaved. {total_done + len(new_rows)} labelled total.")
                _print_report()
                return

    _save(new_rows)
    print(f"\n✅  All done! Run --report for stats, then --merge to push into training data.")
    _print_report()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _print_report():
    if not ANNOTATIONS_PATH.exists():
        print("No targeted annotations yet.")
        return

    ann   = pd.read_csv(ANNOTATIONS_PATH)
    done  = ann[ann["human_label"].notna()]
    queue = pd.read_csv(QUEUE_PATH) if QUEUE_PATH.exists() else pd.DataFrame()

    total_q = len(queue)
    total_d = len(done)

    print(f"\n{'═'*60}")
    print(f"  Targeted annotation report  ({total_d}/{total_q} labelled)")
    print(f"{'═'*60}")

    # By queue_type
    if "stratum" in done.columns:
        print(f"\n  By queue type:")
        for qt, grp in done.groupby("stratum"):
            dist = grp["human_label"].value_counts()
            neg  = dist.get("negative", 0)
            neu  = dist.get("neutral", 0)
            pos  = dist.get("positive", 0)
            hit_rate = pos / len(grp) * 100
            print(f"    {qt:<25}  n={len(grp):>4}  "
                  f"neg={neg:>3}  neu={neu:>3}  pos={pos:>3}  "
                  f"(pos hit rate: {hit_rate:.1f}%)")

    # Overall distribution
    dist  = done["human_label"].value_counts()
    total = len(done)
    print(f"\n  Label distribution (targeted annotations):")
    for lbl in ["negative", "neutral", "positive"]:
        n   = dist.get(lbl, 0)
        bar = "█" * int(n / max(total, 1) * 30)
        print(f"    {lbl:<10}  {n:>4}  ({n/max(total,1)*100:.1f}%)  {bar}")

    # Existing human labels for comparison
    if BLIND_ANN_PATH.exists():
        existing = pd.read_csv(BLIND_ANN_PATH)
        existing = existing[existing["human_label"].notna()]
        print(f"\n  Existing blind_annotation.csv: {len(existing)} rows")
        new_pos   = dist.get("positive", 0)
        exist_pos = (existing["human_label"] == "positive").sum()
        print(f"  Positive class: {exist_pos} existing → +{new_pos} new = {exist_pos + new_pos} total after merge")

    # Merge readiness
    print(f"\n  {'─'*58}")
    if total_d >= 200:
        print(f"  ✅  {total_d} annotations ready — run --merge to push into singbert_train.csv")
    else:
        remaining = 200 - total_d
        print(f"  ⏳  {remaining} more annotations recommended before merging (target: 200+)")
    print(f"  {'─'*58}\n")


# ---------------------------------------------------------------------------
# Merge into training data
# ---------------------------------------------------------------------------
def merge_into_training():
    """Append targeted_annotations.csv into blind_annotation.csv and rebuild singbert_train.csv."""
    if not ANNOTATIONS_PATH.exists():
        print("❌  targeted_annotations.csv not found — annotate first.")
        sys.exit(1)

    ann  = pd.read_csv(ANNOTATIONS_PATH)
    done = ann[ann["human_label"].notna()].copy()
    if len(done) == 0:
        print("❌  No completed annotations found.")
        sys.exit(1)

    # ── Load existing blind_annotation.csv ─────────────────────────────
    if BLIND_ANN_PATH.exists():
        blind = pd.read_csv(BLIND_ANN_PATH)
    else:
        blind = pd.DataFrame(columns=["chunk_id","stratum","doc_type","subreddit",
                                       "has_sg","wc","text","human_label"])

    already_in = set(blind["chunk_id"])
    new_rows   = done[~done["chunk_id"].isin(already_in)].copy()

    if len(new_rows) == 0:
        print("⚠️  All targeted annotations are already in blind_annotation.csv. Nothing to add.")
        return

    # Align columns to blind_annotation.csv
    merge_cols = ["chunk_id", "stratum", "doc_type", "subreddit", "has_sg", "wc", "text", "human_label"]
    new_rows   = new_rows[merge_cols]

    updated_blind = pd.concat([blind, new_rows], ignore_index=True)
    updated_blind.to_csv(BLIND_ANN_PATH, index=False)

    print(f"{'═'*60}")
    print(f"  Merge complete")
    print(f"{'═'*60}")
    print(f"  blind_annotation.csv: {len(blind)} → {len(updated_blind)} rows (+{len(new_rows)})")

    lbl_dist = new_rows["human_label"].value_counts()
    for lbl in ["negative", "neutral", "positive"]:
        print(f"    +{lbl:<10}  {lbl_dist.get(lbl, 0)}")

    # ── Rebuild singbert_train.csv ──────────────────────────────────────
    llm_path  = DATA_DIR / "llm_annotation.csv"
    cot_path  = DATA_DIR / "llm_annotation_cot.csv"
    hold_path = DATA_DIR / "holdout_test.csv"

    # Prefer CoT-annotated LLM labels if they exist and are more complete
    if cot_path.exists():
        llm_df = pd.read_csv(cot_path)
        print(f"\n  Using CoT LLM labels ({len(llm_df):,} rows from llm_annotation_cot.csv)")
    elif llm_path.exists():
        llm_df = pd.read_csv(llm_path)
        print(f"\n  Using standard LLM labels ({len(llm_df):,} rows from llm_annotation.csv)")
    else:
        print("\n  ⚠️  No LLM annotation file found — singbert_train.csv will be human-only.")
        llm_df = pd.DataFrame()

    human_df = pd.read_csv(BLIND_ANN_PATH)
    human_df = human_df[human_df["human_label"].notna() & (human_df["human_label"] != "skip")].copy()
    human_df = human_df.rename(columns={"human_label": "label"})
    human_df["source"] = "human"
    human_df["weight"] = 3.0

    excluded_ids = set(human_df["chunk_id"])
    if hold_path.exists():
        excluded_ids |= set(pd.read_csv(hold_path)["chunk_id"].dropna())

    llm_label_col = "llm_label" if "llm_label" in llm_df.columns else None

    if len(llm_df) > 0 and llm_label_col:
        llm_df = llm_df[llm_df[llm_label_col].notna()].copy()
        llm_df = llm_df.rename(columns={llm_label_col: "label"})
        llm_df = llm_df[~llm_df["chunk_id"].isin(excluded_ids)]
        llm_df["source"] = "llm"
        llm_df["weight"] = 1.0
    else:
        llm_df = pd.DataFrame()

    cols = ["chunk_id", "stratum", "doc_type", "subreddit", "has_sg", "wc",
            "text", "label", "source", "weight"]

    # Ensure all cols exist
    for df_ in [human_df]:
        for c in cols:
            if c not in df_.columns:
                df_[c] = None

    parts = [human_df[cols]]
    if len(llm_df) > 0:
        for c in cols:
            if c not in llm_df.columns:
                llm_df[c] = None
        parts.append(llm_df[cols])

    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(TRAIN_PATH, index=False)

    print(f"\n  singbert_train.csv rebuilt:")
    print(f"    Human rows : {len(human_df):,}  (weight 3×)")
    if len(llm_df) > 0:
        print(f"    LLM rows   : {len(llm_df):,}  (weight 1×)")
    print(f"    Total      : {len(combined):,}")
    print(f"\n  Label distribution:")
    vc = combined["label"].value_counts()
    for lbl in ["negative", "neutral", "positive"]:
        n = vc.get(lbl, 0)
        print(f"    {lbl:<10}  {n:>6,}  ({n/len(combined)*100:.1f}%)")

    print(f"\n  ✅  Saved → {TRAIN_PATH}")
    print(f"\n  Next: upload singbert_train.csv + blind_annotation.csv to Kaggle dataset,")
    print(f"        then run kaggle_finetune_singbert_v1.ipynb with v6 config.")
    print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Targeted annotation CLI for positive-candidate queue."
    )
    parser.add_argument("--report", action="store_true",
                        help="Show annotation progress report (no annotation)")
    parser.add_argument("--merge",  action="store_true",
                        help="Merge targeted_annotations.csv into blind_annotation.csv "
                             "and rebuild singbert_train.csv")
    args = parser.parse_args()

    if args.report:
        _print_report()
    elif args.merge:
        merge_into_training()
    else:
        annotate()


if __name__ == "__main__":
    main()
