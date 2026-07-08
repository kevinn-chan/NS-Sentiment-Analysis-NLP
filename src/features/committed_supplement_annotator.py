"""
Annotation CLI for the committed-candidate supplement batch.

Reads:  data/processed/new/committed_supplement_queue.csv
Writes: back to same file after every keypress (safe to quit and resume).

Shows the trigger term that caused this chunk to be selected — useful for
spotting which lexicon terms are over-firing.

Keys:
  [C] committed    [U] uncommitted    [N] neutral    [S] skip    [B] back    [Q] quit    [?] guide

Stance (axis 2, optional):
  [P] supportive   [K] critical       [N] neutral

Usage:
    python -m src.features.committed_supplement_annotator            # label / resume
    python -m src.features.committed_supplement_annotator --report   # stats only
    python -m src.features.committed_supplement_annotator --merge    # merge into testset.parquet
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DATA        = Path(__file__).resolve().parents[2] / "data" / "processed" / "new"
QUEUE_PATH  = DATA / "committed_supplement_queue.csv"
TESTSET_PATH = DATA / "commitment_testset.parquet"

BUYIN_LABELS  = {"c": "committed", "u": "uncommitted", "n": "neutral", "s": "skip"}
STANCE_LABELS = {"p": "supportive", "k": "critical", "n": "neutral"}

GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║  COMMITTED SUPPLEMENT — label the TRIGGER TERM in context        ║
╠══════════════════════════════════════════════════════════════════╣
║  Ask: is the AUTHOR of this chunk personally committed to NS?    ║
║  (Not the institution. Not a quoted person. The author.)         ║
╠══════════════════════════════════════════════════════════════════╣
║  [C] COMMITTED — author buys in, sees worth, puts in effort,     ║
║      endorses, signed/stays on.                                  ║
║      ✓ "NS made me who I am, grateful for the experience"        ║
║      ✓ "Signing on was the best decision I made"                 ║
║                                                                  ║
║  [U] UNCOMMITTED — author is disengaged / just clearing time /   ║
║      actively opposed to their own service.                      ║
║      ✓ "Just zao liao, don't care about the brotherhood stuff"   ║
║      ✓ "Waste of 2 years, I regret signing on"                   ║
║                                                                  ║
║  [N] NEUTRAL — no clear personal buy-in signal:                  ║
║      • Describing or quoting OTHERS ("my friend signed on")      ║
║      • Discussing the concept without personal stake             ║
║        ("sign on requirements are X")                            ║
║      • Sarcasm or rhetorical use of committed language           ║
║      • Questions about sign-on without stated intent             ║
║      • Future/hypothetical ("thinking of signing on")            ║
╠══════════════════════════════════════════════════════════════════╣
║  ⚠️  The TRIGGER TERM is shown above the text.                    ║
║     Check whether the author uses it in first-person committed   ║
║     sense OR as reference/discussion/sarcasm.                    ║
╚══════════════════════════════════════════════════════════════════╝
"""


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text)))


def load_queue() -> pd.DataFrame:
    if not QUEUE_PATH.exists():
        print(f"❌  {QUEUE_PATH.name} not found.")
        print("    Run: python -m src.features.committed_supplement_sampler")
        sys.exit(1)
    df = pd.read_csv(QUEUE_PATH)
    df["human_label"]  = df["human_label"].fillna("").astype(str)
    df["human_stance"] = df["human_stance"].fillna("").astype(str)
    return df


def print_report(df: pd.DataFrame):
    done     = df[df["human_label"].str.len() > 0]
    labelled = done[done["human_label"] != "skip"]
    print(f"\n{'═'*60}")
    print(f"  Committed supplement  ({len(done)}/{len(df)} done)")
    print(f"{'═'*60}")
    if len(labelled) == 0:
        print("  No labels yet.\n"); return

    print(f"\n  Label distribution:")
    for lbl in ("committed", "uncommitted", "neutral", "skip"):
        cnt = int((done["human_label"] == lbl).sum())
        if cnt:
            print(f"    {lbl:<12} {cnt:>3}  {'█'*cnt}")

    if len(labelled) > 0:
        committed_n = (labelled["human_label"] == "committed").sum()
        print(f"\n  Committed hit rate: {committed_n}/{len(labelled)} = "
              f"{committed_n/len(labelled)*100:.1f}%  "
              f"(baseline testset was 26/257 = 10.1%)")

    # Per-term precision
    if "trigger_term" in df.columns:
        print(f"\n  Precision by trigger term (committed / labelled hits):")
        for term, grp in labelled.groupby("trigger_term"):
            n = len(grp)
            n_c = (grp["human_label"] == "committed").sum()
            bar = "█" * n_c + "░" * (n - n_c)
            print(f"    {term:<30} {n_c:>2}/{n:<3} {n_c/n*100:5.1f}%  {bar}")

    print(f"\n  Run --merge to add to testset.parquet:")
    print(f"  python -m src.features.committed_supplement_annotator --merge")
    print(f"{'═'*60}\n")


def run():
    import termios, tty

    def getch() -> str:
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    df   = load_queue()
    todo = df.index[df["human_label"].str.len() == 0].tolist()

    if not todo:
        print("All chunks labelled.")
        print_report(df)
        return

    print(GUIDE)
    print(f"  {len(df)-len(todo)}/{len(df)} done — resuming\n")
    print("  [C]=committed  [U]=uncommitted  [N]=neutral  [S]=skip  [B]=back  [Q]=quit  [?]=guide")
    print("  Stance: [P]=supportive  [K]=critical  [N]=neutral\n")

    history: list[int] = []
    k = 0
    while k < len(todo):
        idx  = todo[k]
        row  = df.loc[idx]
        text = str(row["text"]).strip().replace("\n", " ")
        pos  = len(df) - len(todo) + len(history) + 1
        term = row.get("trigger_term", "?")

        print(f"  ── [{pos}/{len(df)}] ({row['doc_type']}, r/{row['subreddit']}, "
              f"{_word_count(text)}w) ──")
        print(f"  TRIGGER: '{term}'")
        print(f"\n  {text[:500]}\n")

        # Step 1: buyin
        print("  Buyin [C/U/N/S] > ", end="", flush=True)
        while True:
            ch = getch().lower()
            if ch in BUYIN_LABELS:
                lbl = BUYIN_LABELS[ch]
                print(f"→ {lbl.upper()}")
                df.at[idx, "human_label"] = lbl
                if lbl == "skip":
                    df.at[idx, "human_stance"] = ""
                    df.to_csv(QUEUE_PATH, index=False)
                    history.append(idx)
                    k += 1
                    print()
                    break
                # Step 2: stance
                print("  Stance [P/K/N] > ", end="", flush=True)
                while True:
                    ch2 = getch().lower()
                    if ch2 in STANCE_LABELS:
                        stance = STANCE_LABELS[ch2]
                        print(f"→ {stance.upper()}")
                        df.at[idx, "human_stance"] = stance
                        df.to_csv(QUEUE_PATH, index=False)
                        history.append(idx)
                        k += 1
                        print()
                        break
                    elif ch2 == "q":
                        df.at[idx, "human_label"] = ""
                        df.to_csv(QUEUE_PATH, index=False)
                        print("\nquit"); print_report(df); return
                    elif ch2 == "?":
                        print(); print(GUIDE)
                        print("  Stance [P/K/N] > ", end="", flush=True)
                break
            elif ch == "b":
                print("↩")
                if history:
                    prev = history.pop()
                    df.at[prev, "human_label"]  = ""
                    df.at[prev, "human_stance"] = ""
                    df.to_csv(QUEUE_PATH, index=False)
                    todo.insert(k, prev)
                    print("  ↩  Re-labelling previous chunk\n")
                else:
                    print("  (nothing to undo)\n")
                break
            elif ch == "q":
                print("quit"); df.to_csv(QUEUE_PATH, index=False)
                print_report(df); return
            elif ch == "?":
                print(); print(GUIDE)
                print("  Buyin [C/U/N/S] > ", end="", flush=True)

    print("\nAll done!")
    print_report(df)


def merge_into_testset():
    """Append labelled supplement rows into commitment_testset.parquet."""
    df = load_queue()
    labelled = df[
        (df["human_label"].str.len() > 0) &
        (df["human_label"] != "skip")
    ].copy()

    if len(labelled) == 0:
        print("No labelled rows to merge yet."); return

    ts = pd.read_parquet(TESTSET_PATH)
    existing_ids = set(ts["chunk_id"])
    new_rows = labelled[~labelled["chunk_id"].isin(existing_ids)].copy()
    if len(new_rows) == 0:
        print("All labelled rows already in testset."); return

    # Align columns to testset schema
    new_rows["queue_type"] = "committed_supplement"
    keep_cols = list(ts.columns)
    for c in keep_cols:
        if c not in new_rows.columns:
            new_rows[c] = None
    new_rows = new_rows[keep_cols]

    merged = pd.concat([ts, new_rows], ignore_index=True)
    merged.to_parquet(TESTSET_PATH, index=False)

    print(f"\nMerged {len(new_rows)} new rows into {TESTSET_PATH.name}")
    print(f"  Before: {len(ts)} rows  →  After: {len(merged)} rows")
    print(f"\n  New label distribution in supplement:")
    print(new_rows["human_label"].value_counts().to_string())
    print(f"\nRun recall check:")
    print(f"  python -m src.features.committed_recall_eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--merge",  action="store_true",
                    help="Merge labelled supplement rows into commitment_testset.parquet")
    args = ap.parse_args()
    if args.report:
        print_report(load_queue())
    elif args.merge:
        merge_into_testset()
    else:
        run()


if __name__ == "__main__":
    main()
