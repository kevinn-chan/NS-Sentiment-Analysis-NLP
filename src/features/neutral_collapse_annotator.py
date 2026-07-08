"""
Annotation CLI for the neutral-collapse supplement batches (U and C).

Reads:  data/processed/new/neutral_collapse_queue.csv
Writes: back to same file after every keypress (safe to quit and resume).

Collects BOTH axes in one pass:
  Axis 1 — Buyin:  [C]ommitted / [U]uncommitted / [N]eutral / [S]kip
  Axis 2 — Stance: [P]ositive-supportive / [K]critical / [N]eutral

Usage:
    python -m src.features.neutral_collapse_annotator            # label / resume
    python -m src.features.neutral_collapse_annotator --report   # stats only
    python -m src.features.neutral_collapse_annotator --merge    # merge into training CSV
"""
import argparse, re, sys
from pathlib import Path
import pandas as pd

DATA         = Path(__file__).resolve().parents[2] / "data" / "processed" / "new"
QUEUE_PATH   = DATA / "neutral_collapse_queue.csv"
TRAIN_PATH   = DATA / "commitment_manual_annotations.csv"
TESTSET_PATH = DATA / "commitment_testset.parquet"

BUYIN_LABELS  = {"c": "committed", "u": "uncommitted", "n": "neutral", "s": "skip"}
STANCE_LABELS = {"p": "supportive", "k": "critical",   "n": "neutral"}

BATCH_LABELS  = {"U": "UNCOMMITTED SIGNAL", "C": "COMMITTED SIGNAL"}

GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║  NEUTRAL COLLAPSE ANNOTATION  —  BOTH AXES                       ║
╠══════════════════════════════════════════════════════════════════╣
║  The model called this chunk NEUTRAL on both axes.               ║
║  Trigger phrase shown above — is the model wrong?                ║
╠══════════════════════════════════════════════════════════════════╣
║  AXIS 1 — BUYIN                                                  ║
║  [C] COMMITTED   — author personally buys in / endorses NS       ║
║  [U] UNCOMMITTED — author disengaged / opposed to their service  ║
║  [N] NEUTRAL     — no clear personal signal; model was right     ║
╠══════════════════════════════════════════════════════════════════╣
║  AXIS 2 — STANCE  (overall tone toward NS as an institution)     ║
║  [P] SUPPORTIVE  — positive, endorsing, appreciative             ║
║  [K] CRITICAL    — negative, opposing, complaining               ║
║  [N] NEUTRAL     — informational, balanced, no clear tone        ║
╠══════════════════════════════════════════════════════════════════╣
║  Batch U = uncommitted trigger  |  Batch C = committed trigger   ║
║  Either label is valid for either batch — trust what you read.   ║
╚══════════════════════════════════════════════════════════════════╝
"""


def _wc(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text)))


def load_queue() -> pd.DataFrame:
    if not QUEUE_PATH.exists():
        print(f"❌  {QUEUE_PATH.name} not found.")
        print("    Run: python -m src.features.neutral_collapse_sampler")
        sys.exit(1)
    df = pd.read_csv(QUEUE_PATH)
    for col in ("human_label", "human_stance"):
        df[col] = df[col].fillna("").astype(str)
    return df


def print_report(df: pd.DataFrame):
    done     = df[df["human_label"].str.len() > 0]
    labelled = done[done["human_label"] != "skip"]
    print(f"\n{'═'*60}")
    print(f"  Neutral-collapse supplement  ({len(done)}/{len(df)} done)")
    print(f"{'═'*60}")
    if not len(labelled):
        print("  No labels yet.\n"); return

    for batch in ("U", "C"):
        b = labelled[labelled["batch"] == batch]
        if not len(b): continue
        print(f"\n  Batch {batch} ({len(b)} labelled):")
        for axis, col in [("Buyin", "human_label"), ("Stance", "human_stance")]:
            vals = b[col].value_counts()
            parts = "  ".join(f"{k}={v}" for k, v in vals.items())
            print(f"    {axis}: {parts}")

    non_neutral = (labelled["human_label"] != "neutral").mean() * 100
    print(f"\n  Model-was-wrong rate (buyin): {non_neutral:.1f}%")
    print(f"  Run --merge to add to training set")
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
    # Resume from first row where EITHER axis is unlabelled
    todo = df.index[df["human_label"].str.len() == 0].tolist()

    if not todo:
        print("All chunks labelled.")
        print_report(df)
        return

    print(GUIDE)
    print(f"  {len(df)-len(todo)}/{len(df)} done — resuming\n")
    print("  Buyin : [C]=committed  [U]=uncommitted  [N]=neutral  [S]=skip")
    print("  Stance: [P]=supportive [K]=critical     [N]=neutral\n")
    print("  [B]=back  [Q]=quit  [?]=guide\n")

    history: list[int] = []
    k = 0
    while k < len(todo):
        idx  = todo[k]
        row  = df.loc[idx]
        text = str(row["text"]).strip().replace("\n", " ")
        pos  = len(df) - len(todo) + len(history) + 1

        print(f"  ── [{pos}/{len(df)}] Batch {row.get('batch','')} "
              f"| {BATCH_LABELS.get(str(row.get('batch','')), '')} "
              f"({row['doc_type']}, r/{row['subreddit']}, {_wc(text)}w) ──")
        print(f"  TRIGGER: '{row.get('trigger_phrase', '?')}'")
        print(f"\n  {text[:500]}\n")

        # ── Axis 1: Buyin ──────────────────────────────────────────────────
        print("  Buyin  [C/U/N/S] > ", end="", flush=True)
        buyin_done = False
        while not buyin_done:
            ch = getch().lower()
            if ch in BUYIN_LABELS:
                lbl = BUYIN_LABELS[ch]
                print(f"→ {lbl.upper()}")
                df.at[idx, "human_label"] = lbl
                if lbl == "skip":
                    df.at[idx, "human_stance"] = "skip"
                    df.to_csv(QUEUE_PATH, index=False)
                    history.append(idx); k += 1; print(); buyin_done = True
                    break
                buyin_done = True
            elif ch == "b":
                print("↩")
                if history:
                    prev = history.pop()
                    df.at[prev, "human_label"] = ""; df.at[prev, "human_stance"] = ""
                    df.to_csv(QUEUE_PATH, index=False)
                    todo.insert(k, prev)
                    print("  ↩  Re-labelling previous chunk\n")
                else:
                    print("  (nothing to undo)\n")
                buyin_done = True; k -= 1  # will be incremented below
                break
            elif ch == "q":
                df.to_csv(QUEUE_PATH, index=False); print("quit")
                print_report(df); return
            elif ch == "?":
                print(); print(GUIDE)
                print("  Buyin  [C/U/N/S] > ", end="", flush=True)

        if df.at[idx, "human_label"] in ("skip", ""):
            k += 1; continue

        # ── Axis 2: Stance ─────────────────────────────────────────────────
        print("  Stance [P/K/N]   > ", end="", flush=True)
        while True:
            ch = getch().lower()
            if ch in STANCE_LABELS:
                stance = STANCE_LABELS[ch]
                print(f"→ {stance.upper()}")
                df.at[idx, "human_stance"] = stance
                df.to_csv(QUEUE_PATH, index=False)
                history.append(idx); k += 1; print(); break
            elif ch == "q":
                df.to_csv(QUEUE_PATH, index=False); print("quit")
                print_report(df); return
            elif ch == "?":
                print(); print(GUIDE)
                print("  Stance [P/K/N]   > ", end="", flush=True)

    print("\nAll done!")
    print_report(df)


def merge_into_training():
    """Append labelled rows into commitment_manual_annotations.csv (Kaggle training set)
    AND into commitment_testset.parquet (eval set)."""
    df = load_queue()
    labelled = df[(df["human_label"].str.len() > 0) & (df["human_label"] != "skip")].copy()

    if not len(labelled):
        print("No labelled rows to merge yet."); return

    # ── 1. commitment_manual_annotations.csv (Kaggle training) ──────────────
    train = pd.read_csv(TRAIN_PATH)
    existing_ids = set(train["chunk_id"])
    new_train = labelled[~labelled["chunk_id"].isin(existing_ids)].copy()

    if len(new_train):
        new_train = new_train.rename(columns={
            "human_label":  "manual_buyin",
            "human_stance": "manual_stance",
        })
        # Compute similarity placeholders (0.5 = unknown; Kaggle recomputes from text)
        new_train["buyin_similarity"]  = 0.5
        new_train["stance_similarity"] = 0.5
        keep = ["chunk_id", "text", "buyin_similarity", "stance_similarity",
                "manual_buyin", "manual_stance"]
        new_train = new_train[keep]
        merged_train = pd.concat([train, new_train], ignore_index=True)
        merged_train.to_csv(TRAIN_PATH, index=False)
        print(f"\n✅  commitment_manual_annotations.csv")
        print(f"    Before: {len(train)}  After: {len(merged_train)}  (+{len(new_train)})")
        print(f"    manual_buyin:  {merged_train['manual_buyin'].value_counts().to_dict()}")
        print(f"    manual_stance: {merged_train['manual_stance'].value_counts().to_dict()}")
    else:
        print("All rows already in training CSV.")

    # ── 2. commitment_testset.parquet (eval) ─────────────────────────────────
    ts = pd.read_parquet(TESTSET_PATH)
    existing_ts = set(ts["chunk_id"])
    new_ts = labelled[~labelled["chunk_id"].isin(existing_ts)].copy()

    if len(new_ts):
        new_ts["queue_type"] = "neutral_collapse"
        for c in ts.columns:
            if c not in new_ts.columns:
                new_ts[c] = None
        new_ts = new_ts[list(ts.columns)]
        merged_ts = pd.concat([ts, new_ts], ignore_index=True)
        merged_ts.to_parquet(TESTSET_PATH, index=False)
        print(f"\n✅  commitment_testset.parquet")
        print(f"    Before: {len(ts)}  After: {len(merged_ts)}  (+{len(new_ts)})")

    print(f"\nNext steps:")
    print(f"  1. Upload commitment_manual_annotations.csv to Kaggle")
    print(f"  2. python -m src.features.committed_recall_eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--merge",  action="store_true",
                    help="Merge into commitment_manual_annotations.csv + testset.parquet")
    args = ap.parse_args()
    if args.report:
        print_report(load_queue())
    elif args.merge:
        merge_into_training()
    else:
        run()


if __name__ == "__main__":
    main()
