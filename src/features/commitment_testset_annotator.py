"""
Manual labelling CLI for the Stage 5b commitment TEST set — dual-axis edition.

Hand-labels commitment_testset_queue.csv (200 minority-enriched chunks) on TWO
independent axes:
  Axis 1 (buyin):  committed / uncommitted / neutral — personal investment in own service
  Axis 2 (stance): supportive / critical / neutral   — institutional opinion on NS

Labels are written back into the queue's `human_label` (buyin) and `human_stance`
columns in place, after every keypress, so it is safe to quit (Q) and resume at
any time. Blind by design: the sent_neg / lexicon signals used to select the chunk
are NOT shown.

Keys — Step 1 (buyin):  [C]=committed  [U]=uncommitted  [N]=neutral  [S]=skip  [B]=back  [Q]=quit  [?]=guide
Keys — Step 2 (stance): [S]=supportive  [K]=critical  [N]=neutral

Usage:
    python -m src.features.commitment_testset_annotator                  # label / resume
    python -m src.features.commitment_testset_annotator --report         # progress + label dist
    python -m src.features.commitment_testset_annotator --backfill-stance  # add stance to already-labelled chunks
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DATA_DIR   = Path(__file__).parent.parent.parent / "data" / "processed" / "new"
QUEUE_PATH = DATA_DIR / "commitment_testset_queue.csv"

BUYIN_LABELS  = {"c": "committed", "u": "uncommitted", "n": "neutral", "s": "skip"}
STANCE_LABELS = {"s": "supportive", "k": "critical", "n": "neutral"}  # k for kritical to avoid clash with s=skip

GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║    DUAL-AXIS COMMITMENT LABELLING GUIDE                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  TWO independent axes per chunk. Label both.                     ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  AXIS 1: BUYIN — how invested is the author in THEIR OWN NS?    ║
║  Not their mood — their level of buy-in / effort / belief in NS. ║
╠══════════════════════════════════════════════════════════════════╣
║  [C] COMMITTED — buys in. Takes NS seriously, sees worth in it,  ║
║      puts in effort, endorses it, or signs/stays on.             ║
║      · "NS made me stronger, glad I served"                      ║
║      · "Tough but I gave it my all, signing on"                  ║
║      · "Proud of my unit, the brotherhood is real"               ║
║                                                                  ║
║  [U] UNCOMMITTED — disengaged. Apathy / minimal effort /         ║
║      just clearing time:                                         ║
║      · "Only here to finish 2 years, zao liao, don't care"      ║  ← canonical
║      · "Chao keng all the way, slack until ORD"                  ║
║      · "Counting down, doing the bare minimum"                   ║
║                                                                  ║
║  [N] NEUTRAL — practical/factual, no personal buy-in signal.     ║
║      · "When is my enlistment date?"                             ║
║      · "BMT lasts 9 weeks, here's what to expect"               ║
║      · "Got one chao keng guy in my section" (describing OTHERS  ║
║         → buyin NEUTRAL)                                         ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  AXIS 2: STANCE — does the author endorse or oppose NS as an    ║
║  institution?                                                    ║
╠══════════════════════════════════════════════════════════════════╣
║  [S] SUPPORTIVE — NS is worthwhile, necessary, a positive thing. ║
║      · "NS is necessary, we need it for defence"                 ║
║      · "Proud to serve my country"                               ║
║      · "We need NS with such neighbours"                         ║
║                                                                  ║
║  [K] CRITICAL — NS is unfair, a waste, should be abolished,     ║
║      exploitative, caused lasting harm.                          ║
║      · "NS is a waste of 2 years, total slavery"                 ║
║      · "Doesn't translate to real world at all"                  ║
║      · "Should be abolished, end conscription"                   ║
║                                                                  ║
║  [N] NEUTRAL — no institutional opinion expressed.               ║
║      · Describing OTHERS' views on NS → stance NEUTRAL           ║
║      · Apathy without critique → stance NEUTRAL                  ║
║      · Personal frustrations without institutional critique      ║
║        → stance NEUTRAL                                          ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  ⚠️  KEY INSIGHT: AXES ARE INDEPENDENT                             ║
║                                                                  ║
║  committed + supportive :  "NS made me who I am, glad I served"  ║
║  uncommitted + supportive: "NS is necessary but I just want ORD" ║
║  committed + critical   :  "I gave it my all but system broken"  ║
║  uncommitted + neutral  :  "Only here finish 2 years, zao liao"  ║
║  uncommitted + critical :  "Waste of 2 years, should abolish"    ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  ⚠️  COMMITMENT ≠ SENTIMENT                                       ║
║    "Hate waking at 5am but NS made me stronger" → C buyin        ║
║    "Camp food sucks"  → N buyin, N stance (gripe, not a stance)  ║
║    "Finally ORD, so happy!" → N buyin, N stance (relief)         ║
║                                                                  ║
║  ⚠️  3rd-person mentions of keng/wayang = describing others →     ║
║      NEUTRAL on BOTH axes.                                       ║
╚══════════════════════════════════════════════════════════════════╝
"""


def _word_count(text: str) -> int:
    import re
    return len(re.findall(r"\b\w+\b", str(text)))


def load_queue() -> pd.DataFrame:
    if not QUEUE_PATH.exists():
        print(f"❌  {QUEUE_PATH.name} not found. Run: python -m src.features.commitment_sampler")
        sys.exit(1)
    df = pd.read_csv(QUEUE_PATH)
    if "human_label" not in df.columns:
        df["human_label"] = ""
    df["human_label"] = df["human_label"].fillna("").astype(str)
    if "human_stance" not in df.columns:
        df["human_stance"] = ""
    df["human_stance"] = df["human_stance"].fillna("").astype(str)
    return df


def print_report(df: pd.DataFrame):
    done     = df[df["human_label"].str.len() > 0]
    labelled = done[done["human_label"] != "skip"]
    print(f"\n{'═'*60}")
    print(f"  Commitment TEST-set labelling  ({len(done)}/{len(df)} done)")
    print(f"{'═'*60}")
    if len(done) == 0:
        print("  No labels yet.\n"); return
    print(f"\n  Labelled: {len(labelled)}   Skipped: {(done['human_label']=='skip').sum()}")

    # Buyin distribution
    print(f"\n  BUYIN distribution:")
    for lbl in ("committed", "uncommitted", "neutral"):
        cnt = int((labelled["human_label"] == lbl).sum())
        bar = "█" * cnt
        print(f"    {lbl:<12} {cnt:>3}  {bar}")

    # Stance distribution
    has_stance = labelled[labelled["human_stance"].str.len() > 0]
    if len(has_stance) > 0:
        print(f"\n  STANCE distribution ({len(has_stance)} labelled):")
        for lbl in ("supportive", "critical", "neutral"):
            cnt = int((has_stance["human_stance"] == lbl).sum())
            bar = "█" * cnt
            print(f"    {lbl:<12} {cnt:>3}  {bar}")

        # Joint distribution crosstab
        print(f"\n  Joint distribution (buyin × stance):")
        ct = pd.crosstab(
            has_stance["human_label"], has_stance["human_stance"],
            margins=True, margins_name="Total"
        )
        print(ct.to_string(col_space=10))
    else:
        no_stance = len(labelled) - len(has_stance)
        if no_stance > 0:
            print(f"\n  STANCE: {no_stance} chunks missing stance labels.")
            print(f"  Run:  python -m src.features.commitment_testset_annotator --backfill-stance")

    minority = int(labelled["human_label"].isin(["committed", "uncommitted"]).sum())
    print(f"\n  Minority (committed+uncommitted): {minority}  "
          f"← this is your usable recall test set")
    print(f"{'═'*60}\n")


def run():
    import termios
    import tty

    def getch() -> str:
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    df    = load_queue()
    todo  = df.index[df["human_label"].str.len() == 0].tolist()
    total = len(df)

    if not todo:
        print("All chunks labelled.")
        print_report(df)
        return

    print(GUIDE)
    print(f"  {total - len(todo)}/{total} done — resuming\n")
    print("  Step 1 (buyin):  [C]=committed  [U]=uncommitted  [N]=neutral  [S]=skip  [B]=back  [Q]=quit  [?]=guide")
    print("  Step 2 (stance): [S]=supportive  [K]=critical  [N]=neutral")
    print("  ⚠️  Blind — label on text alone.\n")

    history: list[int] = []   # stack of labelled row-indices this session, for [B]
    k = 0
    while k < len(todo):
        idx  = todo[k]
        row  = df.loc[idx]
        text = str(row["text"]).strip().replace("\n", " ")
        pos  = total - len(todo) + len(history) + 1
        print(f"  ── [{pos}/{total}] ({row['doc_type']}, r/{row['subreddit']}, "
              f"{_word_count(text)}w) ──")
        print(f"\n  {text[:400]}\n")

        # --- Step 1: buyin ---
        print("  Step 1 (buyin) > ", end="", flush=True)
        buyin_done = False
        while not buyin_done:
            ch = getch().lower()
            if ch in BUYIN_LABELS:
                lbl = BUYIN_LABELS[ch]
                print(f"→ {lbl.upper()}")
                df.at[idx, "human_label"] = lbl
                if lbl == "skip":
                    # Skip step 2 too
                    df.at[idx, "human_stance"] = ""
                    df.to_csv(QUEUE_PATH, index=False)
                    history.append(idx)
                    k += 1
                    print()
                    buyin_done = True
                    continue
                buyin_done = True
            elif ch == "q":
                print("quit")
                df.to_csv(QUEUE_PATH, index=False)
                print_report(df)
                return
            elif ch == "b":
                print("↩")
                if history:
                    prev = history.pop()
                    df.at[prev, "human_label"] = ""
                    df.at[prev, "human_stance"] = ""
                    df.to_csv(QUEUE_PATH, index=False)
                    todo.insert(k, prev)   # re-queue it next
                    print("  ↩  Re-labelling previous chunk\n")
                else:
                    print("  (nothing to undo this session)\n")
                buyin_done = True  # break inner loop to re-render
                continue
            elif ch == "?":
                print(); print(GUIDE); print("  Step 1 (buyin) > ", end="", flush=True)

        # If we got here via back or skip, continue to next iteration
        if df.at[idx, "human_label"] == "" or df.at[idx, "human_label"] == "skip":
            continue

        # --- Step 2: stance ---
        print("  Step 2 (stance) > ", end="", flush=True)
        stance_done = False
        while not stance_done:
            ch = getch().lower()
            if ch in STANCE_LABELS:
                lbl = STANCE_LABELS[ch]
                print(f"→ {lbl.upper()}")
                df.at[idx, "human_stance"] = lbl
                df.to_csv(QUEUE_PATH, index=False)
                history.append(idx)
                k += 1
                print()
                stance_done = True
            elif ch == "q":
                # Undo the buyin we just set since stance not done
                df.at[idx, "human_label"] = ""
                print("quit")
                df.to_csv(QUEUE_PATH, index=False)
                print_report(df)
                return
            elif ch == "?":
                print(); print(GUIDE); print("  Step 2 (stance) > ", end="", flush=True)

    print("\nAll done!")
    print_report(df)


def run_backfill_stance():
    """Backfill stance labels for chunks that already have buyin labels."""
    import termios
    import tty

    def getch() -> str:
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    df = load_queue()

    # Find chunks with buyin label but no stance
    needs_stance = df[
        (df["human_label"].str.len() > 0) &
        (df["human_label"] != "skip") &
        (df["human_stance"].str.len() == 0)
    ]

    if len(needs_stance) == 0:
        print("All labelled chunks already have stance labels.")
        print_report(df)
        return

    todo = needs_stance.index.tolist()
    total_backfill = len(todo)

    print(GUIDE)
    print(f"\n  BACKFILL MODE — adding stance to {total_backfill} already-labelled chunks.")
    print(f"  Existing buyin label shown alongside text for context.\n")
    print("  Keys: [S]=supportive  [K]=critical  [N]=neutral  [Q]=quit  [?]=guide\n")

    done_count = 0
    for i, idx in enumerate(todo):
        row  = df.loc[idx]
        text = str(row["text"]).strip().replace("\n", " ")
        buyin = row["human_label"]
        print(f"  ── [{i+1}/{total_backfill}] ({row['doc_type']}, r/{row['subreddit']}, "
              f"{_word_count(text)}w) ──")
        print(f"  Buyin: {buyin.upper()}")
        print(f"\n  {text[:400]}\n")
        print("  Stance > ", end="", flush=True)

        while True:
            ch = getch().lower()
            if ch in STANCE_LABELS:
                lbl = STANCE_LABELS[ch]
                print(f"→ {lbl.upper()}")
                df.at[idx, "human_stance"] = lbl
                df.to_csv(QUEUE_PATH, index=False)
                done_count += 1
                print()
                break
            elif ch == "q":
                print("quit")
                df.to_csv(QUEUE_PATH, index=False)
                print(f"\n  Backfilled {done_count}/{total_backfill} chunks.")
                print_report(df)
                return
            elif ch == "?":
                print(); print(GUIDE); print("  Stance > ", end="", flush=True)

    print(f"\nBackfill complete — {done_count} chunks updated.")
    print_report(df)


def main():
    ap = argparse.ArgumentParser(description="Label the commitment TEST set — dual-axis (blind).")
    ap.add_argument("--report", action="store_true", help="show progress and exit")
    ap.add_argument("--backfill-stance", action="store_true",
                    help="add stance labels to chunks that already have buyin labels")
    args = ap.parse_args()
    if args.report:
        print_report(load_queue())
    elif getattr(args, "backfill_stance", False):
        run_backfill_stance()
    else:
        run()


if __name__ == "__main__":
    main()
