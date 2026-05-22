"""
Interactive sentiment annotation CLI.
Samples 200 chunks stratified by Singlish density and model disagreement,
then walks you through them one at a time for human labelling.

Usage:
    python -m src.features.annotator           # start / resume
    python -m src.features.annotator --report  # accuracy report only (no annotation)

Keys during annotation:
    n — negative    u — neutral    p — positive
    s — skip        q — quit & save

Progress is saved after every label — safe to quit and resume.
Final report is printed automatically when all chunks are labelled.
"""

import argparse
import re
import sys
import termios
import textwrap
import tty
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR        = Path(__file__).parent.parent.parent / "data" / "processed" / "new"
SAMPLE_PATH     = DATA_DIR / "annotation_sample.parquet"
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"

# ---------------------------------------------------------------------------
# Singlish density (same logic as audit)
# ---------------------------------------------------------------------------
_PARTICLES = {"lah","leh","lor","liao","sia","hor","mah","bah","wah","nia"}
_TERMS     = {
    "shiok","song","swee","steady","lobang","slack","lepak","sian","jialat",
    "wayang","chao keng","saikang","siong","tekan","kns","suay","teruk",
    "terok","bo chap","bochap","arrow","gg","bo liao","tok kok","cmi","kena",
    "walao","walau","siao","aiyah","aiyoh","alamak","cheem","liddat",
}
_ALL_MARKERS = _PARTICLES | _TERMS

def singlish_density(text: str) -> float:
    t = str(text).lower()
    tokens = re.findall(r"\b\w+\b", t)
    if not tokens:
        return 0.0
    hits = sum(1 for tok in tokens if tok in _ALL_MARKERS)
    return hits / len(tokens)


# ---------------------------------------------------------------------------
# Build / load sample
# ---------------------------------------------------------------------------
def build_sample() -> pd.DataFrame:
    """Construct stratified 200-chunk sample and save to parquet."""
    print("Building annotation sample …")

    # Load text
    sub = pd.read_parquet(DATA_DIR / "submissions_chunks.parquet",
                          columns=["chunk_id","text","subreddit","doc_type","score"])
    com = pd.read_parquet(DATA_DIR / "comments_chunks.parquet",
                          columns=["chunk_id","text","subreddit","doc_type","score"])
    chunks = pd.concat([sub, com], ignore_index=True)

    # Roberta scores
    rob = pd.read_parquet(DATA_DIR / "chunk_sentiment.parquet")
    rob["roberta_label"] = rob[["sent_neg","sent_neu","sent_pos"]].idxmax(axis=1).map(
        {"sent_neg":"negative","sent_neu":"neutral","sent_pos":"positive"}
    )
    rob["roberta_conf"] = rob[["sent_neg","sent_neu","sent_pos"]].max(axis=1)

    # Lexicon scores
    lex = pd.read_parquet(DATA_DIR / "chunk_sentiment_lexicon.parquet")

    df = (chunks
          .merge(rob[["chunk_id","sent_neg","sent_neu","sent_pos","roberta_label","roberta_conf"]], on="chunk_id")
          .merge(lex,                                                                                on="chunk_id"))

    df["sg_density"] = df["text"].apply(singlish_density)

    # Lexicon majority label
    def lex_label(c):
        if   c >  0.05: return "positive"
        elif c < -0.05: return "negative"
        else:           return "neutral"
    df["lex_label"] = df["sent_lexicon_compound"].apply(lex_label)

    # ── Strata ──────────────────────────────────────────────────────────────
    # 1. High Singlish (>6%): 80 chunks
    hi_sg = df[df["sg_density"] > 0.06].sample(min(80, (df["sg_density"] > 0.06).sum()),
                                                 random_state=42)
    hi_sg = hi_sg.assign(stratum="high_singlish")

    # 2. Medium Singlish (2–6%): 60 chunks
    med_sg = (df[(df["sg_density"] > 0.02) & (df["sg_density"] <= 0.06)]
              .sample(min(60, ((df["sg_density"] > 0.02) & (df["sg_density"] <= 0.06)).sum()),
                      random_state=42))
    med_sg = med_sg.assign(stratum="medium_singlish")

    # 3. Zero Singlish, roberta high-confidence (>0.85): 30 chunks
    hi_conf = (df[(df["sg_density"] == 0) & (df["roberta_conf"] > 0.85)]
               .sample(30, random_state=42))
    hi_conf = hi_conf.assign(stratum="high_conf_english")

    # 4. Roberta vs lexicon strongly disagree: 30 chunks
    disagree = df[
        ((df["roberta_label"] == "positive") & (df["sent_lexicon_compound"] < -0.2)) |
        ((df["roberta_label"] == "negative") & (df["sent_lexicon_compound"] >  0.2)) |
        ((df["roberta_label"] == "neutral")  & (df["sent_lexicon_compound"].abs() > 0.5))
    ]
    disagree = disagree[~disagree["chunk_id"].isin(
        pd.concat([hi_sg, med_sg, hi_conf])["chunk_id"]
    )].sample(min(30, len(disagree)), random_state=42)
    disagree = disagree.assign(stratum="model_disagree")

    sample = (pd.concat([hi_sg, med_sg, hi_conf, disagree], ignore_index=True)
                .drop_duplicates(subset="chunk_id")
                .sample(frac=1, random_state=99)   # shuffle order
                .reset_index(drop=True))

    sample["human_label"] = None
    sample.to_parquet(SAMPLE_PATH, index=False)

    counts = sample["stratum"].value_counts()
    print(f"Sample built: {len(sample)} chunks")
    for s, n in counts.items():
        print(f"  {s:<22} {n}")
    return sample


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------
LABEL_KEYS = {"n": "negative", "u": "neutral", "p": "positive"}

def getch() -> str:
    """Read a single keypress without requiring Enter."""
    fd = sys.stdin.fileno()
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
    codes = {"red":"31","yellow":"33","green":"32","cyan":"36","bold":"1","dim":"2","reset":"0"}
    return f"\033[{codes.get(code,'0')}m{text}\033[0m"


def fmt_label(label: str | None) -> str:
    if label == "negative": return color("negative", "red")
    if label == "positive": return color("positive", "green")
    if label == "neutral":  return color("neutral",  "yellow")
    return color(str(label), "dim")

def fmt_lex(score: float) -> str:
    bar_len  = 20
    mid      = bar_len // 2
    filled   = int(abs(score) * mid)
    if score > 0.05:
        bar = " " * mid + color("█" * filled, "green") + " " * (mid - filled)
        tag = color(f"+{score:.3f}", "green")
    elif score < -0.05:
        bar = " " * (mid - filled) + color("█" * filled, "red") + " " * mid
        tag = color(f"{score:.3f}", "red")
    else:
        bar = " " * mid + " " * mid
        tag = color(f"{score:.3f}", "dim")
    return f"[{bar}] {tag}"


# ---------------------------------------------------------------------------
# Annotation loop
# ---------------------------------------------------------------------------
def annotate():
    # Load or build sample
    if SAMPLE_PATH.exists():
        sample = pd.read_parquet(SAMPLE_PATH)
    else:
        sample = build_sample()

    # Load existing annotations if resuming
    if ANNOTATIONS_CSV.exists():
        done = pd.read_csv(ANNOTATIONS_CSV)
        done_ids = set(done["chunk_id"])
        print(f"Resuming — {len(done_ids)} already labelled.")
    else:
        done      = pd.DataFrame(columns=["chunk_id","human_label","stratum"])
        done_ids  = set()

    todo = sample[~sample["chunk_id"].isin(done_ids)].reset_index(drop=True)

    if len(todo) == 0:
        print("All chunks already labelled. Run with --report to see results.")
        return

    total_done  = len(done_ids)
    total_all   = len(sample)

    print(f"\n{total_done}/{total_all} labelled. {len(todo)} remaining.")
    print("Keys:  n=negative  u=neutral  p=positive  s=skip  q=quit\n")
    input("Press Enter to start …")

    new_rows = []

    for i, row in todo.iterrows():
        clear()

        # Progress
        n_done = total_done + len(new_rows)
        pct    = n_done / total_all * 100
        bar    = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(color(f" [{bar}] {n_done}/{total_all} ({pct:.0f}%)", "cyan"))
        print(color(f" Stratum: {row['stratum']}  |  "
                    f"r/{row['subreddit']}  |  "
                    f"{row['doc_type']}  |  "
                    f"score={int(row['score'])}  |  "
                    f"sg={row['sg_density']:.2f}", "dim"))
        print()

        # Text — wrap at 80 cols
        wrapped = textwrap.fill(str(row["text"]), width=80)
        print(color("┌─ Text " + "─" * 73, "dim"))
        for line in wrapped.split("\n"):
            print(f"│ {line}")
        print(color("└" + "─" * 79, "dim"))
        print()

        # Model scores
        print(f"  RoBERTa  →  {fmt_label(row['roberta_label'])}"
              f"  (neg={row['sent_neg']:.2f} neu={row['sent_neu']:.2f} pos={row['sent_pos']:.2f}  conf={row['roberta_conf']:.2f})")
        print(f"  Lexicon  →  {fmt_label(row['lex_label'])}  {fmt_lex(row['sent_lexicon_compound'])}")
        print()
        print(color("  n=negative  u=neutral  p=positive  s=skip  q=quit", "bold"))
        print("  Your label: ", end="", flush=True)

        # Get keypress
        while True:
            key = getch()
            if key in LABEL_KEYS:
                label = LABEL_KEYS[key]
                print(fmt_label(label))
                new_rows.append({"chunk_id": row["chunk_id"],
                                 "human_label": label,
                                 "stratum": row["stratum"]})
                break
            elif key == "s":
                print(color("skipped", "dim"))
                break
            elif key in ("q", "\x03"):   # q or Ctrl-C
                print(color("quit", "dim"))
                _save(done, new_rows)
                print(f"\nSaved. {total_done + len(new_rows)} labelled so far.")
                return
            # ignore other keys

    _save(done, new_rows)
    print(f"\nAll done! {total_done + len(new_rows)} labelled total.")
    _print_report()


def _save(existing: pd.DataFrame, new_rows: list):
    new_df  = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(ANNOTATIONS_CSV, index=False)


# ---------------------------------------------------------------------------
# Accuracy report
# ---------------------------------------------------------------------------
def _print_report():
    if not ANNOTATIONS_CSV.exists():
        print("No annotations yet.")
        return

    ann    = pd.read_csv(ANNOTATIONS_CSV)
    sample = pd.read_parquet(SAMPLE_PATH)
    df     = ann.merge(sample[["chunk_id","roberta_label","lex_label",
                                "sent_neg","sent_neu","sent_pos",
                                "sent_lexicon_compound","sg_density"]],
                       on="chunk_id", how="left")

    n = len(df)
    if n == 0:
        print("No annotations found.")
        return

    print(f"\n{'═'*60}")
    print(f"  Annotation report  ({n} chunks labelled)")
    print(f"{'═'*60}")

    # Overall
    rob_agree = (df["human_label"] == df["roberta_label"]).mean()
    lex_agree = (df["human_label"] == df["lex_label"]).mean()
    print(f"\n  Overall accuracy")
    print(f"    RoBERTa  vs human:  {rob_agree:.1%}")
    print(f"    Lexicon  vs human:  {lex_agree:.1%}")

    # By stratum
    print(f"\n  RoBERTa accuracy by stratum")
    for stratum, grp in df.groupby("stratum"):
        acc = (grp["human_label"] == grp["roberta_label"]).mean()
        n_s = len(grp)
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        flag = "  ← LOW" if acc < 0.70 else ""
        print(f"    {stratum:<22} [{bar}] {acc:.1%}  (n={n_s}){flag}")

    # Singlish bins
    print(f"\n  RoBERTa accuracy by Singlish density")
    bins   = [-0.001, 0.0, 0.02, 0.06, 0.15, 1.0]
    labels = ["zero","low","medium","high","very high"]
    df["sg_bin"] = pd.cut(df["sg_density"], bins=bins, labels=labels)
    for sg_bin, grp in df.groupby("sg_bin", observed=True):
        if len(grp) < 3:
            continue
        acc = (grp["human_label"] == grp["roberta_label"]).mean()
        n_s = len(grp)
        flag = "  ← LOW" if acc < 0.70 else ""
        print(f"    {str(sg_bin):<12} {acc:.1%}  (n={n_s}){flag}")

    # Confusion matrix
    print(f"\n  Confusion matrix (human rows × roberta cols)")
    lbl_order = ["negative","neutral","positive"]
    print(f"    {'':12}  " + "  ".join(f"{l:>10}" for l in lbl_order))
    for h in lbl_order:
        row_data = df[df["human_label"] == h]["roberta_label"].value_counts()
        counts   = [row_data.get(l, 0) for l in lbl_order]
        print(f"    human={h:<9}" + "  ".join(f"{c:>10}" for c in counts))

    # Human label distribution
    print(f"\n  Human label distribution")
    for lbl, cnt in df["human_label"].value_counts().items():
        print(f"    {lbl:<12} {cnt:>4}  ({cnt/n*100:.1f}%)")

    # Verdict
    print(f"\n  {'─'*56}")
    if rob_agree >= 0.80:
        print("  ✓ RoBERTa accuracy ≥ 80% — acceptable as primary signal.")
    elif rob_agree >= 0.70:
        print("  ~ RoBERTa accuracy 70–80% — usable but note limitations.")
        print("    Supplement with Tier 2 lexicon for Singlish-heavy chunks.")
    else:
        print("  ✗ RoBERTa accuracy < 70% — consider SingBERT fine-tuning.")
        print("    These annotations are your seed training data.")
    print(f"  {'─'*56}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true",
                        help="Print accuracy report without annotating")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild annotation sample from scratch")
    args = parser.parse_args()

    if args.rebuild and SAMPLE_PATH.exists():
        SAMPLE_PATH.unlink()
        print("Cleared existing sample.")

    if args.report:
        _print_report()
    else:
        annotate()
