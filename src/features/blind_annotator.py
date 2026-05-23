"""
Blind annotation CLI for a fair XLM vs RoBERTa head-to-head.

Annotation phase: shows text only — no model scores, no predictions,
no confidence values. You label on the text alone.

Comparison phase (--compare): loads both models post-hoc and scores the
same chunks against your human labels. Produces corpus-weighted accuracy,
per-stratum breakdown, and a disagreement analysis.

Usage:
    python -m src.features.blind_annotator              # annotate (blind)
    python -m src.features.blind_annotator --report     # progress only
    python -m src.features.blind_annotator --compare    # run both models + h2h report
    python -m src.features.blind_annotator --reset      # start fresh

Keys during annotation:
    P   — POSITIVE
    N   — NEGATIVE
    U   — NEUTRAL
    S   — skip (genuinely ambiguous / not NS-related enough)
    Q   — quit and save progress
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
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "new"
ANN_PATH = DATA_DIR / "blind_annotation.csv"
CMP_PATH = DATA_DIR / "blind_annotation_comparison.csv"

CHUNK_COLS = ["chunk_id", "doc_type", "subreddit", "text"]

# Actual corpus proportions (measured from 738,819 chunks)
SINGLISH_PROP = 0.047   # 34,834 chunks with ≥1 Singlish marker
ENGLISH_PROP  = 0.953

# ---------------------------------------------------------------------------
# Singlish detector (word-boundary regex, same vocabulary as spot_checker.py)
# ---------------------------------------------------------------------------
_VOCAB = {
    "lah", "leh", "lor", "liao", "sia", "hor", "mah", "bah", "nia",
    "sian", "jialat", "wayang", "chao keng", "saikang", "siong", "kena",
    "walao", "walau", "siao", "aiyah", "aiyoh", "alamak", "cheem", "liddat",
    "shiok", "song", "swee", "steady", "lobang", "slack", "lepak",
    "bochap", "bo chap", "gg", "tekan", "suay", "teruk", "terok",
    "arrow", "bo liao", "tok kok", "cmi", "kns", "tok gong", "powderful",
    "gao gao", "die die", "wah", "sibei",
}
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_VOCAB, key=len, reverse=True)) + r")\b"
)


def has_singlish(text: str) -> bool:
    return bool(_PATTERN.search(str(text).lower()))


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text)))


# ---------------------------------------------------------------------------
# Sample selection — 100 chunks, stratified on TEXT CHARACTERISTICS ONLY
# No model confidence, no model predictions used in any filter.
# ---------------------------------------------------------------------------
STRATA = [
    # (name,             filter_dict,                                              target_n)
    ("singlish",         {"singlish": True},                                       20),
    ("submissions",      {"doc_type": "submission", "singlish": False},            20),
    ("short_comment",    {"doc_type": "comment", "singlish": False,
                          "wc_max": 80},                                            20),
    ("medium_comment",   {"doc_type": "comment", "singlish": False,
                          "wc_min": 80, "wc_max": 250},                            20),
    ("long_comment",     {"doc_type": "comment", "singlish": False,
                          "wc_min": 250},                                           20),
]

# Corpus weight assigned to each stratum for weighted accuracy calculation.
# singlish = 4.7% of corpus; remaining 95.3% split equally over 4 non-singlish strata.
STRATUM_CORPUS_PROP = {
    "singlish":        SINGLISH_PROP,
    "submissions":     ENGLISH_PROP / 4,
    "short_comment":   ENGLISH_PROP / 4,
    "medium_comment":  ENGLISH_PROP / 4,
    "long_comment":    ENGLISH_PROP / 4,
}


def build_sample(chunks: pd.DataFrame) -> pd.DataFrame:
    df = chunks.copy()
    df["has_sg"] = df["text"].fillna("").apply(has_singlish)
    df["wc"]     = df["text"].fillna("").apply(word_count)
    df["text"]   = df["text"].fillna("").str.strip()
    df           = df[df["text"].str.len() > 0]

    parts = []
    for stratum, filters, n in STRATA:
        g = df.copy()
        if "singlish" in filters:
            g = g[g["has_sg"] == filters["singlish"]]
        if "doc_type" in filters:
            g = g[g["doc_type"] == filters["doc_type"]]
        if "wc_min" in filters:
            g = g[g["wc"] >= filters["wc_min"]]
        if "wc_max" in filters:
            g = g[g["wc"] <  filters["wc_max"]]
        if parts:
            used = pd.concat(parts)["chunk_id"].values
            g = g[~g["chunk_id"].isin(used)]
        sampled          = g.sample(min(n, len(g)), random_state=7)
        sampled          = sampled.copy()
        sampled["stratum"] = stratum
        parts.append(sampled)

    sample = (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=13)
        .reset_index(drop=True)
    )
    return sample[["chunk_id", "stratum", "doc_type", "subreddit",
                   "has_sg", "wc", "text"]]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_results() -> pd.DataFrame:
    if ANN_PATH.exists():
        return pd.read_csv(ANN_PATH)
    return pd.DataFrame(columns=[
        "chunk_id", "stratum", "doc_type", "subreddit",
        "has_sg", "wc", "text", "human_label",
    ])


def save_results(df: pd.DataFrame):
    df.to_csv(ANN_PATH, index=False)


# ---------------------------------------------------------------------------
# Progress report (annotation stats only — no model info)
# ---------------------------------------------------------------------------
def print_progress(results: pd.DataFrame):
    done = results.dropna(subset=["human_label"])
    n    = len(done)
    total = sum(t for _, _, t in STRATA)

    print(f"\n{'═'*55}")
    print(f"  Blind annotation progress  ({n}/{total} done)")
    print(f"{'═'*55}")

    if n == 0:
        print("  No annotations yet.")
        print(f"{'═'*55}\n")
        return

    skipped  = (done["human_label"] == "skip").sum()
    labelled = n - skipped

    print(f"\n  Labelled : {labelled}   Skipped : {skipped}")
    print(f"\n  Label distribution (non-skip):")
    dist = done[done["human_label"] != "skip"]["human_label"].value_counts()
    for lbl in ["negative", "neutral", "positive"]:
        cnt = int(dist.get(lbl, 0))
        bar = "█" * int((cnt / labelled * 25) if labelled else 0)
        print(f"    {lbl:<10} {cnt:>3}  {bar}")

    print(f"\n  By stratum:")
    for stratum, _, target_n in STRATA:
        g   = done[done["stratum"] == stratum]
        cnt = len(g[g["human_label"] != "skip"])
        print(f"    {stratum:<18} {cnt:>2}/{target_n}")

    print(f"\n{'═'*55}\n")
    if n >= total:
        print("  ✅  All chunks labelled. Run --compare to see model results.\n")


# ---------------------------------------------------------------------------
# Annotation loop
# ---------------------------------------------------------------------------
def run():
    import tty, termios

    def getch() -> str:
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print("Loading chunk data ...")
    sub    = pd.read_parquet(DATA_DIR / "submissions_chunks.parquet", columns=CHUNK_COLS)
    com    = pd.read_parquet(DATA_DIR / "comments_chunks.parquet",    columns=CHUNK_COLS)
    chunks = pd.concat([sub, com], ignore_index=True).drop_duplicates("chunk_id")
    del sub, com

    sample  = build_sample(chunks)
    results = load_results()

    done_ids = set(results["chunk_id"].dropna()) if len(results) > 0 else set()
    todo     = sample[~sample["chunk_id"].isin(done_ids)].reset_index(drop=True)

    total    = len(sample)
    reviewed = len(done_ids)

    if len(todo) == 0:
        print("All chunks annotated.")
        print_progress(results)
        print("Run --compare to score both models against your labels.")
        return

    print(f"  {reviewed}/{total} done — resuming from #{reviewed + 1}\n")
    print("  Keys: [P]=positive  [N]=negative  [U]=neutral  [S]=skip  [Q]=quit\n")
    print("  ⚠️  No model predictions shown. Label on text alone.\n")

    for _, row in todo.iterrows():
        reviewed += 1
        sg_flag  = "🇸🇬 " if row["has_sg"] else "   "
        text     = str(row["text"]).strip().replace("\n", " ")

        print(f"  ── [{reviewed}/{total}] {sg_flag} {row['stratum']:<18} "
              f"({row['doc_type']}, r/{row['subreddit']}, {row['wc']}w) ──")
        print(f"\n  {text[:350]}\n")
        print("  > ", end="", flush=True)

        while True:
            ch = getch().lower()
            if ch == "p":
                label = "positive"; print("→ POSITIVE"); break
            elif ch == "n":
                label = "negative"; print("→ NEGATIVE"); break
            elif ch == "u":
                label = "neutral";  print("→ NEUTRAL");  break
            elif ch == "s":
                label = "skip";     print("skip");       break
            elif ch == "q":
                print("quit")
                print_progress(results)
                return

        new_row = row.to_dict()
        new_row["human_label"] = label
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        save_results(results)
        print()

    print("\nAll done!")
    print_progress(results)
    print("Run --compare to score both models against your labels.")


# ---------------------------------------------------------------------------
# Model comparison — runs post-hoc, never touches annotation phase
# ---------------------------------------------------------------------------
def run_comparison():
    results  = load_results()
    labelled = results[
        results["human_label"].notna() & (results["human_label"] != "skip")
    ].copy().reset_index(drop=True)

    if len(labelled) == 0:
        print("No labelled chunks yet. Run annotation first.")
        return

    n = len(labelled)
    print(f"\nRunning comparison on {n} labelled chunks ...")

    texts = labelled["text"].fillna("").str.strip().tolist()

    try:
        import torch
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("transformers not installed. Run: pip install transformers sentencepiece")
        return

    device       = 0 if torch.cuda.is_available() else -1
    device_label = "GPU" if device == 0 else "CPU"
    print(f"Using: {device_label}\n")

    # Normalise labels coming out of HuggingFace (some models return LABEL_0/1/2)
    _LMAP = {
        "negative": "negative", "neutral": "neutral", "positive": "positive",
        "LABEL_0":  "negative", "LABEL_1": "neutral", "LABEL_2":  "positive",
        "Negative": "negative", "Neutral": "neutral", "Positive": "positive",
    }

    MODELS = {
        "XLM":       "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "RoBERTa":   "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "RoBERTa-L": "j-hartmann/sentiment-roberta-large-english-3-classes",
    }

    model_preds = {}
    for name, model_id in MODELS.items():
        print(f"  Loading {name} ({model_id}) ...")
        clf = hf_pipeline(
            "sentiment-analysis",
            model=model_id,
            top_k=None,
            device=device,
            truncation=True,
            max_length=512,
        )
        raw   = clf(texts, batch_size=32)
        preds = []
        for scores in raw:
            score_map = {_LMAP.get(s["label"], s["label"].lower()): s["score"]
                         for s in scores}
            preds.append(max(score_map, key=score_map.get))
        model_preds[name] = preds
        del clf
        print(f"    Done.")

    # Build results dataframe
    df = labelled.copy()
    for name, preds in model_preds.items():
        df[f"pred_{name}"]    = preds
        df[f"correct_{name}"] = df[f"pred_{name}"] == df["human_label"]

    # Corpus weights
    df["corpus_weight"] = df["stratum"].map(STRATUM_CORPUS_PROP)
    total_weight        = df["corpus_weight"].sum()

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  Fair H2H — XLM vs RoBERTa  ({n} chunks, blind annotation)")
    print(f"{'═'*65}")

    # ── Corpus-weighted accuracy ──────────────────────────────────────────────
    print(f"\n  ── Corpus-weighted accuracy ──")
    cw = {}
    for name in MODELS:
        cw[name] = (df[f"correct_{name}"] * df["corpus_weight"]).sum() / total_weight
        marker   = "  ← winner" if cw[name] == max(cw.values()) and len(cw) == len(MODELS) else ""
        print(f"    {name:<10}  {cw[name]:.1%}{marker}")

    # ── Sample accuracy (unweighted) ─────────────────────────────────────────
    print(f"\n  ── Sample accuracy (unweighted) ──")
    for name in MODELS:
        acc = df[f"correct_{name}"].mean()
        print(f"    {name:<10}  {acc:.1%}  (n={n})")

    # ── Per-stratum breakdown ─────────────────────────────────────────────────
    print(f"\n  ── Accuracy by stratum ──")
    for stratum, g in df.groupby("stratum"):
        print(f"\n    {stratum}  (n={len(g)}, corpus_weight={STRATUM_CORPUS_PROP[stratum]:.1%})")
        for name in MODELS:
            acc = g[f"correct_{name}"].mean()
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"      {name:<10} [{bar}] {acc:.1%}")

    # ── Confusion matrices ────────────────────────────────────────────────────
    print(f"\n  ── Confusion (human label → model prediction) ──")
    for name in MODELS:
        print(f"\n    {name}:")
        for true_lbl in ["negative", "neutral", "positive"]:
            g    = df[df["human_label"] == true_lbl]
            if len(g) == 0:
                continue
            dist = g[f"pred_{name}"].value_counts()
            row  = []
            for pred_lbl in ["negative", "neutral", "positive"]:
                cnt  = int(dist.get(pred_lbl, 0))
                mark = "✓" if pred_lbl == true_lbl else " "
                row.append(f"{mark}{pred_lbl[0].upper()}:{cnt}")
            print(f"      human={true_lbl:<10} →  {'  '.join(row)}  (n={len(g)})")

    # ── Disagreement analysis ─────────────────────────────────────────────────
    print(f"\n  ── Where models disagree with each other ──")
    disagree = df[df["pred_XLM"] != df["pred_RoBERTa"]]
    print(f"    {len(disagree)} chunks disagreed ({len(disagree)/n*100:.1f}%)")
    if len(disagree) > 0:
        xlm_right  = (disagree["correct_XLM"]    & ~disagree["correct_RoBERTa"]).sum()
        rob_right  = (disagree["correct_RoBERTa"] & ~disagree["correct_XLM"]).sum()
        both_wrong = (~disagree["correct_XLM"] & ~disagree["correct_RoBERTa"]).sum()
        both_right = (disagree["correct_XLM"]    &  disagree["correct_RoBERTa"]).sum()
        print(f"    XLM correct, RoBERTa wrong  : {xlm_right}")
        print(f"    RoBERTa correct, XLM wrong  : {rob_right}")
        print(f"    Both wrong                  : {both_wrong}")
        print(f"    Both right (different path) : {both_right}")

    # ── Label distribution (human vs models) ─────────────────────────────────
    print(f"\n  ── Label distribution comparison ──")
    for col, label in [("human_label", "Human"), ("pred_XLM", "XLM"),
                        ("pred_RoBERTa", "RoBERTa")]:
        vc    = df[col].value_counts(normalize=True)
        parts = [f"{lbl[0].upper()}:{vc.get(lbl, 0)*100:.0f}%"
                 for lbl in ["negative", "neutral", "positive"]]
        print(f"    {label:<10}  {' / '.join(parts)}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    delta  = cw["XLM"] - cw["RoBERTa"]
    winner = "XLM" if delta > 0 else "RoBERTa"
    margin = abs(delta)

    print(f"\n  ── Verdict ──")
    margin_pp = margin * 100
    if margin_pp < 3:
        print(f"    ⚠️  {winner} wins by {margin_pp:.1f}pp — too close to call at this sample size.")
        print(f"       Cannot conclude one model is definitively better.")
    elif margin_pp < 7:
        print(f"    {winner} wins by {margin_pp:.1f}pp corpus-weighted accuracy.")
        print(f"    Modest margin — direction is reliable, exact magnitude uncertain.")
    else:
        print(f"    {winner} wins clearly by {margin_pp:.1f}pp corpus-weighted accuracy.")

    print(f"{'═'*65}\n")

    # Save full comparison
    df.to_csv(CMP_PATH, index=False)
    print(f"  Full results saved → {CMP_PATH}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Blind annotation CLI for fair XLM vs RoBERTa head-to-head."
    )
    parser.add_argument("--report",  action="store_true", help="Print annotation progress only")
    parser.add_argument("--compare", action="store_true", help="Run both models and print h2h report")
    parser.add_argument("--reset",   action="store_true", help="Delete annotation file and start fresh")
    args = parser.parse_args()

    if args.reset:
        for path in (ANN_PATH, CMP_PATH):
            if path.exists():
                path.unlink()
                print(f"Deleted {path.name}")
        return

    if args.report:
        print_progress(load_results())
        return

    if args.compare:
        run_comparison()
        return

    run()


if __name__ == "__main__":
    main()
