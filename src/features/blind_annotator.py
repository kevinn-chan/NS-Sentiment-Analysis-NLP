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
    python -m src.features.blind_annotator --extend N   # add N more to training set
    python -m src.features.blind_annotator --holdout N  # annotate N chunks into HELD-OUT test set

Keys during annotation:
    P   — POSITIVE
    N   — NEGATIVE
    U   — NEUTRAL
    S   — skip (genuinely ambiguous / not NS-related enough)
    Q   — quit and save progress

IMPORTANT — two separate files:
    blind_annotation.csv  → training data   (used by llm_annotator --build-dataset)
    holdout_test.csv      → evaluation only (NEVER used for training)
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
DATA_DIR     = Path(__file__).parent.parent.parent / "data" / "processed" / "new"
ANN_PATH     = DATA_DIR / "blind_annotation.csv"
HOLDOUT_PATH = DATA_DIR / "holdout_test.csv"
CMP_PATH     = DATA_DIR / "blind_annotation_comparison.csv"

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


def build_extension_sample(chunks: pd.DataFrame, exclude_ids: set,
                           n_total: int, seed: int = 42) -> pd.DataFrame:
    """Draw n_total additional chunks, proportional to stratum sizes, excluding already-done IDs."""
    df = chunks.copy()
    df["has_sg"] = df["text"].fillna("").apply(has_singlish)
    df["wc"]     = df["text"].fillna("").apply(word_count)
    df["text"]   = df["text"].fillna("").str.strip()
    df           = df[(df["text"].str.len() > 0) & (~df["chunk_id"].isin(exclude_ids))]

    # Same proportions as original STRATA (each stratum gets equal share)
    n_per_stratum = n_total // len(STRATA)
    remainder     = n_total % len(STRATA)

    parts = []
    already_sampled = set()
    for i, (stratum, filters, _) in enumerate(STRATA):
        g = df.copy()
        if "singlish" in filters:
            g = g[g["has_sg"] == filters["singlish"]]
        if "doc_type" in filters:
            g = g[g["doc_type"] == filters["doc_type"]]
        if "wc_min" in filters:
            g = g[g["wc"] >= filters["wc_min"]]
        if "wc_max" in filters:
            g = g[g["wc"] <  filters["wc_max"]]
        g = g[~g["chunk_id"].isin(already_sampled)]
        # Give remainder chunks to first stratum
        n = n_per_stratum + (remainder if i == 0 else 0)
        sampled            = g.sample(min(n, len(g)), random_state=seed + i)
        sampled            = sampled.copy()
        sampled["stratum"] = stratum
        already_sampled.update(sampled["chunk_id"].values)
        parts.append(sampled)

    extension = (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=seed + 99)
        .reset_index(drop=True)
    )
    return extension[["chunk_id", "stratum", "doc_type", "subreddit",
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


def load_holdout() -> pd.DataFrame:
    if HOLDOUT_PATH.exists():
        return pd.read_csv(HOLDOUT_PATH)
    return pd.DataFrame(columns=[
        "chunk_id", "stratum", "doc_type", "subreddit",
        "has_sg", "wc", "text", "human_label",
    ])


def save_holdout(df: pd.DataFrame):
    df.to_csv(HOLDOUT_PATH, index=False)


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
# Extended annotation loop (for golden dataset beyond initial 100)
# ---------------------------------------------------------------------------
def run_extended(n: int):
    import tty, termios

    def getch() -> str:
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print(f"Loading chunk data for {n} additional annotations ...")
    sub    = pd.read_parquet(DATA_DIR / "submissions_chunks.parquet", columns=CHUNK_COLS)
    com    = pd.read_parquet(DATA_DIR / "comments_chunks.parquet",    columns=CHUNK_COLS)
    chunks = pd.concat([sub, com], ignore_index=True).drop_duplicates("chunk_id")
    del sub, com

    results     = load_results()
    done_ids    = set(results["chunk_id"].dropna())
    already_n   = len(results.dropna(subset=["human_label"]))

    extension   = build_extension_sample(chunks, exclude_ids=done_ids, n_total=n)
    # Filter any that slipped through (race condition safety)
    extension   = extension[~extension["chunk_id"].isin(done_ids)].reset_index(drop=True)

    if len(extension) == 0:
        print("No new chunks to annotate.")
        return

    total    = already_n + len(extension)
    reviewed = already_n

    print(f"  {already_n} already done — adding {len(extension)} new chunks (target: {total})\n")
    print("  Keys: [P]=positive  [N]=negative  [U]=neutral  [S]=skip  [Q]=quit\n")
    print("  ⚠️  No model predictions shown. Label on text alone.\n")

    for _, row in extension.iterrows():
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

    print(f"\nDone — {reviewed} total annotations.")
    print_progress(results)


# ---------------------------------------------------------------------------
# Holdout annotation loop — saves to holdout_test.csv, NEVER to blind_annotation.csv
# ---------------------------------------------------------------------------
def run_holdout(n: int):
    """Annotate N chunks into a separate held-out test set.

    These labels are used ONLY for final model evaluation.
    They are never passed to llm_annotator --build-dataset.
    """
    import tty, termios

    def getch() -> str:
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print(f"\n{'█'*60}")
    print(f"  HOLDOUT SET — {n} chunks")
    print(f"  These labels go to holdout_test.csv ONLY.")
    print(f"  They will NEVER be used for training.")
    print(f"  Purpose: honest final evaluation of fine-tuned SingBERT.")
    print(f"{'█'*60}\n")

    print("Loading chunk data ...")
    sub    = pd.read_parquet(DATA_DIR / "submissions_chunks.parquet", columns=CHUNK_COLS)
    com    = pd.read_parquet(DATA_DIR / "comments_chunks.parquet",    columns=CHUNK_COLS)
    chunks = pd.concat([sub, com], ignore_index=True).drop_duplicates("chunk_id")
    del sub, com

    # Exclude ALL known IDs — training set AND existing holdout
    exclude_ids = set()
    exclude_ids.update(load_results()["chunk_id"].dropna())
    holdout     = load_holdout()
    exclude_ids.update(holdout["chunk_id"].dropna())

    already_n = len(holdout.dropna(subset=["human_label"]))

    # Use seed=99 so samples differ from --extend (seed=42)
    extension = build_extension_sample(chunks, exclude_ids=exclude_ids,
                                       n_total=n, seed=99)
    extension = extension[~extension["chunk_id"].isin(exclude_ids)].reset_index(drop=True)

    if len(extension) == 0:
        print("No new chunks available for holdout.")
        return

    total    = already_n + len(extension)
    reviewed = already_n

    print(f"  {already_n} holdout already done — annotating {len(extension)} new chunks\n")
    print("  Keys: [P]=positive  [N]=negative  [U]=neutral  [S]=skip  [Q]=quit\n")
    print("  ⚠️  No model predictions shown. Label on text alone.\n")

    for _, row in extension.iterrows():
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
                print(f"\n  Holdout saved: {reviewed - 1} annotations → {HOLDOUT_PATH.name}\n")
                return

        new_row = row.to_dict()
        new_row["human_label"] = label
        holdout = pd.concat([holdout, pd.DataFrame([new_row])], ignore_index=True)
        save_holdout(holdout)
        print()

    # Summary
    done     = holdout.dropna(subset=["human_label"])
    labelled = done[done["human_label"] != "skip"]
    skipped  = (done["human_label"] == "skip").sum()

    print(f"\n{'═'*55}")
    print(f"  Holdout set complete — {len(labelled)} labelled, {skipped} skipped")
    print(f"{'═'*55}")
    vc = labelled["human_label"].value_counts()
    for lbl in ["negative", "neutral", "positive"]:
        cnt = int(vc.get(lbl, 0))
        bar = "█" * int((cnt / len(labelled) * 25) if len(labelled) else 0)
        print(f"    {lbl:<10} {cnt:>3}  {bar}")
    print(f"\n  Saved → {HOLDOUT_PATH}")
    print(f"  ⚠️  Do NOT pass this file to --build-dataset. Evaluation only.")
    print(f"{'═'*55}\n")


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
    parser.add_argument("--extend",  type=int, metavar="N",
                        help="Annotate N more chunks beyond the original 100 (added to training set)")
    parser.add_argument("--holdout", type=int, metavar="N",
                        help="Annotate N chunks into a separate held-out test set (NEVER used for training)")
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

    if args.extend:
        run_extended(args.extend)
        return

    if args.holdout:
        run_holdout(args.holdout)
        return

    run()


if __name__ == "__main__":
    main()
