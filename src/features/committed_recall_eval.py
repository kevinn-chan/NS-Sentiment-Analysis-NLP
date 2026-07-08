"""
Recall/precision evaluation for the commitment classifier on the full testset
(727-row human gold test set).

Computes per-class metrics with Wilson 95% confidence intervals.
Flags if committed recall has dropped significantly vs the original 69.2% baseline.

Usage:
    python -m src.features.committed_recall_eval
    python -m src.features.committed_recall_eval --threshold 0.10   # flag if drop > 10pp
"""
import argparse
import math
from pathlib import Path

import pandas as pd

DATA         = Path(__file__).resolve().parents[2] / "data" / "processed" / "new"
TESTSET_PATH = DATA / "commitment_testset.parquet"
LLM_PATH     = DATA / "chunk_commitment_cascade.parquet"

BASELINE_COMMITTED_RECALL = 0.692   # original reported figure on n=26
BASELINE_COMMITTED_N      = 26


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def flag(val: float, lo: float, hi: float, baseline: float, threshold: float) -> str:
    if hi < baseline - threshold:
        return "  ⚠️  SIGNIFICANT DROP vs baseline"
    if lo > baseline + threshold:
        return "  ✅  SIGNIFICANT IMPROVEMENT vs baseline"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="pp drop that triggers a warning (default 0.10 = 10pp)")
    args = ap.parse_args()

    ts = pd.read_parquet(TESTSET_PATH)
    ts = ts[ts["human_label"].notna() & (ts["human_label"].str.len() > 0)
            & (ts["human_label"] != "skip")].copy()

    print(f"\nTestset: {len(ts)} labelled rows")
    print("  human_label distribution:")
    print(ts["human_label"].value_counts().to_string(header=False))

    if "queue_type" in ts.columns:
        orig_n = (ts["queue_type"] != "committed_supplement").sum()
        supp_n = (ts["queue_type"] == "committed_supplement").sum()
        print(f"\n  Original: {orig_n}  |  Supplement: {supp_n}")

    # Load model predictions
    llm = pd.read_parquet(LLM_PATH, columns=["chunk_id", "buyin_label"])
    merged = ts.merge(llm, on="chunk_id", how="left")
    missing = merged["buyin_label"].isna().sum()
    if missing:
        print(f"\n  ⚠️  {missing} testset chunks not found in chunk_commitment_cascade.parquet")
    merged = merged.dropna(subset=["buyin_label"])

    print(f"\n{'═'*64}")
    print(f"  COMMITMENT CLASSIFIER — PER-CLASS EVALUATION")
    print(f"  Threshold for flagging: >{args.threshold*100:.0f}pp drop from baseline")
    print(f"{'═'*64}")

    classes = ["committed", "uncommitted", "neutral"]
    for cls in classes:
        gold_mask = merged["human_label"] == cls
        pred_mask = merged["buyin_label"] == cls

        n_gold = gold_mask.sum()
        TP = (gold_mask & pred_mask).sum()
        FP = (~gold_mask & pred_mask).sum()
        FN = (gold_mask & ~pred_mask).sum()

        recall    = TP / n_gold if n_gold > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

        r_lo, r_hi = wilson_ci(TP, n_gold)
        p_lo, p_hi = wilson_ci(TP, TP + FP)

        baseline = BASELINE_COMMITTED_RECALL if cls == "committed" else None

        print(f"\n  {cls.upper()}  (n_gold={n_gold})")
        print(f"    Recall    = {recall:6.1%}   95% CI [{r_lo:.1%} – {r_hi:.1%}]", end="")
        if baseline:
            drop = recall - baseline
            print(f"   vs baseline {baseline:.1%}  (Δ{drop:+.1%})", end="")
            print(flag(recall, r_lo, r_hi, baseline, args.threshold), end="")
        print()
        print(f"    Precision = {precision:6.1%}   95% CI [{p_lo:.1%} – {p_hi:.1%}]")

    # Confusion matrix
    print(f"\n{'═'*64}")
    print(f"  CONFUSION MATRIX (rows=gold, cols=predicted)")
    print(f"{'═'*64}")
    ct = pd.crosstab(merged["human_label"], merged["buyin_label"],
                     margins=True, margins_name="Total")
    # Ensure all classes are present
    for c in classes:
        if c not in ct.columns:
            ct[c] = 0
        if c not in ct.index:
            ct.loc[c] = 0
    print(ct[classes + ["Total"]].to_string())

    # Lexicon term breakdown (if supplement column exists)
    if "trigger_term" in merged.columns:
        supp = merged[merged["queue_type"] == "committed_supplement"]
        if len(supp) > 0:
            print(f"\n{'═'*64}")
            print(f"  SUPPLEMENT: COMMITTED PRECISION BY TRIGGER TERM")
            print(f"  (did the model correctly classify chunks where the term fired?)")
            print(f"{'═'*64}")
            for term, grp in supp[supp["human_label"] == "committed"].groupby("trigger_term"):
                n = len(grp)
                correct = (grp["buyin_label"] == "committed").sum()
                print(f"  {term:<35} recall {correct}/{n} = {correct/n*100:.0f}%")

    print(f"\n{'═'*64}")
    print(f"  Overall accuracy: "
          f"{(merged['human_label']==merged['buyin_label']).mean():.1%}  "
          f"(n={len(merged)})")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()
