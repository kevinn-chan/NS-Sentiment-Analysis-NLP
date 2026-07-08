"""
Evaluate cascade model on blind_testset_50.csv + original testset.
Prints κ and macro F1 for both sets side-by-side.
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report

BLIND = "data/processed/new/blind_testset_50.csv"
ORIG  = "data/processed/new/commitment_testset.parquet"
CASC  = "/tmp/results_1/chunk_commitment_cascade.parquet"

def eval_set(human_df, casc, axis, human_col, pred_col, name):
    m = human_df.merge(casc[["chunk_id", pred_col]], on="chunk_id")
    y_true, y_pred = m[human_col], m[pred_col]
    k = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n── {name} | {axis} ({len(m)} rows) ──")
    print(f"   κ={k:.3f}   macro-F1={f1:.3f}")
    print(classification_report(y_true, y_pred, zero_division=0))

casc = pd.read_parquet(CASC)

# Original testset
orig = pd.read_parquet(ORIG)
eval_set(orig, casc, "buyin",  "human_label",  "buyin_label",  "ORIGINAL")
eval_set(orig, casc, "stance", "human_stance", "stance_label", "ORIGINAL")

# Blind 50
blind = pd.read_csv(BLIND)
blind = blind[blind["human_label"].str.strip() != ""]
if len(blind) == 0:
    print("\nBlind testset not annotated yet. Run: python scripts/annotate_blind_50.py")
else:
    eval_set(blind, casc, "buyin",  "human_label",  "buyin_label",  "BLIND-50")
    eval_set(blind, casc, "stance", "human_stance", "stance_label", "BLIND-50")

    # Combined
    combo_b = pd.concat([
        orig[["chunk_id","human_label","human_stance"]],
        blind[["chunk_id","human_label","human_stance"]]
    ])
    eval_set(combo_b, casc, "buyin",  "human_label",  "buyin_label",  "COMBINED")
    eval_set(combo_b, casc, "stance", "human_stance", "stance_label", "COMBINED")
