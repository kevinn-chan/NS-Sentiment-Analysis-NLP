"""
Step through v2 trial disagreements and let user confirm/update labels
under the NEW definition.

Shows: text | old human label | v2 model prediction | ask for final label
Saves back to blind_balanced_100.csv after each row.

Usage: python scripts/relabel_v2_disagreements.py
"""
import csv, os, textwrap
import pandas as pd

CSV     = "data/processed/new/blind_balanced_100.csv"
TRIAL   = "data/processed/new/trial_v2_results.csv"

BUYIN_MAP  = {"c": "committed", "u": "uncommitted", "n": "neutral", "": "keep"}
STANCE_MAP = {"s": "supportive", "cr": "critical",  "n": "neutral", "": "keep"}

def clear(): os.system("clear")

df_human = pd.read_csv(CSV)
df_trial = pd.read_csv(TRIAL)

# Merge on chunk_id
merged = df_human.merge(
    df_trial[["chunk_id","v2_buyin","v2_c2d_strength","v2_stance"]],
    on="chunk_id", how="inner"
)

buyin_disagree  = merged[merged["human_label"].str.strip() != merged["v2_buyin"].str.strip()]
stance_disagree = merged[
    merged["human_stance"].str.strip().fillna("") != merged["v2_stance"].str.strip().fillna("")
]

all_disagree_ids = set(buyin_disagree["chunk_id"]) | set(stance_disagree["chunk_id"])
to_review = merged[merged["chunk_id"].isin(all_disagree_ids)].copy()

print(f"Total disagreements to review: {len(to_review)}")
print(f"  Buyin disagreements:  {len(buyin_disagree)}")
print(f"  Stance disagreements: {len(stance_disagree)}")
print("\nPress Enter to start...\n")
input()

rows_csv = list(csv.DictReader(open(CSV)))
id2idx   = {r["chunk_id"]: i for i, r in enumerate(rows_csv)}

for n, (_, row) in enumerate(to_review.iterrows()):
    clear()
    buyin_match  = row["human_label"].strip() == row["v2_buyin"].strip()
    stance_match = row["human_stance"].strip() == row["v2_stance"].strip()

    print(f"━━━  DISAGREEMENT {n+1}/{len(to_review)}  ━━━")
    print(f"  r/{row['subreddit']}  |  stratum={row['stratum']}")
    print()
    for line in textwrap.wrap(row["text"], 90):
        print("  " + line)
    print()

    # Buyin
    b_flag = "✓" if buyin_match else "✗"
    print(f"  BUYIN  {b_flag}")
    print(f"    Your label : {row['human_label']}")
    print(f"    V2 model   : {row['v2_buyin']}  (c2d={row['v2_c2d_strength']})")

    # Stance
    s_flag = "✓" if stance_match else "✗"
    print(f"  STANCE {s_flag}")
    print(f"    Your label : {row['human_stance']}")
    print(f"    V2 model   : {row['v2_stance']}")

    print()
    print("  BUYIN  — keep [Enter] / [c]ommitted / [u]ncommitted / [n]eutral")
    b_raw = input("  > ").strip().lower()
    new_buyin = BUYIN_MAP.get(b_raw, "keep")

    print("  STANCE — keep [Enter] / [s]upportive / [cr]itical / [n]eutral")
    s_raw = input("  > ").strip().lower()
    new_stance = STANCE_MAP.get(s_raw, "keep")

    # Apply updates
    idx = id2idx.get(row["chunk_id"])
    if idx is not None:
        if new_buyin != "keep":
            rows_csv[idx]["human_label"]  = new_buyin
            print(f"  → buyin updated to: {new_buyin}")
        if new_stance != "keep":
            rows_csv[idx]["human_stance"] = new_stance
            print(f"  → stance updated to: {new_stance}")

    # Save after every row
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_csv[0].keys())
        w.writeheader()
        w.writerows(rows_csv)

print(f"\nDone. Re-run trial: python scripts/trial_prompt_v2.py")
