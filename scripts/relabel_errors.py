"""
Step through buyin mismatches in blind_balanced_100.csv and correct labels.
Shows model prediction + probabilities so you can decide who's right.
"""
import csv, os, textwrap, pandas as pd

CSV  = "data/processed/new/blind_balanced_100.csv"
CASC = "/tmp/results_1/chunk_commitment_cascade.parquet"

BUYIN_LABELS = {"c": "committed", "u": "uncommitted", "n": "neutral", "": "keep"}

def clear(): os.system("clear")

casc = pd.read_parquet(CASC)

rows = list(csv.DictReader(open(CSV)))
id2casc = casc.set_index("chunk_id")[["buyin_label","prob_buyin_committed",
                                      "prob_buyin_uncommitted","prob_buyin_neutral"]].to_dict("index")

errors = [(i, r) for i, r in enumerate(rows)
          if r["human_label"] != id2casc.get(r["chunk_id"], {}).get("buyin_label","")]

print(f"{len(errors)} buyin mismatches to review\n")

for n, (i, row) in enumerate(errors):
    c = id2casc[row["chunk_id"]]
    clear()
    print(f"━━━  ERROR {n+1}/{len(errors)}  ━━━")
    print(f"  stratum={row['stratum']}  |  r/{row['subreddit']}")
    print()
    for line in textwrap.wrap(row["text"], 90):
        print("  " + line)
    print()
    print(f"  YOUR label : {row['human_label']}")
    print(f"  MODEL pred : {c['buyin_label']}  "
          f"(com={c['prob_buyin_committed']:.2f}  "
          f"unc={c['prob_buyin_uncommitted']:.2f}  "
          f"neu={c['prob_buyin_neutral']:.2f})")
    print()
    raw = input("  Keep yours [Enter] / override [c/u/n]: ").strip().lower()
    if raw and raw in BUYIN_LABELS:
        rows[i]["human_label"] = BUYIN_LABELS[raw]
        print(f"  → updated to {BUYIN_LABELS[raw]}")
    else:
        print("  → kept")

    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

print("\nDone. Run: python scripts/eval_blind_50.py")
