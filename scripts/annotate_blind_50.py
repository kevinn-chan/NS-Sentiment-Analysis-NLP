"""
Annotation CLI for blind_testset_50.csv.
Usage: python scripts/annotate_blind_50.py
Labels: buyin = committed/uncommitted/neutral
        stance = supportive/critical/neutral
"""
import csv, os, textwrap, sys

CSV = "data/processed/new/blind_balanced_100.csv"
RELABEL_MODE = False  # overridden by relabel script
BUYIN_LABELS  = {"c": "committed", "u": "uncommitted", "n": "neutral"}
STANCE_LABELS = {"s": "supportive", "cr": "critical",  "n": "neutral"}

def clear(): os.system("clear" if os.name == "posix" else "cls")

def ask(prompt, valid_map):
    while True:
        raw = input(prompt).strip().lower()
        if raw in valid_map:
            return valid_map[raw]
        options = "  ".join(f"[{k}]={v}" for k, v in valid_map.items())
        print(f"  → Options: {options}")

def main():
    rows = list(csv.DictReader(open(CSV)))
    done = 0
    for i, row in enumerate(rows):
        if row["human_label"] and row["human_stance"]:
            done += 1
            continue
        clear()
        pct = f"{done}/{len(rows)} done"
        print(f"━━━  CHUNK {i+1}/{len(rows)}  {pct}  ━━━")
        print(f"  Stratum: {row['stratum']}   |  r/{row['subreddit']}")
        print()
        for line in textwrap.wrap(row["text"], 90):
            print("  " + line)
        print()
        row["human_label"]  = ask("BUYIN  [c]=committed  [u]=uncommitted  [n]=neutral : ", BUYIN_LABELS)
        row["human_stance"] = ask("STANCE [s]=supportive  [cr]=critical  [n]=neutral  : ", STANCE_LABELS)
        done += 1
        # Save after every row
        with open(CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  ✓ saved")

    print(f"\nAll {len(rows)} rows annotated. Run eval script next.")

if __name__ == "__main__":
    main()
