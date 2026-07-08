
"""
Interactive review of gold testset rows where v2.1 prompt disagrees with old human labels.
Steps through gold_v2_disagreements.csv and blind_v2_disagreements.csv.
Updates the source files (commitment_testset.parquet + blind_balanced_100.csv) after each row.

Usage: python scripts/relabel_gold_v2.py
"""
import os, csv, textwrap
import pandas as pd

ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_PARQUET   = os.path.join(ROOT, "data/processed/new/commitment_testset.parquet")
BLIND_CSV      = os.path.join(ROOT, "data/processed/new/blind_balanced_100.csv")
GOLD_DISAGREE  = os.path.join(ROOT, "data/processed/new/gold_v2_disagreements.csv")
BLIND_DISAGREE = os.path.join(ROOT, "data/processed/new/blind_v2_disagreements.csv")
CURSOR_FILE    = os.path.join(ROOT, "data/processed/new/.relabel_cursor")

BUYIN_MAP  = {"c": "committed", "u": "uncommitted", "n": "neutral", "": "keep"}
STANCE_MAP = {"s": "supportive", "cr": "critical",  "n": "neutral", "": "keep"}

def load_cursor():
    if os.path.exists(CURSOR_FILE):
        with open(CURSOR_FILE) as f:
            parts = f.read().strip().split(",")
            return parts[0], int(parts[1]) if len(parts) > 1 else 0
    return "gold", 0

def save_cursor(section, idx):
    with open(CURSOR_FILE, "w") as f:
        f.write(f"{section},{idx}")

def clear(): os.system("clear")

def show_row(n, total, row, source_label):
    clear()
    buyin_match  = str(row.get("human_label","")).strip() == str(row.get("v2_buyin","")).strip()
    stance_match = str(row.get("human_stance","")).strip() == str(row.get("v2_stance","")).strip()

    print(f"━━━  {source_label}  |  {n+1}/{total}  ━━━")
    sub = row.get("subreddit", "—")
    print(f"  r/{sub}")
    print()
    for line in textwrap.wrap(str(row.get("text","")), 90):
        print("  " + line)
    print()

    b_flag = "✓ match" if buyin_match  else "✗ DIFFER"
    s_flag = "✓ match" if stance_match else "✗ DIFFER"

    print(f"  BUYIN  [{b_flag}]")
    print(f"    Old label  : {row.get('human_label','—')}")
    print(f"    V2 model   : {row.get('v2_buyin','—')}  (c2d={row.get('v2_c2d_strength','—')})")
    print(f"  STANCE [{s_flag}]")
    print(f"    Old label  : {row.get('human_stance','—')}")
    print(f"    V2 model   : {row.get('v2_stance','—')}")
    print()
    print("  BUYIN  — keep [Enter] / [c]ommitted / [u]ncommitted / [n]eutral")
    new_buyin = BUYIN_MAP.get(input("  > ").strip().lower(), "keep")
    print("  STANCE — keep [Enter] / [s]upportive / [cr]itical / [n]eutral")
    new_stance = STANCE_MAP.get(input("  > ").strip().lower(), "keep")
    return new_buyin, new_stance


# ── PART 1: Gold testset (parquet) ───────────────────────────────────────────
if os.path.exists(GOLD_DISAGREE) and os.path.exists(GOLD_PARQUET):
    disagree = pd.read_csv(GOLD_DISAGREE)
    gold     = pd.read_parquet(GOLD_PARQUET)

    if len(disagree) == 0:
        print("No gold disagreements to review.")
    else:
        print(f"Gold testset: {len(disagree)} disagreements to review")
        print(f"Source: {GOLD_PARQUET}\n")
        cursor_section, start_idx = load_cursor()
        if cursor_section == "gold" and start_idx > 0:
            print(f"Resuming from row {start_idx + 1}/{len(disagree)}")
        input("Press Enter to start...\n")

        # Index gold by chunk_id for fast update
        gold = gold.set_index("chunk_id")

        for n, (_, row) in enumerate(disagree.iterrows()):
            if cursor_section == "gold" and n < start_idx:
                continue
            new_buyin, new_stance = show_row(n, len(disagree), row, "GOLD TESTSET")
            cid = row.get("chunk_id")

            if cid in gold.index:
                if new_buyin != "keep":
                    gold.at[cid, "human_label"]  = new_buyin
                    print(f"  → buyin  updated: {new_buyin}")
                if new_stance != "keep" and "human_stance" in gold.columns:
                    gold.at[cid, "human_stance"] = new_stance
                    print(f"  → stance updated: {new_stance}")

            # Save after every row
            gold.reset_index().to_parquet(GOLD_PARQUET, index=False)
            save_cursor("gold", n + 1)

        print(f"\n✓ Gold testset saved → {GOLD_PARQUET}")
else:
    print("Gold disagreements file not found — run relabel_silver_v2.py first.")

# ── PART 2: Blind balanced 100 (CSV) ─────────────────────────────────────────
if os.path.exists(BLIND_DISAGREE) and os.path.exists(BLIND_CSV):
    blind_disagree = pd.read_csv(BLIND_DISAGREE)
    blind_rows     = list(csv.DictReader(open(BLIND_CSV)))
    id2idx         = {r["chunk_id"]: i for i, r in enumerate(blind_rows)}

    if len(blind_disagree) == 0:
        print("\nNo remaining blind set disagreements to review.")
    else:
        print(f"\nBlind set: {len(blind_disagree)} remaining disagreements to review")
        cursor_section, blind_start = load_cursor()
        if cursor_section == "blind" and blind_start > 0:
            print(f"Resuming from row {blind_start + 1}/{len(blind_disagree)}")
        input("Press Enter to start...\n")

        for n, (_, row) in enumerate(blind_disagree.iterrows()):
            if cursor_section == "blind" and n < blind_start:
                continue
            new_buyin, new_stance = show_row(n, len(blind_disagree), row, "BLIND SET")
            idx = id2idx.get(str(row.get("chunk_id","")))

            if idx is not None:
                if new_buyin != "keep":
                    blind_rows[idx]["human_label"]  = new_buyin
                    print(f"  → buyin  updated: {new_buyin}")
                if new_stance != "keep":
                    blind_rows[idx]["human_stance"] = new_stance
                    print(f"  → stance updated: {new_stance}")

            # Save after every row
            with open(BLIND_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=blind_rows[0].keys())
                w.writeheader()
                w.writerows(blind_rows)
            save_cursor("blind", n + 1)

        print(f"\n✓ Blind set saved → {BLIND_CSV}")

print("\nAll done. Next: python scripts/build_v2_training_set.py")
