"""
Inject FAISS uncommitted + critical candidates into the LLM enrich queue
as unlabeled rows, ready for --annotate-enrich.

Run once before --annotate-enrich:
    python scripts/inject_faiss_to_llm_queue.py
"""
from pathlib import Path
import pandas as pd

ROOT     = Path(__file__).parent.parent
DATA_NEW = ROOT / "data" / "processed" / "new"

FAISS_FILES = [
    DATA_NEW / "commitment_faiss_enrich_buyin_committed.csv",
    DATA_NEW / "commitment_faiss_enrich_stance_supportive.csv",
]

queue_path = DATA_NEW / "commitment_llm_enrich_queue.csv"
queue = pd.read_csv(queue_path)
existing_ids = set(queue["chunk_id"].astype(str))

print(f"Existing queue: {len(queue):,} rows")

new_rows = []
for f in FAISS_FILES:
    if not f.exists():
        print(f"WARNING: {f.name} not found — skipping")
        continue
    df = pd.read_csv(f)[["chunk_id", "text"]]
    df = df[~df["chunk_id"].astype(str).isin(existing_ids)]
    existing_ids.update(df["chunk_id"].astype(str))
    new_rows.append(df)
    print(f"  {f.name}: {len(df):,} new candidates")

if not new_rows:
    print("Nothing to add.")
else:
    to_add = pd.concat(new_rows, ignore_index=True)
    # Add required columns with nulls — annotate-enrich fills llm_buyin/llm_stance
    for col in ["doc_type", "subreddit", "queue_type", "sent_neg",
                "lex_committed", "lex_uncommitted", "llm_label",
                "llm_buyin", "llm_stance"]:
        to_add[col] = None

    queue_new = pd.concat([queue, to_add[queue.columns]], ignore_index=True)
    queue_new.to_csv(queue_path, index=False)

    unlabeled = queue_new["llm_buyin"].isna().sum()
    est_cost = unlabeled * 0.00055
    print(f"\nQueue: {len(queue):,} → {len(queue_new):,} rows")
    print(f"Unlabeled (ready to annotate): {unlabeled:,}")
    print(f"Estimated LLM cost: ~${est_cost:.2f}")
    print(f"\nNow run:")
    print(f"  .venv/bin/python -m src.features.commitment_llm_annotator --annotate-enrich")
