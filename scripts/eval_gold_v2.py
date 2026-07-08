"""
Evaluate v2.1 prompt against gold testset using gpt-4o.
- Throttled to 3 RPM for free-tier compliance
- Saves progress after every row (resume-safe)
- Prints classification report at the end

Usage: python scripts/eval_gold_v2.py
"""
import os, time, json, asyncio, argparse
import pandas as pd
from sklearn.metrics import classification_report
from openai import AsyncOpenAI
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.prompts.commitment_v2_prompt import SYSTEM_PROMPT

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_PATH = os.path.join(ROOT, "data/processed/new/commitment_testset.parquet")

ap = argparse.ArgumentParser()
ap.add_argument("--balanced", action="store_true", help="Sample equal counts per human_label class")
ap.add_argument("--limit", type=int, default=None, help="Total rows (balanced: split across classes)")
ap.add_argument("--rpm", type=int, default=3, help="Requests per minute (use higher on paid tier)")
ap.add_argument("--model", default="gpt-4o-mini")
ap.add_argument("--output", default=os.path.join(ROOT, "data/processed/new/gold_eval_v2.csv"))
cli = ap.parse_args()

OUT_PATH  = cli.output
MODEL     = cli.model
RPM_LIMIT = cli.rpm
DELAY     = 60.0 / RPM_LIMIT

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


def build_user_message(text: str) -> str:
    return f"Classify this text:\n\n{text}"


async def label_one(text: str) -> dict:
    resp = await client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=60,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(text)},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


async def main():
    gold = pd.read_parquet(GOLD_PATH)

    if cli.balanced:
        per_class = (cli.limit or 90) // gold["human_label"].nunique()
        gold = pd.concat([
            g.sample(min(len(g), per_class), random_state=0)
            for _, g in gold.groupby("human_label")
        ])
        print(f"Balanced sample: {gold['human_label'].value_counts().to_dict()}")
    elif cli.limit:
        gold = gold.head(cli.limit)

    # Resume from existing output
    if os.path.exists(OUT_PATH):
        done = pd.read_csv(OUT_PATH)
        done_ids = set(done["chunk_id"].tolist())
        print(f"Resuming — {len(done_ids)} already done, {len(gold) - len(done_ids)} remaining")
    else:
        done = pd.DataFrame()
        done_ids = set()

    remaining = gold[~gold["chunk_id"].isin(done_ids)].copy()
    total = len(remaining)
    print(f"Gold testset: {len(gold)} rows | To label: {total} | Model: {MODEL} | {RPM_LIMIT} RPM")
    print(f"Estimated cost: ~${total * 0.002:.2f} | Time: ~{total * DELAY / 60:.0f} min\n")

    rows = []
    for i, (_, row) in enumerate(remaining.iterrows()):
        result = await label_one(row["text"])
        rows.append({
            "chunk_id":        row["chunk_id"],
            "human_label":     row.get("human_label", ""),
            "human_stance":    row.get("human_stance", ""),
            "v2_buyin":        result.get("buyin",        "neutral"),
            "v2_stance":       result.get("stance",       "neutral"),
            "v2_c2d_strength": result.get("c2d_strength", ""),
        })

        # Save after every row
        out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        out.to_csv(OUT_PATH, index=False)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{total}...")

        if i < total - 1:
            time.sleep(DELAY)

    # Final report
    out = pd.read_csv(OUT_PATH)
    out = out.dropna(subset=["human_label", "v2_buyin"])

    print("\n── BUYIN ──")
    print(classification_report(out["human_label"], out["v2_buyin"],
                                labels=["committed","uncommitted","neutral"], zero_division=0))
    stance = out.dropna(subset=["human_stance", "v2_stance"])
    if len(stance):
        print("── STANCE ──")
        print(classification_report(stance["human_stance"], stance["v2_stance"],
                                    labels=["supportive","critical","neutral"], zero_division=0))

    print(f"\nSaved → {OUT_PATH}")


asyncio.run(main())
