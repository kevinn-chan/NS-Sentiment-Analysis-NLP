"""
Re-label silver rows using v2.1 prompt with async parallel calls.
Tier 3 OpenAI: up to 5,000 RPM — uses 50 concurrent workers.

Usage: python scripts/relabel_silver_v2.py
"""
import json, os, sys, asyncio
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, "src")
from features.prompts.commitment_v2_prompt import SYSTEM_PROMPT, build_user_message

client  = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL   = "gpt-4.1-mini"
WORKERS = 50
OUT_DIR = "data/processed/new/cascade"

async def label_one(sem, text: str, idx: int) -> dict:
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_message(text)},
                    ],
                    max_tokens=60,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as e:
                await asyncio.sleep(2 ** attempt)
        return {"buyin": "neutral", "c2d_strength": None, "stance": "neutral"}

async def label_all(texts: list[str], label: str) -> list[dict]:
    sem = asyncio.Semaphore(WORKERS)
    tasks = [label_one(sem, t, i) for i, t in enumerate(texts)]
    results = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
        done += 1
        if done % 200 == 0:
            print(f"  {label}: {done}/{len(texts)} done...")
    print(f"  {label}: {len(texts)}/{len(texts)} done.")
    # as_completed returns out of order — reorder
    # Use gather instead for ordered results
    return results

async def label_ordered(texts: list[str], label: str) -> list[dict]:
    sem = asyncio.Semaphore(WORKERS)
    tasks = [label_one(sem, t, i) for i, t in enumerate(texts)]
    total = len(tasks)
    results = [None] * total
    done = 0

    async def wrap(i, coro):
        nonlocal done
        r = await coro
        results[i] = r
        done += 1
        if done % 200 == 0:
            print(f"  {label}: {done}/{total}...")
        return r

    await asyncio.gather(*[wrap(i, t) for i, t in enumerate(tasks)])
    print(f"  {label}: {total}/{total} done.")
    return results

async def main():
    # ── Stage 2a silver ───────────────────────────────────────────────────────
    print("=" * 60)
    print("PART 1: stage2a silver (buyin axis)")
    print("=" * 60)
    s2a = pd.read_csv(f"{OUT_DIR}/stage2a_train.csv")
    silver_mask = s2a["source"] == "llm_silver"
    silver = s2a[silver_mask].copy()
    print(f"Rows: {len(silver):,}  |  workers: {WORKERS}")

    results = await label_ordered(silver["text"].tolist(), "stage2a")
    silver["v2_buyin"]        = [r.get("buyin",  "neutral") for r in results]
    silver["v2_c2d_strength"] = [r.get("c2d_strength")      for r in results]
    silver["v2_stance"]       = [r.get("stance", "neutral") for r in results]

    old = silver["label"].value_counts()
    new = silver["v2_buyin"].value_counts()
    print("\nLabel shift (silver stage2a):")
    print(f"  {'Label':<15} {'Old':>8} {'New':>8} {'Δ':>8}")
    for lbl in ["committed","uncommitted","neutral"]:
        print(f"  {lbl:<15} {old.get(lbl,0):>8,} {new.get(lbl,0):>8,} {new.get(lbl,0)-old.get(lbl,0):>+8,}")
    com = silver[silver["v2_buyin"]=="committed"]
    print(f"\n  c2d_strength breakdown:\n{com['v2_c2d_strength'].value_counts().to_string()}")

    out = f"{OUT_DIR}/stage2a_silver_v2.csv"
    silver.to_csv(out, index=False)
    print(f"  Saved → {out}")

    # ── Stage 2b silver ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 2: stage2b silver (stance axis)")
    print("=" * 60)
    s2b = pd.read_csv(f"{OUT_DIR}/stage2b_train.csv")
    silver2b = s2b[s2b["source"] == "llm_silver"].copy()
    print(f"Rows: {len(silver2b):,}  |  workers: {WORKERS}")

    results2b = await label_ordered(silver2b["text"].tolist(), "stage2b")
    silver2b["v2_stance"] = [r.get("stance","neutral") for r in results2b]

    old2b = silver2b["label"].value_counts()
    new2b = silver2b["v2_stance"].value_counts()
    print("\nLabel shift (silver stage2b):")
    print(f"  {'Label':<15} {'Old':>8} {'New':>8} {'Δ':>8}")
    for lbl in ["supportive","critical","neutral"]:
        print(f"  {lbl:<15} {old2b.get(lbl,0):>8,} {new2b.get(lbl,0):>8,} {new2b.get(lbl,0)-old2b.get(lbl,0):>+8,}")

    out2b = f"{OUT_DIR}/stage2b_silver_v2.csv"
    silver2b.to_csv(out2b, index=False)
    print(f"  Saved → {out2b}")

    # ── Gold testset ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 3: gold testset disagreements")
    print("=" * 60)
    gold = pd.read_parquet("data/processed/new/commitment_testset.parquet")
    print(f"Rows: {len(gold):,}")

    gold_results = await label_ordered(gold["text"].tolist(), "gold")
    gold["v2_buyin"]        = [r.get("buyin",  "neutral") for r in gold_results]
    gold["v2_c2d_strength"] = [r.get("c2d_strength")      for r in gold_results]
    gold["v2_stance"]       = [r.get("stance", "neutral") for r in gold_results]

    disagree = gold[gold["human_label"].str.strip() != gold["v2_buyin"].str.strip()]
    print(f"  Gold disagreements: {len(disagree)}")
    disagree.to_csv("data/processed/new/gold_v2_disagreements.csv", index=False)

    # ── Blind set ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 4: blind_balanced_100 full re-check")
    print("=" * 60)
    blind = pd.read_csv("data/processed/new/blind_balanced_100.csv")
    blind = blind[blind["human_label"].str.strip() != ""].copy()
    print(f"Rows: {len(blind)}")

    blind_results = await label_ordered(blind["text"].tolist(), "blind")
    blind["v2_buyin"]        = [r.get("buyin",  "neutral") for r in blind_results]
    blind["v2_c2d_strength"] = [r.get("c2d_strength")      for r in blind_results]
    blind["v2_stance"]       = [r.get("stance", "neutral") for r in blind_results]

    blind_dis = blind[
        (blind["human_label"].str.strip() != blind["v2_buyin"].str.strip()) |
        (blind["human_stance"].str.strip().fillna("") != blind["v2_stance"].str.strip().fillna(""))
    ]
    print(f"  Blind disagreements: {len(blind_dis)}")
    blind_dis.to_csv("data/processed/new/blind_v2_disagreements.csv", index=False)

    print("\n✓ Done. Run: python scripts/relabel_gold_v2.py")

asyncio.run(main())
