"""
Benchmark local Ollama model against gold testset.
Uses Ollama's native API with structured output (format parameter)
for reliable JSON responses.

Usage:
    python bench_local.py
    python bench_local.py --model qwen3:14b
    python bench_local.py --model qwen3.6:latest --limit 100

Requires:
    pip install requests pandas pyarrow scikit-learn
    ollama serve  (running in background)
"""
import argparse, json, time, os, requests
import pandas as pd
from sklearn.metrics import classification_report

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "qwen3.6:latest"
OLLAMA_URL    = "http://localhost:11434/api/chat"

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "buyin":        {"type": "string", "enum": ["committed", "uncommitted", "neutral"]},
        "c2d_strength": {"type": "string", "enum": ["explicit", "demonstrated", ""]},
        "stance":       {"type": "string", "enum": ["supportive", "critical", "neutral"]},
    },
    "required": ["buyin", "stance"],
}

# ── Load prompt ───────────────────────────────────────────────────────────────
# Always use the full v2 prompt (condensed version proved too lossy)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SYSTEM_PROMPT = None
for mod in ["commitment_v2_prompt"]:
    try:
        SYSTEM_PROMPT = __import__(mod).SYSTEM_PROMPT
        print(f"Loaded prompt from {mod}")
        break
    except ImportError:
        pass
if SYSTEM_PROMPT is None:
    for name in ["commitment_v2_prompt.py"]:
        prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        if os.path.exists(prompt_file):
            ns = {}
            exec(open(prompt_file).read(), ns)
            SYSTEM_PROMPT = ns["SYSTEM_PROMPT"]
            print(f"Loaded prompt from {name}")
            break
if SYSTEM_PROMPT is None:
    raise FileNotFoundError("No prompt file found. Place commitment_local_prompt.py or commitment_v2_prompt.py in the same folder.")


def label_one(text: str, model: str) -> dict:
    resp = requests.post(OLLAMA_URL, json={
        "model":  model,
        "stream": False,
        "format": JSON_SCHEMA,
        "options": {"temperature": 0, "num_predict": 60},
        "messages": [
            {"role": "system", "content": "/no_think\n\n" + SYSTEM_PROMPT},
            {"role": "user",   "content": f"Classify this text:\n\n{text}"},
        ],
    })
    resp.raise_for_status()
    raw = resp.json().get("message", {}).get("content", "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--limit",  type=int, default=None, help="Only run N rows (for quick test)")
    parser.add_argument("--output", default="bench_results.csv")
    args = parser.parse_args()

    # Load gold testset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_candidates = [
        os.path.join(script_dir, "commitment_testset.parquet"),
        os.path.join(script_dir, "..", "data", "processed", "new", "commitment_testset.parquet"),
    ]
    gold = None
    for p in parquet_candidates:
        if os.path.exists(p):
            gold = pd.read_parquet(p)
            print(f"Loaded gold testset: {p}")
            break
    if gold is None:
        raise FileNotFoundError("commitment_testset.parquet not found.")

    if args.limit:
        gold = gold.head(args.limit)

    total = len(gold)
    print(f"Model: {args.model} | Rows: {total}")
    print("Starting...\n")

    rows = []
    t0 = time.time()
    for i, (_, row) in enumerate(gold.iterrows()):
        result = label_one(row["text"], args.model)
        rows.append({
            "chunk_id":     row["chunk_id"],
            "human_label":  row.get("human_label", ""),
            "human_stance": row.get("human_stance", ""),
            "pred_buyin":   result.get("buyin",  "neutral"),
            "pred_stance":  result.get("stance", "neutral"),
            "pred_c2d":     result.get("c2d_strength", ""),
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = rate * (total - i - 1)
            print(f"  {i+1}/{total} | {rate:.1f}s/row | ETA {remaining/60:.0f} min")

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min ({elapsed/total:.1f}s/row)")
    print(f"Saved → {args.output}\n")

    # ── Classification report ─────────────────────────────────────────────────
    valid = out.dropna(subset=["human_label", "pred_buyin"])
    valid = valid[valid["human_label"] != ""]

    print("══ BUYIN ══")
    print(classification_report(
        valid["human_label"], valid["pred_buyin"],
        labels=["committed", "uncommitted", "neutral"], zero_division=0
    ))

    stance = out.dropna(subset=["human_stance", "pred_stance"])
    stance = stance[stance["human_stance"] != ""]
    if len(stance):
        print("══ STANCE ══")
        print(classification_report(
            stance["human_stance"], stance["pred_stance"],
            labels=["supportive", "critical", "neutral"], zero_division=0
        ))

    print(f"\nFor 700k rows at {elapsed/total:.1f}s/row → ~{700000*(elapsed/total)/3600/24:.1f} days")


if __name__ == "__main__":
    main()
