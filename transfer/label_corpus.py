"""
Label the silver corpus locally using Ollama.
Resume-safe: saves progress after every row.
Chain after benchmarks to use remaining overnight time.

Usage:
    python label_corpus.py --model mistral-small3.2:24b
    python label_corpus.py --model qwen3.6:latest --limit 500
"""
import argparse, json, time, os, requests
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/chat"

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "buyin":        {"type": "string", "enum": ["committed", "uncommitted", "neutral"]},
        "c2d_strength": {"type": "string", "enum": ["explicit", "demonstrated", ""]},
        "stance":       {"type": "string", "enum": ["supportive", "critical", "neutral"]},
    },
    "required": ["buyin", "stance"],
}

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SYSTEM_PROMPT = __import__("commitment_v2_prompt").SYSTEM_PROMPT


def label_one(text: str, model: str) -> dict:
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model":  model,
            "stream": False,
            "think":  False,
            "format": JSON_SCHEMA,
            "options": {"temperature": 0, "num_predict": 200, "num_ctx": 16384},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Classify this text:\n\n{text}"},
            ],
        }, timeout=180)
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "{}")
        return json.loads(raw)
    except Exception as e:
        print(f"  Error: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="Path to parquet/csv with chunk_id and text columns")
    parser.add_argument("--output", default="corpus_labels.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load input
    if args.input.endswith(".parquet"):
        corpus = pd.read_parquet(args.input)
    else:
        corpus = pd.read_csv(args.input)

    assert "chunk_id" in corpus.columns and "text" in corpus.columns, \
        f"Input must have chunk_id and text columns, got: {list(corpus.columns)}"

    # Resume from existing output
    done_ids = set()
    if os.path.exists(args.output):
        done = pd.read_csv(args.output)
        done_ids = set(done["chunk_id"].tolist())
        print(f"Resuming — {len(done_ids)} already done")

    remaining = corpus[~corpus["chunk_id"].isin(done_ids)]
    if args.limit:
        remaining = remaining.head(args.limit)

    total = len(remaining)
    print(f"Model: {args.model} | To label: {total} | Output: {args.output}")

    t0 = time.time()
    batch = []
    for i, (_, row) in enumerate(remaining.iterrows()):
        result = label_one(row["text"], args.model)
        batch.append({
            "chunk_id":     row["chunk_id"],
            "pred_buyin":   result.get("buyin", "neutral"),
            "pred_stance":  result.get("stance", "neutral"),
            "pred_c2d":     result.get("c2d_strength", ""),
        })

        # Save every 10 rows
        if len(batch) >= 10 or i == total - 1:
            new_rows = pd.DataFrame(batch)
            if os.path.exists(args.output):
                existing = pd.read_csv(args.output)
                pd.concat([existing, new_rows], ignore_index=True).to_csv(args.output, index=False)
            else:
                new_rows.to_csv(args.output, index=False)
            batch = []

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining_time = rate * (total - i - 1)
            print(f"  {i+1}/{total} | {rate:.1f}s/row | ETA {remaining_time/60:.0f} min")

    elapsed = time.time() - t0
    final = pd.read_csv(args.output)
    print(f"\nDone: {len(final)} rows labelled in {elapsed/60:.1f} min ({elapsed/max(total,1):.1f}s/row)")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
