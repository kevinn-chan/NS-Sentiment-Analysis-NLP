"""
Label the full corpus locally using Ollama.
Resume-safe: saves after every row with atomic writes.

Usage:
    python label_corpus.py --model gemma4:12b --input ../data/processed/new/comments_chunks.parquet --output corpus_comments.csv
    python label_corpus.py --model gemma4:12b --input ../data/processed/new/submissions_chunks.parquet --output corpus_submissions.csv
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
    for attempt in range(5):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model":  model,
                "stream": False,
                "think":  False,
                "keep_alive": -1,
                "format": JSON_SCHEMA,
                "options": {"temperature": 0, "num_predict": 200, "num_ctx": 16384},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Classify this text:\n\n{text}"},
                ],
            }, timeout=300)
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "{}")
            parsed = json.loads(raw) if raw else {}
            if parsed.get("buyin") or parsed.get("stance"):
                return parsed
            print(f"  Empty response, retry {attempt+1}/5...")
            time.sleep(10)
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            wait = min(30 * (attempt + 1), 120)
            print(f"  Error: {e.__class__.__name__}, retry {attempt+1}/5 in {wait}s...")
            time.sleep(wait)
    print("  All retries failed, defaulting to neutral")
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="corpus_labels.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.input.endswith(".parquet"):
        corpus = pd.read_parquet(args.input, columns=["doc_id", "text"])
    else:
        corpus = pd.read_csv(args.input)

    # Create chunk_id if missing (doc_id + row suffix for uniqueness)
    if "chunk_id" not in corpus.columns:
        counts = {}
        chunk_ids = []
        for doc_id in corpus["doc_id"]:
            idx = counts.get(doc_id, 0)
            chunk_ids.append(f"{doc_id}_{idx}")
            counts[doc_id] = idx + 1
        corpus["chunk_id"] = chunk_ids

    # Resume
    done_ids = set()
    done = pd.DataFrame()
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
    rows = []
    for i, (_, row) in enumerate(remaining.iterrows()):
        result = label_one(row["text"], args.model)
        rows.append({
            "chunk_id":     row["chunk_id"],
            "pred_buyin":   result.get("buyin", "neutral"),
            "pred_stance":  result.get("stance", "neutral"),
            "pred_c2d":     result.get("c2d_strength", ""),
        })

        # Atomic save every row
        out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        tmp = args.output + ".tmp"
        out.to_csv(tmp, index=False)
        try:
            os.replace(tmp, args.output)
        except PermissionError:
            pass

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining_time = rate * (total - i - 1)
            print(f"  {i+1}/{total} | {rate:.1f}s/row | ETA {remaining_time/3600:.1f} hrs")

    elapsed = time.time() - t0
    final = pd.read_csv(args.output)
    print(f"\nDone: {len(final)} rows in {elapsed/3600:.1f} hrs ({elapsed/max(total,1):.1f}s/row)")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
