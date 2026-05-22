"""
Tier 3 sentiment audit: samples ~500 chunks and labels them via local Ollama.
Measures how well twitter-roberta performs on Singlish-heavy NS content.

Run AFTER chunk_sentiment.parquet exists (i.e. after the Kaggle Tier 1 notebook).

Usage:
    python -m src.features.sentiment_audit

Requires Ollama running locally (`ollama serve`). Defaults to gemma3:1b.
Free, no API key, data stays local. ~10–15 min on Apple Silicon.

Decision gate: if agreement on high-Singlish chunks is < 85%, consider
fine-tuning SingBERT on these gold labels (needs ~500 more annotations).
"""

import json
import time
from pathlib import Path

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"   # change if you pull a larger model

# ---------------------------------------------------------------------------
# Singlish markers used for density estimation (not scoring)
# ---------------------------------------------------------------------------
_SINGLISH_PARTICLES = {"lah", "leh", "lor", "liao", "sia", "hor", "mah", "bah", "wah", "nia"}
_SINGLISH_TERMS = {
    "shiok", "song", "swee", "steady", "lobang", "slack", "lepak", "ho seh",
    "ord", "sian", "jialat", "wayang", "chao keng", "saikang", "siong", "tekan",
    "guai lan", "kns", "suay", "sway", "teruk", "terok", "bo chap", "bochap",
    "kan cheong", "arrow", "gg", "bo liao", "tok kok", "cmi", "gao gao",
    "shiok leh", "jia lat", "tok gong", "powderful", "kena",
}
SINGLISH_MARKERS = _SINGLISH_PARTICLES | _SINGLISH_TERMS


def singlish_density(text: str) -> float:
    """Fraction of whitespace-split tokens that match a Singlish marker."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in SINGLISH_MARKERS) / len(tokens)


# ---------------------------------------------------------------------------
# Ollama labelling
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a sentiment classifier for Singapore National Service (NS) Reddit posts. "
    "Singlish is common: 'sian' = fed up, 'shiok' = great, 'wayang' = fake, "
    "'jialat' = bad situation, 'siong' = tough, 'lobang' = good opportunity. "
    "Always reply with valid JSON only."
)

_USER_TMPL = """\
Classify the sentiment of this Reddit post or comment about Singapore's National Service.

Classify as one of: negative, neutral, positive

Rules:
- Judge the author's emotional tone, not the topic
- Complaints, frustration, criticism → negative
- Factual, informational, balanced → neutral
- Pride, satisfaction, enthusiasm, relief → positive
- If clearly sarcastic, invert the surface sentiment

Reply with ONLY this JSON (no other text):
{{"label": "negative or neutral or positive", "confidence": 0.0-1.0, "reason": "under 10 words"}}

Text:
{text}"""


def _label_one(text: str) -> dict:
    payload = {
        "model":  OLLAMA_MODEL,
        "format": "json",          # Ollama enforces valid JSON output
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER_TMPL.format(text=text[:1200])},
        ],
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    result  = json.loads(content)

    # Normalise label to lowercase and validate
    label = result.get("label", "").lower().strip()
    if label not in {"negative", "neutral", "positive"}:
        raise ValueError(f"Unexpected label: {label!r}")
    result["label"] = label
    return result


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------
def run_audit(
    sentiment_path: Path,
    sub_path: Path,
    com_path: Path,
    n_sample: int = 500,
    out_path: Path | None = None,
    high_singlish_fraction: float = 0.5,
) -> pd.DataFrame:
    """
    Sample n_sample chunks, label with Ollama, compare against twitter-roberta.
    Returns a DataFrame with both labels and a per-row agreement column.
    """
    # Verify Ollama is reachable before starting
    try:
        requests.get("http://localhost:11434", timeout=5).raise_for_status()
    except Exception:
        raise RuntimeError(
            "Ollama is not reachable at localhost:11434. "
            "Start it with: ollama serve"
        )

    # Load scores + text
    scores = pd.read_parquet(sentiment_path)
    sub    = pd.read_parquet(sub_path, columns=["chunk_id", "text"])
    com    = pd.read_parquet(com_path, columns=["chunk_id", "text"])
    df     = pd.concat([sub, com], ignore_index=True).merge(scores, on="chunk_id", how="inner")
    del sub, com, scores

    df["singlish_density"] = df["text"].apply(singlish_density)

    # Stratified sample: high-Singlish + general
    n_high = int(n_sample * high_singlish_fraction)
    n_rest = n_sample - n_high

    high_sg = (
        df[df["singlish_density"] > 0.04]
        .sample(min(n_high, (df["singlish_density"] > 0.04).sum()), random_state=42)
    )
    rest = (
        df[~df["chunk_id"].isin(high_sg["chunk_id"])]
        .sample(n_rest, random_state=42)
    )
    sample = pd.concat([high_sg, rest], ignore_index=True)
    print(f"Model:  {OLLAMA_MODEL}")
    print(f"Sample: {len(sample)} total  ({len(high_sg)} high-Singlish, {len(rest)} general)")

    # Ollama labelling
    gold_labels, gold_conf, gold_reasons = [], [], []
    t0 = time.time()

    for i, row in enumerate(sample.itertuples()):
        try:
            result = _label_one(row.text)
            gold_labels.append(result["label"])
            gold_conf.append(result.get("confidence", 1.0))
            gold_reasons.append(result.get("reason", ""))
        except Exception as e:
            print(f"  [WARN] chunk {row.chunk_id}: {e}")
            gold_labels.append(None)
            gold_conf.append(None)
            gold_reasons.append(None)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (len(sample) - i - 1)
            print(f"  {i+1:>4} / {len(sample)}  |  ETA {eta/60:.1f} min")

    sample = sample.copy()
    sample["gold_label"]      = gold_labels
    sample["gold_confidence"] = gold_conf
    sample["gold_reason"]     = gold_reasons

    # RoBERTa predicted label
    label_map = {"sent_neg": "negative", "sent_neu": "neutral", "sent_pos": "positive"}
    sample["roberta_label"] = (
        sample[["sent_neg", "sent_neu", "sent_pos"]].idxmax(axis=1).map(label_map)
    )
    sample["agree"] = sample["gold_label"] == sample["roberta_label"]

    # ── Summary ──────────────────────────────────────────────────────────────
    valid   = sample.dropna(subset=["gold_label"])
    overall = valid["agree"].mean()
    print(f"\n── Overall agreement: {overall:.1%} (n={len(valid)}) ──")

    bins        = [0, 0.02, 0.06, 0.15, 1.0]
    bin_labels  = ["low (0–2%)", "medium (2–6%)", "high (6–15%)", "very high (>15%)"]
    valid       = valid.copy()
    valid["sg_bin"] = pd.cut(valid["singlish_density"], bins=bins, labels=bin_labels)

    print("\n── Agreement by Singlish density ──")
    for sg_bin, grp in valid.groupby("sg_bin", observed=True):
        acc  = grp["agree"].mean()
        n    = len(grp)
        flag = "  ← BELOW THRESHOLD" if acc < 0.85 and n >= 20 else ""
        print(f"  {str(sg_bin):<22}  {acc:.1%}  (n={n}){flag}")

    print("\n── Confusion matrix (gold rows × roberta cols) ──")
    lbl_order = ["negative", "neutral", "positive"]
    for gold in lbl_order:
        row_data = valid[valid["gold_label"] == gold]["roberta_label"].value_counts()
        counts   = [row_data.get(lbl, 0) for lbl in lbl_order]
        print(f"  gold={gold:<10}  " + "  ".join(f"{lbl}={c:>4}" for lbl, c in zip(lbl_order, counts)))

    print("\n── Recommendation ──")
    high_grp    = valid[valid["sg_bin"] == "high (6–15%)"]
    high_sg_acc = high_grp["agree"].mean() if len(high_grp) >= 10 else overall
    if overall >= 0.85:
        print("  ✓ twitter-roberta acceptable (overall ≥ 85%). Ship Tier 1 as primary signal.")
    elif high_sg_acc < 0.80:
        print("  ✗ High-Singlish accuracy < 80%. Consider fine-tuning SingBERT on gold labels.")
        print("    Action: annotate 500 more chunks → fine-tune zanelim/singbert on Kaggle.")
    else:
        print("  ~ Moderate divergence on Singlish chunks. Use Tier 2 lexicon as supplementary signal.")
        print("    Consider weighting roberta score by singlish_density in downstream aggregation.")

    if out_path:
        sample.to_parquet(out_path, index=False)
        print(f"\nSaved audit results → {out_path}")

    return sample


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "new"

    run_audit(
        sentiment_path = DATA_DIR / "chunk_sentiment.parquet",
        sub_path       = DATA_DIR / "submissions_chunks.parquet",
        com_path       = DATA_DIR / "comments_chunks.parquet",
        n_sample       = 500,
        out_path       = DATA_DIR / "sentiment_audit.parquet",
    )
