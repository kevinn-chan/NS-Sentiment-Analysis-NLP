"""
Patches chunk_sentiment.parquet by re-scoring Singlish-containing chunks
with cardiffnlp/twitter-xlm-roberta-base-sentiment.

Why: twitter-roberta-base-sentiment-latest scored 40% on controlled Singlish
     test; XLM scored 80% on the same sentences. XLM handles Singlish because
     its multilingual pretraining covers Malay vocabulary overlap and
     code-switching patterns.

What changes: only chunks with at least one Singlish marker (word-boundary
     regex) get re-scored — ~33k chunks (~4.5% of corpus). The remaining
     ~704k English-dominant chunks keep their existing RoBERTa scores.

Runtime: ~15–20 min on Apple Silicon CPU (batch_size=64).

Usage:
    python -m src.features.patch_sentiment_xlm
"""

import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
XLM_MODEL  = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH_SIZE = 64
DATA_DIR   = Path(__file__).parent.parent.parent / "data" / "processed" / "new"

# ---------------------------------------------------------------------------
# Singlish detector — word-boundary regex, superset of lexicon markers
# ---------------------------------------------------------------------------
_SINGLISH_VOCAB = {
    # particles
    "lah","leh","lor","liao","sia","hor","mah","bah","wah","nia",
    # sentiment terms
    "shiok","song","swee","steady","lobang","slack","lepak","sian","jialat",
    "wayang","chao keng","saikang","siong","tekan","kns","suay","teruk",
    "terok","bo chap","bochap","arrow","gg","bo liao","tok kok","cmi","kena",
    # extended
    "walao","walau","siao","aiyah","aiyoh","alamak","cheem","liddat",
    "shiok leh","tok gong","powderful","gao gao","die die",
}

# Build one compiled pattern for speed
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_SINGLISH_VOCAB, key=len, reverse=True)) + r")\b"
)

def has_singlish(text: str) -> bool:
    return bool(_PATTERN.search(str(text).lower()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ── Load existing sentiment scores ──────────────────────────────────────
    print("Loading chunk_sentiment.parquet …")
    sent = pd.read_parquet(DATA_DIR / "chunk_sentiment.parquet")
    print(f"  {len(sent):,} rows  |  cols: {sent.columns.tolist()}")

    # ── Load chunk text ──────────────────────────────────────────────────────
    print("Loading chunk text …")
    sub = pd.read_parquet(DATA_DIR / "submissions_chunks.parquet", columns=["chunk_id","text"])
    com = pd.read_parquet(DATA_DIR / "comments_chunks.parquet",    columns=["chunk_id","text"])
    chunks = pd.concat([sub, com], ignore_index=True).drop_duplicates(subset="chunk_id")
    del sub, com

    # Merge text into scores dataframe
    df = sent.merge(chunks[["chunk_id","text"]], on="chunk_id", how="left")
    del chunks

    # ── Identify Singlish-containing chunks ──────────────────────────────────
    print("Detecting Singlish chunks …")
    df["has_sg"] = df["text"].apply(has_singlish)
    sg_mask      = df["has_sg"]
    n_sg         = sg_mask.sum()
    print(f"  Singlish chunks: {n_sg:,} / {len(df):,}  ({n_sg/len(df)*100:.1f}%)")

    # ── Load XLM model ───────────────────────────────────────────────────────
    device = 0 if torch.cuda.is_available() else -1
    print(f"\nLoading {XLM_MODEL}  (device={'GPU' if device==0 else 'CPU'}) …")
    xlm = pipeline(
        "sentiment-analysis",
        model=XLM_MODEL,
        top_k=None,
        device=device,
        truncation=True,
        max_length=512,
    )

    # Verify label set
    test_out = xlm(["test"])
    labels   = sorted(s["label"] for s in test_out[0])
    print(f"  Label set: {labels}")
    assert set(labels) == {"negative","neutral","positive"}, f"Unexpected labels: {labels}"

    # ── Batch inference on Singlish chunks ───────────────────────────────────
    sg_df    = df[sg_mask].copy().reset_index(drop=True)
    texts    = sg_df["text"].fillna("").str.strip().tolist()
    chunk_ids = sg_df["chunk_id"].tolist()
    n        = len(texts)

    print(f"\nScoring {n:,} Singlish chunks with XLM …")
    new_records = []
    t0 = time.time()

    for start in range(0, n, BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]
        batch_ids   = chunk_ids[start : start + BATCH_SIZE]
        raw         = xlm(batch_texts, batch_size=BATCH_SIZE)

        for cid, scores in zip(batch_ids, raw):
            score_map = {s["label"]: s["score"] for s in scores}
            new_records.append({
                "chunk_id": cid,
                "sent_neg": score_map.get("negative", float("nan")),
                "sent_neu": score_map.get("neutral",  float("nan")),
                "sent_pos": score_map.get("positive", float("nan")),
            })

        done    = start + len(batch_texts)
        elapsed = time.time() - t0
        rate    = done / elapsed if elapsed > 0 else 0
        eta     = (n - done) / rate if rate > 0 else 0
        if done % 5000 < BATCH_SIZE or done >= n:
            print(f"  {done:>6,} / {n:,}  |  {rate:.0f} chunks/s  |  ETA {eta/60:.1f} min")

    print(f"Done in {(time.time()-t0)/60:.1f} min")

    # ── Patch original dataframe ─────────────────────────────────────────────
    new_df = pd.DataFrame(new_records).set_index("chunk_id")
    sent   = sent.set_index("chunk_id")
    sent.update(new_df)          # in-place replace rows that match chunk_id
    sent   = sent.reset_index()

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = DATA_DIR / "chunk_sentiment.parquet"
    sent.to_parquet(out_path, index=False)
    print(f"\nSaved patched parquet → {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    label_map = {"sent_neg":"negative","sent_neu":"neutral","sent_pos":"positive"}
    sent["majority_label"] = sent[["sent_neg","sent_neu","sent_pos"]].idxmax(axis=1).map(label_map)
    print("\n── Final label distribution (full corpus after patch) ──")
    vc = sent["majority_label"].value_counts()
    for lbl, cnt in vc.items():
        print(f"  {lbl:<10} {cnt:>8,}  ({cnt/len(sent)*100:.1f}%)")

    print("\n── XLM scores on patched Singlish chunks ──")
    patched = sent[sent["chunk_id"].isin(new_df.index)]
    patched_labels = patched[["sent_neg","sent_neu","sent_pos"]].idxmax(axis=1).map(label_map)
    xlm_vc = patched_labels.value_counts()
    for lbl, cnt in xlm_vc.items():
        print(f"  {lbl:<10} {cnt:>8,}  ({cnt/len(patched)*100:.1f}%)")


if __name__ == "__main__":
    main()
