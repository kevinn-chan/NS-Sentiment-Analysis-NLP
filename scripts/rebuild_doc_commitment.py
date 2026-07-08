"""Rebuild doc-level commitment columns in doc_sentiment.parquet from cascade output.

Also fills commitment_divergence in doc_divergence_v2.parquet (Stage 7 patch).
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data/processed/new")

# ── Load ──
casc = pd.read_parquet(DATA / "chunk_commitment_cascade.parquet")
meta = pd.read_parquet(DATA / "chunk_metadata.parquet",
                       columns=["chunk_id", "doc_id", "sent_neg", "upvotes"])
doc_sent = pd.read_parquet(DATA / "doc_sentiment.parquet")

print(f"Cascade: {len(casc):,}  Metadata: {len(meta):,}  Doc: {len(doc_sent):,}")

# ── Join cascade to metadata ──
df = meta.merge(casc, on="chunk_id", how="inner")
df["upvote_w"] = df["upvotes"].fillna(0) + 1

df["is_unc"]  = (df["buyin_label"] == "uncommitted").astype(int)
df["is_com"]  = (df["buyin_label"] == "committed").astype(int)
df["is_crit"] = (df["stance_label"] == "critical").astype(int)
df["is_sup"]  = (df["stance_label"] == "supportive").astype(int)
df["is_neg"]  = (df["sent_neg"] > 0.6).astype(int)
df["is_pos"]  = (df["sent_neg"] < 0.3).astype(int)
df["is_broad_unc"]  = ((df["is_unc"] == 1) | ((df["buyin_label"] == "neutral") & (df["sent_neg"] > 0.70))).astype(int)
df["is_broad_crit"] = ((df["is_crit"] == 1) | ((df["stance_label"] == "neutral") & (df["sent_neg"] > 0.70))).astype(int)

# ── Aggregate per doc ──
grp = df.groupby("doc_id")
doc_commit = grp.agg(
    chunk_count_commit = ("chunk_id", "count"),
    total_upvotes_commit = ("upvotes", "sum"),
    n_unc      = ("is_unc", "sum"),
    n_com      = ("is_com", "sum"),
    n_crit     = ("is_crit", "sum"),
    n_sup      = ("is_sup", "sum"),
    n_neg      = ("is_neg", "sum"),
    n_pos      = ("is_pos", "sum"),
    n_broad_unc  = ("is_broad_unc", "sum"),
    n_broad_crit = ("is_broad_crit", "sum"),
).reset_index()

cc = doc_commit["chunk_count_commit"].clip(lower=1)
doc_commit["pct_uncommitted"]       = doc_commit["n_unc"]       / cc
doc_commit["pct_committed"]         = doc_commit["n_com"]       / cc
doc_commit["pct_critical"]          = doc_commit["n_crit"]      / cc
doc_commit["pct_supportive"]        = doc_commit["n_sup"]       / cc
doc_commit["pct_negative"]          = doc_commit["n_neg"]       / cc
doc_commit["pct_positive"]          = doc_commit["n_pos"]       / cc
doc_commit["pct_broad_uncommitted"] = doc_commit["n_broad_unc"] / cc
doc_commit["pct_broad_critical"]    = doc_commit["n_broad_crit"]/ cc

# Upvote-weighted rates per doc
wtd_rows = []
for doc_id, g in df.groupby("doc_id"):
    w = g["upvote_w"]
    tw = w.sum()
    wtd_rows.append({
        "doc_id": doc_id,
        "wtd_uncommitted":       (g["is_unc"]       * w).sum() / tw if tw else 0,
        "wtd_committed":         (g["is_com"]       * w).sum() / tw if tw else 0,
        "wtd_critical":          (g["is_crit"]      * w).sum() / tw if tw else 0,
        "wtd_supportive":        (g["is_sup"]       * w).sum() / tw if tw else 0,
        "wtd_negative":          (g["is_neg"]       * w).sum() / tw if tw else 0,
        "wtd_positive":          (g["is_pos"]       * w).sum() / tw if tw else 0,
        "wtd_broad_uncommitted": (g["is_broad_unc"] * w).sum() / tw if tw else 0,
        "wtd_broad_critical":    (g["is_broad_crit"]* w).sum() / tw if tw else 0,
    })
wtd = pd.DataFrame(wtd_rows)
doc_commit = doc_commit.merge(wtd, on="doc_id")
doc_commit["net_disposition"] = (doc_commit["wtd_committed"] + doc_commit["wtd_supportive"]
                                  - doc_commit["wtd_uncommitted"] - doc_commit["wtd_critical"])

# ── Drop old commitment columns from doc_sentiment, merge new ──
commit_cols = ["chunk_count_commit", "total_upvotes_commit",
               "pct_uncommitted", "pct_committed", "pct_critical", "pct_supportive",
               "pct_negative", "pct_positive", "pct_broad_uncommitted", "pct_broad_critical",
               "wtd_uncommitted", "wtd_committed", "wtd_critical", "wtd_supportive",
               "wtd_negative", "wtd_positive", "wtd_broad_uncommitted", "wtd_broad_critical",
               "net_disposition"]
existing_commit_cols = [c for c in commit_cols if c in doc_sent.columns]
doc_sent = doc_sent.drop(columns=existing_commit_cols)
doc_sent = doc_sent.merge(doc_commit[["doc_id"] + commit_cols], on="doc_id", how="left")

out_path = DATA / "doc_sentiment.parquet"
doc_sent.to_parquet(out_path, index=False)
print(f"\nSaved {out_path}  ({doc_sent.shape})")
print(f"Commitment columns: {commit_cols}")

# ── Stage 7 patch: fill commitment_divergence in doc_divergence_v2 ──
div_path = DATA / "doc_divergence_v2.parquet"
if div_path.exists():
    div = pd.read_parquet(div_path)
    # Per-post: pct_uncommitted(comments) - pct_uncommitted(submission)
    df_with_doctype = df.merge(
        pd.read_parquet(DATA / "chunk_metadata.parquet", columns=["chunk_id", "doc_id"]).drop_duplicates("chunk_id"),
        on="chunk_id", how="left", suffixes=("", "_meta")
    )
    # Need doc_type — get from comments/submissions chunks
    comments = pd.read_parquet(DATA / "comments_chunks.parquet", columns=["doc_id", "post_id", "doc_type"])
    submissions = pd.read_parquet(DATA / "submissions_chunks.parquet", columns=["doc_id", "doc_type"])
    submissions["post_id"] = submissions["doc_id"]
    doc_types = pd.concat([
        comments[["doc_id", "post_id", "doc_type"]].drop_duplicates("doc_id"),
        submissions[["doc_id", "post_id", "doc_type"]].drop_duplicates("doc_id"),
    ])

    # Join doc_type and post_id to commitment data
    dc = doc_commit[["doc_id", "pct_uncommitted"]].merge(doc_types, on="doc_id", how="inner")

    # Submission uncommitted rate per post
    sub_unc = dc[dc["doc_type"] == "submission"].set_index("post_id")["pct_uncommitted"]
    # Comment uncommitted rate per post (mean across all comment docs)
    com_unc = dc[dc["doc_type"] == "comment"].groupby("post_id")["pct_uncommitted"].mean()

    commit_div = (com_unc - sub_unc).dropna()
    div["commitment_divergence"] = div["post_id"].map(commit_div)
    div.to_parquet(div_path, index=False)
    filled = div["commitment_divergence"].notna().sum()
    print(f"\nPatched {div_path}: {filled:,}/{len(div):,} posts with commitment_divergence")
else:
    print(f"\n{div_path} not found — skipping Stage 7 patch")
