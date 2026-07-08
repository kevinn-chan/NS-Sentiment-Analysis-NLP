"""Rebuild temporal_commitment.parquet from cascade inference output.

Joins chunk_commitment_cascade.parquet with chunk_metadata.parquet,
aggregates by (year_month, subreddit, topic_macro).
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data/processed/new")

# ── Load ──
casc = pd.read_parquet(DATA / "chunk_commitment_cascade.parquet")
meta = pd.read_parquet(DATA / "chunk_metadata.parquet")
print(f"Cascade: {len(casc):,} rows   Metadata: {len(meta):,} rows")

# ── Join ──
df = meta.merge(casc, on="chunk_id", how="inner")
df["year_month"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
df["upvote_w"] = df["upvotes"].fillna(0) + 1
print(f"Joined: {len(df):,} rows")

# ── Binary flags ──
df["is_unc"]  = (df["buyin_label"] == "uncommitted").astype(int)
df["is_com"]  = (df["buyin_label"] == "committed").astype(int)
df["is_crit"] = (df["stance_label"] == "critical").astype(int)
df["is_sup"]  = (df["stance_label"] == "supportive").astype(int)
df["is_neg"]  = (df["sent_neg"] > 0.6).astype(int)
df["is_pos"]  = (df["sent_pos"] > 0.6).astype(int)

# Broad: explicit label OR (neutral + high sentiment negativity)
df["is_broad_unc"]  = ((df["is_unc"] == 1) | ((df["buyin_label"] == "neutral") & (df["sent_neg"] > 0.70))).astype(int)
df["is_broad_crit"] = ((df["is_crit"] == 1) | ((df["stance_label"] == "neutral") & (df["sent_neg"] > 0.70))).astype(int)

# ── Aggregate ──
grp = df.groupby(["year_month", "subreddit", "topic_macro"])

counts = grp.agg(
    chunk_count   = ("chunk_id", "count"),
    total_upvotes = ("upvotes", "sum"),
    n_unc         = ("is_unc", "sum"),
    n_com         = ("is_com", "sum"),
    n_crit        = ("is_crit", "sum"),
    n_sup         = ("is_sup", "sum"),
    n_neg         = ("is_neg", "sum"),
    n_pos         = ("is_pos", "sum"),
    n_broad_unc   = ("is_broad_unc", "sum"),
    n_broad_crit  = ("is_broad_crit", "sum"),
).reset_index()

cc = counts["chunk_count"].clip(lower=1)
counts["pct_uncommitted"]      = counts["n_unc"]       / cc
counts["pct_committed"]        = counts["n_com"]       / cc
counts["pct_critical"]         = counts["n_crit"]      / cc
counts["pct_supportive"]       = counts["n_sup"]       / cc
counts["pct_negative"]         = counts["n_neg"]       / cc
counts["pct_positive"]         = counts["n_pos"]       / cc
counts["pct_broad_uncommitted"]= counts["n_broad_unc"] / cc
counts["pct_broad_critical"]   = counts["n_broad_crit"]/ cc

# ── Upvote-weighted rates ──
def weighted_rate(group, flag_col):
    w = group["upvote_w"]
    return (group[flag_col] * w).sum() / w.sum() if w.sum() > 0 else 0.0

wtd_rows = []
for key, grp_df in df.groupby(["year_month", "subreddit", "topic_macro"]):
    w = grp_df["upvote_w"]
    total_w = w.sum()
    row = {
        "year_month": key[0], "subreddit": key[1], "topic_macro": key[2],
        "wtd_uncommitted":      (grp_df["is_unc"]       * w).sum() / total_w if total_w else 0,
        "wtd_committed":        (grp_df["is_com"]       * w).sum() / total_w if total_w else 0,
        "wtd_critical":         (grp_df["is_crit"]      * w).sum() / total_w if total_w else 0,
        "wtd_supportive":       (grp_df["is_sup"]       * w).sum() / total_w if total_w else 0,
        "wtd_negative":         (grp_df["is_neg"]       * w).sum() / total_w if total_w else 0,
        "wtd_positive":         (grp_df["is_pos"]       * w).sum() / total_w if total_w else 0,
        "wtd_broad_uncommitted":(grp_df["is_broad_unc"] * w).sum() / total_w if total_w else 0,
        "wtd_broad_critical":   (grp_df["is_broad_crit"]* w).sum() / total_w if total_w else 0,
    }
    wtd_rows.append(row)

wtd = pd.DataFrame(wtd_rows)
out = counts.merge(wtd, on=["year_month", "subreddit", "topic_macro"])

# Net disposition: (committed + supportive - uncommitted - critical) weighted
out["net_disposition"] = (out["wtd_committed"] + out["wtd_supportive"] - out["wtd_uncommitted"] - out["wtd_critical"])

out = out.sort_values(["year_month", "subreddit", "topic_macro"]).reset_index(drop=True)

out_path = DATA / "temporal_commitment.parquet"
out.to_parquet(out_path, index=False)
print(f"\nSaved {out_path}  ({out.shape})")
print(f"Year-months: {out['year_month'].nunique()}")
print(f"Sample:\n{out.head(3).to_string()}")
