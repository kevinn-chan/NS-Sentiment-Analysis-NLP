"""
Rebuild doc_author_monthly.parquet with truly unique authors per topic×month,
deduplicated across subreddits.

Join doc_sentiment (has topic labels) → raw parquets (have author) on post_id/id,
then nunique(author) per topic_macro × topic_sub × year_month.
"""
import pandas as pd
from pathlib import Path

DATA  = Path("data/processed/new")
INTER = Path("data/interim")

print("Loading doc_sentiment...")
doc = pd.read_parquet(DATA / "doc_sentiment.parquet",
                      columns=["doc_id", "post_id", "doc_type", "created_utc",
                               "topic_macro", "topic_sub"])
doc = doc.dropna(subset=["topic_macro"])
doc["year_month"] = pd.to_datetime(doc["created_utc"], utc=True).dt.to_period("M").astype(str)

print(f"  {len(doc):,} docs with topic labels")

print("Loading raw author data...")
subs = pd.read_parquet(INTER / "submissions_raw.parquet", columns=["id", "author"])
subs = subs.rename(columns={"id": "post_id"})

# For comments: join on comment's own id (doc_id), not post_id (parent thread)
coms = pd.read_parquet(INTER / "comments_raw.parquet", columns=["id", "author"])
coms = coms.rename(columns={"id": "doc_id"})

for df in [subs, coms]:
    df["author"] = df["author"].where(~df["author"].isin(["[deleted]", "AutoModerator", "", None]))

print("Joining authors to doc_sentiment...")
doc_subs = doc[doc["doc_type"] == "submission"].merge(subs, on="post_id", how="left")
doc_coms = doc[doc["doc_type"] == "comment"].merge(coms, on="doc_id", how="left")
merged = pd.concat([doc_subs, doc_coms], ignore_index=True)
print(f"  {merged['author'].isna().sum():,} docs with no author match (deleted/bot)")

# Truly unique authors per topic_macro × topic_sub × year_month (no subreddit split)
print("Aggregating unique authors...")
da = (
    merged
    .groupby(["topic_macro", "topic_sub", "year_month"])["author"]
    .nunique()
    .reset_index()
    .rename(columns={"author": "unique_authors"})
)

out = DATA / "doc_author_monthly.parquet"
da.to_parquet(out, index=False)
print(f"Saved {len(da):,} rows → {out}")

# Global unique authors per month (no topic grouping — avoids double-counting)
da_global = (
    merged.dropna(subset=["author"])
    .groupby("year_month")["author"]
    .nunique()
    .reset_index()
    .rename(columns={"author": "unique_authors"})
)
out_global = DATA / "doc_author_monthly_global.parquet"
da_global.to_parquet(out_global, index=False)
print(f"Saved {len(da_global):,} rows → {out_global}")
print(da_global.sort_values("year_month").tail(10))
