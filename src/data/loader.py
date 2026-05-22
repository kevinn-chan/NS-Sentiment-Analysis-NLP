import zstandard
import json
import os
import logging
from pathlib import Path

import pandas as pd

from config import (
    TARGET_SUBREDDITS,
    DATA_RAW,
    DATA_INTERIM,
)

log = logging.getLogger(__name__)

SUBMISSION_FIELDS = {
    "id", "title", "selftext", "score", "upvote_ratio",
    "num_comments", "created_utc", "link_flair_text",
    "author", "permalink", "subreddit",
}

COMMENT_FIELDS = {
    "id", "body", "author", "score", "created_utc",
    "subreddit", "link_id", "parent_id", "permalink",
}


# ── ZST streaming core (adapted from Watchful1/PushshiftDumps) ───────────────

def _read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    # Handles UTF-8 codepoints that straddle chunk boundaries.
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        return _read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def _stream_zst(file_path: str):
    """Yield (line, bytes_processed) from a ZST file without loading it fully into memory."""
    with open(file_path, "rb") as fh:
        buffer = ""
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
        while True:
            chunk = _read_and_decode(reader, 2**27, (2**29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line, fh.tell()
            buffer = lines[-1]
        reader.close()


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_submissions(file_path: str) -> pd.DataFrame:
    file_size = os.stat(file_path).st_size
    records, bad_lines, total_lines = [], 0, 0

    for line, bytes_processed in _stream_zst(file_path):
        total_lines += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue

        if obj.get("subreddit", "").lower() not in TARGET_SUBREDDITS:
            continue

        records.append({field: obj.get(field) for field in SUBMISSION_FIELDS})

        if total_lines % 100_000 == 0:
            log.info(
                f"[submissions] {total_lines:,} lines | {bad_lines:,} bad | "
                f"{(bytes_processed / file_size) * 100:.1f}%"
            )

    log.info(
        f"[submissions] done — {total_lines:,} total | {bad_lines:,} bad | {len(records):,} kept"
    )

    df = pd.DataFrame(records)
    df["created_utc"] = pd.to_datetime(
        pd.to_numeric(df["created_utc"], errors="coerce"), unit="s", utc=True
    )
    for col in ("id", "author", "title", "selftext", "subreddit", "permalink", "link_flair_text"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["doc_type"] = "submission"
    return df


def load_comments(file_path: str) -> pd.DataFrame:
    file_size = os.stat(file_path).st_size
    records, bad_lines, total_lines = [], 0, 0

    for line, bytes_processed in _stream_zst(file_path):
        total_lines += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue

        if obj.get("subreddit", "").lower() not in TARGET_SUBREDDITS:
            continue

        records.append({field: obj.get(field) for field in COMMENT_FIELDS})

        if total_lines % 100_000 == 0:
            log.info(
                f"[comments] {total_lines:,} lines | {bad_lines:,} bad | "
                f"{(bytes_processed / file_size) * 100:.1f}%"
            )

    log.info(
        f"[comments] done — {total_lines:,} total | {bad_lines:,} bad | {len(records):,} kept"
    )

    df = pd.DataFrame(records)
    df["created_utc"] = pd.to_datetime(
        pd.to_numeric(df["created_utc"], errors="coerce"), unit="s", utc=True
    )
    # Pushshift dumps sometimes store these as int — cast to str to ensure uniform type for parquet
    for col in ("id", "author", "body", "subreddit", "link_id", "parent_id", "permalink"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["doc_type"] = "comment"
    # Strip 't3_' prefix → bare submission id, used to join comments back to their post
    df["post_id"] = df["link_id"].str.replace(r"^t3_", "", regex=True)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def load_all(data_dir: Path = DATA_RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan data_dir for all ZST files. Routes by filename to the correct loader.

    Each file is saved to its own parquet immediately after loading to avoid
    holding multiple large DataFrames in memory simultaneously. The combined
    DataFrames are only materialised at the end for return — callers that only
    need the parquet files (e.g. the cleaner) can skip the return value.
    """
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    sub_dfs, com_dfs = [], []

    for zst_file in sorted(data_dir.glob("*.zst")):
        name = zst_file.name.lower()
        stem = zst_file.stem                        # e.g. "singapore_comments"
        log.info(f"Loading: {zst_file.name}")

        if "submission" in name:
            df = load_submissions(str(zst_file))
            out = DATA_INTERIM / f"{stem}.parquet"
            df.to_parquet(out, index=False)
            log.info(f"Saved {len(df):,} rows → {out.name}")
            sub_dfs.append(df)

        elif "comment" in name:
            df = load_comments(str(zst_file))
            out = DATA_INTERIM / f"{stem}.parquet"
            df.to_parquet(out, index=False)
            log.info(f"Saved {len(df):,} rows → {out.name}")
            com_dfs.append(df)

        else:
            log.warning(
                f"Cannot infer doc type from '{zst_file.name}' — skipped. "
                "Filename must contain 'submission' or 'comment'."
            )

    submissions = pd.concat(sub_dfs, ignore_index=True) if sub_dfs else pd.DataFrame()
    comments    = pd.concat(com_dfs, ignore_index=True) if com_dfs else pd.DataFrame()
    log.info(f"Total: {len(submissions):,} submissions | {len(comments):,} comments")

    # Save merged files under the names cleaner.py expects for a clean end-to-end run.
    submissions.to_parquet(DATA_INTERIM / "submissions_raw.parquet", index=False)
    comments.to_parquet(DATA_INTERIM / "comments_raw.parquet", index=False)
    log.info("Saved submissions_raw.parquet and comments_raw.parquet")

    return submissions, comments
