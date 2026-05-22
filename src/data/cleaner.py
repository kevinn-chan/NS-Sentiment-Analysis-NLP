import re
import logging
import numpy as np
import pandas as pd

from config import (
    DATA_INTERIM,
    DATA_PROCESSED,
    SUBMISSIONS_CLEAN,
    COMMENTS_CLEAN,
)

log = logging.getLogger(__name__)

# ── Bot authors ───────────────────────────────────────────────────────────────
BOT_AUTHORS = {"automoderator", "sneakpeek_bot", "microtechanalysis"}

# ── NS relevance keywords ─────────────────────────────────────────────────────
PHRASE_KEYWORDS = {
    # Identity & core
    "national service", "nsman", "ns man", "nsmen",
    # Policy discourse
    "conscription", "serve nation", "defend singapore", "duty to country",
    "ns disruption", "ns deferment",
    # Milestones & timeline
    "operationally ready", "enlistment", "ns training",
    "book in", "book out", "confined to camp", "in-camp",
    # Medical
    "pes status", "downpes", "medical board",
    # Units, vocations & locations
    "officer cadet", "mindef", "tekong", "bmtc",
    "naval diving", "infantry", "commandos", "storeman",
    # Training & duties
    "route march", "guard duty", "outfield", "sign extra",
    # Slang & camp culture
    "chao keng", "siao on", "tekkan", "arrowed",
    # Admin
    "off day", "leave pass", "emart",
}

WORD_KEYWORDS = {
    # Core acronyms
    "ns", "bmt", "pes", "ord", "saf", "spf", "scdf", "nsf", "ndu",
    # Training & assessment
    "ippt", "ict", "rsi", "rso",
    # Ranks & roles
    "recruit", "enlistee", "enlisted", "enlist",
    "ocs", "scs", "vocation", "sergeant", "encik",
    # Admin & slang
    "db", "wayang", "reservist", "conscript",
}

# Single compiled regex covering both tiers — built once at module load
# Phrases first (longer/more specific), then word-boundary tokens
_ns_pattern = re.compile(
    '|'.join(
        [re.escape(p) for p in sorted(PHRASE_KEYWORDS, key=len, reverse=True)]
        + [r'\b' + re.escape(k) + r'\b' for k in WORD_KEYWORDS]
    ),
    re.IGNORECASE,
)

# ── Random-word detection ─────────────────────────────────────────────────────
FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "i", "my", "you",
    "your", "we", "he", "she", "it", "they", "and", "or", "but",
    "in", "on", "at", "to", "for", "of", "with", "this", "that",
    "have", "has", "not", "be", "do", "did", "will", "would", "can",
    "could", "should",
    # Singapore particles — genuine posts use these
    "lah", "lor", "leh", "sia", "meh", "hor",
}


def _is_suspicious(text: str) -> bool:
    """
    Returns True only if ALL five conditions are met.
    Applied after NS filter so the NS-keyword gate rarely fires.
    """
    words = text.lower().split()
    n = len(words)
    if n <= 10:
        return False
    if len(set(words)) / n <= 0.95:       # TTR gate
        return False
    avg_len = sum(len(w) for w in words) / n
    if 3 <= avg_len <= 9:                  # avg word length gate
        return False
    if any(w in FUNCTION_WORDS for w in words):  # function word gate
        return False
    if _ns_pattern.search(text):           # NS keyword gate
        return False
    return True


# ── Shared helpers ────────────────────────────────────────────────────────────

def _apply_bot_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    bot_mask = (
        df["author"].str.lower().isin(BOT_AUTHORS)
        | df["author"].str.contains(r"[Bb]ot", regex=True, na=False)
    )
    return df[~bot_mask].copy(), before - (~bot_mask).sum()


# ── Submissions cleaner ───────────────────────────────────────────────────────

def clean_submissions(df: pd.DataFrame) -> pd.DataFrame:
    original = len(df)
    log.info(f"Cleaning {original:,} submissions")

    # Rule 1 — Normalise removed / deleted selftext to empty string
    bad_selftext = df["selftext"].isin(["[removed]", "[deleted]", "None", "nan"])
    df.loc[bad_selftext, "selftext"] = ""
    log.info(f"  Rule 1: {bad_selftext.sum():,} selftext fields cleared")

    # Rule 5 — Combine title + selftext (vectorised)
    df["combined_text"] = np.where(
        df["selftext"] != "",
        df["title"] + ". " + df["selftext"],
        df["title"],
    )

    # Rule 6 — Mark text source
    df["text_source"] = np.where(df["selftext"] != "", "full", "title_only")

    # Rule 0 — NS relevance filter
    # r/NationalServiceSG is bypassed — every post there is NS-relevant by context,
    # even without explicit keywords (e.g. venting posts, mood posts, short replies)
    ns_mask = (
        df["subreddit"].str.lower() == "nationalservicesg"
    ) | df["combined_text"].str.contains(_ns_pattern, na=False)
    df = df[ns_mask].copy()
    log.info(f"  Rule 0: {original - len(df):,} non-NS posts removed → {len(df):,} remain")

    # Rule 2 — AutoModerator
    before = len(df)
    df = df[df["author"].str.lower() != "automoderator"].copy()
    log.info(f"  Rule 2: {before - len(df):,} AutoModerator posts removed")

    # Rule 3 — Bot accounts
    df, n_bots = _apply_bot_filter(df)
    log.info(f"  Rule 3: {n_bots:,} bot posts removed")

    # Rule 4 — Random-word detection (flag only, never auto-drop)
    df["is_suspicious"] = df["combined_text"].apply(_is_suspicious)
    log.info(f"  Rule 4: {df['is_suspicious'].sum():,} posts flagged as suspicious")

    # Rule 7 — Log weight (floors negative scores at 0 before log)
    df["log_weight"] = np.log1p(df["score"].clip(lower=0))

    log.info(
        f"Submissions clean: {original:,} → {len(df):,} "
        f"({len(df) / original * 100:.1f}% retained)"
    )
    return df


# ── Comments cleaner ──────────────────────────────────────────────────────────

def clean_comments(df: pd.DataFrame) -> pd.DataFrame:
    original = len(df)
    log.info(f"Cleaning {original:,} comments")

    # Rule 1 — Drop comments with no recoverable body (no title fallback)
    bad_body = df["body"].isin(["[removed]", "[deleted]", "None", "nan", ""])
    df = df[~bad_body].copy()
    log.info(f"  Rule 1: {bad_body.sum():,} removed/deleted comments dropped")

    # Rule 0 — NS relevance filter
    # r/NationalServiceSG bypassed — contextually NS by definition
    before = len(df)
    ns_mask = (
        df["subreddit"].str.lower() == "nationalservicesg"
    ) | df["body"].str.contains(_ns_pattern, na=False)
    df = df[ns_mask].copy()
    log.info(f"  Rule 0: {before - len(df):,} non-NS comments removed → {len(df):,} remain")

    # Rule 2 — AutoModerator
    before = len(df)
    df = df[df["author"].str.lower() != "automoderator"].copy()
    log.info(f"  Rule 2: {before - len(df):,} AutoModerator comments removed")

    # Rule 3 — Bot accounts
    df, n_bots = _apply_bot_filter(df)
    log.info(f"  Rule 3: {n_bots:,} bot comments removed")

    # Rule 4 — Random-word detection (flag only)
    df["is_suspicious"] = df["body"].apply(_is_suspicious)
    log.info(f"  Rule 4: {df['is_suspicious'].sum():,} comments flagged as suspicious")

    # Rule 7 — Log weight
    df["log_weight"] = np.log1p(df["score"].clip(lower=0))

    log.info(
        f"Comments clean: {original:,} → {len(df):,} "
        f"({len(df) / original * 100:.1f}% retained)"
    )
    return df


# ── Thread depth ─────────────────────────────────────────────────────────────

def compute_thread_depth(df: pd.DataFrame) -> pd.Series:
    """
    Compute thread depth for each comment in the filtered dataset.
      1  = direct reply to post (parent_id starts with t3_)
      N  = Nth-level reply (full chain found in dataset)
     -1  = parent comment was filtered out — depth unknown, but > 1
    """
    id_set = set(df["id"])
    id_to_depth: dict[str, int] = {}

    # Pass 1 — direct replies to posts → depth 1
    direct_mask = df["parent_id"].str.startswith("t3_")
    for cid in df.loc[direct_mask, "id"]:
        id_to_depth[cid] = 1

    # Pass 2 — t1_ replies whose parent is not in dataset → depth -1
    t1_mask = df["parent_id"].str.startswith("t1_")
    t1_df = df[t1_mask].copy()
    t1_df["_parent_bare"] = t1_df["parent_id"].str[3:]
    orphan_mask = ~t1_df["_parent_bare"].isin(id_set)
    for cid in t1_df.loc[orphan_mask, "id"]:
        id_to_depth[cid] = -1

    # Pass 3 — iterative resolution for in-dataset chains
    remaining = set(t1_df.loc[~orphan_mask, "id"])
    parent_of  = dict(zip(t1_df["id"], t1_df["_parent_bare"]))

    for _ in range(50):   # guard against pathological nesting
        if not remaining:
            break
        resolved: set[str] = set()
        for cid in remaining:
            parent = parent_of[cid]
            if parent in id_to_depth:
                pd_ = id_to_depth[parent]
                id_to_depth[cid] = pd_ + 1 if pd_ >= 0 else -1
                resolved.add(cid)
        if not resolved:
            for cid in remaining:   # no progress — mark remaining as unknown
                id_to_depth[cid] = -1
            break
        remaining -= resolved

    return df["id"].map(id_to_depth).fillna(-1).astype("int16")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_cleaning():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Submissions — small enough to process in one shot
    log.info("=" * 60)
    log.info("SUBMISSIONS")
    log.info("=" * 60)
    subs = pd.read_parquet(DATA_INTERIM / "submissions_raw.parquet")
    subs_clean = clean_submissions(subs)
    subs_clean.to_parquet(SUBMISSIONS_CLEAN, index=False)
    log.info(f"Saved → {SUBMISSIONS_CLEAN}")
    del subs, subs_clean

    # Comments — process in chunks to avoid memory spike
    log.info("=" * 60)
    log.info("COMMENTS")
    log.info("=" * 60)
    CHUNK_SIZE = 1_000_000
    coms = pd.read_parquet(DATA_INTERIM / "comments_raw.parquet")
    chunks = []
    for start in range(0, len(coms), CHUNK_SIZE):
        chunk = coms.iloc[start: start + CHUNK_SIZE].copy()
        log.info(f"Processing chunk {start // CHUNK_SIZE + 1} "
                 f"(rows {start:,}–{min(start + CHUNK_SIZE, len(coms)):,})")
        chunks.append(clean_comments(chunk))
    del coms

    coms_clean = pd.concat(chunks, ignore_index=True)
    del chunks

    # Thread depth — computed on the full filtered dataset so parent chains resolve correctly
    log.info("Computing thread depth...")
    coms_clean["depth"] = compute_thread_depth(coms_clean)
    log.info(f"  Depth distribution:\n{coms_clean['depth'].value_counts().sort_index()}")

    coms_clean.to_parquet(COMMENTS_CLEAN, index=False)
    log.info(f"Saved → {COMMENTS_CLEAN}")
    del coms_clean

    log.info("Cleaning complete.")


if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    run_cleaning()
