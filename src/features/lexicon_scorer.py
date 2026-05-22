"""
Tier 2 sentiment scorer: VADER + custom NS/Singlish lexicon.
Runs locally (CPU only, no GPU), completes in under a minute on 737k chunks.
Output: chunk_sentiment_lexicon.parquet  (chunk_id, sent_lexicon_compound)

Usage:
    python -m src.features.lexicon_scorer
"""

import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------------------
# Custom lexicon — NS jargon + Singlish sentiment terms
# Scores follow VADER's [-4, +4] scale.
# VADER already handles standard English (sucks, terrible, great, love, etc.)
# so we only add terms it doesn't know or systematically mis-scores here.
# ---------------------------------------------------------------------------
NS_SINGLISH_LEXICON: dict[str, float] = {

    # ── Positive ────────────────────────────────────────────────────────────
    "shiok":          3.0,   # great, awesome (universal positive)
    "shiok leh":      3.5,
    "shiok one":      3.0,
    "song":           2.5,   # enjoyable, satisfying
    "song ah":        2.5,
    "swee":           2.0,   # nice, good
    "steady":         2.0,   # reliable, impressive (VADER scores neutral)
    "steady lah":     2.0,
    "tok gong":       3.0,   # excellent (Hokkien)
    "powderful":      2.5,   # powerful — Singlish spelling VADER misses
    "lobang":         2.0,   # good opportunity / deal
    "lobang king":    2.5,   # one who always finds good opportunities
    "good lobang":    2.5,
    "slack":          1.5,   # easy duty — positive in NS context (VADER is neutral)
    "lepak":          1.5,   # chill, relax
    "ho seh":         2.0,   # great, OK (Hokkien)
    "ho seh lah":     2.0,
    "on lah":         1.5,   # enthusiastic / keen
    "ord loh":        3.5,   # celebratory ORD discharge phrase
    "ord lo":         3.5,

    # ── Negative ────────────────────────────────────────────────────────────
    "sian":          -2.5,   # bored, fed up (core Singlish negative)
    "sian ah":       -2.5,
    "sian leh":      -2.5,
    "jialat":        -2.5,   # bad situation (Hokkien)
    "jia lat":       -2.5,
    "wayang":        -2.0,   # fake, performative — strong NS negative
    "chao keng":     -2.0,   # malingering — negative judgement of others
    "chao keng lah": -2.5,
    "saikang":       -1.5,   # menial/dirty work
    "siong":         -1.5,   # tough, gruelling (usually a complaint)
    "tekan":         -2.0,   # harassed / bullied by superior
    "guai lan":      -2.5,   # obstinate, difficult (Hokkien profanity-adjacent)
    "kns":           -3.0,   # Kan Ni Sai — strong Hokkien profanity
    "suay":          -2.0,   # unlucky
    "sway":          -2.0,   # alternate spelling of suay
    "teruk":         -2.5,   # terrible
    "terok":         -2.5,   # alternate spelling
    "bo chap":       -1.5,   # don't care, apathetic
    "bochap":        -1.5,
    "kan cheong":    -1.0,   # anxious, nervous (negative but mild)
    "arrow":         -1.5,   # assigned an unwanted task
    "kena arrow":    -2.0,
    "kena sai":      -3.0,   # (vulgar) got screwed over
    "gg liao":       -2.5,   # game over, done for
    "bo liao":       -1.5,   # meaningless, boring, pointless
    "tok kok":       -1.5,   # talking nonsense
    "cannot make it": -2.0,  # failure / incompetence
    "cmi":           -2.0,   # "cannot make it" abbreviation
    "waste time lah": -2.0,
    "waste of time lah": -2.5,
    "rubbish lah":   -2.0,
    "chao on":       -0.5,   # excessively eager — mild negative social judgement
    "kena burn":     -2.0,   # got burned / failed
    "ownself check ownself": -1.5,  # sarcastic phrase about self-accountability in SAF

    # ── Context-neutral intensifiers (score 0 so VADER rule engine still applies) ──
    "gao gao":        0.0,   # intensifier — extreme; let surrounding words carry valence
    "die die must":   0.0,   # "must do at all costs" — intensity, not valence
    "lah":            0.0,   # particle — no inherent valence
    "leh":            0.0,
    "lor":            0.0,
    "sia":            0.0,
}


def build_analyzer() -> SentimentIntensityAnalyzer:
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(NS_SINGLISH_LEXICON)
    return analyzer


def score_chunks(df: pd.DataFrame, text_col: str = "text") -> pd.Series:
    """Return VADER compound scores ([-1, +1]) with custom lexicon applied."""
    analyzer = build_analyzer()
    return df[text_col].apply(lambda t: analyzer.polarity_scores(str(t))["compound"])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "new"
    OUT_PATH = DATA_DIR / "chunk_sentiment_lexicon.parquet"

    print("Loading chunks …")
    sub = pd.read_parquet(DATA_DIR / "submissions_chunks.parquet", columns=["chunk_id", "text"])
    com = pd.read_parquet(DATA_DIR / "comments_chunks.parquet",    columns=["chunk_id", "text"])
    df  = pd.concat([sub, com], ignore_index=True)
    del sub, com
    print(f"Loaded {len(df):,} chunks")

    print("Scoring …")
    df["sent_lexicon_compound"] = score_chunks(df)

    out = df[["chunk_id", "sent_lexicon_compound"]]
    out.to_parquet(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}  ({len(out):,} rows)")

    print("\n── Score distribution ──")
    print(df["sent_lexicon_compound"].describe().round(4))

    pos = (df["sent_lexicon_compound"] >  0.05).sum()
    neu = (df["sent_lexicon_compound"].between(-0.05, 0.05)).sum()
    neg = (df["sent_lexicon_compound"] < -0.05).sum()
    total = len(df)
    print(f"\nPositive (>0.05):        {pos:>8,}  ({pos/total*100:.1f}%)")
    print(f"Neutral  (-0.05 – 0.05): {neu:>8,}  ({neu/total*100:.1f}%)")
    print(f"Negative (<-0.05):       {neg:>8,}  ({neg/total*100:.1f}%)")
