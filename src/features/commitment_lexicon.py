"""
Lexicon-based NS commitment scorer — Stage 5b parallel track.

Two-bucket seed lexicon: COMMITTED vs UNCOMMITTED terms.
Runs locally in seconds (no GPU) while the Kaggle NLI job runs in parallel.

Sanity check: after both jobs complete, compare lex_net vs commit_net at the
topic_macro level (Spearman rank correlation). Large divergence means BART-
large-mnli is not capturing Singlish commitment language — not random noise,
a systematic gap (bo chup, chao keng, keng, wayang are invisible to the NLI).

Usage:
    python -m src.features.commitment_lexicon

Output:
    data/processed/new/chunk_commitment_lexicon.parquet
    columns: chunk_id, lex_committed, lex_uncommitted, lex_net
"""

import re
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Seed lexicon — two buckets
# ---------------------------------------------------------------------------
# Add / remove terms here; scoring picks them up automatically.
# Phrases match with word boundaries on first and last token.
# Single-word terms get strict \b...\b matching.

COMMITTED = [
    # Affective commitment
    "proud to serve",
    "honour to serve",
    "honor to serve",
    "love serving",
    "love ns",
    "enjoyed ns",
    "enjoy ns",
    # Value / worth framing — NOTE: "worth it" deliberately excluded.
    # It fires inside "not worth it" (uncommitted), causing false committed hits
    # that cancel the uncommitted signal. Use "worth serving" / "worth the sacrifice".
    "worth serving",
    "worth the sacrifice",
    "meaningful experience",
    "gives me purpose",
    "gave me purpose",
    # Growth framing (classic NS narratives)
    "made me a man",
    "built character",
    "builds character",
    "made me stronger",
    "makes me stronger",
    "shaped who i am",
    "grateful for ns",
    "grateful for national service",
    # Duty / defence framing
    "defend singapore",
    "protect singapore",
    "protect our country",
    "national duty",
    "serve the nation",
    "serve our country",
    "duty to serve",
    # Career commitment — first-person sign-on only.
    # Bare "sign on" / "signed on" fires on third-person references and quotes;
    # restricted to first-person collocations to reduce false committed hits.
    "i signed on",
    "i want to sign on",
    "i plan to sign on",
    "planning to sign on",
    "i'm signing on",
    "signed on and",   # "I signed on and ..." construction
    # Unit cohesion
    "brotherhood",
    # General support
    "ns is important",
    "national service is important",
    "ns is necessary",
    "national service is necessary",
    "important for singapore",
    "important for defence",
    "important for defense",
    "necessary sacrifice",
]

UNCOMMITTED = [
    # Singlish uncommitment — exact blind spot of BART-large-mnli
    "bo chup",          # don't care / completely apathetic (only 4 hits; see variants below)
    "bo chap",          # common alternate spelling of bo chup
    "bochup",           # no-space variant
    "bochap",           # no-space variant
    "chao keng",        # malinger to avoid NS duties (very strong signal)
    # NOTE: "keng" fires separately inside "chao keng" — intentional double-count
    # since chao keng is a stronger uncommitment signal than keng alone.
    "keng",             # skive / avoid duties (word-boundary matched)
    "wayang",           # performing without substance; putting on a show
    # Waste framing
    "waste of time",
    "waste 2 years",
    "waste two years",
    "2 years wasted",
    "two years wasted",
    "waste of life",
    "waste of my life",
    "ns is a waste",
    "national service is a waste",
    # Negative valuation
    "pointless",
    "not worth it",
    "not worth serving",
    # Explicit opposition
    "hate ns",
    "hate national service",
    "despise ns",
    "despise national service",
    "abolish ns",
    "abolish national service",
    "get rid of ns",
    # Coercion / exploitation framing
    "slavery",
    "forced labour",
    "forced labor",
    "exploitation",
    # Regret / avoidance intent
    "regret serving",
    "regret enlisting",
    "rather not serve",
    "don't see the point",
    "do not see the point",
    "why bother",
    # Additional Singlish disengagement — common blind spots for non-local models
    "sian",             # bored / fed up (extremely common NS complaint)
    "sian half",        # intensified sian
    "sianz",            # alternate spelling
    "zao",              # run away / leave ASAP
    "ord lo",           # ORD countdown / can't wait to leave
    "ord mood",         # mentally checked out before ORD
    "smoke",            # smoke screen / pretend to work (wayang variant)
    "chao recruit",     # derogatory self-label
    "cannot make it",   # dismissing NS / unit as useless
    "ns sucks",
    "ns is terrible",
    "ns is horrible",
    "ns is trash",
    "ns ruined",
    "wasted my time",
    "counting down",    # counting down to ORD — strong disengagement signal
    "just want to ord", # explicit disengagement
]

# ---------------------------------------------------------------------------
# Compile patterns once at import time
# ---------------------------------------------------------------------------

def _build_patterns(terms):
    """Word-boundary regex for each term. Phrase-internal spaces → \\s+."""
    return [
        re.compile(
            r'\b' + r'\s+'.join(re.escape(w) for w in t.lower().split()) + r'\b',
            re.IGNORECASE,
        )
        for t in terms
    ]


_COMMITTED_RE   = _build_patterns(COMMITTED)
_UNCOMMITTED_RE = _build_patterns(UNCOMMITTED)


def score_chunk(text: str) -> dict:
    """Return committed/uncommitted hit counts and net score for one chunk."""
    t = text.lower()
    committed   = sum(len(p.findall(t)) for p in _COMMITTED_RE)
    uncommitted = sum(len(p.findall(t)) for p in _UNCOMMITTED_RE)
    return {
        'lex_committed':   committed,
        'lex_uncommitted': uncommitted,
        'lex_net':         committed - uncommitted,
    }


# ---------------------------------------------------------------------------
# Per-term hit counts — for diagnostics / spotting misfires
# ---------------------------------------------------------------------------

def term_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of (bucket, term, hit_count) sorted by count desc."""
    rows = []
    texts = df['text'].str.lower()
    for term, pat in zip(COMMITTED, _COMMITTED_RE):
        count = texts.apply(lambda t: len(pat.findall(t))).sum()
        rows.append({'bucket': 'committed', 'term': term, 'hits': int(count)})
    for term, pat in zip(UNCOMMITTED, _UNCOMMITTED_RE):
        count = texts.apply(lambda t: len(pat.findall(t))).sum()
        rows.append({'bucket': 'uncommitted', 'term': term, 'hits': int(count)})
    return pd.DataFrame(rows).sort_values('hits', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

SUBMISSIONS_CHUNKS = ROOT / 'data/processed/new/submissions_chunks.parquet'
COMMENTS_CHUNKS    = ROOT / 'data/processed/new/comments_chunks.parquet'
OUT_PATH           = ROOT / 'data/processed/new/chunk_commitment_lexicon.parquet'


def main():
    print('Loading chunk parquets ...')
    sub = pd.read_parquet(SUBMISSIONS_CHUNKS, columns=['chunk_id', 'text'])
    com = pd.read_parquet(COMMENTS_CHUNKS,    columns=['chunk_id', 'text'])
    df  = pd.concat([sub, com], ignore_index=True)
    del sub, com

    df = df.drop_duplicates(subset='chunk_id')
    df['text'] = df['text'].fillna('').str.strip()
    df = df[df['text'].str.len() > 0].reset_index(drop=True)
    print(f'Chunks to score: {len(df):,}')

    print('Scoring ...')
    scores  = df['text'].apply(score_chunk)
    results = pd.DataFrame(scores.tolist())
    results.insert(0, 'chunk_id', df['chunk_id'].values)

    print(f'Saving → {OUT_PATH}')
    results.to_parquet(OUT_PATH, index=False)

    # Coverage summary
    any_hit = (results['lex_committed'] + results['lex_uncommitted'] > 0).sum()
    print(f'\n── Lexicon coverage ──')
    print(f'  Chunks with any hit:  {any_hit:,} ({any_hit / len(results) * 100:.1f}%)')
    print(f'  Mean lex_committed:   {results["lex_committed"].mean():.4f}')
    print(f'  Mean lex_uncommitted: {results["lex_uncommitted"].mean():.4f}')
    print(f'  Mean lex_net:         {results["lex_net"].mean():.4f}')

    committed_only   = ((results['lex_committed'] > 0) & (results['lex_uncommitted'] == 0)).sum()
    uncommitted_only = ((results['lex_uncommitted'] > 0) & (results['lex_committed'] == 0)).sum()
    mixed            = ((results['lex_committed'] > 0) & (results['lex_uncommitted'] > 0)).sum()
    print(f'  Committed-only hits:  {committed_only:,}')
    print(f'  Uncommitted-only:     {uncommitted_only:,}')
    print(f'  Mixed signals:        {mixed:,}')

    # Per-term breakdown
    print('\n── Term hit counts ──')
    tc = term_counts(df)
    for _, row in tc[tc['hits'] > 0].iterrows():
        print(f'  [{row["bucket"]:>11s}]  {row["hits"]:>6,}  {row["term"]}')

    print('\nDone.')


if __name__ == '__main__':
    main()
