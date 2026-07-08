"""
RAG Phase 2 — Generate rag_temporal_narratives.json using gpt-4.1-mini.

For each month in the corpus (2018-01 to 2025-12):
  1. Pull stats from rag_fact_table.parquet
  2. Pull top-10 chunks by upvotes from chunk_metadata.parquet
  3. Cross-reference ns_events.json for overlapping events
  4. Call gpt-4.1-mini to generate a ~150-word narrative
  5. Store result in rag_temporal_narratives.json

PREREQUISITE: Stage 8 complete + rag_fact_table.parquet built.

Cost estimate: ~$3 total (~96 months × 300 input + 200 output tokens, gpt-4.1-mini pricing)

Run:
    python -m scripts.rag.build_temporal_narratives
    python -m scripts.rag.build_temporal_narratives --year 2019   # single year
    python -m scripts.rag.build_temporal_narratives --resume      # skip done months
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.rag.config import CHUNK_METADATA, FACT_TABLE, NS_EVENTS, TEMPORAL_NARRATIVES

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

NARRATIVE_PROMPT = """You are writing a brief analytical note about NS (National Service) Reddit \
discourse for a specific month in Singapore. This note will be retrieved to answer contextual \
questions like "what was happening in NS discourse in {period}?"

Month: {period}
Chunk volume: {chunk_count:,} chunks ({volume_label})
Sentiment: {neg_pct:.0f}% negative / {neu_pct:.0f}% neutral / {pos_pct:.0f}% positive
Top topics this month: {topic_dist}
Subreddit breakdown: {sr_dist}

Known NS events in or near this period:
{events_text}

Top 10 chunks by upvotes (authentic Reddit text):
{chunks_text}

Write a 120–150 word analytical note covering:
1. Was this month notable? (high volume, sentiment spike, specific controversy)
2. What dominated the discourse?
3. If a specific event drove discourse, explain its impact.
4. What is the overall public mood toward NS this month?

If the month was unremarkable, say so in 2-3 sentences.
Write factually. Do not invent data not present above. Do not use markdown headers."""


def format_events(events: list, year: int, month: int) -> str:
    if not events:
        return "No recorded NS events for this period."
    lines = []
    for e in events:
        lines.append(f"- {e['title']} ({e['date_start']}): {e['description'][:200]}…")
    return "\n".join(lines)


def format_top_chunks(meta: pd.DataFrame, year: int, month: int) -> str:
    subset = meta[(meta["year"] == year) & (meta["month"] == month)]
    if subset.empty:
        return "No chunks available for this month."
    top = subset.nlargest(10, "upvotes")
    lines = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        text = str(row["text_snippet"]).replace("\n", " ").strip()[:200]
        lines.append(f"[{i}] {row['subreddit']} ↑{int(row['upvotes'])} — \"{text}\"")
    return "\n".join(lines)


def generate_narrative(
    client: OpenAI,
    year: int,
    month: int,
    meta: pd.DataFrame,
    fact_table: pd.DataFrame | None,
    all_events: list,
    baseline_count: float,
) -> dict:
    from calendar import month_abbr

    subset = meta[(meta["year"] == year) & (meta["month"] == month)]
    chunk_count = len(subset)
    pct_vs_baseline = ((chunk_count / baseline_count) - 1) * 100 if baseline_count else 0
    volume_label = (
        f"+{pct_vs_baseline:.0f}% vs baseline" if pct_vs_baseline >= 5 else
        f"{pct_vs_baseline:.0f}% vs baseline" if pct_vs_baseline <= -5 else
        "near baseline"
    )

    # Stats from fact table
    neg_pct = neu_pct = pos_pct = 33.0
    topic_dist = sr_dist = "N/A"
    if fact_table is not None:
        ft_sub = fact_table[(fact_table.get("year", pd.Series()) == year) &
                            (fact_table.get("month", pd.Series()) == month)
                            if "month" in fact_table.columns else
                            fact_table["year"] == year]
        if not ft_sub.empty:
            neg_pct = float(ft_sub.get("mean_sent_neg", pd.Series([0.33])).mean()) * 100
            neu_pct = float(ft_sub.get("mean_sent_neu", pd.Series([0.33])).mean()) * 100
            pos_pct = float(ft_sub.get("mean_sent_pos", pd.Series([0.33])).mean()) * 100

    # Top topics
    if not subset.empty and "topic_macro" in subset.columns:
        top_topics = subset["topic_macro"].value_counts().head(3)
        topic_dist = ", ".join(f"{t} ({n})" for t, n in top_topics.items())
        top_sr = subset["subreddit"].value_counts().head(3)
        sr_dist = ", ".join(f"{s} ({n})" for s, n in top_sr.items())

    # Matching events
    period_events = []
    for event in all_events:
        try:
            e_start = pd.Timestamp(event["date_start"])
            e_end   = pd.Timestamp(event["date_end"])
            p_start = pd.Timestamp(year=year, month=month, day=1)
            p_end   = pd.Timestamp(year=year, month=month, day=28)
            if e_start <= p_end and e_end >= p_start:
                period_events.append(event)
        except Exception:
            pass

    period_key = f"{year}-{month:02d}"
    period_label = f"{month_abbr[month]} {year}"

    prompt = NARRATIVE_PROMPT.format(
        period       = period_label,
        chunk_count  = chunk_count,
        volume_label = volume_label,
        neg_pct      = neg_pct,
        neu_pct      = neu_pct,
        pos_pct      = pos_pct,
        topic_dist   = topic_dist,
        sr_dist      = sr_dist,
        events_text  = format_events(period_events, year, month),
        chunks_text  = format_top_chunks(meta, year, month),
    )

    log.info(f"  Calling gpt-4.1-mini for {period_label} ({chunk_count} chunks) …")
    t0 = time.time()
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.3,
    )
    elapsed = time.time() - t0
    narrative_text = response.choices[0].message.content.strip()
    log.info(f"    Done in {elapsed:.1f}s")

    return {
        "period":      period_key,
        "volume":      chunk_count,
        "volume_pct_vs_baseline": round(pct_vs_baseline, 1),
        "sentiment":   {"neg": round(neg_pct/100, 3), "neu": round(neu_pct/100, 3), "pos": round(pos_pct/100, 3)},
        "dominant_topics": topic_dist,
        "events":      [e["id"] for e in period_events],
        "notable":     bool(period_events) or abs(pct_vs_baseline) > 50,
        "narrative":   narrative_text,
        "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",   type=int, help="Generate for single year only")
    parser.add_argument("--resume", action="store_true", help="Skip already-generated months")
    args = parser.parse_args()

    if not CHUNK_METADATA.exists():
        log.error("chunk_metadata.parquet not found. Run build_index.py first.")
        sys.exit(1)
    if not NS_EVENTS.exists():
        log.error("ns_events.json not found.")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    log.info("Loading metadata …")
    meta = pd.read_parquet(CHUNK_METADATA)
    all_events = json.load(open(NS_EVENTS))
    all_events = all_events if isinstance(all_events, list) else all_events.get("events", [])

    fact_table = pd.read_parquet(FACT_TABLE) if FACT_TABLE.exists() else None
    baseline_count = len(meta) / 96  # rough monthly baseline

    # Determine months to process
    years = [args.year] if args.year else list(range(2018, 2026))
    periods = [(y, m) for y in years for m in range(1, 13)]

    # Load existing
    existing = {}
    if args.resume and TEMPORAL_NARRATIVES.exists():
        existing = json.load(open(TEMPORAL_NARRATIVES))
        log.info(f"Resuming: {len(existing)} months already done")

    results = dict(existing)

    for i, (year, month) in enumerate(periods, 1):
        key = f"{year}-{month:02d}"
        if args.resume and key in existing:
            continue

        log.info(f"\n[{i}/{len(periods)}] {key}")
        try:
            result = generate_narrative(client, year, month, meta, fact_table, all_events, baseline_count)
            results[key] = result

            # Incremental save
            with open(TEMPORAL_NARRATIVES, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            log.error(f"  FAILED for {key}: {e}")
            results[key] = {"period": key, "error": str(e)}

        if i < len(periods):
            time.sleep(0.5)

    log.info(f"\nDone: {len(results)} months → {TEMPORAL_NARRATIVES}")


if __name__ == "__main__":
    main()
