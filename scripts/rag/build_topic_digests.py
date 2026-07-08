"""
RAG Phase 1 — Generate rag_topic_digests.json using gpt-4.1.

For each of the 17 macro topics:
  1. Pull top-50 chunks by upvotes from chunk_metadata.parquet
  2. Include sentiment labels and chunk text
  3. Call gpt-4.1 to generate a ~300-word analytical narrative
  4. Store result in rag_topic_digests.json

Prerequisites:
  - chunk_metadata.parquet must exist (run build_index.py first)
  - OPENAI_API_KEY must be set in environment

Cost estimate: ~$2 total (17 topics × ~500 input + 400 output tokens, gpt-4.1 pricing)

Run:
    python -m scripts.rag.build_topic_digests
    python -m scripts.rag.build_topic_digests --topic "NS Policy & Society"  # single topic
    python -m scripts.rag.build_topic_digests --resume  # skip already-generated topics
"""

import argparse
import json
import logging
import os
import sys
import time

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.rag.config import CHUNK_METADATA, TOPIC_DIGESTS
from src.models.topic_labels import MACRO_CATEGORIES

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

DIGEST_PROMPT = """You are summarizing Reddit discourse about National Service (NS) in Singapore for an \
analytical knowledge base. This summary will be retrieved at query time to help answer questions about this topic area.

Topic area: {topic}
Total chunks in corpus for this topic: {n_chunks:,}
Sentiment profile (SingBERT v7, κ=0.602): {neg_pct:.0f}% negative / {neu_pct:.0f}% neutral / {pos_pct:.0f}% positive
Date range: {year_min} – {year_max}
Subreddits: {subreddits}

Top {n_shown} chunks by upvotes (authentic Reddit quotes — do not edit or paraphrase):
{chunks_text}

Write a structured analytical summary using EXACTLY these section headers:

DOMINANT_THEMES:
[3-5 bullet points of what people most discuss in this topic area]

COMMON_CONCERNS:
[The frustrations, complaints, or anxieties most commonly expressed — be specific]

COMMUNITY_TONE:
[Describe the emotional register. Go beyond the statistics — what does it feel like to read this topic's posts?]

UNCOMMITTED_EXAMPLES:
[What does "uncommitted" or disengaged look like in this topic? Give concrete language examples from the quotes]

NOTABLE_PATTERNS:
[Any shifts over time, subreddit differences, recurring debates, or standout controversies]

REPRESENTATIVE_QUOTES:
[Pick 3 quotes from the provided chunks that best capture this topic's discourse. Format each as:
 "quote text" — {{subreddit}}, {{year}}, {{upvotes}} upvotes]

Be specific and analytical. Write for a government department analyst who needs to understand \
what Singaporean men actually say about NS online. Do not invent data. Do not editorialize beyond what the quotes show."""


def format_chunks_for_prompt(chunks: pd.DataFrame, max_chunks: int = 50) -> str:
    lines = []
    for i, (_, row) in enumerate(chunks.head(max_chunks).iterrows(), 1):
        sentiment = (
            "negative" if (row.get("sent_neg", 0) or 0) > 0.5 else
            "positive" if (row.get("sent_pos", 0) or 0) > 0.5 else
            "neutral"
        )
        text = str(row["text_snippet"]).replace("\n", " ").strip()
        lines.append(
            f"[{i}] {row['subreddit']} · {int(row['year'])} · ↑{int(row['upvotes'])} · {sentiment}\n"
            f"    \"{text}\""
        )
    return "\n\n".join(lines)


def generate_digest(client: OpenAI, topic: str, meta: pd.DataFrame) -> dict:
    topic_df = meta[meta["topic_macro"] == topic].copy()
    n_chunks = len(topic_df)

    if n_chunks == 0:
        log.warning(f"  No chunks found for topic: {topic}")
        return {"topic": topic, "error": "no_chunks", "chunk_count": 0}

    # Sentiment averages (fill NA with 0)
    neg = topic_df["sent_neg"].fillna(0).mean()
    neu = topic_df["sent_neu"].fillna(0).mean()
    pos = topic_df["sent_pos"].fillna(0).mean()

    # Top 50 by upvotes
    top_chunks = topic_df.nlargest(50, "upvotes")
    year_min = int(topic_df["year"].dropna().min()) if topic_df["year"].notna().any() else "?"
    year_max = int(topic_df["year"].dropna().max()) if topic_df["year"].notna().any() else "?"
    subreddits = ", ".join(sorted(topic_df["subreddit"].unique().tolist()))

    prompt = DIGEST_PROMPT.format(
        topic=topic,
        n_chunks=n_chunks,
        neg_pct=neg * 100,
        neu_pct=neu * 100,
        pos_pct=pos * 100,
        year_min=year_min,
        year_max=year_max,
        subreddits=subreddits,
        n_shown=min(50, len(top_chunks)),
        chunks_text=format_chunks_for_prompt(top_chunks),
    )

    log.info(f"  Calling gpt-4.1 for '{topic}' ({n_chunks:,} chunks) …")
    t0 = time.time()
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=0.3,
    )
    elapsed = time.time() - t0
    raw_text = response.choices[0].message.content.strip()
    log.info(f"    Done in {elapsed:.1f}s  ({response.usage.total_tokens} tokens)")

    # Parse sections
    sections = {}
    current_section = None
    current_lines = []

    for line in raw_text.split("\n"):
        stripped = line.strip()
        if stripped.endswith(":") and stripped.rstrip(":").upper() == stripped.rstrip(":"):
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = stripped.rstrip(":").lower().replace(" ", "_")
            current_lines = []
        else:
            current_lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()

    return {
        "topic": topic,
        "chunk_count": n_chunks,
        "sentiment_profile": {"neg": round(neg, 3), "neu": round(neu, 3), "pos": round(pos, 3)},
        "year_range": [year_min, year_max],
        "subreddits": subreddits,
        "dominant_themes": sections.get("dominant_themes", ""),
        "common_concerns": sections.get("common_concerns", ""),
        "community_tone": sections.get("community_tone", ""),
        "uncommitted_examples": sections.get("uncommitted_examples", ""),
        "notable_patterns": sections.get("notable_patterns", ""),
        "representative_quotes": sections.get("representative_quotes", ""),
        "raw_text": raw_text,
        "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", help="Generate digest for a single topic only")
    parser.add_argument("--resume", action="store_true", help="Skip topics already in output file")
    args = parser.parse_args()

    # Check prerequisites
    if not CHUNK_METADATA.exists():
        log.error(f"chunk_metadata.parquet not found at {CHUNK_METADATA}")
        log.error("Run build_index.py first.")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load metadata
    log.info(f"Loading chunk metadata from {CHUNK_METADATA} …")
    meta = pd.read_parquet(CHUNK_METADATA)
    log.info(f"  {len(meta):,} chunks, {meta['topic_macro'].nunique()} macros")

    # Load existing digests if resuming
    existing = {}
    if args.resume and TOPIC_DIGESTS.exists():
        existing = json.load(open(TOPIC_DIGESTS))
        log.info(f"  Resuming: {len(existing)} digests already exist")

    # Determine topics to process
    if args.topic:
        topics = [args.topic]
    else:
        topics = MACRO_CATEGORIES

    results = dict(existing)

    for i, topic in enumerate(topics, 1):
        if args.resume and topic in existing:
            log.info(f"[{i}/{len(topics)}] Skipping '{topic}' (already done)")
            continue

        log.info(f"\n[{i}/{len(topics)}] Generating digest: '{topic}'")
        try:
            digest = generate_digest(client, topic, meta)
            results[topic] = digest

            # Save incrementally after each topic (safe to interrupt)
            TOPIC_DIGESTS.parent.mkdir(parents=True, exist_ok=True)
            with open(TOPIC_DIGESTS, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            log.info(f"  Saved incrementally → {TOPIC_DIGESTS}")

        except Exception as e:
            log.error(f"  FAILED for '{topic}': {e}")
            results[topic] = {"topic": topic, "error": str(e)}

        # Rate limit pause between calls
        if i < len(topics):
            time.sleep(1)

    log.info(f"\n{'='*60}")
    log.info(f"Topic digests complete: {len(results)}/{len(topics)} topics")
    log.info(f"Output: {TOPIC_DIGESTS}")
    log.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
