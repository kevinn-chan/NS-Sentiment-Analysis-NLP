"""
RAG Phase 1 — Smoke tests and retrieval validation.

Tests the chatbot pipeline end-to-end using the 20-query test suite
from RAG_DESIGN.md Section 9. Run after build_index.py completes.

Usage:
    python -m scripts.rag.test_rag                  # full 20-query suite
    python -m scripts.rag.test_rag --quick           # 5-query quick check
    python -m scripts.rag.test_rag --retrieval-only  # no LLM synthesis
"""

import argparse
import logging
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))

from src.rag.chatbot import NSChatbot
from src.rag.config import FAISS_INDEX, CHUNK_METADATA

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Test queries ──────────────────────────────────────────────────────────────

QUANTITATIVE_QUERIES = [
    "What was the negative sentiment rate for NS Policy & Society in 2024?",
    "Which macro topic had the highest negative sentiment across 2018 to 2025?",
    "How did uncommitted sentiment change year over year from 2022 to 2024?",
    "Compare negative sentiment in r/singapore vs r/askSingapore in 2023.",
    "What was the net disposition score in 2025?",
    "Which year had the highest discourse intensity for Pay & Benefits?",
    "What percentage of NS Policy & Society chunks were negative in 2022 vs 2021?",
    "How much did committed sentiment change between 2018 and 2024?",
    "Which subreddit had the highest upvote-weighted negative sentiment in 2024?",
    "What was the month with the highest negative sentiment in 2019?",
]

QUALITATIVE_QUERIES = [
    "Explain the sharp drop in public sentiment in January 2019.",
    "What do NSmen say about NS pay?",
    "Why did critical sentiment spike in 2022 to 2024?",
    "What does zao liao culture look like in the data?",
    "Explain the 2025 sentiment reversal.",
    "What are the most common concerns about BMT and training?",
    "How did COVID affect NS sentiment in 2020?",
    "What do people in r/NationalServiceSG and r/singapore feel differently about?",
    "Are NSmen more committed or uncommitted in discussions about mental health?",
    "Show me examples of cynicism or sarcasm about NS.",
]

QUICK_QUERIES = [
    "Explain the sharp drop in public sentiment in January 2019.",   # event annotation test
    "What do NSmen say about NS pay?",                               # topic digest test
    "What does zao liao culture look like in the data?",            # lexicon / apathy test
    "How did COVID affect NS sentiment in 2020?",                    # event + temporal test
    "What are the most common concerns about BMT?",                  # topic retrieval test
]


def run_retrieval_test(bot: NSChatbot, query: str) -> dict:
    """Test retrieval quality without calling the LLM."""
    t0 = time.time()
    routed = bot.router.route(query)
    chunks = bot.retriever.retrieve(query, routed.filters)
    elapsed = time.time() - t0

    return {
        "query":    query,
        "intent":   routed.intent,
        "filters":  str(routed.filters),
        "n_chunks": len(chunks),
        "top_sim":  round(chunks[0].faiss_sim, 3) if chunks else None,
        "top_chunk_preview": chunks[0].text[:120] if chunks else None,
        "elapsed_ms": round(elapsed * 1000),
    }


def run_full_test(bot: NSChatbot, query: str) -> dict:
    """Full pipeline test including LLM synthesis."""
    t0 = time.time()
    answer, chunks = bot.answer(query)
    elapsed = time.time() - t0

    return {
        "query":    query,
        "answer":   answer,
        "n_chunks": len(chunks),
        "elapsed_s": round(elapsed, 1),
        "word_count": len(answer.split()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",          action="store_true", help="Run 5-query quick check")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM synthesis")
    parser.add_argument("--query",          type=str, help="Test a single custom query")
    args = parser.parse_args()

    # Prerequisites check
    if not FAISS_INDEX.exists():
        log.error(f"FAISS index not found at {FAISS_INDEX}")
        log.error("Run: python -m scripts.rag.build_index")
        sys.exit(1)
    if not CHUNK_METADATA.exists():
        log.error(f"Chunk metadata not found at {CHUNK_METADATA}")
        sys.exit(1)

    log.info("Initialising chatbot (loading assets) …")
    t0 = time.time()
    bot = NSChatbot()
    log.info(f"Startup in {time.time()-t0:.1f}s")

    # Determine queries
    if args.query:
        queries = [args.query]
        retrieval_only = args.retrieval_only
    elif args.quick:
        queries = QUICK_QUERIES
        retrieval_only = args.retrieval_only
    else:
        queries = QUALITATIVE_QUERIES + QUANTITATIVE_QUERIES
        retrieval_only = args.retrieval_only

    log.info(f"\nRunning {len(queries)} queries (retrieval_only={retrieval_only})\n{'='*60}")

    passed = 0
    for i, query in enumerate(queries, 1):
        log.info(f"\n[{i}/{len(queries)}] {query}")
        try:
            if retrieval_only:
                result = run_retrieval_test(bot, query)
                log.info(f"  intent={result['intent']}  n_chunks={result['n_chunks']}  "
                         f"top_sim={result['top_sim']}  {result['elapsed_ms']}ms")
                if result['top_chunk_preview']:
                    log.info(f"  Top chunk: \"{result['top_chunk_preview']}\"")
                ok = result['n_chunks'] > 0
            else:
                result = run_full_test(bot, query)
                log.info(f"  {result['elapsed_s']}s  {result['word_count']} words  {result['n_chunks']} sources")
                log.info(f"  Answer preview: {result['answer'][:200]} …")
                ok = result['word_count'] > 20 and result['elapsed_s'] < 15

            status = "✓ PASS" if ok else "✗ FAIL"
            log.info(f"  {status}")
            if ok:
                passed += 1

        except Exception as e:
            log.error(f"  ✗ ERROR: {e}")

    log.info(f"\n{'='*60}")
    log.info(f"Results: {passed}/{len(queries)} passed")
    if passed == len(queries):
        log.info("All tests passed ✓")
    else:
        log.warning(f"{len(queries) - passed} test(s) failed")
    log.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
