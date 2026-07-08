"""
RAG Chatbot — configuration constants and paths.
All runtime assets are loaded from DATA_PROCESSED/new/.
"""

import os
from pathlib import Path

# Load .env from project root (if python-dotenv is installed)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
DATA_NEW = ROOT / "data" / "processed" / "new"

# ── Source parquets (read-only) ───────────────────────────────────────────────
SUBMISSIONS_CHUNKS  = DATA_NEW / "submissions_chunks.parquet"
COMMENTS_CHUNKS     = DATA_NEW / "comments_chunks.parquet"
CHUNK_SENTIMENT     = DATA_NEW / "chunk_sentiment.parquet"      # sent_neg/neu/pos
CHUNK_TOPICS        = DATA_NEW / "chunk_topics.parquet"         # topic_id_fine (fallback if not in chunks)

# ── RAG build outputs ─────────────────────────────────────────────────────────
FAISS_INDEX         = DATA_NEW / "chunk_faiss.index"
CHUNK_METADATA      = DATA_NEW / "chunk_metadata.parquet"
NS_EVENTS           = DATA_NEW / "ns_events.json"
TOPIC_DIGESTS       = DATA_NEW / "rag_topic_digests.json"
TEMPORAL_NARRATIVES = DATA_NEW / "rag_temporal_narratives.json"
FACT_TABLE          = DATA_NEW / "rag_fact_table.parquet"

# ── Embedding model (must match chunker.py) ───────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM   = 768

# ── Retrieval parameters ──────────────────────────────────────────────────────
FAISS_CANDIDATE_K  = 50    # retrieve this many from FAISS before reranking
RETURN_K           = 10    # return this many to context assembler
UPVOTE_ALPHA       = 0.3   # weight for log-upvote boost in reranking
MIN_COSINE_SIM     = 0.25  # drop chunks below this similarity threshold

# ── Context window ────────────────────────────────────────────────────────────
# Groq Llama 3.3 has a 128k context; we budget ~3k tokens for rich longitudinal context
MAX_CONTEXT_CHARS  = 10000  # ~2500 tokens; fits stats + timeline + events + chunks
MAX_CHUNK_TEXT_LEN = 300    # truncate each chunk text to this many chars in context

# ── LLM synthesis ─────────────────────────────────────────────────────────────
SYNTHESIS_MODEL    = "claude-haiku-4-5"   # fast + cheap; synthesis-only role
SYNTHESIS_MAX_TOKENS = 600

# ── Quantitative query defaults ───────────────────────────────────────────────
DEFAULT_SUBREDDIT  = "all"
DEFAULT_TOPIC      = "all"
