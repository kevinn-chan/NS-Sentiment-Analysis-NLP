from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

# ── Subreddits ────────────────────────────────────────────────────────────────
TARGET_SUBREDDITS = {"singapore", "asksingapore", "nationalservicesg"}

# ── Interim / processed filenames ────────────────────────────────────────────
SUBMISSIONS_CLEAN   = DATA_PROCESSED / "submissions_clean.parquet"
COMMENTS_CLEAN      = DATA_PROCESSED / "comments_clean.parquet"
SUBMISSIONS_CHUNKS  = DATA_PROCESSED / "new" / "submissions_chunks.parquet"
COMMENTS_CHUNKS     = DATA_PROCESSED / "new" / "comments_chunks.parquet"

# ── Stage 4 outputs ───────────────────────────────────────────────────────────
CHUNK_TOPICS        = DATA_PROCESSED / "new" / "chunk_topics.parquet"
HIERARCHICAL_TOPICS = DATA_PROCESSED / "new" / "hierarchical_topics.parquet"
TOPIC_KEYWORDS_FINE   = DATA_PROCESSED / "new" / "topic_keywords_fine.csv"
TOPIC_KEYWORDS_COARSE = DATA_PROCESSED / "new" / "topic_keywords_coarse.csv"
MODELS_DIR          = ROOT / "models"
BERTOPIC_FINE       = MODELS_DIR / "new" / "bertopic_fine"
BERTOPIC_COARSE     = MODELS_DIR / "new" / "bertopic_coarse"

# ── NLP / modelling ───────────────────────────────────────────────────────────
EMBEDDING_MODEL       = "sentence-transformers/all-mpnet-base-v2"
CHUNK_MIN_SENTENCES   = 2
CHUNK_MAX_SENTENCES   = 6
SIMILARITY_THRESHOLD  = 0.5    # cosine similarity drop below this triggers a new chunk
