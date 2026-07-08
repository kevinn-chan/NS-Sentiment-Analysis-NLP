"""
RAG — FAISS Retriever.

Loads the FAISS index and chunk_metadata.parquet at startup.
Retrieves top-k semantically similar chunks with optional metadata filtering.
Reranks by cosine_similarity × (1 + α × log(1 + upvotes)).
"""

import logging
import os
from dataclasses import dataclass

# Must be set before SentenceTransformer import to prevent tokenizer deadlocks on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Limit OpenMP threads: prevents segfault when venv is built on anaconda Python,
# where faiss and torch compete over OpenMP/BLAS thread state.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# CRITICAL: SentenceTransformer (torch) must be imported BEFORE faiss on macOS.
# Importing faiss first modifies OpenMP/BLAS thread state, causing a segfault when
# PyTorch subsequently tries to initialise its own threading context during model load.
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

from src.rag.config import (
    FAISS_INDEX, CHUNK_METADATA, EMBEDDING_MODEL,
    FAISS_CANDIDATE_K, RETURN_K, UPVOTE_ALPHA, MIN_COSINE_SIM,
)
from src.rag.query_router import QueryFilters

log = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    chunk_id:    str
    text:        str
    subreddit:   str
    year:        int
    month:       int
    topic:       str
    upvotes:     int
    sent_neg:    float
    sent_pos:    float
    sent_neu:    float
    faiss_sim:   float     # raw cosine similarity from FAISS
    score:       float     # combined score after upvote reranking


class Retriever:

    def __init__(self):
        # IMPORTANT: Load SentenceTransformer (torch) BEFORE faiss.read_index().
        # faiss.read_index() for a large index modifies OpenMP/BLAS thread state;
        # loading torch after that causes a segfault on macOS during model.encode().
        log.info(f"Loading embedding model: {EMBEDDING_MODEL} …")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        log.info("  Embedder ready")

        log.info("Loading FAISS index …")
        self.index = faiss.read_index(str(FAISS_INDEX))
        log.info(f"  FAISS index loaded: {self.index.ntotal:,} vectors")

        log.info("Loading chunk metadata …")
        self.meta = pd.read_parquet(CHUNK_METADATA)
        log.info(f"  Metadata loaded: {len(self.meta):,} rows")

        # Verify alignment
        assert self.index.ntotal == len(self.meta), (
            f"FAISS ntotal ({self.index.ntotal}) != metadata rows ({len(self.meta)}). "
            "Re-run build_index.py."
        )

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query and L2-normalise to match FAISS inner-product space."""
        emb = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    def _build_mask(self, filters: QueryFilters) -> np.ndarray | None:
        """Return boolean mask over metadata rows, or None (= no filtering)."""
        mask = pd.Series([True] * len(self.meta), index=self.meta.index)

        if filters.years:
            mask &= self.meta["year"].isin(filters.years)
        if filters.months:
            mask &= self.meta["month"].isin(filters.months)
        if filters.subreddits:
            mask &= self.meta["subreddit"].isin(filters.subreddits)
        if filters.topics:
            mask &= self.meta["topic_macro"].isin(filters.topics)

        n_candidates = mask.sum()
        if n_candidates == 0:
            log.warning("Filter produced 0 candidates — falling back to unfiltered search")
            return None

        log.debug(f"Filter mask: {n_candidates:,} candidates")
        return mask

    def retrieve(
        self,
        query: str,
        filters: QueryFilters,
        top_k: int = FAISS_CANDIDATE_K,
        return_k: int = RETURN_K,
    ) -> list[ChunkResult]:
        """
        Embed query, search full FAISS index, post-filter by metadata, rerank by upvotes.
        Returns up to return_k ChunkResults.

        Strategy: search top-(top_k × 5) from full index, then filter to candidates.
        IndexFlatIP is fast enough (~80ms for 737k) that pre-filtering is unnecessary.
        """
        q_emb = self._embed_query(query)
        mask  = self._build_mask(filters)

        # Adaptive n_search: when a filter is sparse (e.g. a single month = 0.4% of corpus)
        # we must search deep enough that filtered candidates are well-represented.
        # Formula: max(top_k × 5, ceil(total / n_candidates) × return_k × 3)
        # Cap at 30,000 to keep FAISS search under ~3s on IndexFlatIP with 737k vectors.
        MAX_N_SEARCH = 30_000
        if mask is not None:
            n_candidates = int(mask.sum())
            if n_candidates > 0:
                sparsity_factor = max(1, self.index.ntotal // n_candidates)
                n_search = max(top_k * 5, sparsity_factor * return_k * 3)
            else:
                n_search = top_k * 5
        else:
            n_search = top_k
        n_search = min(n_search, MAX_N_SEARCH, self.index.ntotal)
        log.debug(f"FAISS n_search={n_search:,}")

        distances, indices = self.index.search(q_emb, n_search)

        # Build valid_set for fast O(1) membership test
        if mask is not None:
            valid_set = set(np.where(mask.values)[0].tolist())
        else:
            valid_set = None

        results: list[ChunkResult] = []
        seen_chunk_ids: set[str] = set()

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.meta):
                continue
            if valid_set is not None and idx not in valid_set:
                continue
            if float(dist) < MIN_COSINE_SIM:
                continue
            if len(results) >= top_k:
                break

            row = self.meta.iloc[int(idx)]
            chunk_id = str(row["chunk_id"])
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            upvotes = max(0, int(row.get("upvotes", 0) or 0))
            upvote_boost = np.log1p(upvotes)
            combined_score = float(dist) * (1.0 + UPVOTE_ALPHA * upvote_boost)

            results.append(ChunkResult(
                chunk_id  = chunk_id,
                text      = str(row.get("text_snippet", ""))[:300],
                subreddit = str(row.get("subreddit", "")),
                year      = int(row.get("year", 0) or 0),
                month     = int(row.get("month", 0) or 0),
                topic     = str(row.get("topic_macro", "Unknown")),
                upvotes   = upvotes,
                sent_neg  = float(row.get("sent_neg", 0) or 0),
                sent_pos  = float(row.get("sent_pos", 0) or 0),
                sent_neu  = float(row.get("sent_neu", 0) or 0),
                faiss_sim = float(dist),
                score     = combined_score,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:return_k]
