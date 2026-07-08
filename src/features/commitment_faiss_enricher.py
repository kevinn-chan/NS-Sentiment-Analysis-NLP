"""
Commitment FAISS Enricher
=========================
Finds semantically similar chunks to confirmed committed/supportive minority
examples using the existing FAISS index.

Purpose: when the LLM-labelled enrich queue has < 5% committed or < 5%
supportive, use this to surface more candidates without random sampling.

Usage:
    python -m src.features.commitment_faiss_enricher --axis buyin --label committed --k 3000
    python -m src.features.commitment_faiss_enricher --axis stance --label supportive --k 3000

Output:
    data/processed/new/commitment_faiss_enrich_{axis}_{label}.csv
    Columns: chunk_id, text, faiss_similarity, source
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Prevent tokenizer / OpenMP conflicts before any torch/faiss import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# SentenceTransformer (torch) MUST be imported before faiss on macOS
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent.parent
DATA_NEW    = ROOT / "data" / "processed" / "new"
FAISS_INDEX = DATA_NEW / "chunk_faiss.index"
CHUNK_META  = DATA_NEW / "chunk_metadata.parquet"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Enrich queue / label sources
ENRICH_QUEUE    = DATA_NEW / "commitment_enrich_queue.parquet"
TEST_SET        = DATA_NEW / "commitment_testset.parquet"

# Per-axis label column names (LLM / human labels)
AXIS_LABEL_COLS = {
    "buyin":  ["human_label", "llm_buyin"],
    "stance": ["human_stance", "llm_stance"],
}

# Valid target labels per axis
AXIS_VALID_LABELS = {
    "buyin":  {"committed", "uncommitted", "neutral"},
    "stance": {"supportive", "critical", "neutral"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find FAISS neighbours of confirmed minority-class chunks."
    )
    parser.add_argument(
        "--axis",
        required=True,
        choices=["buyin", "stance"],
        help="Which labelling axis to enrich (buyin or stance).",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Target label to find more of (e.g. committed, supportive).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3000,
        help="Number of nearest neighbours to return per seed (before dedup). Default: 3000.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=500,
        help="Maximum number of seed chunks to query (most central seeds first). Default: 500.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="FAISS search batch size (number of seed vectors per call). Default: 50.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_NEW,
        help=f"Output directory. Default: {DATA_NEW}",
    )
    return parser.parse_args()


def load_labelled_chunk_ids(axis: str, target_label: str) -> tuple[set[str], set[str]]:
    """
    Load all chunk_ids that already have ANY label for this axis (to exclude from results),
    and the subset confirmed as target_label (seed set).

    Returns:
        labelled_ids  — all chunk_ids with a known label (exclusion set)
        seed_ids      — chunk_ids confirmed as target_label (seed set)
    """
    label_cols = AXIS_LABEL_COLS[axis]
    frames = []

    for path in [ENRICH_QUEUE, TEST_SET]:
        if path.exists():
            df = pd.read_parquet(path)
            frames.append(df)
            log.info(f"  Loaded {path.name}: {len(df):,} rows")
        else:
            log.warning(f"  {path.name} not found — skipping")

    if not frames:
        log.warning("No labelled data found. Returning empty sets.")
        return set(), set()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["chunk_id"])
    log.info(f"  Combined labelled pool: {len(combined):,} unique chunks")

    labelled_ids: set[str] = set()
    seed_ids: set[str] = set()

    for col in label_cols:
        if col not in combined.columns:
            continue
        has_label = combined[col].notna() & (combined[col] != "")
        labelled_ids.update(combined.loc[has_label, "chunk_id"].astype(str).tolist())

        is_target = has_label & (combined[col] == target_label)
        seed_ids.update(combined.loc[is_target, "chunk_id"].astype(str).tolist())

    log.info(f"  Total labelled (exclusion set): {len(labelled_ids):,}")
    log.info(f"  Seeds confirmed as '{target_label}': {len(seed_ids):,}")
    return labelled_ids, seed_ids


def load_index_and_meta() -> tuple[faiss.Index, pd.DataFrame]:
    """Load FAISS index and chunk metadata. Validates alignment."""
    log.info(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    # Load model early so torch initialises before faiss modifies BLAS state
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    log.info("  Embedder ready (warming BLAS before faiss load)")
    del embedder  # free GPU memory; we only needed the BLAS init side-effect

    log.info(f"Loading FAISS index: {FAISS_INDEX} ...")
    if not FAISS_INDEX.exists():
        log.error(f"FAISS index not found: {FAISS_INDEX}")
        sys.exit(1)
    index = faiss.read_index(str(FAISS_INDEX))
    log.info(f"  FAISS index loaded: {index.ntotal:,} vectors")

    log.info(f"Loading chunk metadata: {CHUNK_META} ...")
    if not CHUNK_META.exists():
        log.error(f"chunk_metadata.parquet not found: {CHUNK_META}")
        sys.exit(1)
    meta = pd.read_parquet(CHUNK_META)
    log.info(f"  Metadata loaded: {len(meta):,} rows")

    if index.ntotal != len(meta):
        log.error(
            f"FAISS ntotal ({index.ntotal}) != metadata rows ({len(meta)}). "
            "Re-run build_index.py."
        )
        sys.exit(1)

    return index, meta


def get_seed_vectors(
    meta: pd.DataFrame,
    index: faiss.Index,
    seed_ids: set[str],
    max_seeds: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Retrieve embedding vectors for seed chunk_ids directly from the FAISS index
    using their faiss_idx positions.

    Returns (vectors array [n_seeds × dim], list of resolved chunk_ids).
    """
    # Build lookup: chunk_id → faiss_idx
    meta_str = meta.copy()
    meta_str["chunk_id"] = meta_str["chunk_id"].astype(str)

    seed_meta = meta_str[meta_str["chunk_id"].isin(seed_ids)].copy()
    if seed_meta.empty:
        log.warning("No seed chunks found in metadata. Cannot proceed.")
        return np.empty((0, index.d), dtype=np.float32), []

    # Use faiss_idx if available, else fall back to positional iloc index
    if "faiss_idx" in seed_meta.columns:
        seed_meta = seed_meta.dropna(subset=["faiss_idx"])
        seed_meta["_fidx"] = seed_meta["faiss_idx"].astype(int)
    else:
        log.info("  faiss_idx column not found — using positional index as faiss_idx")
        seed_meta["_fidx"] = seed_meta.index.astype(int)

    # Clamp to valid range
    seed_meta = seed_meta[seed_meta["_fidx"] < index.ntotal].copy()
    log.info(f"  Seeds with valid faiss_idx: {len(seed_meta):,}")

    # Limit to max_seeds (FAISS reconstruct is O(n) per call on IndexFlatIP)
    if len(seed_meta) > max_seeds:
        log.info(f"  Capping seeds at {max_seeds:,} (of {len(seed_meta):,} total)")
        seed_meta = seed_meta.sample(n=max_seeds, random_state=42)

    fidx_list = seed_meta["_fidx"].tolist()
    chunk_id_list = seed_meta["chunk_id"].tolist()

    log.info(f"  Reconstructing {len(fidx_list):,} seed vectors from FAISS index ...")
    vectors = np.zeros((len(fidx_list), index.d), dtype=np.float32)
    for i, fidx in enumerate(fidx_list):
        index.reconstruct(int(fidx), vectors[i])

    log.info(f"  Reconstructed {len(vectors):,} vectors (dim={index.d})")
    return vectors, chunk_id_list


def batch_search(
    index: faiss.Index,
    seed_vectors: np.ndarray,
    k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run batched FAISS search for all seed vectors.
    Returns (all_distances, all_indices) each shape [n_seeds × k].
    """
    n = len(seed_vectors)
    all_distances = np.empty((n, k), dtype=np.float32)
    all_indices   = np.empty((n, k), dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = seed_vectors[start:end]
        D, I = index.search(batch, k)
        all_distances[start:end] = D
        all_indices[start:end]   = I
        if (start // batch_size) % 10 == 0:
            log.info(f"  Searched {end:,}/{n:,} seeds ...")

    return all_distances, all_indices


def build_candidate_table(
    meta: pd.DataFrame,
    distances: np.ndarray,
    indices: np.ndarray,
    labelled_ids: set[str],
) -> pd.DataFrame:
    """
    Flatten FAISS results into a deduplicated candidate DataFrame,
    excluding already-labelled chunk_ids.

    For each candidate chunk_id, keeps the highest similarity score seen.
    """
    meta_str = meta.copy()
    meta_str["chunk_id"] = meta_str["chunk_id"].astype(str)
    meta_str = meta_str.reset_index(drop=True)

    # text column: prefer 'text_snippet', fall back to 'text'
    text_col = "text_snippet" if "text_snippet" in meta_str.columns else "text"

    best: dict[str, float] = {}   # chunk_id → best similarity

    n_seeds, k = indices.shape
    for i in range(n_seeds):
        for j in range(k):
            idx = int(indices[i, j])
            if idx == -1 or idx >= len(meta_str):
                continue
            sim = float(distances[i, j])
            cid = str(meta_str.at[idx, "chunk_id"])
            if cid in labelled_ids:
                continue
            if cid not in best or sim > best[cid]:
                best[cid] = sim

    log.info(f"  Unique unlabelled candidates: {len(best):,}")

    if not best:
        return pd.DataFrame(columns=["chunk_id", "text", "faiss_similarity", "source"])

    cids = list(best.keys())
    sims = [best[c] for c in cids]

    # Build chunk_id → row lookup
    id_to_idx = {str(row["chunk_id"]): idx for idx, row in meta_str.iterrows()}

    rows = []
    for cid, sim in zip(cids, sims):
        if cid in id_to_idx:
            row = meta_str.iloc[id_to_idx[cid]]
            text = str(row.get(text_col, ""))
        else:
            text = ""
        rows.append({"chunk_id": cid, "text": text, "faiss_similarity": sim, "source": "faiss"})

    result = pd.DataFrame(rows)
    result = result.sort_values("faiss_similarity", ascending=False).reset_index(drop=True)
    return result


def main() -> None:
    args = parse_args()

    axis  = args.axis
    label = args.label.lower()
    k     = args.k

    valid = AXIS_VALID_LABELS[axis]
    if label not in valid:
        log.error(f"Invalid label '{label}' for axis '{axis}'. Valid: {valid}")
        sys.exit(1)

    log.info(f"=== Commitment FAISS Enricher: axis={axis}, label={label}, k={k} ===")

    # ── 1. Load labelled pools ────────────────────────────────────────────────
    log.info("Loading labelled chunk pools ...")
    labelled_ids, seed_ids = load_labelled_chunk_ids(axis, label)

    if len(seed_ids) == 0:
        log.error(
            f"No seed chunks found for axis='{axis}', label='{label}'. "
            "Run Stage 5b LLM annotation first."
        )
        sys.exit(1)

    log.info(f"Seeds: {len(seed_ids):,}  |  Exclusion set: {len(labelled_ids):,}")

    # ── 2. Load FAISS index + metadata ───────────────────────────────────────
    index, meta = load_index_and_meta()

    # ── 3. Retrieve seed vectors ──────────────────────────────────────────────
    log.info("Retrieving seed vectors ...")
    seed_vectors, used_chunk_ids = get_seed_vectors(meta, index, seed_ids, args.max_seeds)

    if len(seed_vectors) == 0:
        log.error("Could not reconstruct any seed vectors. Aborting.")
        sys.exit(1)

    log.info(f"Using {len(seed_vectors):,} seeds for FAISS search (k={k} per seed)")

    # ── 4. Batch FAISS search ─────────────────────────────────────────────────
    log.info("Running FAISS nearest-neighbour search ...")
    distances, indices = batch_search(index, seed_vectors, k=k, batch_size=args.batch_size)

    # ── 5. Build candidate table ──────────────────────────────────────────────
    log.info("Building candidate table (deduping, excluding labelled) ...")
    candidates = build_candidate_table(meta, distances, indices, labelled_ids)

    log.info(f"\nCandidate summary:")
    log.info(f"  Total unique candidates: {len(candidates):,}")
    if len(candidates) > 0:
        log.info(f"  Similarity range: {candidates['faiss_similarity'].min():.4f} – {candidates['faiss_similarity'].max():.4f}")
        log.info(f"  Median similarity: {candidates['faiss_similarity'].median():.4f}")

    # ── 6. Save output ────────────────────────────────────────────────────────
    out_path = args.output_dir / f"commitment_faiss_enrich_{axis}_{label}.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_path, index=False)
    log.info(f"\nSaved → {out_path}")
    log.info(
        f"\nNext step: review or LLM-annotate {out_path.name}, "
        "then merge into the enrich queue."
    )


if __name__ == "__main__":
    main()
