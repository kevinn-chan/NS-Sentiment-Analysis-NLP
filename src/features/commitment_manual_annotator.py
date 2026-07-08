"""
Manual annotation tool for FAISS-enriched commitment candidates.
Labels BOTH buyin and stance per chunk in a single pass.

Modes:
  --mode both        (default) union of committed+supportive FAISS candidates
  --mode uncommitted buyin-uncommitted FAISS candidates
  --mode critical    stance-critical FAISS candidates
  --mode negative    union of uncommitted+critical FAISS candidates

Usage:
    python -m src.features.commitment_manual_annotator --n 300
    python -m src.features.commitment_manual_annotator --mode uncommitted --n 200
    python -m src.features.commitment_manual_annotator --mode critical --n 150
    python -m src.features.commitment_manual_annotator --mode negative --n 300
    python -m src.features.commitment_manual_annotator --resume
    python -m src.features.commitment_manual_annotator --show-stats

Output:
    data/processed/new/commitment_manual_annotations.csv
    Columns: chunk_id, text, buyin_similarity, stance_similarity, manual_buyin, manual_stance
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).parent.parent.parent
DATA_NEW = ROOT / "data" / "processed" / "new"

BUYIN_LABELS  = {"c": "committed", "u": "uncommitted", "n": "neutral", "s": "skip"}
STANCE_LABELS = {"s": "supportive", "k": "critical",   "n": "neutral", "x": "skip"}

OUTPUT = DATA_NEW / "commitment_manual_annotations.csv"

FAISS_FILES = {
    "committed":   "commitment_faiss_enrich_buyin_committed.csv",
    "uncommitted": "commitment_faiss_enrich_buyin_uncommitted.csv",
    "supportive":  "commitment_faiss_enrich_stance_supportive.csv",
    "critical":    "commitment_faiss_enrich_stance_critical.csv",
}

HEADER_BOTH = """
╔══════════════════════════════════════════════════════════════════╗
║            DUAL-AXIS COMMITMENT ANNOTATION                      ║
╠══════════════════════════════════════════════════════════════════╣
║  BUYIN   — does the author personally endorse their NS service? ║
║    c = committed   (pride, endorses own service, stays by choice)║
║    u = uncommitted (rejects/resents own NS obligation)          ║
║    n = neutral     (narrating, no clear personal stance)        ║
║    s = skip        (can't tell / too ambiguous)                 ║
╠══════════════════════════════════════════════════════════════════╣
║  STANCE  — what is the author's view of NS as an institution?  ║
║    s = supportive  (NS is necessary/good for Singapore)         ║
║    k = critical    (NS policy/system is wrong or unfair)        ║
║    n = neutral     (describing NS, not evaluating it)           ║
║    x = skip        (can't tell / too ambiguous)                 ║
╚══════════════════════════════════════════════════════════════════╝
  Quit anytime: type 'q' — progress saves after every chunk.
"""

HEADER_NEGATIVE = """
╔══════════════════════════════════════════════════════════════════╗
║       ANNOTATION — UNCOMMITTED / CRITICAL FOCUS                 ║
╠══════════════════════════════════════════════════════════════════╣
║  These chunks were retrieved because they resemble UNCOMMITTED  ║
║  or CRITICAL posts. Label both axes honestly — many will be     ║
║  neutral, that's fine.                                          ║
╠══════════════════════════════════════════════════════════════════╣
║  BUYIN   c=committed / u=uncommitted / n=neutral / s=skip       ║
║    uncommitted = rejects, avoids, resents own NS obligation     ║
║    Key signals: chao keng, OOC, MC, waste of time, ORD ASAP    ║
╠══════════════════════════════════════════════════════════════════╣
║  STANCE  s=supportive / k=critical / n=neutral / x=skip         ║
║    critical = thinks NS policy/system is wrong or unfair        ║
║    Key signals: unfair, abolish, slavery, no point, reform      ║
╚══════════════════════════════════════════════════════════════════╝
  Quit anytime: type 'q' — progress saves after every chunk.
"""


def load_faiss(keys: list[str], n: int) -> pd.DataFrame:
    frames = []
    for key in keys:
        path = DATA_NEW / FAISS_FILES[key]
        if not path.exists():
            print(f"WARNING: {path.name} not found — skipping. Run FAISS enricher first.")
            continue
        df = pd.read_csv(path)
        sim_col = f"{key}_similarity"
        df = df.rename(columns={"faiss_similarity": sim_col})
        df["_source_key"] = key
        frames.append(df)

    if not frames:
        print("ERROR: No FAISS candidate files found.")
        sys.exit(1)

    merged = frames[0]
    for df in frames[1:]:
        key_col = [c for c in df.columns if c.endswith("_similarity")][0]
        merged = pd.merge(
            merged, df[["chunk_id", "text", key_col]],
            on="chunk_id", how="outer", suffixes=("", "_r")
        )
        if "text_r" in merged.columns:
            merged["text"] = merged["text"].fillna(merged["text_r"])
            merged = merged.drop(columns=["text_r"])

    # Fill missing similarity scores with 0 and compute combined rank score
    sim_cols = [c for c in merged.columns if c.endswith("_similarity")]
    for col in sim_cols:
        merged[col] = merged[col].fillna(0)
    merged["_score"] = merged[sim_cols].sum(axis=1)
    merged = merged.sort_values("_score", ascending=False).drop(columns="_score")

    return merged.head(n).reset_index(drop=True)


def load_done() -> pd.DataFrame:
    if OUTPUT.exists() and OUTPUT.stat().st_size > 0:
        return pd.read_csv(OUTPUT)
    return pd.DataFrame(columns=["chunk_id", "text", "buyin_similarity",
                                  "stance_similarity", "manual_buyin", "manual_stance"])


def save(rows: list[dict]):
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)


def show_stats():
    if not OUTPUT.exists() or OUTPUT.stat().st_size == 0:
        print("No annotations yet.")
        return
    df = pd.read_csv(OUTPUT)
    labelled = df[df["manual_buyin"].notna() | df["manual_stance"].notna()]
    print(f"\nAnnotated so far: {len(labelled)} / {len(df)}")
    print("\nBuyin:")
    print(df["manual_buyin"].value_counts(dropna=False).to_string())
    print("\nStance:")
    print(df["manual_stance"].value_counts(dropna=False).to_string())
    print()


def prompt_label(question: str, valid: dict) -> str | None:
    options = " / ".join(f"{k}={v}" for k, v in valid.items())
    while True:
        try:
            inp = input(f"  {question} [{options}]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return "__quit__"
        if inp == "q":
            return "__quit__"
        if inp in valid:
            val = valid[inp]
            return None if val == "skip" else val
        print(f"    Invalid. Options: {options}")


def run(mode: str, n: int):
    # Determine which FAISS files to use based on mode
    mode_map = {
        "both":       (["committed", "supportive"], HEADER_BOTH),
        "uncommitted":(["uncommitted"],              HEADER_NEGATIVE),
        "critical":   (["critical"],                 HEADER_NEGATIVE),
        "negative":   (["uncommitted", "critical"],  HEADER_NEGATIVE),
    }
    keys, header = mode_map[mode]

    candidates = load_faiss(keys, n)
    done_df    = load_done()
    done_ids   = set(done_df["chunk_id"].astype(str))

    todo = candidates[~candidates["chunk_id"].astype(str).isin(done_ids)].reset_index(drop=True)

    if todo.empty:
        print("All candidates already annotated.")
        show_stats()
        return

    print(header)
    print(f"  Mode: {mode} | {len(todo)} chunks to annotate (of {n} requested, {len(done_ids)} already done).\n")

    rows = done_df.to_dict("records")
    sim_cols = [c for c in todo.columns if c.endswith("_similarity")]

    for i, row in todo.iterrows():
        sim_tag = "  ".join(f"{c.replace('_similarity','')}={row[c]:.3f}" for c in sim_cols if row[c] > 0)

        print(f"\n── [{i+1}/{len(todo)}] ({sim_tag}) ──")
        print(f"  {row['text']}\n")

        buyin_label = prompt_label("Buyin  (c/u/n/s)", BUYIN_LABELS)
        if buyin_label == "__quit__":
            save(rows)
            print(f"\nSaved {len([r for r in rows if r.get('manual_buyin')])} labels. Exiting.")
            sys.exit(0)

        stance_label = prompt_label("Stance (s/k/n/x)", STANCE_LABELS)
        if stance_label == "__quit__":
            rows.append({
                "chunk_id":          str(row["chunk_id"]),
                "text":              row["text"],
                "buyin_similarity":  row.get("committed_similarity", row.get("uncommitted_similarity", 0)),
                "stance_similarity": row.get("supportive_similarity", row.get("critical_similarity", 0)),
                "manual_buyin":      buyin_label,
                "manual_stance":     None,
            })
            save(rows)
            print(f"\nSaved progress. Exiting.")
            sys.exit(0)

        rows.append({
            "chunk_id":          str(row["chunk_id"]),
            "text":              row["text"],
            "buyin_similarity":  row.get("committed_similarity", row.get("uncommitted_similarity", 0)),
            "stance_similarity": row.get("supportive_similarity", row.get("critical_similarity", 0)),
            "manual_buyin":      buyin_label,
            "manual_stance":     stance_label,
        })
        save(rows)

    print("\nAll candidates annotated!")
    show_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["both", "uncommitted", "critical", "negative"],
                        default="both",
                        help="Which FAISS candidates to present (default: both=committed+supportive)")
    parser.add_argument("--n", type=int, default=300,
                        help="Number of top FAISS candidates to annotate (default: 300)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-labelled chunks and continue")
    parser.add_argument("--show-stats", action="store_true")
    args = parser.parse_args()

    if args.show_stats:
        show_stats()
    else:
        run(args.mode, args.n)
