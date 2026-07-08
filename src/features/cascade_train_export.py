"""
Export 4 cascade-ready training CSVs for Kaggle.

Stage 1a: buyin_relevant (committed|uncommitted → 1, neutral → 0)
Stage 2a: committed vs uncommitted (binary, drop neutrals)
Stage 1b: has_stance (supportive|critical → 1, neutral → 0)
Stage 2b: supportive vs critical (binary, drop neutrals)

Sources (in priority order):
  1. commitment_testset.parquet          — gold, weight 1.0
  2. commitment_manual_annotations.csv   — gold, weight 1.0
  3. neutral_collapse_queue.csv          — gold (if annotated), weight 1.0
  4. commitment_llm_enrich_queue.csv     — silver (llm_buyin/llm_stance), weight varies by stage
  5. committed_supplement_queue.csv      — gold (if annotated), weight 1.0

Outputs to data/processed/new/cascade/:
  stage1a_train.csv  — columns: chunk_id, text, label (0/1), weight, source
  stage2a_train.csv  — columns: chunk_id, text, label (committed/uncommitted), weight, source
  stage1b_train.csv  — columns: chunk_id, text, label (0/1), weight, source
  stage2b_train.csv  — columns: chunk_id, text, label (supportive/critical), weight, source

Usage:
    python -m src.features.cascade_train_export
    python -m src.features.cascade_train_export --silver-weight-1a 0.3 --silver-weight-2a 0.4
"""
import argparse
from pathlib import Path
import pandas as pd

DATA    = Path(__file__).resolve().parents[2] / "data" / "processed" / "new"
OUT_DIR = DATA / "cascade"


def load_gold() -> pd.DataFrame:
    """Merge all human-annotated sources into one frame with buyin + stance columns."""
    frames = []

    # commitment_testset.parquet
    ts = pd.read_parquet(DATA / "commitment_testset.parquet")
    ts = ts.rename(columns={"human_label": "buyin", "human_stance": "stance"})
    ts["source"] = "testset"
    frames.append(ts[["chunk_id", "text", "buyin", "stance", "source"]])

    # commitment_manual_annotations.csv
    ma = pd.read_csv(DATA / "commitment_manual_annotations.csv")
    ma = ma.rename(columns={"manual_buyin": "buyin", "manual_stance": "stance"})
    ma["source"] = "manual"
    frames.append(ma[["chunk_id", "text", "buyin", "stance", "source"]])

    # neutral_collapse_queue.csv (if annotated)
    ncq = DATA / "neutral_collapse_queue.csv"
    if ncq.exists():
        nc = pd.read_csv(ncq)
        nc = nc[nc["human_label"].str.len() > 0]
        nc = nc[nc["human_label"] != "skip"]
        if len(nc):
            nc = nc.rename(columns={"human_label": "buyin", "human_stance": "stance"})
            nc["source"] = "neutral_collapse"
            frames.append(nc[["chunk_id", "text", "buyin", "stance", "source"]])
            print(f"  neutral_collapse queue: {len(nc)} labelled rows")

    # committed_supplement_queue.csv (if annotated)
    csq = DATA / "committed_supplement_queue.csv"
    if csq.exists():
        cs = pd.read_csv(csq)
        cs = cs[cs["human_label"].str.len() > 0]
        cs = cs[cs["human_label"] != "skip"]
        if len(cs):
            cs = cs.rename(columns={"human_label": "buyin"})
            cs["stance"] = None
            cs["source"] = "committed_supplement"
            frames.append(cs[["chunk_id", "text", "buyin", "stance", "source"]])
            print(f"  committed_supplement queue: {len(cs)} labelled rows")

    gold = pd.concat(frames, ignore_index=True)
    gold = gold.drop_duplicates(subset="chunk_id", keep="first")
    gold["weight"] = 1.0
    return gold


def load_silver() -> pd.DataFrame:
    """Load LLM-labelled enrich queue as silver labels."""
    eq = pd.read_csv(DATA / "commitment_llm_enrich_queue.csv")
    eq = eq[eq["llm_buyin"].notna() & (eq["llm_buyin"].str.len() > 0)].copy()
    eq = eq.rename(columns={"llm_buyin": "buyin", "llm_stance": "stance"})
    eq["source"] = "llm_silver"
    return eq[["chunk_id", "text", "buyin", "stance", "source"]]


def build_stage1a(gold: pd.DataFrame, silver: pd.DataFrame, silver_weight: float) -> pd.DataFrame:
    """Binary: commitment-relevant (1) vs not (0)."""
    RELEVANT = {"committed", "uncommitted"}

    g = gold[gold["buyin"].isin(RELEVANT | {"neutral"})].copy()
    g["label"]  = g["buyin"].apply(lambda x: 1 if x in RELEVANT else 0)
    g["weight"] = 1.0

    # Silver: use LLM "relevant" predictions as positive (78% precise — trustworthy)
    # For negatives: only use human-labelled neutrals, NOT LLM neutrals
    # (LLM neutral recall is only 47% — half of true relevant get marked neutral)
    s_pos = silver[silver["buyin"].isin(RELEVANT)].copy()
    s_pos["label"]  = 1
    s_pos["weight"] = silver_weight

    df = pd.concat([g[["chunk_id","text","label","weight","source"]],
                    s_pos[["chunk_id","text","label","weight","source"]]], ignore_index=True)
    df = df.drop_duplicates(subset="chunk_id", keep="first")
    return df


def build_stage2a(gold: pd.DataFrame, silver: pd.DataFrame, silver_weight: float) -> pd.DataFrame:
    """Binary: committed vs uncommitted (neutrals dropped)."""
    CLASSES = {"committed", "uncommitted"}

    g = gold[gold["buyin"].isin(CLASSES)].copy()
    g["label"]  = g["buyin"]
    g["weight"] = 1.0

    s = silver[silver["buyin"].isin(CLASSES)].copy()
    s["label"]  = s["buyin"]
    s["weight"] = silver_weight

    df = pd.concat([g[["chunk_id","text","label","weight","source"]],
                    s[["chunk_id","text","label","weight","source"]]], ignore_index=True)
    df = df.drop_duplicates(subset="chunk_id", keep="first")
    return df


def build_stage1b(gold: pd.DataFrame, silver: pd.DataFrame, silver_weight: float) -> pd.DataFrame:
    """Binary: has-stance (1) vs neutral (0)."""
    HAS_STANCE = {"supportive", "critical"}

    g = gold[gold["stance"].notna() & gold["stance"].isin(HAS_STANCE | {"neutral"})].copy()
    g["label"]  = g["stance"].apply(lambda x: 1 if x in HAS_STANCE else 0)
    g["weight"] = 1.0

    s_pos = silver[silver["stance"].isin(HAS_STANCE)].copy()
    s_pos["label"]  = 1
    s_pos["weight"] = silver_weight

    s_neg = silver[silver["stance"] == "neutral"].copy()
    s_neg["label"]  = 0
    s_neg["weight"] = silver_weight * 0.5  # ponytail: down-weight silver negatives
    s_neg = s_neg.sample(min(len(s_neg), len(s_pos) * 3), random_state=42)  # cap neutral oversampling

    df = pd.concat([g[["chunk_id","text","label","weight","source"]],
                    s_pos[["chunk_id","text","label","weight","source"]],
                    s_neg[["chunk_id","text","label","weight","source"]]], ignore_index=True)
    df = df.drop_duplicates(subset="chunk_id", keep="first")
    return df


def build_stage2b(gold: pd.DataFrame, silver: pd.DataFrame, silver_weight: float) -> pd.DataFrame:
    """Binary: supportive vs critical (neutrals dropped)."""
    CLASSES = {"supportive", "critical"}

    g = gold[gold["stance"].notna() & gold["stance"].isin(CLASSES)].copy()
    g["label"]  = g["stance"]
    g["weight"] = 1.0

    s = silver[silver["stance"].isin(CLASSES)].copy()
    s["label"]  = s["stance"]
    s["weight"] = silver_weight

    df = pd.concat([g[["chunk_id","text","label","weight","source"]],
                    s[["chunk_id","text","label","weight","source"]]], ignore_index=True)
    df = df.drop_duplicates(subset="chunk_id", keep="first")
    return df


def report(name: str, df: pd.DataFrame):
    print(f"\n  {name}: {len(df):,} rows")
    print(f"    label dist: {df['label'].value_counts().to_dict()}")
    print(f"    source dist: {df['source'].value_counts().to_dict()}")
    gold_n  = (df['weight'] == 1.0).sum()
    silver_n = (df['weight'] < 1.0).sum()
    print(f"    gold: {gold_n:,}  silver: {silver_n:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-weight-1a", type=float, default=0.5,
                    help="Sample weight for LLM silver positives in Stage 1a (default 0.5)")
    ap.add_argument("--silver-weight-2a", type=float, default=0.4,
                    help="Sample weight for LLM silver in Stage 2a (default 0.4)")
    ap.add_argument("--silver-weight-1b", type=float, default=0.7,
                    help="Sample weight for LLM silver in Stage 1b (default 0.7 — higher quality)")
    ap.add_argument("--silver-weight-2b", type=float, default=0.5,
                    help="Sample weight for LLM silver in Stage 2b (default 0.5)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print("Loading gold labels …")
    gold = load_gold()
    print(f"  Gold rows: {len(gold):,}")
    print(f"  buyin:  {gold['buyin'].value_counts().to_dict()}")
    print(f"  stance: {gold['stance'].value_counts().to_dict()}")

    print("\nLoading silver labels …")
    silver = load_silver()
    print(f"  Silver rows: {len(silver):,}")

    print(f"\n{'='*60}")
    print(f"  Building cascade training sets")
    print(f"{'='*60}")

    stages = {
        "stage1a_train.csv": build_stage1a(gold, silver, args.silver_weight_1a),
        "stage2a_train.csv": build_stage2a(gold, silver, args.silver_weight_2a),
        "stage1b_train.csv": build_stage1b(gold, silver, args.silver_weight_1b),
        "stage2b_train.csv": build_stage2b(gold, silver, args.silver_weight_2b),
    }

    for fname, df in stages.items():
        df.to_csv(OUT_DIR / fname, index=False)
        report(fname, df)

    print(f"\n{'='*60}")
    print(f"  Saved to: {OUT_DIR}/")
    print(f"  Upload all 4 CSVs to Kaggle as a dataset.")
    print(f"\n  Kaggle training notes:")
    print(f"    - Use 'weight' column in loss: CrossEntropyLoss(reduction='none') * weight")
    print(f"    - Stage 1a/1b: binary classification (label 0/1)")
    print(f"    - Stage 2a/2b: binary classification (label 0 vs 1 from LabelEncoder)")
    print(f"    - Train each stage independently, evaluate on commitment_testset.parquet")
    print(f"    - Check model.config.id2label after each run")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
