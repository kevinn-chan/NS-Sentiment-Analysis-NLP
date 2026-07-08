"""
Generate all 5 cascade Kaggle notebooks as .ipynb files.
Run once: python notebooks/cascade/build_notebooks.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10.0"}},
        "cells": cells,
    }


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src if isinstance(src, list) else [src]}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}


# ── Shared boilerplate ────────────────────────────────────────────────────

INSTALL = "!pip install -q -U transformers datasets scikit-learn"

IMPORTS = """\
import glob, json, random, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding,
)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""

FIND_FILE = """\
def find_input_file(*names):
    for name in names:
        for pattern in [f"/kaggle/input/**/{name}", f"/kaggle/input/datasets/kevinnchan/**/{name}"]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return Path(matches[0])
        if Path(name).exists():
            return Path(name)
    raise FileNotFoundError(f"Cannot find any of {names}")
"""

DATASET_CLASS = """\
class CascadeDataset(Dataset):
    def __init__(self, encodings, label_ids, weights=None):
        self.encodings = encodings
        self.labels    = label_ids
        self.weights   = weights if weights is not None else [1.0] * len(label_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"]  = torch.tensor(self.labels[idx], dtype=torch.long)
        item["weights"] = torch.tensor(self.weights[idx], dtype=torch.float)
        return item
"""

WEIGHTED_TRAINER = """\
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        weights = inputs.pop("weights")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss    = (loss_fn(outputs.logits, labels) * weights).mean()
        return (loss, outputs) if return_outputs else loss
"""

METRICS = """\
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "kappa":    cohen_kappa_score(labels, preds),
        "accuracy": accuracy_score(labels, preds),
    }
"""

SAVE_ZIP = """\
model_out = OUTPUT_DIR / "best_model"
trainer.save_model(str(model_out))
tokenizer.save_pretrained(str(model_out))
with open(model_out / "id2label.json", "w") as f:
    json.dump({"axis": AXIS, "id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)
print(f"Model saved → {model_out}")

zip_path = str(OUTPUT_DIR.parent / f"cascade_{AXIS.replace('/', '_')}")
shutil.make_archive(zip_path, "zip", str(model_out))
print(f"Zipped → {zip_path}.zip  ({Path(zip_path+'.zip').stat().st_size/1e6:.0f} MB)")
print("Upload this zip as a new Kaggle dataset for inference.")
"""

# ── Training notebook factory ─────────────────────────────────────────────

def training_nb(stage, axis, train_file, label_col, id2label, val_frac=0.15,
                model_name="kevinnchan/singbert-ns-sentiment",
                max_len=128, batch_size=32, epochs=6, lr=2e-5, grad_accum=2):
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    is_binary = num_labels == 2

    config = f"""\
# ── Config ────────────────────────────────────────────────────────────────
STAGE      = "{stage}"
AXIS       = "{axis}"
MODEL_NAME = "{model_name}"   # replace with your saved SingBERT Kaggle dataset path if needed
MAX_LEN    = {max_len}
BATCH_SIZE = {batch_size}
EPOCHS     = {epochs}
LR         = {lr}
GRAD_ACCUM = {grad_accum}
VAL_FRAC   = {val_frac}

ID2LABEL   = {json.dumps(id2label)}
LABEL2ID   = {json.dumps(label2id)}
NUM_LABELS = {num_labels}

OUTPUT_DIR = Path(f"/kaggle/working/{{STAGE}}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Stage: {{STAGE}}  |  Axis: {{AXIS}}  |  Labels: {{LABEL2ID}}")
"""

    load_data = f"""\
# ── Load training data ────────────────────────────────────────────────────
train_path = find_input_file("{train_file}")
print(f"Loading: {{train_path}}")
df = pd.read_csv(train_path)
print(f"Total rows: {{len(df):,}}")
print(df["{label_col}"].value_counts().to_string())

# Encode labels
df = df[df["{label_col}"].notna()].copy()
df["label_id"] = df["{label_col}"].map(LABEL2ID)
df = df[df["label_id"].notna()].copy()
df["label_id"] = df["label_id"].astype(int)
df["weight"]   = df["weight"].fillna(1.0).astype(float)
df["text"]     = df["text"].fillna("").astype(str).str.strip()
df = df[df["text"].str.len() > 5].reset_index(drop=True)

train_df, val_df = train_test_split(
    df, test_size=VAL_FRAC, random_state=SEED, stratify=df["label_id"]
)
print(f"Train: {{len(train_df):,}}  Val: {{len(val_df):,}}")
print(f"Train label dist: {{train_df['label_id'].value_counts().to_dict()}}")
"""

    tokenize = """\
# ── Tokenise ──────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_enc = tokenizer(train_df["text"].tolist(), truncation=True, max_length=MAX_LEN, padding=False)
val_enc   = tokenizer(val_df["text"].tolist(),   truncation=True, max_length=MAX_LEN, padding=False)

train_dataset = CascadeDataset(train_enc, train_df["label_id"].tolist(), train_df["weight"].tolist())
val_dataset   = CascadeDataset(val_enc,   val_df["label_id"].tolist(),   [1.0]*len(val_df))
print(f"Train dataset: {len(train_dataset)}  Val dataset: {len(val_dataset)}")
"""

    train_model = """\
# ── Model + Trainer ───────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS,
    id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
)
model.to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

training_args = TrainingArguments(
    output_dir                  = str(OUTPUT_DIR),
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE * 2,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate               = LR,
    weight_decay                = 0.01,
    warmup_ratio                = 0.06,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    save_total_limit            = 1,
    load_best_model_at_end      = True,
    metric_for_best_model       = "kappa",
    greater_is_better           = True,
    fp16                        = torch.cuda.is_available(),
    dataloader_num_workers      = 2,
    report_to                   = "none",
)

trainer = WeightedTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    data_collator   = DataCollatorWithPadding(tokenizer),
)

train_result = trainer.train()
print(f"Best val kappa: {trainer.state.best_metric:.4f}")
"""

    eval_testset = f"""\
# ── Evaluate on commitment_testset.parquet ────────────────────────────────
test_path = find_input_file("commitment_testset.parquet")
test_df   = pd.read_parquet(test_path)

# Pick the right column for this axis
GOLD_COL = {json.dumps('human_label' if 'buyin' in axis else 'human_stance')}
test_df = test_df[test_df[GOLD_COL].isin(LABEL2ID)].reset_index(drop=True)
print(f"Evaluable testset rows: {{len(test_df)}}  (labels: {{LABEL2ID}})")
print(test_df[GOLD_COL].value_counts().to_string())

test_enc = tokenizer(test_df["text"].tolist(), truncation=True, max_length=MAX_LEN, padding=False)
test_labels = test_df[GOLD_COL].map(LABEL2ID).tolist()
test_dataset = CascadeDataset(test_enc, test_labels)

out = trainer.predict(test_dataset)
probs = torch.softmax(torch.tensor(out.predictions, dtype=torch.float32), dim=-1).numpy()
preds = np.argmax(out.predictions, axis=-1)
preds_named = [ID2LABEL[p] for p in preds]
true_named  = test_df[GOLD_COL].tolist()

print("\\n" + "="*72)
print(f"  {{STAGE}} — {{AXIS}} — Testset Evaluation")
print("="*72)
print(f"  Accuracy: {{accuracy_score(test_labels, preds):.1%}}")
print(f"  Kappa:    {{cohen_kappa_score(test_labels, preds):.3f}}")
print()
print(classification_report(true_named, preds_named, digits=3))
print(confusion_matrix(true_named, preds_named))
print("="*72)

# Save predictions
eval_df = test_df.copy()
eval_df["pred_label"] = preds_named
for i, lbl in ID2LABEL.items():
    eval_df[f"prob_{{lbl}}"] = probs[:, i]
eval_df.to_csv(OUTPUT_DIR / f"testset_eval_{{STAGE}}.csv", index=False)
print(f"Saved eval → {{OUTPUT_DIR}}/testset_eval_{{STAGE}}.csv")
"""

    cells = [
        md(f"# Cascade {stage} — {axis}\n\nBinary classifier. Part of the 4-stage NS commitment cascade."),
        code(INSTALL),
        code(IMPORTS),
        code(FIND_FILE),
        code(DATASET_CLASS),
        code(WEIGHTED_TRAINER),
        code(METRICS),
        code(config),
        code(load_data),
        code(tokenize),
        code(train_model),
        code(eval_testset),
        code(SAVE_ZIP),
    ]
    return nb(cells)


# ── Inference notebook ────────────────────────────────────────────────────

def inference_nb():
    cells = [
        md("# Cascade inference — full corpus\n\nRuns all 4 cascade models on 737k chunks and outputs `chunk_commitment_cascade.parquet`."),
        code(INSTALL),
        code("""\
import glob, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN   = 128
BATCH     = 128
OUT_DIR   = Path("/kaggle/working")
print(f"Device: {DEVICE}")
"""),
        code(FIND_FILE.replace("find_input_file(*names)", "find_input_file(*names)").replace(
            'if Path(name).exists():\n            return Path(name)',
            'local = Path(name)\n        if local.exists(): return local'
        )),
        code("""\
def find_input_dir(*names):
    for name in names:
        for pattern in [f"/kaggle/input/**/{name}", f"/kaggle/input/datasets/kevinnchan/**/{name}"]:
            matches = glob.glob(pattern, recursive=True)
            dirs = [m for m in matches if Path(m).is_dir()]
            if dirs: return Path(dirs[0])
    raise FileNotFoundError(f"Cannot find dir: {names}")
"""),
        code("""\
# ── Load all 4 models ─────────────────────────────────────────────────────
def load_model(dir_name):
    model_dir = find_input_dir(dir_name)
    if (model_dir / "best_model").is_dir():
        model_dir = model_dir / "best_model"
    with open(model_dir / "id2label.json") as f:
        meta = json.load(f)
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    label2id = {v: k for k, v in id2label.items()}
    tok   = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval().to(DEVICE)
    print(f"Loaded {dir_name}: axis={meta['axis']} labels={id2label}")
    return tok, model, id2label, label2id

tok1a, model1a, id2label1a, label2id1a = load_model("cascade-stage1a-buyin-relevance")
tok2a, model2a, id2label2a, label2id2a = load_model("cascade-stage2a-committed-uncommitted")
tok1b, model1b, id2label1b, label2id1b = load_model("cascade-stage1b-stance-relevance")
tok2b, model2b, id2label2b, label2id2b = load_model("cascade-stage2b-supportive-critical")
"""),
        code("""\
# ── Load chunk text ───────────────────────────────────────────────────────
def load_chunks():
    cc_path = find_input_file("comments_chunks.parquet")
    sc_path = find_input_file("submissions_chunks.parquet")
    cc = pd.read_parquet(cc_path, columns=["chunk_id", "doc_id", "text"])
    sc = pd.read_parquet(sc_path, columns=["chunk_id", "doc_id", "text"])
    df = pd.concat([cc, sc], ignore_index=True).drop_duplicates("chunk_id")
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 5].reset_index(drop=True)
    print(f"Chunks loaded: {len(df):,}")
    return df

chunks = load_chunks()
texts = chunks["text"].tolist()
chunk_ids = chunks["chunk_id"].tolist()
"""),
        code("""\
# ── Inference helper ──────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tok   = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}


@torch.no_grad()
def run_inference(texts, tokenizer, model, batch_size=BATCH, desc=""):
    ds     = TextDataset(texts, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    all_probs = []
    for i, batch in enumerate(loader):
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        if i % 50 == 0:
            print(f"  {desc} {i*batch_size:,}/{len(texts):,}", end="\\r")
    print(f"  {desc} done — {len(texts):,} chunks")
    return np.concatenate(all_probs, axis=0)
"""),
        code("""\
# ── Stage 1a: buyin relevance ─────────────────────────────────────────────
print("Stage 1a: buyin relevance ...")
probs1a  = run_inference(texts, tok1a, model1a, desc="1a")
# id2label1a: {0: 'not_relevant', 1: 'relevant'} (or whichever order the model used)
rel_col  = label2id1a.get("relevant", label2id1a.get(1, 1))
not_col  = label2id1a.get("not_relevant", label2id1a.get(0, 0))
is_relevant = probs1a[:, rel_col] >= 0.5
print(f"  Relevant: {is_relevant.sum():,} / {len(is_relevant):,} ({is_relevant.mean():.1%})")
"""),
        code("""\
# ── Stage 2a: committed vs uncommitted (relevant chunks only) ─────────────
print("Stage 2a: committed vs uncommitted ...")
rel_texts    = [texts[i] for i in range(len(texts)) if is_relevant[i]]
rel_indices  = [i for i in range(len(texts)) if is_relevant[i]]
probs2a      = run_inference(rel_texts, tok2a, model2a, desc="2a")
com_col      = label2id2a.get("committed",   0)
unc_col      = label2id2a.get("uncommitted", 1)

# Build full-corpus buyin arrays (default: neutral)
prob_buyin_committed   = np.zeros(len(texts))
prob_buyin_uncommitted = np.zeros(len(texts))
prob_buyin_neutral     = np.ones(len(texts))

for arr_idx, corpus_idx in enumerate(rel_indices):
    prob_buyin_committed[corpus_idx]   = probs2a[arr_idx, com_col]
    prob_buyin_uncommitted[corpus_idx] = probs2a[arr_idx, unc_col]
    prob_buyin_neutral[corpus_idx]     = 0.0

# Non-relevant chunks: set neutral prob from stage 1a
for i in range(len(texts)):
    if not is_relevant[i]:
        prob_buyin_neutral[i]     = probs1a[i, not_col]
        prob_buyin_committed[i]   = probs1a[i, rel_col] * 0.5
        prob_buyin_uncommitted[i] = probs1a[i, rel_col] * 0.5

buyin_label = []
for i in range(len(texts)):
    if not is_relevant[i]:
        buyin_label.append("neutral")
    else:
        if probs2a[rel_indices.index(i) if i in rel_indices else 0, com_col] >= \
           probs2a[rel_indices.index(i) if i in rel_indices else 0, unc_col]:
            buyin_label.append("committed")
        else:
            buyin_label.append("uncommitted")

# Rebuild properly
buyin_label = ["neutral"] * len(texts)
for arr_idx, corpus_idx in enumerate(rel_indices):
    if probs2a[arr_idx, com_col] >= probs2a[arr_idx, unc_col]:
        buyin_label[corpus_idx] = "committed"
    else:
        buyin_label[corpus_idx] = "uncommitted"

from collections import Counter
print(f"  Buyin label dist: {Counter(buyin_label)}")
"""),
        code("""\
# ── Stage 1b: stance relevance ────────────────────────────────────────────
print("Stage 1b: stance relevance ...")
probs1b     = run_inference(texts, tok1b, model1b, desc="1b")
has_col     = label2id1b.get("has_stance", label2id1b.get(1, 1))
no_col      = label2id1b.get("no_stance",  label2id1b.get(0, 0))
has_stance  = probs1b[:, has_col] >= 0.5
print(f"  Has stance: {has_stance.sum():,} ({has_stance.mean():.1%})")
"""),
        code("""\
# ── Stage 2b: supportive vs critical ──────────────────────────────────────
print("Stage 2b: supportive vs critical ...")
st_texts   = [texts[i] for i in range(len(texts)) if has_stance[i]]
st_indices = [i for i in range(len(texts)) if has_stance[i]]
probs2b    = run_inference(st_texts, tok2b, model2b, desc="2b")
sup_col    = label2id2b.get("supportive", 0)
crit_col   = label2id2b.get("critical",   1)

prob_stance_supportive = np.zeros(len(texts))
prob_stance_critical   = np.zeros(len(texts))
prob_stance_neutral    = np.ones(len(texts))

for arr_idx, corpus_idx in enumerate(st_indices):
    prob_stance_supportive[corpus_idx] = probs2b[arr_idx, sup_col]
    prob_stance_critical[corpus_idx]   = probs2b[arr_idx, crit_col]
    prob_stance_neutral[corpus_idx]    = 0.0

for i in range(len(texts)):
    if not has_stance[i]:
        prob_stance_neutral[i]    = probs1b[i, no_col]
        prob_stance_supportive[i] = probs1b[i, has_col] * 0.5
        prob_stance_critical[i]   = probs1b[i, has_col] * 0.5

stance_label = ["neutral"] * len(texts)
for arr_idx, corpus_idx in enumerate(st_indices):
    if probs2b[arr_idx, sup_col] >= probs2b[arr_idx, crit_col]:
        stance_label[corpus_idx] = "supportive"
    else:
        stance_label[corpus_idx] = "critical"

print(f"  Stance label dist: {Counter(stance_label)}")
"""),
        code("""\
# ── Assemble output parquet ───────────────────────────────────────────────
result = pd.DataFrame({
    "chunk_id":               chunk_ids,
    "buyin_label":            buyin_label,
    "stance_label":           stance_label,
    "prob_buyin_committed":   prob_buyin_committed.round(5),
    "prob_buyin_uncommitted": prob_buyin_uncommitted.round(5),
    "prob_buyin_neutral":     prob_buyin_neutral.round(5),
    "prob_stance_supportive": prob_stance_supportive.round(5),
    "prob_stance_critical":   prob_stance_critical.round(5),
    "prob_stance_neutral":    prob_stance_neutral.round(5),
})

out_path = OUT_DIR / "chunk_commitment_cascade.parquet"
result.to_parquet(out_path, index=False)
print(f"\\nSaved → {out_path}  ({result.shape})")
print(f"Buyin:  {result['buyin_label'].value_counts().to_dict()}")
print(f"Stance: {result['stance_label'].value_counts().to_dict()}")
"""),
        code("""\
# ── Quick eval against testset ────────────────────────────────────────────
ts_path = find_input_file("commitment_testset.parquet")
ts = pd.read_parquet(ts_path)
merged = ts.merge(result, on="chunk_id", how="inner")
print(f"Testset rows with predictions: {len(merged)}")

from sklearn.metrics import classification_report
for gold_col, pred_col, name in [
    ("human_label",  "buyin_label",  "BUYIN"),
    ("human_stance", "stance_label", "STANCE"),
]:
    sub = merged[merged[gold_col].isin(["committed","uncommitted","neutral",
                                         "supportive","critical"])]
    print(f"\\n{'='*60}")
    print(f"  CASCADE {name} — n={len(sub)}")
    print(classification_report(sub[gold_col], sub[pred_col], digits=3))
"""),
    ]
    return nb(cells)


# ── Write notebooks ───────────────────────────────────────────────────────

stages = [
    dict(
        stage="stage1a", axis="buyin_relevance",
        train_file="stage1a_train.csv", label_col="label",
        id2label={0: "not_relevant", 1: "relevant"},
        filename="cascade_stage1a_buyin_relevance.ipynb",
    ),
    dict(
        stage="stage2a", axis="buyin_direction",
        train_file="stage2a_train.csv", label_col="label",
        id2label={0: "committed", 1: "uncommitted"},
        filename="cascade_stage2a_committed_uncommitted.ipynb",
    ),
    dict(
        stage="stage1b", axis="stance_relevance",
        train_file="stage1b_train.csv", label_col="label",
        id2label={0: "no_stance", 1: "has_stance"},
        filename="cascade_stage1b_stance_relevance.ipynb",
    ),
    dict(
        stage="stage2b", axis="stance_direction",
        train_file="stage2b_train.csv", label_col="label",
        id2label={0: "critical", 1: "supportive"},
        filename="cascade_stage2b_supportive_critical.ipynb",
    ),
]

for s in stages:
    fname = s.pop("filename")
    notebook = training_nb(**s)
    path = OUT / fname
    path.write_text(json.dumps(notebook, indent=1))
    print(f"Written: {path.name}")

infer = inference_nb()
infer_path = OUT / "cascade_inference.ipynb"
infer_path.write_text(json.dumps(infer, indent=1))
print(f"Written: {infer_path.name}")

print("\nDone. Upload these to Kaggle:")
print("  Notebooks: 4 training + 1 inference")
print("\nKaggle datasets needed:")
print("  1. cascade-train-data      → data/processed/new/cascade/*.csv")
print("  2. ns-commitment-testset   → data/processed/new/commitment_testset.parquet")
print("  3. ns-corpus-chunks        → data/processed/new/comments_chunks.parquet")
print("                               data/processed/new/submissions_chunks.parquet")
print("  4. singbert-base           → your existing SingBERT model (from prior Kaggle runs)")
print("\nTraining order: 1a → 2a → 1b → 2b → inference")
print("After each training run, download the zipped model and re-upload as a new dataset.")
