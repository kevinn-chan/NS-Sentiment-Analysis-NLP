# NS Sentiment — Project Handoff (updated 2026-05-26)

## Goal

Quantify Singaporean public sentiment and commitment to National Service (NS) over time,
using Reddit data from three subreddits:
- r/singapore
- r/askSingapore
- r/NationalServiceSG

The end product is a Streamlit dashboard with Seaborn visualisations showing:
- Topic distribution across NS discourse (BERTopic, hierarchical)
- Sentiment trends over time (monthly rollups, score-weighted)
- Commitment-to-defence scoring (zero-shot NLI)
- Divergence between post sentiment and comment section sentiment
- Thread depth analysis

---

## Pipeline Stages — Current Status

| Stage | Description | Status |
|---|---|---|
| 1 | Data loading (ZST → parquet) | ✅ Done |
| 2 | Cleaning + NS relevance filter + thread depth | ✅ Done |
| 3 | Semantic chunking + embeddings | ✅ Done |
| 4 | BERTopic topic modelling (v3) | ✅ Done |
| 4b | Noise topic removal | ✅ Done |
| 4c | Manual 4-layer taxonomy + hierarchy encoding | ✅ Done |
| 5a | Sentiment classification — XLM base (all 737k chunks) | ⚠️ Superseded — see 5a-h2h |
| 5a-verify | Human annotation (197 chunks, 81.3% corpus-weighted) | ✅ Done (anchored — see 5a-h2h) |
| 5a-xlm | XLM base vs RoBERTa corpus-weighted comparison | ✅ Done |
| 5a-spot | Manual spot-check (100 chunks, 78.0% accuracy) | ✅ Done |
| 5a-h2h | Fair blind head-to-head — RoBERTa wins, fine-tune decision pending | 🔄 Decision pending |
| 5a-llm-validate | LLM kappa gate — gpt-4.1, κ=0.765 ✅ (corrected prompt) | ✅ Done |
| 5a-llm-annotate | Bulk annotation complete — 7,946 chunks, singbert_train.csv built | ✅ Done |
| 5a-holdout-eval | Holdout accuracy: 77.9%, κ=0.597 (195 unseen rows, pre-QC) | ✅ Done |
| 5a-holdout-qc | Blind QC review: 28 positive-class disagreements → 9 corrected | ✅ Done |
| **5a-singbert** | **Fine-tune zanelim/singbert-large-sg on singbert_train.csv** | ⏳ **Next** |
| **5b** | **Commitment scoring (zero-shot NLI)** | ✅ **Done** |
| 6 | Document-level aggregation | ⏳ Not started |
| 7 | Divergence score (post vs comments) | ⏳ Not started |
| 8 | Temporal aggregation | ⏳ Not started |
| 9 | Streamlit dashboard + Seaborn viz | ⏳ Not started |

---

## Active Output Files

All live files are under `data/processed/new/` and `models/new/`.

| File | Rows | Notes |
|---|---|---|
| `data/processed/new/comments_chunks.parquet` | 637,660 | Post-noise-removal |
| `data/processed/new/submissions_chunks.parquet` | 101,159 | Post-noise-removal |
| `data/processed/new/chunk_topics.parquet` | 737,583 | `topic_id_fine`, `topic_id_coarse`; 20,486 outliers (-1) |
| `data/processed/new/chunk_sentiment.parquet` | 737,274 | ★ `chunk_id`, `sent_neg`, `sent_neu`, `sent_pos` — XLM base (FINAL) |
| `data/processed/new/chunk_sentiment_lexicon.parquet` | 738,819 | `chunk_id`, `sent_lexicon_compound` (VADER+NS lexicon) |
| `data/processed/new/chunk_commitment.parquet` | 737,274 | ★ `chunk_id`, `commit_support`, `commit_critical`, `commit_neutral` — **DONE** |
| `data/processed/new/chunk_commitment_lexicon.parquet` | 737,274 | `chunk_id`, `lex_committed`, `lex_uncommitted`, `lex_net` — **DONE** |
| `data/processed/new/annotation_sample.parquet` | 197 | Stratified sample for human labelling |
| `data/processed/new/annotations.csv` | 197 | Human labels (complete — 81.3% corpus-weighted vs XLM) |
| `data/processed/new/blind_annotation.csv` | 197 | Reviewed human labels (weight 3× in training) |
| `data/processed/new/holdout_test.csv` | 199 | Held-out test set — NEVER use for training |
| `data/processed/new/llm_annotation.csv` | 7,946 | gpt-4.1 bulk annotations (κ=0.765 validated prompt) |
| `data/processed/new/llm_validation.csv` | 197 | Validation run results (κ=0.765) |
| `data/processed/new/holdout_eval.csv` | 195 | Holdout eval: 81.0% acc, κ=0.653, positive F1=60.4% (post-QC) |
| `data/processed/new/singbert_train.csv` | 8,095 | **Training set for SingBERT** (human 3× + LLM 1×, 0 holdout overlap) |
| `notebooks/kaggle_finetune_singbert_v1.ipynb` | — | **SingBERT fine-tune notebook** — upload to Kaggle, run on T4 |
| `data/processed/new/spot_check.csv` | 100 | Manual spot-check results (78.0% accuracy) |
| `data/processed/new/sentiment_audit.parquet` | 500 | llama3.2:3b gold labels vs roberta (supplementary, use with caution) |
| `data/processed/new/topic_keywords_fine.csv` | 359 topics | Post-noise-removal keywords |
| `data/processed/new/topic_keywords_coarse.csv` | — | Coarse model keywords |
| `data/processed/new/hierarchical_topics_manual.parquet` | 358 rows | Manual 4-layer taxonomy as binary dendrogram |
| `data/processed/new/dendrogram_manual.html` | — | Manual taxonomy visualisation |
| `data/processed/new/dendrogram_bertopic.html` | — | BERTopic Ward clustering visualisation |
| `models/new/bertopic_fine/` | — | 359 topics, safetensors format, post-noise-removal |
| `models/new/bertopic_coarse/` | — | Coarse model, safetensors format |

**Old files** (do not use): `data/processed/` (without `/new/`) and `models/old/`

---

## Code Structure

```
ns_sentiment/
├── config.py
├── src/
│   ├── data/
│   │   ├── loader.py                    # ZST → parquet
│   │   └── cleaner.py                   # NS filter, bot removal, thread depth
│   ├── features/
│   │   ├── chunker.py                   # Semantic chunking + embeddings
│   │   ├── lexicon_scorer.py            # VADER + NS/Singlish lexicon (supplementary)
│   │   ├── commitment_lexicon.py        # ★ Stage 5b lexicon scorer (committed/uncommitted buckets)
│   │   ├── patch_sentiment_xlm.py       # Singlish patch (SUPERSEDED — kept as record)
│   │   ├── sentiment_audit.py           # Tier 3: Ollama audit (llama3.2:3b)
│   │   ├── annotator.py                 # ★ Human annotation CLI (complete, 197 chunks)
│   │   └── spot_checker.py              # ★ Manual spot-check CLI (complete, 100 chunks)
│   └── models/
│       ├── topic_model.py
│       ├── topic_labels.py              # ★ Full 4-layer taxonomy (359 topics)
│       ├── build_manual_dendrogram.py
│       └── plot_manual_dendrogram.py
├── notebooks/
│   ├── kaggle_commitment_v1.ipynb       # ★ Stage 5b — bart-large-mnli NLI commitment scoring
│   ├── kaggle_xlm_base_v1.ipynb         # ★ Stage 5a FINAL — XLM base all 737k chunks
│   ├── kaggle_xlm_patch_v1.ipynb        # Stage 5a Singlish patch (SUPERSEDED — kept as record)
│   ├── kaggle_sentiment_v1.ipynb        # Stage 5a original RoBERTa run (SUPERSEDED)
│   ├── kaggle_chunker_v2.ipynb
│   ├── kaggle_topic_model_v3.ipynb
│   └── kaggle_noise_removal.ipynb
├── models/new/
└── data/processed/new/
```

---

## Stage 4 — What Was Done

### BERTopic v3 (final model)
- **359 fine topics** after noise removal (started from 375+, removed 16 noise topics)
- **Noise topics removed:** t58, t71, t77, t79, t89, t94, t97, t117, t127, t128, t195, t207, t255, t257, t261, t372
- **UMAP:** n_components=5, n_neighbors=15, min_dist=0.0, metric=cosine, low_memory=True
- **HDBSCAN:** min_cluster_size=50, metric=euclidean, prediction_data=True
- Outlier rescue via batched cosine similarity to topic centroids

### Manual 4-layer taxonomy
All 359 topics vetted and labelled in `src/models/topic_labels.py`:
```python
TOPIC_LABELS = {
    0: {"name": "Vocational Appointments",
        "macro": "Vocations & Units",
        "sub":   "SCDF & SPF",
        "sub_sub": "SCDF"},
    ...
}
```

**17 Macro categories:** Vocations & Units · BMT & Training · Physical Fitness & IPPT ·
Medical & Health · Mental Health · NS Life & Culture · Enlistment & Pre-NS ·
Reservist & ICT · ORD & Post-NS · Pay & Benefits · Discipline & Misconduct ·
Relationships & Social · Admin & Logistics · Gear & Equipment · NS Policy & Society ·
Career & Sign-on · Gender & Diversity

---

## Stage 5a — Sentiment Classification (DONE ✅)

### ⚠️ Current state: model selection decision pending

`chunk_sentiment.parquet` currently contains **XLM base scores** (40.3% neg / 47.6% neu / 12.2% pos).
This file needs to be **replaced** — see fair h2h findings below. Do not proceed to Stage 6
until the model decision is resolved and the full corpus is re-scored.

**Two paths open:**
1. **Re-run full corpus with RoBERTa-base** — 65.9% blind accuracy, 1 Kaggle job (~2 hrs)
2. **Fine-tune SingBERT** — expected 72–76% blind accuracy, requires ~500 more annotation hours + training job

### Model that was run (XLM base — now superseded)
**Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment` — run on all 737,274 chunks
**Notebook:** `notebooks/kaggle_xlm_base_v1.ipynb` (Kaggle T4 x2)
**Output:** `chunk_sentiment.parquet` — `chunk_id`, `sent_neg`, `sent_neu`, `sent_pos`
**Status:** XLM was shown to over-predict negative (54% predicted vs 32% human) — do not use

### Why XLM over the initial RoBERTa hybrid (historical — now reversed)

Initial design used `cardiffnlp/twitter-roberta-base-sentiment-latest` as base with
an XLM patch on Singlish chunks. Head-to-head evaluation on 197 human-annotated chunks
revealed the annotation sample over-represented Singlish by **15.2x** (71.6% of sample
vs 4.7% of real corpus), making RoBERTa appear to win overall (75.6% vs 69.5%).

Corpus-weighted accuracy (95.3% English / 4.7% Singlish):
- RoBERTa base: **76.7%**
- **XLM base: 81.3%** ← winner

XLM is stronger on English-dominant text (82.1% vs 76.8%), which is 95.3% of the corpus.
The hybrid was the worst of both worlds — extra complexity with lower accuracy.
XLM as sole model is simpler and more accurate.

**How the 4.7% Singlish density was measured:**
The Singlish detector (word-boundary regex over ~50 NS/Singlish terms) was applied
to all 738,819 chunks. Result: 34,834 chunks matched (4.7%). Corpus proportion used
for weighting is 95.3% English / 4.7% Singlish.

### What was evaluated and dropped

| Approach | Outcome |
|---|---|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | 76.7% corpus-weighted — dropped in favour of XLM |
| RoBERTa + XLM Singlish patch hybrid | Complex, still loses to plain XLM — dropped |
| `facebook/bart-large-mnli` (zero-shot NLI) | Wrong tool for sentiment — retained for Stage 5b |
| VADER + NS lexicon | 47.7% human agreement — supplementary signal only |
| `gemma3:1b` (Ollama) | 92.8% negative — unusable |
| `llama3.2:3b` (Ollama audit) | Misleading metric; human annotation is ground truth |
| Fine-tuned transformer | Requires 500+ labelled examples — deferred, not dropped |

### Validation results

| Method | Accuracy | Notes |
|---|---|---|
| Human annotation — sample-weighted | 69.5% | Misleading (Singlish 15x over-represented in sample) |
| Human annotation — corpus-weighted | **81.3%** | True accuracy; use this number |
| Manual spot-check (100 chunks) | **78.0%** | Independent random-ish sample, no model anchoring |

Spot-check by stratum: `high_neu`=100%, `high_pos`=95.5%, `high_neg`=81.8%,
`singlish`=65.0%, `low_conf`=44.4%

### Known failure modes
- Over-predicts negative for factual NS questions ("Forced to downpes due to rash problem?")
- Misses Singlish sentiment cues in both directions ("sian max" → neutral, "lepak" → neutral)
- Low-confidence chunks (18.4% of corpus) are genuinely ambiguous — acceptable at aggregation level
- Chunks that are reactions to other posts (not standalone NS sentiment) are correctly labelled
  for surface tone but may not reflect NS-directed sentiment specifically

### Fair blind head-to-head — COMPLETED (2026-05-25)

**Script:** `src/features/blind_annotator.py`
**Annotation:** 98 blind-labelled chunks (no model scores shown during annotation)
**Strata:** model-agnostic — singlish/submissions/short+medium+long comments (20 each)
**Human label distribution:** 32% neg / 52% neu / 16% pos

| Model | Corpus-weighted accuracy | Notes |
|---|---|---|
| XLM (`twitter-xlm-roberta-base-sentiment`) | 60.5% | Over-predicts negative (54% vs human 32%) |
| **RoBERTa (`twitter-roberta-base-sentiment-latest`)** | **65.9%** | **Winner — neutral recall 82%** |
| RoBERTa-Large (`j-hartmann/sentiment-roberta-large-english-3-classes`) | 65.9% | Ties RoBERTa-base exactly — no gain from larger model |

**Key finding:** XLM systematically over-predicts negative — 41% of human-neutral chunks called negative vs 13% for RoBERTa. This structural flaw explains the 40.3% negative rate in the full corpus (vs ~23% from RoBERTa on this benchmark).

**Pre-trained model ceiling:** ~66%. Swapping models or going larger does not help — domain mismatch (Twitter-trained models on Reddit NS discourse) is the bottleneck, not model capacity.

**Fine-tuning path:** `zanelim/singbert-large-sg` (pre-trained on r/singapore + HardwareZone). Needs ~600 labeled examples. Currently have 98 reliable blind labels. Need ~500 more annotations (~5 hrs) to reach the training floor. Expected accuracy: 72–76%.

**Project context:** This is enterprise-level work for department/supervisor presentation. 65.9% is borderline; fine-tuning to 72–76% is the recommended path.

### Annotation bias discovery (historical context — resolved)
The original 197-chunk annotation was not model-neutral — RoBERTa predictions shown on screen
during labelling, strata selected on RoBERTa confidence. The fair blind h2h (above) overturned
the earlier apparent XLM-wins conclusion. The 81.3% figure from the anchored annotation
should not be cited as a validated XLM accuracy number — it was biased.

### Validation files
- `data/processed/new/annotations.csv` — 197 human labels (chunk_id, human_label, stratum)
- `data/processed/new/annotation_sample.parquet` — stratified annotation sample with chunk text
- `data/processed/new/spot_check.csv` — 100-chunk manual spot-check (human_verdict, corrected_label)

### Lexicon cross-check (supplementary)
**Script:** `src/features/lexicon_scorer.py`
**Output:** `chunk_sentiment_lexicon.parquet` — `sent_lexicon_compound` [-1, +1]
VADER + 60-entry custom NS/Singlish lexicon. 65.5% directional agreement with XLM.
Use as interpretability signal only, not primary classifier.

---

## Stage 5b — Commitment Scoring

**Model:** `facebook/bart-large-mnli` (zero-shot NLI, `multi_label=False`)
**Input:** chunk text from chunk parquets + chunk_ids (all 737k chunks)
**Run on:** Kaggle GPU (T4 x2) — 3 NLI forward passes per chunk

Hypotheses — 3-way softmax competition (scores sum to 1 per chunk):
```
"The author supports National Service"                                  → commit_support
"The author is critical of National Service"                            → commit_critical
"The author is discussing National Service without expressing a strong opinion"  → commit_neutral
```

`multi_label=False`: all 3 compete in a single softmax, mirroring the sent_neg/neu/pos structure.
Net score for aggregation: `commit_net = commit_support − commit_critical`.
Neutral chunks (factual questions, experience descriptions) score high on H3 and are excluded
from the net signal rather than diluting it.

**Output:** `chunk_commitment.parquet` — `chunk_id`, `commit_support`, `commit_critical`, `commit_neutral`

**Notebook:** `notebooks/kaggle_commitment_v1.ipynb` ✅ completed
- Final settings: `BATCH_SIZE=256`, `PIPE_BATCH_SIZE=128`, `max_length=256`, fp16, DataParallel T4 x2
- Issues encountered and fixed: pipeline single-GPU bottleneck (10 chunks/s → dropped pipeline),
  OOM at PIPE_BATCH_SIZE=512 (BART decoder FFN fc1 ~1GB/layer × 12), fixed with 128 + max_length=256

Input dataset on Kaggle: `ns-sentiment-chunks-v3` (same as Stage 5a).

### Lexicon parallel track — DONE ✅

**Script:** `src/features/commitment_lexicon.py`
**Output:** `chunk_commitment_lexicon.parquet` — `chunk_id`, `lex_committed`, `lex_uncommitted`, `lex_net`

Seed lexicon: 38 committed terms / 37 uncommitted terms.
Run: `python -m src.features.commitment_lexicon` (completes in ~1 min locally, no GPU).

**Run results (corrected — post bug-fix re-run, 737k chunks):**
- Chunks with any hit: 19,026 (2.6%) — slightly down from 2.8% (fewer false committed hits after "worth it" removal)
- Mean lex_net ≈ −0.0021 — slightly uncommitted on balance (correct direction: NS criticism is common)
- Mixed signals: 153 (down from 771 before fix — removing "worth it" eliminated false cancellations)
- Committed-only: 10,132 chunks; uncommitted-only: 8,741 chunks
- Top committed signals: `sign on` (9,671), `signed on` (1,187), `brotherhood` (835)
- Top uncommitted signals: `keng` (4,965), `chao keng` (3,489), `wayang` (1,777), `waste of time` (809)
- Notable: `slavery` (422 hits) — common NS hyperbole, genuine uncommitment signal

**Bugs found and fixed:**
1. `worth it` removed from COMMITTED — it was firing inside "not worth it" (uncommitted),
   cancelling the uncommitted signal. Net effect: "not worth it" was scoring 0 instead of -1.
   Fix: removed `worth it`; unambiguous phrases `worth serving` / `worth the sacrifice` remain.
2. `bo chup` had only 4 hits — Reddit Singlish spelling is inconsistent.
   Added variants: `bo chap`, `bochup`, `bochap`.

**Known limitation:** `sign on` / `signed on` fires on ANY mention (e.g. "my friend signed on",
"thinking of signing on?"), not only first-person author commitment. Treat as weak signal;
use NLI `commit_support` to disambiguate at topic-level aggregation.

### NLI output validation — DONE ✅

**Row count:** 737,274 — matches corpus exactly, 0 nulls
**Score integrity:** every row sums to exactly 1.000 (softmax confirmed working)

**Majority label distribution:**
- neutral:   89.2% (657,650) — expected; most NS discourse is factual/descriptive
- support:    6.2%  (45,436)
- critical:   4.6%  (34,188)

**Net commitment:** mean=+0.048, median=+0.056
- 65.2% net positive (support > critical)
- 34.7% net negative (critical > support)
- Corpus is slightly pro-NS overall

**Lexicon cross-check (Spearman on 20,296 signal chunks):**
- Spearman r = 0.291, p≈0 — moderate agreement, statistically significant
- Directional agreement: 65.6%
- Committed lexicon chunks: commit_net = +0.068 (vs baseline +0.048) ✓
- Uncommitted lexicon chunks: commit_net = −0.100 (vs baseline +0.048) ✓
- NLI moves in the right direction when lexicon has signal

**Known failure modes (spot-checked):**
- Reports ABOUT criticism scored as critical: "girl goes on Instagram rant dissing NS" → critical=0.998
  (the chunk is describing someone else's criticism, not the author being critical)
- Definitional statements scored as support: "Mandatory military service. Stands for NS." → support=0.992
- "sign on" (9,671 lexicon hits) mostly fires on third-person mentions, not author commitment
- These are inherent NLI limitations; acceptable at aggregation level where individual errors cancel

---

## Stage 6 — Document-level Aggregation (after 5b)

Join: `chunk_sentiment` + `chunk_commitment` + `chunk_topics` + chunk parquets (for `log_weight`, `doc_id`)
Group by `doc_id`, aggregate scores weighted by `log_weight` (already in chunk parquets).

Before joining, add taxonomy columns to `chunk_topics`:
```python
from src.models.topic_labels import TOPIC_LABELS
chunk_topics['topic_macro']   = chunk_topics['topic_id_fine'].map(lambda t: TOPIC_LABELS.get(t, {}).get('macro'))
chunk_topics['topic_sub']     = chunk_topics['topic_id_fine'].map(lambda t: TOPIC_LABELS.get(t, {}).get('sub'))
chunk_topics['topic_sub_sub'] = chunk_topics['topic_id_fine'].map(lambda t: TOPIC_LABELS.get(t, {}).get('sub_sub'))
```

Output: `doc_sentiment.parquet`

---

## Stage 7 — Divergence Score

For each `post_id`, compare submission chunk scores vs comment chunk scores.
Divergence metric: e.g. mean(sent_neg_comments) − mean(sent_neg_submissions) per post.
Output: `doc_divergence.parquet`

---

## Stage 8 — Temporal Aggregation

**Must normalise datetime first** (see Known Issues).
Monthly rollup grouped by subreddit + topic_macro.
Output: `temporal_sentiment.parquet`

---

## Stage 9 — Streamlit Dashboard + Seaborn Viz

Key views to build:
1. Topic distribution (BERTopic hierarchical dendrogram, macro-coloured)
2. Sentiment trend over time (monthly, score-weighted, by topic_macro)
3. Commitment-to-defence score trend (from Stage 5b)
4. Post vs comment divergence heatmap (from Stage 7)
5. Thread depth analysis

---

## Emotion Classification — Deferred Optional Layer

Model: `cardiffnlp/twitter-roberta-base-emotion-multilabel-latest`
Output: `chunk_emotion.parquet` — 11 emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust)

**Status:** Deferred pending supervisor consultation. Gate already cleared (81.3% > 70%).
Do NOT add until Stages 5b–9 have a working v1.
If added, this is a second Kaggle notebook (~45 min effort) alongside `chunk_sentiment.parquet`.
Known limitation: no XLM version for Singlish — acceptable caveat.

---

## What Worked

- **Kaggle "Save & Run All"** (committed mode) — only reliable way to run long jobs
- **Corpus-weighted accuracy** — always weight annotation results by actual corpus proportions,
  not stratified sample proportions. Sample bias can flip the apparent winner.
  Formula: Σ (corpus_proportion_i × accuracy_i) over strata.
- **Batched cosine similarity** for outlier rescue — avoids OOM
- **UMAP checkpoint** (monkey-patch saves `umap_embeddings.npy`) — saves 2+ hours if HDBSCAN fails
- **Noise removal via `update_topics()`** — cleanly removes noise and recalculates c-TF-IDF
- **Manual taxonomy top-down** — macro → sub → sub_sub semantically, then assign fine topics
- **Dendrogram via scipy `link_color_func`** — only reliable way to get macro-coloured dendrograms
- **`top_k=None` in HuggingFace pipeline** — correct replacement for deprecated `return_all_scores=True`
- **Blind annotation design** — hiding model scores during labelling is essential; the original
  anchored annotation (scores shown on screen) produced a false XLM-wins result that the blind
  h2h overturned. Always annotate blind, then compare models post-hoc.
- **RoBERTa-base over XLM** — 65.9% vs 60.5% corpus-weighted on blind benchmark; XLM's
  systematic negative over-prediction (54% predicted vs 32% human) makes it unsuitable
- **VADER + custom NS lexicon** — fast interpretable signal; 60 entries covering key NS/Singlish terms
- **Ollama `format: "json"`** — enforces valid JSON output from local LLMs reliably
- **Resume-safe CLIs** — both `annotator.py` and `spot_checker.py` save after every keypress;
  safe to quit at any time and resume

---

## What Didn't Work / Gotchas

- **`return_all_scores=True` in transformers pipeline** — deprecated; use `top_k=None`
- **gemma3:1b for sentiment labelling** — 92.8% of labels were "negative"; too small, miscalibrated
- **llama3.2:3b audit agreement % as primary metric** — misleading; use human annotation as ground truth
- **Singlish density via exact whitespace token match** — undercounts; "sian." (with period) misses.
  Use `re.findall(r'\b\w+\b', text)` or word-boundary regex instead
- **BERTopic `visualize_hierarchy` with filtered `topics=` list** — crashes; use scipy directly
- **`color_threshold` in `visualize_hierarchy`** — doesn't apply colours with custom hierarchical_topics
- **`set_topic_labels()` with partial topic coverage** — crashes if max topic ID > len(custom_labels_)
- **Bottom-up dendrogram clustering** — cutting at distance thresholds gave meaningless groupings
- **MPS (Apple GPU) for embeddings** — OOM at batch_size=256; use CPU
- **Local HDBSCAN with core_dist_n_jobs=-1** — spawns 9 workers on 809k points → OOM; fix: n_jobs=1
- **`calculate_probabilities=True` in BERTopic** — OOM on Kaggle
- **Hierarchical `reduce_topics()` return value** — newer BERTopic returns model object, not tuple
- **`sentencepiece` missing for XLM locally** — add to pip install; also ensure venv python is used,
  not system anaconda (`python` → `.venv/bin/python`)
- **Annotation CSV merge with parquet sample** — both had 'stratum' column; pandas renames to
  stratum_x/stratum_y. Fix: drop 'stratum' from one side before merging.
- **Spot-checker UI quirk** — pressing N on an already-negative chunk shows [negative → negative]
  in the mislabelled list. Not a bug; the verdict is still "disagree".

---

## Known Issues (carry-forward)

### datetime precision mismatch
- `submissions_chunks`: `datetime64[ms, UTC]`
- `comments_chunks`: `datetime64[ns, UTC]`
- Normalise both to `ms` before Stage 8 temporal aggregation

### chunk_topics.parquet missing taxonomy columns
- No `topic_macro` / `topic_sub` / `topic_sub_sub` columns yet
- Join from `topic_labels.py` before Stage 6 (see Stage 6 section above)

### chunk_sentiment row count vs chunk parquets
- `chunk_sentiment.parquet`: 737,274 rows (unique chunk_ids)
- Combined chunk parquets: 738,819 rows (1,545 duplicate chunk_ids from batch boundary issue)
- Coverage is 100% of unique chunks — the 1,545 gap is duplicates in source, not missing scores

### hierarchical_topics.parquet is stale
- Auto-generated BERTopic dendrogram was pre-noise-removal (contains removed topic IDs)
- Use `hierarchical_topics_manual.parquet` (current) for all downstream work

---

## Config (`config.py`)

```python
CHUNK_MIN_SENTENCES   = 2
CHUNK_MAX_SENTENCES   = 6
CHUNK_TOKEN_THRESHOLD = 400
CHARS_PER_TOKEN       = 4
SIMILARITY_THRESHOLD  = 0.5
EMBEDDING_MODEL       = "sentence-transformers/all-mpnet-base-v2"
```

---

## Immediate Next Actions

### ✅ Stage 5a LLM annotation — COMPLETE

**Files produced:**
- `singbert_train.csv` — 8,095 rows (197 human ×3 + 7,898 LLM ×1), zero holdout overlap ✅
- `llm_annotation.csv` — 7,946 raw LLM labels
- `llm_validation.csv` — 197-row κ=0.765 validation result
- `holdout_eval.csv` — 195-row holdout accuracy result

**Quality summary:**
| Metric | Score | Notes |
|---|---|---|
| κ (validation, 197 rows) | 0.765 | Human labels reviewed with LLM — not fully independent |
| Accuracy (holdout, 195 rows) | **81.0%** | Clean, post-QC corrected labels — use this number |
| κ (holdout, post-QC) | 0.653 | Honest inter-annotator agreement on cleaned labels |
| Positive F1 (holdout, post-QC) | 60.4% | Improved from 46.2% after fixing 9 annotation errors |
| Negative F1 (holdout, post-QC) | 82.0% | Solid |
| Neutral F1 (holdout, post-QC) | 85.2% | Strong |
| _Pre-QC accuracy_ | _77.9%_ | _Before blind QC review — kept for reference_ |
| _Pre-QC positive F1_ | _46.2%_ | _Human annotation noise confirmed and resolved_ |

**Label distribution (singbert_train.csv):** neutral 60%, negative 27.5%, positive 12.4%
**Total API cost:** ~$9 (gpt-4.1, Tier 3, across all validation + bulk runs)

---

### 🔜 Next: Fine-tune SingBERT (Stage 5a-singbert)

**Base model:** `zanelim/singbert-large-sg` (BERT-large, pre-trained on r/singapore + HardwareZone)

**Training data:** `singbert_train.csv` (8,095 rows, weighted: human 3×, LLM 1×)
**Eval data:** `holdout_test.csv` (199 rows, 195 evaluable — the clean test set)

**Key training decisions:**
- Use **class weights** for positive class (weight ~1.5–2×) — positive F1 is now 60.4% post-QC (was 46.2%), reduced need vs original estimate
- Use `human_label` weighted sampling or `weight` column for loss weighting
- Target metric: weighted F1 on holdout_test.csv
- Expected accuracy: 75–82% (SingBERT domain pre-training should beat gpt-4.1's 77.9%)

**Kaggle notebook:** `notebooks/kaggle_finetune_singbert_v1.ipynb` ✅ BUILT
- Loads `singbert_train.csv` + `holdout_test.csv` from dataset `ns-sentiment-labels-v1`
- Replicates human rows 3× (from weight column) before train/val split
- Balanced class weights (positive ~2.7×, negative ~1.2×) via `WeightedTrainer`
- Custom `WeightedTrainer` applies class weights in cross-entropy loss
- 90/10 stratified train/val split; early stopping patience=2 on val κ
- Evaluates on sealed holdout: accuracy, F1, κ, classification report
- Saves best model + tokenizer to `/kaggle/working/singbert_ns_sentiment/best_model/`
- Zips for Kaggle dataset upload → use in inference notebook

**To run on Kaggle:**
1. Upload `data/processed/new/singbert_train.csv` + `data/processed/new/holdout_test.csv` as Kaggle dataset `ns-sentiment-labels-v1`
2. Import notebook, add that dataset as input
3. Run on T4 x2 (GPU), committed mode
4. Expected runtime: ~45–90 min (BERT-large, 5 epochs, ~8.6k train rows)

**After SingBERT trains:**
```bash
# Build inference notebook to score full 737k corpus
# Output: chunk_sentiment_singbert.parquet (replaces chunk_sentiment.parquet)
```
Then Stage 6 (doc aggregation) is unblocked.

**Provider history (LLM annotator):**
- Groq 8B: 6k TPM wall
- Groq 70B: 1k RPD exhausted
- Cerebras Qwen 235B: free, slow, κ=0.55 (model issue)
- OpenAI gpt-4.1-mini Tier 0: 200 RPD wall
- OpenAI gpt-4.1 Tier 3: 10k RPM, κ=0.765 ✅ USED

   **Kappa history:**
   | Round | Model | κ | Notes |
   |---|---|---|---|
   | 1 | Groq 8B complex prompt | 0.410 | Too complex for 8B |
   | 2 | Groq 8B simplified | 0.747 | At 25 rows only — hit RPD wall |
   | 3 | Cerebras Qwen 235B | 0.550 | Model neutral-biased |
   | 4 | OpenAI gpt-4.1-mini | 0.555 | Same neutral bias |
   | 5 | OpenAI gpt-4.1 | 0.595 | Better but below gate |
   | 6 | **gpt-4.1 + 48 label corrections** | **0.820** | ✅ User reviewed all disagreements |

   **Key insight:** κ=0.55 plateau was annotation noise (single annotator, ambiguous NS posts).
   Resolved by `python -m src.features.blind_annotator --review` — 48 disagreements shown
   side-by-side with gpt-4.1 labels; user corrected/confirmed each one.

   **Files:**
   - `data/processed/new/blind_annotation.csv` — 197 reviewed human labels (training, weight 3×)
   - `data/processed/new/holdout_test.csv` — 195 held-out labels (eval only, NEVER train)
   - `data/processed/new/llm_validation.csv` — gpt-4.1 labels for all 197 validation rows
   - `data/processed/new/llm_annotation.csv` — bulk LLM labels (in progress)

⚠️ Do NOT proceed to Stage 6 until `chunk_sentiment.parquet` is replaced AND
   `llm_annotation.csv` is complete + `singbert_train.csv` is built.

---

### Track B — Stage 5b (commitment scoring) ✅ COMPLETE

**Stage 5b does NOT depend on Stage 5a.**

   - Notebook: `notebooks/kaggle_commitment_v1.ipynb` ✅ complete
   - Lexicon scorer: `src/features/commitment_lexicon.py` ✅ complete — re-run with bug fixes applied
   - Output: `chunk_commitment.parquet` ✅ 737,274 rows — `chunk_id`, `commit_support`, `commit_critical`, `commit_neutral`
   - Output: `chunk_commitment_lexicon.parquet` ✅ 737,274 rows — `chunk_id`, `lex_committed`, `lex_uncommitted`, `lex_net`
   - **No further action needed on Track B**

---

### After both tracks complete

3. **Stage 6 — Document-level aggregation**
   Join: `chunk_sentiment` + `chunk_commitment` + `chunk_topics` + chunk parquets
   Group by `doc_id`, weight by `log_weight`. Add taxonomy columns from TOPIC_LABELS dict.
   Output: `doc_sentiment.parquet`

4. **Stage 7 — Divergence score**
   For each `post_id`, compare submission chunk scores vs comment chunk scores.
   Output divergence metric per post.

5. **Stage 8 — Temporal aggregation**
   Normalise datetime to `ms` first (see known issues).
   Monthly rollup grouped by subreddit + topic_macro.

6. **Stage 9 — Streamlit dashboard + Seaborn viz**
   See Stage 9 section above for required views.
