# NS Sentiment — Project Handoff (updated 2026-05-25)

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
| **5b** | **Commitment scoring (zero-shot NLI)** | 🔄 **IN PROGRESS (separate session)** |
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
| `data/processed/new/annotation_sample.parquet` | 197 | Stratified sample for human labelling |
| `data/processed/new/annotations.csv` | 197 | Human labels (complete — 81.3% corpus-weighted vs XLM) |
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

## Stage 5b — Commitment Scoring (NEXT)

**Model:** `facebook/bart-large-mnli` (zero-shot NLI)
**Input:** chunk text from chunk parquets + chunk_ids (all 737k chunks)
**Run on:** Kaggle GPU (T4 x2) — most expensive step; NLI runs 3 forward passes per chunk

Hypotheses to score per chunk:
```
"The author supports National Service"
"The author is critical of National Service"
"The author feels positively about serving in the military"
```

**Output:** `chunk_commitment.parquet` — `chunk_id`, `commit_support`, `commit_critical`, `commit_positive`

Build a new Kaggle notebook: `notebooks/kaggle_commitment_v1.ipynb`
Use `kaggle_xlm_base_v1.ipynb` as the structural template.
Key differences from sentiment run:
- `BATCH_SIZE=64` (NLI is much heavier than classification — 3 passes per chunk)
- `CHECKPOINT_N=25_000` (smaller to guard against session timeouts)
- Model: `facebook/bart-large-mnli` — classification labels will be `ENTAILMENT`, `NEUTRAL`, `CONTRADICTION`
- For each hypothesis, the `ENTAILMENT` score is the commitment score
- Run each hypothesis as a separate pipeline call OR batch all 3 together as NLI pairs

Input dataset on Kaggle: `ns-sentiment-chunks-v3` (same as Stage 5a — no new upload needed).

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

## Immediate Next Actions (two parallel tracks)

### Track A — Stage 5a (sentiment model — BLOCKED on decision)

**Decision required:** Fine-tune SingBERT or proceed with RoBERTa-base at 65.9%?

**Option A1 — Fine-tune SingBERT (recommended for enterprise presentation)**
   - Base: `zanelim/singbert-large-sg` (pre-trained on r/singapore + HardwareZone)
   - Data needed: ~600 labeled examples total (have 98 reliable blind labels)
   - User effort: ~500 more blind annotations via `python -m src.features.blind_annotator`
   - Then: I build `notebooks/kaggle_finetune_singbert_v1.ipynb` + training pipeline
   - Expected accuracy: 72–76% corpus-weighted
   - Then: re-run full 737k corpus → new `chunk_sentiment.parquet`

**Option A2 — Proceed with RoBERTa-base (faster, lower accuracy)**
   - Build `notebooks/kaggle_roberta_v1.ipynb` (swap model name in xlm_base_v1, ~30 min)
   - Run on Kaggle T4 x2 (~2 hrs)
   - Output: new `chunk_sentiment.parquet` (RoBERTa scores, ~23% neg vs XLM's 40%)
   - Documented accuracy: 65.9% blind corpus-weighted

⚠️ Do NOT proceed to Stage 6 until `chunk_sentiment.parquet` is replaced.

---

### Track B — Stage 5b (commitment scoring — INDEPENDENT, ready to start)

**Stage 5b does NOT depend on Stage 5a.** Can be built and run now in a separate session.

   - Notebook: `notebooks/kaggle_commitment_v1.ipynb` (not yet created)
   - Model: `facebook/bart-large-mnli` (zero-shot NLI)
   - Input: `submissions_chunks.parquet` + `comments_chunks.parquet` (chunk text only)
   - Hypotheses per chunk (3 NLI passes each):
     - "The author supports National Service"
     - "The author is critical of National Service"
     - "The author feels positively about serving in the military"
   - Output: `chunk_commitment.parquet` — `chunk_id`, `commit_support`, `commit_critical`, `commit_positive`
   - BATCH_SIZE=64 (NLI is 3× heavier than classification)
   - CHECKPOINT_N=25_000
   - Template: `notebooks/kaggle_xlm_base_v1.ipynb`
   - Input dataset on Kaggle: `ns-sentiment-chunks-v3` (already uploaded — no new upload needed)
   - Expected runtime: ~4–6 hrs on T4 x2 (3 forward passes × 737k chunks)

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
