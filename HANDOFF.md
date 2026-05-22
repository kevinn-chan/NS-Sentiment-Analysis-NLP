# NS Sentiment — Project Handoff (updated 2026-05-22)

## Goal

Quantify Singaporean public sentiment and commitment to National Service (NS) defence
over time, using Reddit data from three subreddits:
- r/singapore
- r/askSingapore
- r/NationalServiceSG

The end product is a Streamlit dashboard with Seaborn visualisations showing:
- Topic distribution across NS discourse (BERTopic, hierarchical)
- Sentiment trends over time (monthly rollups, score-weighted)
- Commitment-to-defence scoring (zero-shot + lexicon hybrid)
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
| 5a | Sentiment classification (3-tier hybrid) | ✅ Done |
| 5a-verify | Human annotation accuracy check | ✅ Done |
| 5a-xlm | XLM base model evaluation + corpus-weighted comparison | ✅ Done |
| 5a-spot | Manual spot-check (100 chunks, 78.0% accuracy) | ✅ Done |
| **5b** | **Commitment scoring (zero-shot NLI)** | ⏳ **NEXT STEP** |
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
| `data/processed/new/chunk_sentiment.parquet` | 737,274 | ★ `chunk_id`, `sent_neg`, `sent_neu`, `sent_pos` — XLM-patched |
| `data/processed/new/chunk_sentiment_lexicon.parquet` | 738,819 | `chunk_id`, `sent_lexicon_compound` (VADER+NS lexicon) |
| `data/processed/new/annotation_sample.parquet` | 197 | Stratified sample for human labelling |
| `data/processed/new/annotations.csv` | in progress | Human labels — run `annotator.py --report` when done |
| `data/processed/new/sentiment_audit.parquet` | 500 | llama3.2:3b gold labels vs roberta (see caveats below) |
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
│   │   ├── lexicon_scorer.py            # ★ Tier 2: VADER + NS/Singlish lexicon
│   │   ├── patch_sentiment_xlm.py       # ★ Tier 1b: XLM patch for Singlish chunks
│   │   ├── sentiment_audit.py           # Tier 3: Ollama audit (llama3.2:3b)
│   │   └── annotator.py                 # ★ Human annotation CLI (resume-safe)
│   └── models/
│       ├── topic_model.py
│       ├── topic_labels.py              # ★ Full 4-layer taxonomy (359 topics)
│       ├── build_manual_dendrogram.py
│       └── plot_manual_dendrogram.py
├── notebooks/
│   ├── kaggle_sentiment_v1.ipynb        # ★ Stage 5a Kaggle run (already executed)
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

### Final model
**Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment` — run on all 737,274 chunks
**Notebook:** `notebooks/kaggle_xlm_base_v1.ipynb` (Kaggle T4 x2)
**Output:** `chunk_sentiment.parquet` — `chunk_id`, `sent_neg`, `sent_neu`, `sent_pos`

Final corpus distribution: negative 40.3% / neutral 47.6% / positive 12.2%

### Why XLM over the initial RoBERTa hybrid

Initial design used `cardiffnlp/twitter-roberta-base-sentiment-latest` as base with
an XLM patch on Singlish chunks. Head-to-head evaluation on 197 human-annotated chunks
revealed the annotation sample over-represented Singlish by **15.2x** (71.6% of sample
vs 4.7% of real corpus), making RoBERTa appear to win overall (75.6% vs 69.5%).

Corpus-weighted accuracy (95.3% English / 4.7% Singlish):
- RoBERTa base: 76.7%
- **XLM base: 81.3%** ← winner

XLM is stronger on English-dominant text (82.1% vs 76.8%), which is 95.3% of the corpus.
The hybrid was the worst of both worlds. XLM as sole model is simpler and more accurate.

### What was evaluated and dropped

- **Fine-tuned transformer:** requires 500+ labelled examples — deferred, not permanently dropped
- **Zero-shot NLI (bart-large-mnli):** wrong tool for sentiment — retained for Stage 5b
- **Lexicon only:** 47.7% human agreement — supplementary signal, not classifier
- **gemma3:1b Ollama:** 92.8% negative — unusable
- **llama3.2:3b Ollama:** misleading audit metric; human annotation is the reliable path
- **RoBERTa+XLM patch hybrid:** annotation sample bias masked XLM's corpus-level superiority

### Validation results

| Method | Accuracy |
|---|---|
| Human annotation (197 chunks, sample-weighted) | 69.5% |
| Human annotation (197 chunks, corpus-weighted) | **81.3%** |
| Manual spot-check (100 chunks) | **78.0%** |

Spot-check by stratum: high_neu=100%, high_pos=95.5%, high_neg=81.8%,
singlish=65.0%, low_conf=44.4%

Known failure modes:
- Over-predicts negative for factual NS questions ("Forced to downpes due to rash problem?")
- Misses Singlish sentiment cues in both directions ("sian max" called neutral, "lepak" called neutral)
- Low-confidence chunks (18.4%) are genuinely ambiguous — acceptable at aggregation level

### Validation files
- `data/processed/new/annotations.csv` — 197 human labels
- `data/processed/new/annotation_sample.parquet` — stratified annotation sample
- `data/processed/new/spot_check.csv` — 100-chunk manual spot-check results

### Lexicon cross-check (supplementary)
**Script:** `src/features/lexicon_scorer.py`
**Output:** `chunk_sentiment_lexicon.parquet` — `sent_lexicon_compound` [-1, +1]
VADER + 60-entry custom NS/Singlish lexicon. 65.5% directional agreement with XLM.
Use as interpretability signal, not primary classifier.

---

## Stage 5b — Commitment Scoring (NEXT)

**Model:** `facebook/bart-large-mnli` (zero-shot NLI)
**Input:** `chunk_sentiment.parquet` chunk_ids + text from chunk parquets
**Run on:** Kaggle GPU (T4 x2) — most expensive step, multiple forward passes per chunk

Hypotheses to score per chunk:
```
"The author supports National Service"
"The author is critical of National Service"
"The author feels positively about serving in the military"
```

**Output:** `chunk_commitment.parquet` — `chunk_id`, `commit_support`, `commit_critical`, `commit_positive`

Build a new Kaggle notebook following the same pattern as `kaggle_sentiment_v1.ipynb`.
Use BATCH_SIZE=64 (smaller than sentiment — NLI runs 3 passes per chunk).
Checkpoint every 25k rows.

---

## What Worked

- **Kaggle "Save & Run All"** (committed mode) — only reliable way to run long jobs
- **Batched cosine similarity** for outlier rescue — avoids OOM
- **UMAP checkpoint** (monkey-patch saves `umap_embeddings.npy`) — saves 2+ hours if HDBSCAN fails
- **Noise removal via `update_topics()`** — cleanly removes noise and recalculates c-TF-IDF
- **Manual taxonomy top-down** — macro → sub → sub_sub semantically, then assign fine topics
- **Dendrogram via scipy `link_color_func`** — only reliable way to get macro-coloured dendrograms
- **`top_k=None` in HuggingFace pipeline** — correct replacement for deprecated `return_all_scores=True`
- **XLM-RoBERTa as sole base model** — 81.3% corpus-weighted accuracy vs 76.7% for RoBERTa; annotation sample bias (15.2x Singlish over-representation) initially masked this
- **Corpus-weighted accuracy** — always weight annotation results by actual corpus proportions, not stratified sample proportions
- **VADER + custom NS lexicon** — fast interpretable signal; 60 entries covering key NS/Singlish terms
- **Ollama `format: "json"`** — enforces valid JSON output from local LLMs reliably

---

## What Didn't Work / Gotchas

- **`return_all_scores=True` in transformers pipeline** — deprecated; silently returns single score dict instead of list; use `top_k=None`
- **gemma3:1b for sentiment labelling** — 92.8% of labels were "negative"; model too small and miscalibrated for this task
- **llama3.2:3b audit agreement % as primary metric** — 39% overall agreement sounds bad but sample was deliberately 50% Singlish-heavy; actual corpus-weighted accuracy is ~73–75%. Use human annotation, not LLM agreement, as the truth signal.
- **Singlish density via exact whitespace token match** — undercounts; "sian." (with period) doesn't match "sian". Use `re.findall(r'\b\w+\b', text)` instead.
- **BERTopic `visualize_hierarchy` with filtered `topics=` list** — crashes; use scipy directly
- **`color_threshold` in `visualize_hierarchy`** — doesn't apply colours with custom hierarchical_topics; all traces one colour
- **`set_topic_labels()` with partial topic coverage** — crashes if max topic ID > len(custom_labels_)
- **Bottom-up dendrogram clustering** — cutting at distance thresholds gave meaningless groupings
- **MPS (Apple GPU) for embeddings** — OOM at batch_size=256; use CPU
- **Local HDBSCAN with core_dist_n_jobs=-1** — spawns 9 workers on 809k points → OOM; fix: core_dist_n_jobs=1
- **`calculate_probabilities=True` in BERTopic** — OOM on Kaggle
- **Hierarchical `reduce_topics()` return value** — newer BERTopic returns model object, not (topics, probs) tuple

---

## Known Issues (carry-forward)

### datetime precision mismatch
- `submissions_chunks`: `datetime64[ms, UTC]`
- `comments_chunks`: `datetime64[ns, UTC]`
- Normalise both to `ms` before Stage 8 temporal aggregation

### chunk_topics.parquet missing taxonomy columns
- No `topic_macro` / `topic_sub` / `topic_sub_sub` columns yet
- Join from `topic_labels.py` before Stage 6:
  ```python
  chunk_topics['topic_macro'] = chunk_topics['topic_id_fine'].map(
      lambda t: TOPIC_LABELS.get(t, {}).get('macro')
  )
  ```

### chunk_sentiment row count vs chunk parquets
- `chunk_sentiment.parquet`: 737,274 rows (unique chunk_ids)
- Combined chunk parquets: 738,819 rows (1,545 duplicate chunk_ids from batch boundary issue)
- Coverage is 100% of unique chunks — the 1,545 gap is duplicates in source, not missing scores

### hierarchical_topics.parquet is stale
- Auto-generated BERTopic dendrogram was pre-noise-removal (contains t71, t117 etc.)
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

## Immediate Next Actions (in order)

1. **Finish human annotation** — `python -m src.features.annotator` (~1.5 hrs remaining)
   Then run `python -m src.features.annotator --report` to get accuracy verdict.

2. **Build Stage 5b Kaggle notebook** — commitment scoring via `facebook/bart-large-mnli`
   Follow `kaggle_sentiment_v1.ipynb` as template. BATCH_SIZE=64, checkpoint every 25k.
   Input datasets: `ns-sentiment-chunks-v3` (for text) — no new dataset needed.

3. **Stage 6 — document-level aggregation**
   Join: `chunk_sentiment` + `chunk_commitment` + `chunk_topics` + chunk parquets
   Group by `doc_id`, aggregate weighted by `log_weight` (already in chunk parquets).
   Add `topic_macro` column from `TOPIC_LABELS` dict.
   Output: `doc_sentiment.parquet`

4. **Stage 7 — divergence score**
   For each `post_id`, compare submission chunk scores vs comment chunk scores.
   Output divergence metric per post.

5. **Stage 8 — temporal aggregation**
   Normalise datetime to `ms` first (see known issues above).
   Monthly rollup grouped by subreddit + topic_macro.
