# NS Sentinel — Pipeline Documentation

This document describes every stage of the NS Sentinel NLP pipeline in detail: what each stage does, how it works, what it produces, and the key design decisions behind it.

---

## Table of contents

1. [Architecture overview](#architecture-overview)
2. [Stage 1 — Data ingestion](#stage-1--data-ingestion)
3. [Stage 2 — Cleaning & NS relevance filtering](#stage-2--cleaning--ns-relevance-filtering)
4. [Stage 3 — Semantic chunking](#stage-3--semantic-chunking)
5. [Stage 4 — Topic modelling](#stage-4--topic-modelling)
6. [Stage 5a — Sentiment classification (SingBERT)](#stage-5a--sentiment-classification-singbert)
7. [Stage 5b — Commitment classification (4-stage cascade)](#stage-5b--commitment-classification-4-stage-cascade)
8. [Stage 6 — Document-level aggregation](#stage-6--document-level-aggregation)
9. [Stage 7 — Divergence analysis](#stage-7--divergence-analysis)
10. [Stage 8 — Temporal aggregation](#stage-8--temporal-aggregation)
11. [Stage 9 — Dashboard](#stage-9--dashboard)
12. [RAG chatbot pipeline](#rag-chatbot-pipeline)
13. [Data flow diagram](#data-flow-diagram)

---

## Architecture overview

The pipeline processes Reddit data through 9 sequential stages, from raw ZST dumps to an interactive Streamlit dashboard. Each stage reads from the previous stage's outputs and writes parquets or CSVs to `data/processed/new/`.

**Core principle**: chunk-level analysis. Every Reddit document (post or comment) is split into 1–3 semantic text chunks. All NLP models (sentiment, commitment, topic) operate at the chunk level. Document-level and temporal scores are computed by aggregating chunk scores in later stages.

**Compute model**: Stages 1–3 and 6–8 run locally (CPU). Stages 4–5 require GPU and run on Kaggle T4 notebooks. The dashboard (Stage 9) is a Streamlit app that reads pre-computed parquets.

---

## Stage 1 — Data ingestion

**Script**: `src/data/loader.py`

**Input**: ZST-compressed Reddit dump files from Pushshift archives in `data/raw/`:
- `singapore_comments.zst` (1.2 GB)
- `singapore_submissions.zst` (109 MB)
- `askSingapore_comments.zst` (492 MB)
- `askSingapore_submissions.zst` (56 MB)
- `NationalServiceSG_comments.zst` (51 MB)
- `NationalServiceSG_submissions.zst` (11 MB)

**Output**: `data/interim/comments_raw.parquet` (2 GB), `data/interim/submissions_raw.parquet` (125 MB)

**Process**:
1. Stream-decompress each ZST file using the `zstandard` library
2. Parse each line as JSON, extract relevant fields (id, body/selftext, author, score, created_utc, subreddit, etc.)
3. Filter to target subreddits: r/singapore, r/askSingapore, r/NationalServiceSG
4. Concatenate all subreddit data into two dataframes (submissions, comments)
5. Write to parquet with pyarrow compression

**Key fields extracted**:
- Submissions: id, title, selftext, score, upvote_ratio, num_comments, created_utc, link_flair_text, author, subreddit
- Comments: id, body, author, score, created_utc, subreddit, link_id, parent_id

---

## Stage 2 — Cleaning & NS relevance filtering

**Script**: `src/data/cleaner.py`

**Input**: Interim parquets from Stage 1

**Output**: `data/processed/submissions_clean.parquet`, `data/processed/comments_clean.parquet`

**Process**:

1. **Bot removal**: Filter out posts from known bots (AutoModerator, sneakpeek_bot, microtechanalysis)

2. **Text cleaning**:
   - Remove Reddit markdown artifacts (URLs, quote prefixes, code blocks)
   - Normalise whitespace and Unicode
   - Strip deleted/removed content markers (`[deleted]`, `[removed]`)
   - Combine submission title + selftext into a single `text` column

3. **NS relevance filtering**: A keyword-based filter identifies NS-relevant content. The keyword list includes:
   - Core terms: "national service", "nsman", "conscription"
   - Policy terms: "ns disruption", "ns deferment"
   - Milestones: "enlistment", "book in", "book out", "operationally ready", "ord"
   - Vocations & units: specific military unit names and vocations
   - The filter is intentionally broad — downstream chunking applies a tighter semantic filter

4. **Deduplication**: Remove exact-text duplicates within each subreddit

5. **Date parsing**: Convert `created_utc` (Unix timestamp) to datetime, extract year and month

**Design decisions**:
- r/NationalServiceSG content is treated as 100% NS-relevant (no keyword filter needed)
- For r/singapore and r/askSingapore, the keyword filter runs against the full text including title
- Comments inherit relevance from their parent post — if the post is NS-relevant, all its comments are included

---

## Stage 3 — Semantic chunking

**Script**: `src/features/chunker.py`

**Input**: Cleaned parquets from Stage 2

**Output**: `data/processed/new/submissions_chunks.parquet` (460 MB), `data/processed/new/comments_chunks.parquet` (2.9 GB)

**Process**:

1. **Sentence splitting**: Use spaCy (`en_core_web_sm`) to split each document into sentences

2. **Embedding**: Embed all sentences using `sentence-transformers/all-mpnet-base-v2` (768-dim)

3. **Similarity-based grouping**: Walk through sentences sequentially. Compute cosine similarity between consecutive sentence embeddings. When similarity drops below 0.5, start a new chunk. This groups semantically coherent sentences together.

4. **Size constraints**:
   - Minimum chunk: 2 sentences (configurable via `CHUNK_MIN_SENTENCES`)
   - Maximum chunk: 6 sentences (configurable via `CHUNK_MAX_SENTENCES`)
   - Minimum characters: 200. Trailing fragments below this are merged into the previous chunk
   - Whole documents under 200 chars are kept as a single chunk

5. **NS relevance re-filter**: Each chunk is independently checked for NS relevance using a tighter keyword + semantic filter. Non-relevant chunks are discarded. This is important because a long r/singapore post might discuss NS in only one paragraph — only that chunk is kept.

6. **Context prefix embedding**: Final chunks are re-embedded with a context prefix for downstream topic modelling. This improves topic assignment by providing disambiguation context.

7. **Metadata**: Each chunk row includes doc_id, chunk_id, text, embedding, subreddit, created_utc, score (upvotes), doc_type (submission/comment), post_id (for comments: which submission they reply to), and log_weight (log1p of upvote score, floored at 1.0).

**Output schema**: ~737K chunks across both parquets, with 768-dim embedding vectors stored inline.

**Design decisions**:
- All documents go through the full sentence-split + similarity pipeline regardless of length. An earlier version had a shortcut for short documents (single chunk, single embedding), but this caused mixed-topic short documents to receive a single topic label, losing the second topic's signal.
- The similarity threshold of 0.5 was chosen empirically to balance between over-splitting (each sentence is its own chunk) and under-splitting (entire document is one chunk).

---

## Stage 4 — Topic modelling

**Notebook**: `notebooks/kaggle_topic_model_v3.ipynb` (Kaggle T4 GPU)

**Input**: Chunk embeddings from Stage 3

**Output**:
- `models/new/bertopic_fine/` — fine-grained BERTopic model (359 topics)
- `models/new/bertopic_coarse/` — coarse BERTopic model (17 macro topics)
- `data/processed/new/chunk_topics.parquet` — topic assignments per chunk
- `data/processed/new/topic_keywords_fine.csv`, `topic_keywords_coarse.csv`
- `data/processed/new/hierarchical_topics_manual.parquet` — manual topic taxonomy

**Process**:

1. **UMAP dimensionality reduction**: Reduce 768-dim embeddings to 5 dimensions for clustering

2. **HDBSCAN clustering**: Cluster the reduced embeddings. BERTopic uses HDBSCAN's soft clustering to assign topic probabilities per chunk.

3. **c-TF-IDF topic representations**: Compute class-based TF-IDF to extract representative keywords per topic

4. **Manual taxonomy**: The 359 leaf topics are manually organised into a 4-level hierarchy:
   - 359 leaf topics → 112 cluster topics → 52 sub topics → 17 macro topics
   - This taxonomy is defined in `src/models/topic_labels.py` (the `TOPIC_LABELS` dictionary)
   - The hierarchy enables drill-down in the dashboard from "NS Life & Culture" (macro) → "NS Culture & Humour" (sub) → "NS Identity & Reflections" (cluster) → individual fine topics

5. **Topic assignment**: Each chunk receives a `topic_id_fine` (0–358) and `topic_prob_fine` (probability). Chunks that don't fit any cluster are assigned topic_id_fine = -1 (outlier).

**The 17 macro topics**:
NS Life & Culture, BMT & Training, NS Policy & Society, Enlistment & Pre-NS, Admin & Logistics, Physical Fitness & IPPT, Pay & Benefits, ORD & Post-NS, Relationships & Social Life, Medical & Health, Vocations & Units, Reservist & ICT, Gear & Equipment, Mental Health, Food & Cookhouse, Technology & Digital NS, NS & Education

---

## Stage 5a — Sentiment classification (SingBERT)

**Notebooks**: `notebooks/kaggle_finetune_singbert_v1.ipynb` (training), `notebooks/kaggle_infer_singbert_v1.ipynb` (inference)

**Model**: `zanelim/singbert-large-sg` fine-tuned for 3-class sentiment (positive / neutral / negative)

**Input**: All 737K chunks from Stage 3

**Output**: `data/processed/new/chunk_sentiment.parquet` — columns `sent_neg`, `sent_neu`, `sent_pos` (softmax probabilities per chunk)

**Training data**: `data/processed/new/singbert_train.csv` — 3,289 rows of human-annotated + LLM-annotated chunks, class-balanced

**Training process**:
1. Base model: `zanelim/singbert-large-sg` — a RoBERTa-large model pre-trained on Singapore English text (Singlish, code-switching, local slang)
2. Fine-tune with a classification head for 3 classes
3. Train on Kaggle T4 GPU for ~3 epochs with learning rate warmup
4. The model is saved as `models/new/singbert_v7/`

**Why SingBERT?**: Standard English sentiment models (VADER, TextBlob, general BERT) perform poorly on Singaporean English due to code-switching (English + Mandarin + Malay + Hokkien), Singlish grammar, and culturally specific expressions. SingBERT was pre-trained on Singapore web text and handles these phenomena natively.

**Evaluation** (on 217-row human gold set):

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.82 | 0.90 | 0.86 |
| Neutral | 0.83 | 0.82 | 0.83 |
| Positive | 0.88 | 0.78 | 0.83 |
| **Macro avg** | **0.84** | **0.83** | **0.78** |

**Earlier approaches tried and discarded**:
- VADER lexicon: κ = 0.31 — too poor for code-switched text
- Generic BERT (bert-base-uncased): κ = 0.52 — misses Singlish
- GPT-4.1-mini zero-shot: κ = 0.68 — good but expensive at 737K scale

---

## Stage 5b — Commitment classification (4-stage cascade)

**Notebooks**: `notebooks/cascade/cascade_stage1a_buyin_relevance.ipynb`, `cascade_stage1b_stance_relevance.ipynb`, `cascade_stage2a_committed_uncommitted.ipynb`, `cascade_stage2b_supportive_critical.ipynb`

**Inference**: `notebooks/cascade/cascade_inference_v2.ipynb`

**Input**: All 737K chunks from Stage 3

**Output**: `data/processed/new/chunk_commitment_cascade.parquet`

**Architecture — two independent axes**:

The commitment model classifies chunks along two orthogonal axes:

1. **Buyin axis** (personal investment in NS):
   - Stage 1a: Is this chunk relevant to buyin? (binary: relevant / not relevant)
   - Stage 2a: If relevant → committed or uncommitted?

2. **Stance axis** (opinion on NS policy):
   - Stage 1b: Does this chunk express a stance? (binary: has stance / no stance)
   - Stage 2b: If has stance → supportive or critical?

Chunks that are not relevant to an axis default to **neutral** on that axis. This means a chunk can be "committed but critical" (personally invested in NS but disagrees with policy) or "uncommitted but supportive" (not personally invested but thinks NS is good policy).

**Why two axes?**: A single committed/uncommitted dimension conflates personal investment with policy opinion. Someone who served NS dutifully for 10 years (committed) might still criticise NS policy (critical). The two-axis model separates these distinct signals:
- **Buyin** is used for the Commitment Decline analysis (are people becoming less personally invested over time?)
- **Stance** is used for Topic Analysis (which topics draw the most criticism?)

**Training data**: Each stage is trained on a mix of:
- Human annotations (`commitment_manual_annotations.csv`, `commitment_annotation.csv`)
- LLM-generated silver labels (GPT-4.1-mini with chain-of-thought prompting)
- FAISS-enriched examples (semantically similar chunks to existing gold labels)

Training data for each stage is in `data/processed/new/cascade/stage{1a,1b,2a,2b}_train.csv`.

**Model**: Each of the 4 stages is a fine-tuned `zanelim/singbert-large-sg` classifier. All 4 models run on Kaggle T4 GPUs.

**Cascade flow**:
```
Input chunk
    ├── Stage 1a: buyin_relevant? ──► No → buyin_neutral
    │                                Yes ──► Stage 2a: committed / uncommitted
    │
    └── Stage 1b: has_stance? ──► No → stance_neutral
                                 Yes ──► Stage 2b: supportive / critical
```

**Evaluation** (on 727-row human gold set):

| Stage | Task | Accuracy | F1 |
|-------|------|----------|----|
| 1a | Buyin relevance | 96.2% | 0.95 |
| 1b | Stance relevance | 95.1% | 0.93 |
| 2a | Committed vs Uncommitted | 75.8% | 0.714 |
| 2b | Supportive vs Critical | 80.3% | 0.784 |

---

## Stage 6 — Document-level aggregation

**Script**: `src/analysis/stage6_doc_aggregation.py`

**Input**: chunk_sentiment.parquet, chunk_commitment_cascade.parquet, chunk_topics.parquet, submissions_chunks.parquet, comments_chunks.parquet

**Output**: `data/processed/new/doc_sentiment.parquet` — one row per document with aggregated scores

**Process**:

1. **Join**: Merge chunk-level sentiment, commitment, and topic labels onto each chunk's metadata

2. **Weighted aggregation**: For each document, compute weighted averages of chunk scores:
   - Weight = `topic_prob_fine × log_weight` (topic confidence × upvote weight)
   - `log_weight` is `log1p(upvote_score)`, floored at 1.0 so zero-upvote chunks still contribute

3. **Topic assignment**: Each document's topic is the topic with the highest total weight across its chunks:
   - `topic_macro = argmax(Σ(topic_prob_fine × log_weight))` per topic per doc
   - Outlier chunks (topic_id_fine = -1) are included in score aggregation but excluded from topic assignment

4. **Sentiment columns**: `sent_neg`, `sent_neu`, `sent_pos` (mean across chunks), plus `sentiment_label` (argmax)

5. **Commitment columns**: `buyin_committed`, `buyin_uncommitted`, `buyin_neutral`, `stance_supportive`, `stance_critical`, `stance_neutral` (proportions of chunks in each class)

---

## Stage 7 — Divergence analysis

**Script**: `src/analysis/stage7_divergence_v2.py`

**Input**: chunk_sentiment.parquet, submissions_chunks.parquet, comments_chunks.parquet, chunk_topics.parquet

**Output**: `data/processed/new/doc_divergence_v2.parquet`, `data/processed/new/topic_discourse_intensity.parquet`

**What it measures**: How often does the comment section's mood differ from the original post's mood? This identifies threads where the community pushes back against or amplifies the original poster's sentiment.

**Key metric — upvote-weighted divergence**:
```
upvote_weighted_div = Σ(|neg_i − pos_i| × log(1 + upvotes_i)) / Σ(log(1 + upvotes_i))
```
Computed only over opinionated chunks (where `max(sent_neg, sent_pos) > 0.5`). This filters out the ~89% of neutral chunks that would otherwise dominate the signal.

**Additional metrics per thread**:
- `within_thread_variance`: Variance of sent_neg across comment chunks (disagreement signal)
- `opinion_chunk_pct`: Proportion of chunks in the thread that are opinionated
- `total_upvotes`: Sum of upvotes across post + all comments
- `tone_shift_matrix`: Categorisation of post sentiment vs comment sentiment (pos→neg, neg→pos, etc.)

**Discourse intensity per topic**: `topic_discourse_intensity.parquet` aggregates `Σ(upvotes × |neg − pos|)` per macro topic, identifying which topics generate the most heated discussion.

---

## Stage 8 — Temporal aggregation

**Script**: `src/analysis/stage8_temporal.py`

**Input**: doc_sentiment.parquet, chunk_commitment_cascade.parquet, chunk_metadata.parquet

**Output**:
- `data/processed/new/temporal_sentiment.parquet` — monthly × subreddit × topic_macro
- `data/processed/new/temporal_sentiment_overall.parquet` — monthly × subreddit (no topic)
- `data/processed/new/temporal_commitment.parquet` — monthly commitment metrics

**Process**:

1. Group documents by (year_month, subreddit, topic_macro)
2. Compute monthly means of sentiment and commitment scores
3. Flag low-volume months (doc_count < 30) to prevent noisy signal
4. For commitment, compute upvote-weighted metrics:
   ```
   wtd_committed = Σ(log(1 + score_i) × is_committed_i) / Σ(log(1 + score_i))
   ```

**Temporal columns** (commitment):
- Buyin: `pct_buyin_committed`, `pct_buyin_uncommitted`, `net_buyin` (committed − uncommitted)
- Stance: `pct_stance_supportive`, `pct_stance_critical`, `net_stance` (supportive − critical)
- Combined: `net_disposition` (overall positive − negative signal)

---

## Stage 9 — Dashboard

**Script**: `app/dashboard.py`

**Input**: All parquets from Stages 6–8, plus topic models, keywords, and RAG assets

The dashboard is documented separately in [docs/dashboard_guide.md](dashboard_guide.md).

---

## RAG chatbot pipeline

The Sentinel Bot (dashboard page 7) uses Retrieval-Augmented Generation to answer questions about NS sentiment.

### Knowledge base build

Four scripts in `scripts/rag/` build the RAG knowledge base:

1. **`build_index.py`** — FAISS index
   - Loads chunk embeddings from submissions_chunks + comments_chunks parquets
   - Builds a flat FAISS index over all 737K chunk embeddings (768-dim)
   - Saves `chunk_faiss.index` (2.1 GB) and `chunk_metadata.parquet` (89 MB)

2. **`build_topic_digests.py`** — Topic summaries
   - For each of the 17 macro topics, generates a statistical digest:
     - Total chunk count, sentiment breakdown, top keywords
     - Most negative / most positive sub-topics
     - Temporal trend summary
   - Saves `rag_topic_digests.json`

3. **`build_temporal_narratives.py`** — Timeline narratives
   - Pre-computes natural-language summaries of sentiment trends for each topic × year
   - Identifies inflection points, peaks, troughs
   - Saves `rag_temporal_narratives.json`

4. **`build_fact_table.py`** — Quantitative facts
   - Pre-computes common statistics (% negative, mean sentiment, doc counts) for every (topic, subreddit, year) combination
   - Saves `rag_fact_table.parquet` — enables instant quantitative answers without scanning full data

### Query processing

When a user asks a question, the RAG pipeline follows this flow:

1. **Query routing** (`src/rag/query_router.py`):
   - Classifies the query as quantitative ("what % of...", "how many...") or qualitative ("what do people say about...", "why do people feel...")
   - Extracts topic, subreddit, and time period filters from the query
   - Routes quantitative queries directly to the fact table; qualitative queries to FAISS retrieval

2. **Retrieval** (`src/rag/retriever.py`):
   - Embeds the query using the same sentence-transformer model
   - Retrieves top-50 candidates from FAISS
   - Reranks by: cosine similarity + upvote boost (`UPVOTE_ALPHA = 0.3 × log(1 + upvotes)`)
   - Filters below `MIN_COSINE_SIM = 0.25`
   - Returns top-10 chunks

3. **Spike detection** (`src/rag/spike_detector.py`):
   - Checks if the query period coincides with any known NS events (from `ns_events.json`)
   - Identifies sentiment spikes in the temporal data
   - Adds event context to the prompt

4. **Context assembly** (`src/rag/context_assembler.py`):
   - Combines: topic digest + temporal narrative + relevant events + retrieved chunks + fact table stats
   - Budgets total context to ~10,000 chars (~2,500 tokens)
   - Prioritises statistical context over raw chunks

5. **Synthesis** (`src/rag/synthesizer.py`):
   - Sends assembled context + user query to Anthropic Claude Haiku
   - System prompt instructs the model to cite statistics, reference specific time periods, and ground answers in the retrieved chunks
   - Returns a concise, data-backed answer

---

## Data flow diagram

```
ZST dumps (data/raw/)
    │
    ▼  Stage 1: loader.py
Interim parquets (data/interim/)
    │
    ▼  Stage 2: cleaner.py
Clean parquets (data/processed/)
    │
    ▼  Stage 3: chunker.py
Chunk parquets ──────────────────────────┐
    │                                    │
    ├──► Stage 4: BERTopic ──► chunk_topics.parquet
    │                                    │
    ├──► Stage 5a: SingBERT ──► chunk_sentiment.parquet
    │                                    │
    ├──► Stage 5b: Cascade ──► chunk_commitment_cascade.parquet
    │                                    │
    ▼                                    ▼
Stage 6: doc aggregation ──► doc_sentiment.parquet
    │                        doc_commitment.parquet
    │
    ├──► Stage 7: divergence ──► doc_divergence_v2.parquet
    │                            topic_discourse_intensity.parquet
    │
    ├──► Stage 8: temporal ──► temporal_sentiment.parquet
    │                          temporal_commitment.parquet
    │
    └──► RAG build ──► chunk_faiss.index
                       rag_topic_digests.json
                       rag_temporal_narratives.json
                       rag_fact_table.parquet
                           │
                           ▼
                    Stage 9: Dashboard
                    (app/dashboard.py)
```
