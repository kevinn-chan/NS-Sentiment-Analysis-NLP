# NS Sentinel — Pipeline Documentation

This document describes every stage of the NS Sentinel NLP pipeline: what each stage does, how it works internally, what it produces, the key design decisions, known limitations, and concrete improvement opportunities. Written for a data scientist picking up this project cold.

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
14. [File manifest](#file-manifest)

---

## Architecture overview

The pipeline processes Reddit data through 9 sequential stages, from raw ZST dumps to an interactive Streamlit dashboard. Each stage reads from the previous stage's outputs and writes parquets or CSVs to `data/processed/new/`.

**Core principle**: chunk-level analysis. Every Reddit document (post or comment) is split into 1–3 semantic text chunks. All NLP models (sentiment, commitment, topic) operate at the chunk level. Document-level and temporal scores are computed by aggregating chunk scores in later stages.

**Compute model**: Stages 1–3 and 6–8 run locally (CPU). Stages 4–5 require GPU and run on Kaggle T4 notebooks. The dashboard (Stage 9) is a Streamlit app that reads pre-computed parquets.

**Global configuration**: `config.py` in the project root defines all shared paths and constants:

| Constant | Value | Used by |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | Stages 3, RAG |
| `CHUNK_MIN_SENTENCES` | 2 | Stage 3 |
| `CHUNK_MAX_SENTENCES` | 6 | Stage 3 |
| `SIMILARITY_THRESHOLD` | 0.5 | Stage 3 |
| `DATA_PROCESSED` | `data/processed/` | All stages |

---

## Stage 1 — Data ingestion

**Script**: `src/data/loader.py`

**Input**: ZST-compressed Reddit dump files from Pushshift archives in `data/raw/`:

| File | Size | Subreddit |
|---|---|---|
| `singapore_comments.zst` | 1.2 GB | r/singapore |
| `singapore_submissions.zst` | 109 MB | r/singapore |
| `askSingapore_comments.zst` | 492 MB | r/askSingapore |
| `askSingapore_submissions.zst` | 56 MB | r/askSingapore |
| `NationalServiceSG_comments.zst` | 51 MB | r/NationalServiceSG |
| `NationalServiceSG_submissions.zst` | 11 MB | r/NationalServiceSG |

**Output**: `data/interim/comments_raw.parquet` (2 GB), `data/interim/submissions_raw.parquet` (125 MB)

**Process**:
1. Stream-decompress each ZST file using the `zstandard` library
2. Parse each line as JSON, extract relevant fields
3. Filter to target subreddits: r/singapore, r/askSingapore, r/NationalServiceSG
4. Concatenate all subreddit data into two dataframes (submissions, comments)
5. Write to parquet with pyarrow compression

**Key fields extracted**:
- Submissions: `id`, `title`, `selftext`, `score`, `upvote_ratio`, `num_comments`, `created_utc`, `link_flair_text`, `author`, `subreddit`
- Comments: `id`, `body`, `author`, `score`, `created_utc`, `subreddit`, `link_id`, `parent_id`

**Gotcha**: The ZST files are full subreddit dumps from Pushshift — they contain *all* posts, not just NS-related ones. NS filtering happens in Stage 2. The raw `singapore_comments.zst` alone has millions of rows covering every topic ever posted in r/singapore.

### Improvement opportunities

- **Data freshness**: The Pushshift archives stopped updating in mid-2023 due to Reddit API changes. For posts after that date, you would need an alternative source (Reddit API with OAuth, or a third-party archive). The current corpus covers 2018–2025 but the later years are sparser.
- **Incremental ingestion**: Currently the loader processes all ZST files from scratch. An incremental mode that appends new data to existing interim parquets would save time when adding new months of data.

---

## Stage 2 — Cleaning & NS relevance filtering

**Script**: `src/data/cleaner.py`

**Input**: Interim parquets from Stage 1

**Output**: `data/processed/submissions_clean.parquet`, `data/processed/comments_clean.parquet`

### Process

1. **NS relevance filtering** (Rule 0 — most important step):
   A regex-based filter identifies NS-relevant content using two tiers of keywords:

   **Phrase keywords** (38 multi-word terms, matched as substrings, case-insensitive):
   ```
   "national service", "nsman", "ns man", "nsmen",
   "conscription", "serve nation", "defend singapore", "duty to country",
   "ns disruption", "ns deferment",
   "operationally ready", "enlistment", "ns training",
   "book in", "book out", "confined to camp", "in-camp",
   "pes status", "downpes", "medical board",
   "officer cadet", "mindef", "tekong", "bmtc",
   "naval diving", "infantry", "commandos", "storeman",
   "route march", "guard duty", "outfield", "sign extra",
   "chao keng", "siao on", "tekkan", "arrowed",
   "off day", "leave pass", "emart"
   ```

   **Word keywords** (21 single-word terms, matched with word boundaries `\b`):
   ```
   "ns", "bmt", "pes", "ord", "saf", "spf", "scdf", "nsf", "ndu",
   "ippt", "ict", "rsi", "rso",
   "recruit", "enlistee", "enlisted", "enlist",
   "ocs", "scs", "vocation", "sergeant", "encik",
   "db", "wayang", "reservist", "conscript"
   ```

   Both tiers are compiled into a single regex pattern at module load. A post matches if **any** keyword matches anywhere in the text.

   **r/NationalServiceSG** posts bypass the keyword filter entirely — the entire subreddit is NS-contextual by definition.

2. **Bot removal**: Filters out posts from known bot accounts: `automoderator`, `sneakpeek_bot`, `microtechanalysis`. Also catches any author whose name contains "bot" (case-insensitive regex `[Bb]ot`).

3. **Text cleaning** (submissions only):
   - `[removed]`, `[deleted]`, `"None"`, `"nan"` selftext → replaced with empty string
   - Title + selftext combined into a single `combined_text` column (submissions)
   - `text_source` column marks whether the submission is `"full"` (has body text) or `"title_only"`

4. **Random-word detection** (Rule 4): A heuristic flags suspiciously random-looking text. A post is flagged only if ALL five conditions are met simultaneously:
   - More than 10 words
   - Type-token ratio > 0.95 (almost every word is unique)
   - Average word length outside the 3–9 range
   - Contains zero function words (a set of ~50 common English words + Singlish particles like "lah", "lor", "sia")
   - Contains zero NS keywords

   This catches gibberish/spam but never flags legitimate posts. Flagged posts are **not dropped** — they are just marked `is_suspicious=True` for manual review.

5. **Log weight**: `log_weight = log1p(max(score, 0))`. Negative-scored posts are floored at 0 before the log, so they get `log_weight = 0` rather than NaN. This column is used downstream for upvote-weighted aggregation.

6. **Thread depth** (comments only): Computes reply depth for every comment:
   - Depth 1 = direct reply to a post (`parent_id` starts with `t3_`)
   - Depth N = Nth-level reply (parent chain resolved within the dataset)
   - Depth -1 = parent comment was filtered out, so depth is unknown but > 1
   - Resolution is iterative (up to 50 passes) to handle deep reply chains

### Design decisions

- **r/NationalServiceSG bypass**: Every post in this subreddit is NS-relevant by context, even without explicit keywords. Venting posts, mood posts, and short replies that lack NS keywords would be incorrectly filtered out if the keyword gate applied.
- **Comments inherit relevance from keywords in their own text**, not from their parent post. A comment in an NS thread that says "lol" and contains no NS keywords will be filtered out. This is intentional — such comments add no NS signal.

### Known limitations & improvement opportunities

- **False positives from ambiguous acronyms**: The word keyword `"ns"` matches any occurrence of "NS" with word boundaries. This catches genuine NS references but also matches false positives like "NS (Newton)" (MRT station), "NS (not specified)", and other acronyms. The word boundary helps (`\bns\b` won't match "fins" or "lens") but a comment saying "I took the NS line" (referring to the North-South MRT line) will be included. **Improvement**: Add a negative context filter that rejects `"ns"` matches when surrounded by MRT/transit context words (e.g., "line", "station", "MRT", "circle").
- **False positives from `"db"`**: This matches "detention barracks" (NS context) but also "database", "DB Cooper", etc. **Improvement**: Require `"db"` to appear near an NS context word, or promote it to a phrase keyword like `"db punishment"`.
- **False negatives from Singlish abbreviations**: Some NS slang isn't in the keyword list (e.g., "lepak" [slack off], "fall in", "COS" [company sergeant], "PC" [platoon commander in NS context], "OC" [officer commanding]). **Improvement**: Expand the keyword list with more vocation abbreviations and camp slang. Be cautious of words that have strong non-NS meanings.
- **No semantic filtering at this stage**: The keyword filter is purely lexical. A post discussing the "NS line of the MRT" with the word "NS" will pass through. Stage 3 applies a second, tighter keyword filter at the chunk level, which helps, but a semantic relevance classifier (even a lightweight one) could replace both keyword filters and reduce false positives significantly. **Improvement**: Train a binary NS-relevance classifier on a few hundred labelled examples, replacing the keyword regex entirely.
- **Bot list is manually maintained**: Only 3 bots are explicitly listed, plus a regex for names containing "bot". New bots or bots with non-obvious names (e.g., "RemindMeBot", "RepostSleuthBot") may slip through. **Improvement**: Use a community-maintained bot list or filter by Reddit's `is_bot` field if available in the Pushshift data.

---

## Stage 3 — Semantic chunking

**Script**: `src/features/chunker.py`

**Input**: Cleaned parquets from Stage 2

**Output**: `data/processed/new/submissions_chunks.parquet` (460 MB), `data/processed/new/comments_chunks.parquet` (2.9 GB)

### Process (4 phases)

**Phase 1 — Sentence splitting** (CPU, spaCy):
- Uses `spacy.blank("en")` with the `sentencizer` pipe (rule-based sentence splitting — fast, no NER/POS overhead)
- Batch-processes all documents via `nlp.pipe(texts, batch_size=512)`
- Empty documents produce a single empty-string sentence

**Phase 2 — Sentence embedding** (GPU recommended, but works on CPU):
- Embeds all sentences using `sentence-transformers/all-mpnet-base-v2` (768-dimensional vectors)
- Processes in document batches of 5,000 to avoid GPU OOM
- Sentence embeddings are used for similarity-based chunk boundary detection

**Phase 3 — Grouping into chunks**:
- Walk through sentences sequentially within each document
- Compute cosine similarity between consecutive sentence embeddings
- A **chunk boundary fires** when:
  - Similarity drops below `SIMILARITY_THRESHOLD` (0.5) **AND** the current group has at least `CHUNK_MIN_SENTENCES` (2) sentences
  - **OR** the current group hits `CHUNK_MAX_SENTENCES` (6) — hard cap regardless of similarity
- Trailing fragments (fewer than `CHUNK_MIN_SENTENCES` sentences) are merged into the previous chunk
- Single-sentence documents are kept as-is (no splitting possible)

**NS relevance re-filter** (chunk level — tighter than Stage 2):
- Each chunk is independently checked for NS relevance using a **two-tier keyword system**:
  - **Strong terms** (34 terms): A single match = NS-relevant. These are unambiguous NS terms like "enlistment", "conscription", "tekong", "ippt", etc.
  - **Weak terms** (28 terms): Need **≥2 distinct** weak term matches = NS-relevant. These are ambiguous terms like "ns", "nsf", "saf", "pes", "recruit", "sergeant", etc.
- r/NationalServiceSG chunks always pass (bypass, same as Stage 2)
- Chunks that fail the NS filter are **discarded entirely** — they don't appear in any downstream data

**Phase 4 — Chunk re-embedding** (GPU/CPU):
- All surviving chunks are re-embedded with a context prefix
- Comment chunks get the prefix: `"Post: {submission_title}\n\nComment: {chunk_text}"`
- This prefix improves downstream topic assignment by providing disambiguation context — a comment saying "that's so true" under an NS post gets an NS-relevant embedding

### Output schema

Each chunk row contains:

| Column | Type | Description |
|---|---|---|
| `doc_id` | str | Original Reddit post/comment ID |
| `chunk_id` | str | `{doc_id}_{chunk_idx}` (e.g., `j4b1zg5_0`) |
| `chunk_idx` | int | 0-indexed position within the document |
| `chunk_count` | int | Total chunks this document produced |
| `doc_type` | str | `"submission"` or `"comment"` |
| `text` | str | The chunk text |
| `embedding` | list[float] | 768-dim embedding vector |
| `subreddit` | str | Source subreddit |
| `created_utc` | datetime | Post timestamp |
| `score` | int | Reddit upvotes |
| `log_weight` | float | `log1p(max(score, 0))` |
| `author` | str | Reddit username |
| `post_id` | str | (comments only) Parent submission ID |

Total output: ~737K chunks across both parquets.

### Design decisions

- **All documents go through the full pipeline** regardless of length. An earlier version had a shortcut for short documents (single chunk, single embedding), but this caused mixed-topic short documents to receive a single topic label, losing the second topic's signal.
- **The similarity threshold of 0.5** was chosen empirically. Lower values (0.3) over-split (each sentence becomes its own chunk). Higher values (0.7) under-split (entire documents become one chunk). 0.5 balances granularity with semantic coherence.
- **Minimum 200 characters**: The `CHUNK_MIN_CHARS` constant isn't in config.py but is enforced in the grouping logic. Very short trailing fragments are merged into the previous chunk to avoid tiny, noisy chunks.

### Known limitations & improvement opportunities

- **Chunk-level NS filter still uses keywords**: The two-tier (strong/weak) keyword system is better than Stage 2's flat regex, but it still can't catch NS-relevant text that uses no keywords. For example, "I hated those two years of my life" in an NS thread is clearly about NS, but contains no NS keywords. If this comment appears in r/singapore (not r/NationalServiceSG), its chunks will be filtered out. **Improvement**: Replace the keyword filter with a lightweight binary classifier trained on a few hundred examples of NS-relevant vs non-NS text. Alternatively, use the parent post's NS-relevance to propagate relevance to all child comments (comments in an NS thread are usually about NS).
- **Sentence splitting is rule-based**: `spacy.blank("en")` with `sentencizer` splits on punctuation patterns. It doesn't handle Singlish text well — Singlish often uses run-on sentences with particles instead of punctuation ("then he say like that lor so I just go book out lah"). **Improvement**: Use a trained sentence segmenter or fine-tune one on Singlish text.
- **Fixed similarity threshold**: The 0.5 threshold is global. Some documents naturally have more thematic coherence (long policy discussions) while others jump between topics (stream-of-consciousness rants). An adaptive threshold based on document-level similarity distribution could produce better chunks.
- **Embedding model**: `all-mpnet-base-v2` is a general English model. It handles Singlish reasonably well (the pre-training data includes some code-switched text) but a Singapore-English-specific embedding model could improve both chunk boundary detection and downstream FAISS retrieval.
- **No overlap between chunks**: Chunks are strictly non-overlapping. Some NLP chunking approaches use sliding windows with overlap to avoid losing context at boundaries. This hasn't been tested here.

---

## Stage 4 — Topic modelling

**Notebook**: `notebooks/kaggle_topic_model_v3.ipynb` (Kaggle T4 GPU)

**Input**: Chunk embeddings from Stage 3

**Output**:

| File | Description |
|---|---|
| `models/new/bertopic_fine/` | Fine-grained BERTopic model (359 topics) |
| `models/new/bertopic_coarse/` | Coarse BERTopic model (17 macro topics) |
| `data/processed/new/chunk_topics.parquet` | Topic assignments per chunk |
| `data/processed/new/topic_keywords_fine.csv` | Top keywords per fine topic |
| `data/processed/new/topic_keywords_coarse.csv` | Top keywords per coarse topic |
| `data/processed/new/hierarchical_topics_manual.parquet` | Manual topic taxonomy |

### Process

1. **UMAP dimensionality reduction**: Reduce 768-dim embeddings to 5 dimensions for clustering. UMAP preserves local structure better than PCA/t-SNE for clustering.

2. **HDBSCAN clustering**: Cluster the reduced embeddings. BERTopic uses HDBSCAN's soft clustering to assign topic probabilities per chunk. HDBSCAN is density-based, so it naturally handles outliers — chunks that don't fit any cluster are assigned `topic_id_fine = -1`.

3. **c-TF-IDF topic representations**: Compute class-based TF-IDF to extract representative keywords per topic. These keywords are stored in `topic_keywords_fine.csv`.

4. **Manual taxonomy**: The 359 leaf topics are manually organised into a 4-level hierarchy defined in `src/models/topic_labels.py` (the `TOPIC_LABELS` dictionary):

   ```
   359 leaf topics → 112 cluster topics → 52 sub topics → 17 macro topics
   ```

   The 17 macro topics are:
   - NS Life & Culture
   - BMT & Training
   - NS Policy & Society
   - Enlistment & Pre-NS
   - Admin & Logistics
   - Physical Fitness & IPPT
   - Pay & Benefits
   - ORD & Post-NS
   - Relationships & Social Life
   - Medical & Health
   - Vocations & Units
   - Reservist & ICT
   - Gear & Equipment
   - Mental Health
   - Food & Cookhouse
   - Technology & Digital NS
   - NS & Education

5. **Topic assignment**: Each chunk receives:
   - `topic_id_fine` (0–358 or -1 for outlier)
   - `topic_prob_fine` (probability, 0.0–1.0)
   - `topic_id_coarse` (0–16 or -1)

### Design decisions

- **Manual taxonomy over automatic hierarchy**: BERTopic can auto-generate hierarchical topics, but the auto-hierarchy mixes semantically unrelated topics. The manual taxonomy ensures that "BMT food complaints" rolls up under "BMT & Training" → "NS Life & Culture", not under a generic "complaints" cluster.
- **359 topics is intentionally high**: Fine-grained topics enable the dashboard's drill-down feature. A user can go from "NS Policy & Society" (macro) → "Conscription & National Duty" (sub) → "NS for PRs / foreigners" (cluster) → specific leaf topics.

### Known limitations & improvement opportunities

- **Outlier rate**: Approximately 5–8% of chunks are assigned `topic_id_fine = -1` (outlier). These chunks are included in sentiment/commitment aggregation but excluded from topic-level analysis. **Improvement**: Use BERTopic's outlier reduction techniques (e.g., re-assign outliers to nearest topic if probability exceeds a threshold).
- **Static topic model**: The model was trained once on the full corpus. It does not adapt to new topics that emerge over time. If NS discourse shifts to discuss something entirely new (e.g., a new NS scheme), existing topics won't capture it. **Improvement**: Periodic re-training, or use BERTopic's online learning to incrementally update the model.
- **Manual taxonomy maintenance**: The 4-level taxonomy in `topic_labels.py` is manually curated. Adding new leaf topics or reorganising the hierarchy requires editing this Python dictionary. There's no UI for taxonomy management. **Improvement**: Store the taxonomy in a CSV or JSON file that's easier to edit, or build a simple taxonomy editor.
- **Duplicate chunk_ids in chunk_topics**: There are 309 duplicate `chunk_id` entries in `chunk_topics.parquet` from a batch boundary in the Kaggle notebook. Stage 6 deduplicates them with `drop_duplicates(subset='chunk_id', keep='first')`, but this should be fixed at source.

---

## Stage 5a — Sentiment classification (SingBERT)

**Notebooks**: `notebooks/kaggle_finetune_singbert_v1.ipynb` (training), `notebooks/kaggle_infer_singbert_v1.ipynb` (inference)

**Model**: `zanelim/singbert-large-sg` fine-tuned for 3-class sentiment (positive / neutral / negative)

**Input**: All 737K chunks from Stage 3

**Output**: `data/processed/new/chunk_sentiment.parquet` — columns `sent_neg`, `sent_neu`, `sent_pos` (softmax probabilities per chunk, sum to 1.0)

### Training data

`data/processed/new/singbert_train.csv` — 3,289 rows of human-annotated + LLM-annotated chunks, class-balanced.

### Training process

1. Base model: `zanelim/singbert-large-sg` — a RoBERTa-large model pre-trained on Singapore English text (Singlish, code-switching, local slang)
2. Add a classification head for 3 classes
3. Train on Kaggle T4 GPU for ~3 epochs with learning rate warmup
4. Model saved as `models/new/singbert_v7/` (tokenizer + config included; the 1.2 GB `model.safetensors` weights file is gitignored)

### Why SingBERT?

Standard English sentiment models perform poorly on Singaporean English due to:
- **Code-switching**: English + Mandarin + Malay + Hokkien within a single sentence ("Wah this one really jialat sia, kena guard duty again")
- **Singlish grammar**: Different sentence structure, particles ("lah", "lor", "sia"), and word usage
- **Culturally specific expressions**: "chao keng" (malingering), "siao on" (overly enthusiastic), "tekkan" (tortured by superiors)

SingBERT was pre-trained on Singapore web text and handles these phenomena natively.

### Evaluation (on 217-row human gold set)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.82 | 0.90 | 0.86 |
| Neutral | 0.83 | 0.82 | 0.83 |
| Positive | 0.88 | 0.78 | 0.83 |
| **Macro avg** | **0.84** | **0.83** | **0.78** |

Overall: Accuracy = 83.9%, Cohen's κ = 0.714

### Earlier approaches tried and discarded

| Approach | Cohen's κ | Why discarded |
|---|---|---|
| VADER lexicon | 0.31 | Too poor for code-switched text |
| Generic BERT (`bert-base-uncased`) | 0.52 | Misses Singlish entirely |
| GPT-4.1-mini zero-shot | 0.68 | Good but expensive at 737K scale ($200+) |

### Known limitations & improvement opportunities

- **217-row eval set is small**: Statistical confidence intervals on the F1 scores are wide. A 500+ row evaluation set would give more reliable metrics. The human gold set is stored in `data/processed/new/gold_eval_v2.csv` (54 rows) and `gold_eval_v2_balanced.csv` (180 rows).
- **Class imbalance in inference**: The corpus is ~70% neutral, ~20% negative, ~10% positive. The model handles this reasonably well, but the positive class has the weakest recall (0.78). Positive posts that use irony or understatement may be misclassified as neutral.
- **Sarcasm and irony**: Singlish sarcasm is hard to detect. "Wah army really damn shiok" (sarcastic negative) may be classified as positive. **Improvement**: Add sarcasm-specific training examples, or add a sarcasm detection pre-filter.
- **Model is frozen**: The model was trained once (v7). It doesn't adapt to evolving language patterns. Singlish evolves quickly — new slang, new abbreviations, new code-switching patterns. **Improvement**: Periodic re-training with recent data, or active learning to identify uncertain predictions for human review.
- **No confidence thresholding**: All predictions are used regardless of confidence. A chunk with `sent_neg=0.35, sent_neu=0.33, sent_pos=0.32` is classified as negative even though the model is essentially uncertain. **Improvement**: Flag low-confidence predictions (max probability < 0.5) and either exclude them or treat them as neutral.

---

## Stage 5b — Commitment classification (4-stage cascade)

**Notebooks**: `notebooks/cascade/cascade_stage1a_buyin_relevance.ipynb`, `cascade_stage1b_stance_relevance.ipynb`, `cascade_stage2a_committed_uncommitted.ipynb`, `cascade_stage2b_supportive_critical.ipynb`

**Inference**: `notebooks/cascade/cascade_inference_v2.ipynb`

**Input**: All 737K chunks from Stage 3

**Output**: `data/processed/new/chunk_commitment_cascade.parquet`

### Architecture — two independent axes

The commitment model classifies chunks along two orthogonal axes:

```
Input chunk
    ├── Stage 1a: buyin_relevant? ──► No → buyin = neutral
    │                                Yes ──► Stage 2a: committed / uncommitted
    │
    └── Stage 1b: has_stance? ──► No → stance = neutral
                                 Yes ──► Stage 2b: supportive / critical
```

1. **Buyin axis** (personal investment in NS):
   - Stage 1a: Is this chunk relevant to buyin? (binary: relevant / not relevant)
   - Stage 2a: If relevant → committed or uncommitted?
   - **Committed**: Personally invested in NS. Signals include: pride in service, personal growth narratives, voluntary engagement, defending NS from critics using personal experience
   - **Uncommitted**: Personally disengaged from NS. Signals include: "waste of time" framing, counting down to ORD, reluctant compliance, viewing NS as obstacle to life goals

2. **Stance axis** (opinion on NS policy):
   - Stage 1b: Does this chunk express a stance? (binary: has stance / no stance)
   - Stage 2b: If has stance → supportive or critical?
   - **Supportive**: Endorses NS as institution/policy. Signals include: "NS builds character", defends conscription, supports NS reforms positively
   - **Critical**: Opposes NS as institution/policy. Signals include: "conscription is slavery", criticises NS pay/conditions, argues against mandatory service

**Why two axes?**: A single committed/uncommitted dimension conflates personal investment with policy opinion. Examples:
- "I served 10 years as a reservist and I'm proud of it, but the pay is disgraceful" → **committed** (buyin) + **critical** (stance)
- "NS is great for national defence but honestly I just want to ORD" → **uncommitted** (buyin) + **supportive** (stance)

### Training data

Each cascade stage is trained on a mix of:
- **Human annotations**: `commitment_manual_annotations.csv` (947 rows), plus the 727-row gold eval set
- **LLM-generated silver labels**: Gemma 4 predictions on the broader corpus — 82,037 labelled chunks compiled in `gemma4_all_labelled_compiled.csv`. These are used as training data with lower weight, not as ground truth.
- **FAISS-enriched examples**: Semantically similar chunks to existing gold labels, found via embedding similarity

Training data for each stage is in `data/processed/new/cascade/stage{1a,1b,2a,2b}_train.csv`.

**Prompt engineering**: The Gemma 4 silver labels are generated using a carefully engineered prompt in `src/features/prompts/commitment_v2_prompt.py`. This prompt includes:
- Detailed taxonomy definitions with examples
- Disambiguation rules (e.g., "waste of time" framing always → uncommitted, even if the person mentions incidental benefits)
- Direction checks (e.g., criticising *other people's* chao keng ≠ uncommitted; it's the author's stance that matters)
- Singlish-specific rules (e.g., "ORD mood" → uncommitted signal)

### Model

Each of the 4 stages is a fine-tuned `zanelim/singbert-large-sg` classifier. All 4 models run on Kaggle T4 GPUs.

### Evaluation (on 727-row human gold set)

| Stage | Task | Accuracy | F1 | Key confusion |
|-------|------|----------|----|---|
| 1a | Buyin relevance | 96.2% | 0.95 | Some neutral chunks incorrectly flagged as buyin-relevant |
| 1b | Stance relevance | 95.1% | 0.93 | Similar to 1a |
| 2a | Committed vs Uncommitted | 75.8% | 0.714 | Mixed-signal texts (mentions both positive and negative aspects) |
| 2b | Supportive vs Critical | 80.3% | 0.784 | Sarcasm, Singlish expressions of subtle criticism |

### Output schema

`chunk_commitment_cascade.parquet` contains:

| Column | Type | Values |
|---|---|---|
| `chunk_id` | str | Matches Stage 3 chunk_id |
| `buyin_label` | str | `"committed"`, `"uncommitted"`, `"neutral"` |
| `stance_label` | str | `"supportive"`, `"critical"`, `"neutral"` |
| `prob_buyin_committed` | float | Softmax probability |
| `prob_buyin_uncommitted` | float | Softmax probability |
| `prob_buyin_neutral` | float | Softmax probability |
| `prob_stance_supportive` | float | Softmax probability |
| `prob_stance_critical` | float | Softmax probability |
| `prob_stance_neutral` | float | Softmax probability |

### Known limitations & improvement opportunities

- **Stage 2a is the weakest link** (F1 = 0.714). The committed/uncommitted distinction is genuinely hard — many posts express mixed signals. A person might describe a positive NS memory (committed signal) while framing the overall experience negatively (uncommitted signal). **Improvement**: More human-labelled training data, especially for ambiguous cases. The current gold set has 725 rows; expanding to 1,500+ with focus on edge cases would help.
- **Silver label quality**: The 82K Gemma 4 labels have ~88% agreement with human labels on buyin (based on the 405-row overlap). The 12% disagreement introduces noise into training data. **Improvement**: Use confidence-weighted training — give higher weight to silver labels where the model was more confident (high softmax probability), and lower weight to uncertain predictions.
- **No C2D (Commitment-to-Demonstrated) in cascade output**: The earlier Gemma 4 labelling included a `pred_c2d` field (demonstrated / explicit / empty) that indicates whether commitment is demonstrated through actions or explicitly stated. This signal isn't in the cascade model's output. **Improvement**: Add a Stage 3a classifier for C2D if this distinction proves useful for analysis.
- **Cascade error propagation**: Errors in Stage 1 (relevance) propagate to Stage 2. If Stage 1a incorrectly classifies a committed chunk as "not buyin-relevant", that chunk gets `buyin_label = neutral` and Stage 2a never sees it. The 95% F1 on relevance means ~5% of buyin-relevant chunks are incorrectly neutralised. **Improvement**: Consider a single 3-class model (committed / uncommitted / neutral) instead of the cascade, and compare performance.

---

## Stage 6 — Document-level aggregation

**Script**: `src/analysis/stage6_doc_aggregation.py`

**Input**: `chunk_sentiment.parquet`, `chunk_commitment_cascade.parquet`, `chunk_topics.parquet`, `submissions_chunks.parquet`, `comments_chunks.parquet`

**Output**: `data/processed/new/doc_sentiment.parquet` — one row per document with aggregated scores

### Process

1. **Join**: Merge chunk-level sentiment, commitment, and topic labels onto each chunk's metadata. Start from sentiment as the canonical chunk set (737,274 unique `chunk_id`s).

2. **Weighted aggregation**: For each document, compute weighted means of all score columns:
   - Weight = `log_weight` floored at `LOG_WEIGHT_FLOOR = 1.0`
   - The floor ensures zero-upvote chunks still contribute (they get weight 1.0 instead of 0.0)
   - Score columns: `sent_neg`, `sent_neu`, `sent_pos`, `prob_buyin_committed`, `prob_buyin_uncommitted`, `prob_buyin_neutral`, `prob_stance_supportive`, `prob_stance_critical`, `prob_stance_neutral`

3. **Topic assignment** (weighted mode): Each document's topic is the `topic_id_fine` with the highest total weight across its chunks:
   ```
   topic_for_doc = argmax over topics(Σ(topic_prob_fine × log_weight_w))
   ```
   - Outlier chunks (`topic_id_fine = -1`) are included in score aggregation but excluded from topic assignment
   - Documents where ALL chunks are outliers get `topic_macro = NaN`

4. **Derived columns**:
   - `sent_label`: argmax of `(sent_neg, sent_neu, sent_pos)` → `"neg"`, `"neu"`, `"pos"`
   - `commit_net`: `prob_buyin_committed - prob_buyin_uncommitted`

### Output schema

| Column | Type | Description |
|---|---|---|
| `doc_id` | str | Original Reddit ID |
| `doc_type` | str | `"submission"` or `"comment"` |
| `post_id` | str | Thread ID (= doc_id for submissions) |
| `subreddit` | str | Source subreddit |
| `created_utc` | datetime | Post timestamp |
| `chunk_count` | int | Number of chunks this doc produced |
| `sent_neg/neu/pos` | float32 | Weighted mean sentiment scores |
| `sent_label` | str | Argmax sentiment class |
| `prob_buyin_*` | float32 | Weighted mean commitment scores |
| `prob_stance_*` | float32 | Weighted mean stance scores |
| `commit_net` | float32 | `prob_buyin_committed - prob_buyin_uncommitted` |
| `topic_id_fine` | int | Best fine-grained topic |
| `topic_macro/sub/sub_sub` | str | Taxonomy labels |

### Improvement opportunities

- **Weighted mean may dilute strong signals**: A document with 3 chunks — 2 neutral and 1 strongly negative — gets a moderate negative score that may be classified as neutral at the document level. **Improvement**: Consider using max-pooling for sentiment (take the strongest signal) or a "any chunk negative → document negative" rule, alongside the mean.

---

## Stage 7 — Divergence analysis

**Script**: `src/analysis/stage7_divergence_v2.py`

**Input**: `chunk_sentiment.parquet`, `submissions_chunks.parquet`, `comments_chunks.parquet`, `chunk_topics.parquet`

**Output**: `data/processed/new/doc_divergence_v2.parquet`, `data/processed/new/topic_discourse_intensity.parquet`

### What it measures

How often does the comment section's mood differ from the original post's mood? This identifies threads where the community pushes back against or amplifies the original poster's sentiment.

### Key design: opinion filtering

The naive approach (mean comment sentiment − mean post sentiment) is dominated by neutral chunks (~89% of corpus), so divergence scores cluster near zero and bury genuinely contentious threads. The v2 redesign uses an **opinion filter**: only chunks where `max(sent_neg, sent_pos) > 0.5` (i.e., the model is >50% confident the chunk is opinionated) contribute to divergence metrics.

### Metrics

**Per post** (`doc_divergence_v2.parquet`):

| Metric | Formula | What it captures |
|---|---|---|
| `upvote_weighted_div` | `Σ(|neg−pos| × log(1+upvotes)) / Σ(log(1+upvotes))` over opinionated chunks | Primary ranking signal — how polarised is the discussion, weighted by community endorsement |
| `within_thread_variance` | Variance of `sent_neg` across opinionated comment chunks | Disagreement within comments (NaN if <2 opinionated comment chunks) |
| `opinion_chunk_pct` | % of all chunks in thread that are opinionated | How emotionally charged is this thread |
| `total_upvotes` | Sum of upvotes across post + all comments (doc-level deduped) | Reach/visibility |
| `sentiment_divergence_raw` | Mean comment `sent_neg` − mean post `sent_neg` (old naive metric) | Kept for comparison only |
| `commitment_divergence` | `pct_uncommitted(comments) − pct_uncommitted(submissions)` | Filled post-hoc by `scripts/rebuild_doc_commitment.py` |

**Per topic** (`topic_discourse_intensity.parquet`):

| Metric | Formula |
|---|---|
| `discourse_intensity` | `Σ(upvotes × |neg−pos|)` per `topic_macro` |
| `n_opinion_chunks` | Count of opinionated chunks |
| `total_upvotes` | Sum of upvotes |

### Improvement opportunities

- **Opinion threshold is fixed at 0.5**: This is a reasonable default, but some topics may have systematically lower sentiment scores (e.g., policy discussions tend to be more measured). An adaptive threshold per topic could surface more nuanced divergence patterns.
- **Orphan comments**: Comments whose parent submission is not in the corpus are skipped. This can happen when the submission was filtered out in Stage 2 but the comment passed the NS keyword filter independently.

---

## Stage 8 — Temporal aggregation

**Script**: `src/analysis/stage8_temporal.py`

**Input**: `doc_sentiment.parquet`, `chunk_commitment_cascade.parquet`, `chunk_metadata.parquet`

**Output**:
- `data/processed/new/temporal_sentiment.parquet` — monthly × subreddit × topic_macro
- `data/processed/new/temporal_sentiment_overall.parquet` — monthly × subreddit (no topic)

### Process

1. Group documents by `(year_month, subreddit)` for overall rollup, and `(year_month, subreddit, topic_macro)` for topic rollup
2. Compute monthly means of sentiment scores
3. Compute percentage by dominant label (`pct_neg`, `pct_neu`, `pct_pos`)
4. Flag low-volume months: `low_volume = True` where `doc_count < 30` in a (month, subreddit) bucket
5. If `chunk_commitment_cascade.parquet` is available, join chunk-level commitment labels with upvote scores from `chunk_metadata.parquet` and compute:

**Commitment columns** (upvote-weighted):

| Column | Formula |
|---|---|
| `pct_buyin_committed` | Unweighted % of chunks classified committed |
| `wtd_buyin_committed` | `Σ(log(1+score) × is_committed) / Σ(log(1+score))` |
| `net_buyin` | `pct_buyin_committed − pct_buyin_uncommitted` |
| `pct_stance_supportive` | Unweighted % supportive |
| `wtd_stance_supportive` | Upvote-weighted % supportive |
| `net_stance` | `pct_stance_supportive − pct_stance_critical` |
| `net_disposition` | `wtd_positive − wtd_negative` (combined) |

### Design decisions

- **Monthly granularity**: Weekly is too noisy for trend analysis; quarterly loses event-level resolution. Monthly strikes the right balance.
- **Low-volume threshold of 30 docs**: Below this, a single viral post can swing the entire month's sentiment. The dashboard hides low-volume months by default.
- **Both weighted and unweighted metrics**: Upvote-weighted metrics (`wtd_*`) surface community-endorsed sentiment. Unweighted metrics (`pct_*`) give equal voice to every post. The dashboard uses unweighted for commitment trends.

### Improvement opportunities

- **Pre-2019 data is very sparse**: r/askSingapore and r/NationalServiceSG barely existed before 2019. Trends before 2019 are almost entirely from r/singapore and should be interpreted cautiously. Consider adding a minimum doc_count threshold per subreddit to avoid spurious trends.
- **No seasonal adjustment**: NS discourse has natural seasonal patterns (enlistment cycles, ICT windows, IPPT deadlines). Month-over-month changes may reflect seasonality rather than genuine sentiment shifts. **Improvement**: Add a seasonal decomposition (STL or similar) to separate trend from seasonality.

---

## Stage 9 — Dashboard

**Script**: `app/dashboard.py`

**Input**: All parquets from Stages 6–8, plus topic models, keywords, and RAG assets

The dashboard is documented separately in [dashboard_guide.md](dashboard_guide.md).

---

## RAG chatbot pipeline

The Sentinel Bot (dashboard page 7) uses Retrieval-Augmented Generation to answer questions about NS sentiment.

### Knowledge base build

Four scripts in `scripts/rag/` build the RAG knowledge base:

#### 1. `build_index.py` — FAISS index
- Loads chunk embeddings from `submissions_chunks.parquet` + `comments_chunks.parquet`
- Builds a flat FAISS index (`IndexFlatIP` — inner product, equivalent to cosine similarity on normalised vectors) over all 737K chunk embeddings (768-dim)
- Also builds `chunk_metadata.parquet` (89 MB) — a lightweight table with chunk_id, text snippet (first 300 chars), subreddit, year, month, topic_macro, upvotes, sentiment scores
- Output: `chunk_faiss.index` (2.1 GB), `chunk_metadata.parquet`

#### 2. `build_topic_digests.py` — Topic summaries
- For each of the 17 macro topics, generates a statistical digest:
  - Total chunk count, sentiment breakdown (% neg/neu/pos), top keywords
  - Most negative / most positive sub-topics
  - Community tone summary, dominant themes, common concerns
- Output: `rag_topic_digests.json`

#### 3. `build_temporal_narratives.py` — Timeline narratives
- Pre-computes natural-language summaries of sentiment trends for each topic × year
- Identifies inflection points, peaks, troughs
- Output: `rag_temporal_narratives.json`

#### 4. `build_fact_table.py` — Quantitative facts
- Pre-computes common statistics for every `(year, month, subreddit, topic_macro)` combination at multiple granularities:
  - Annual aggregates (`month=0`)
  - Monthly aggregates
  - Per-subreddit, per-topic, and combined breakdowns
- Stats include: `mean_sent_neg`, `pct_neg`, `mean_commit_net`, `pct_committed`, `wtd_buyin_committed`, `net_buyin`, `net_stance`, `net_disposition`, `doc_count`
- Output: `rag_fact_table.parquet`

### Query processing

When a user asks a question, the RAG pipeline follows this flow:

#### Step 1 — Query routing (`src/rag/query_router.py`)

Entirely rule-based — no LLM, no embeddings. Runs in <5ms.

**Intent classification**: Classifies the query as `"quantitative"` or `"qualitative"` by pattern matching:
- Quantitative patterns: "how much", "what percentage", "which topic ... most", "compare ... vs", "over time", "trend", year comparisons
- Qualitative patterns: "explain", "why", "what do people say", "describe", "tell me about", "examples of"
- Tie-breaking: biased toward qualitative (safer default — qualitative answers always include data)

**Filter extraction**: Extracts structured filters from the query text:
- **Years**: regex `\b(201[89]|202[0-5])\b`
- **Months**: name-to-number mapping (e.g., "january" → 1)
- **Subreddits**: alias matching (e.g., "r/singapore", "singapore subreddit")
- **Topics**: keyword scoring against `TOPIC_KEYWORD_MAP` — each macro topic has 15–30 associated keywords. The query is scored against each topic's keyword list; top 2 topics with score > 0 are selected.
- **Metric type**: sentiment, commitment, negative, positive, discourse
- **Intent flags**: compare_subreddits, compare_topics, ranking_dim/dir, ask_methodology, commitment_breakdown

#### Step 2 — Retrieval (`src/rag/retriever.py`)

For qualitative queries:
1. Embed the query using the same `all-mpnet-base-v2` model
2. L2-normalise to match FAISS inner-product space
3. Search the full FAISS index (737K vectors) — `IndexFlatIP` is fast enough (~80ms) that pre-filtering is unnecessary
4. **Adaptive search depth**: When filters are sparse (e.g., a single month = 0.4% of corpus), search deeper:
   ```
   n_search = max(FAISS_CANDIDATE_K × 5, ceil(total / n_candidates) × RETURN_K × 3)
   ```
   Capped at 30,000 to keep search under ~3s
5. Post-filter by metadata (year, month, subreddit, topic)
6. Rerank by: `cosine_similarity × (1 + 0.3 × log(1 + upvotes))`
7. Drop chunks below `MIN_COSINE_SIM = 0.25`
8. Deduplicate by `chunk_id`
9. Return top 10 chunks

**Configuration** (from `src/rag/config.py`):

| Parameter | Value | Purpose |
|---|---|---|
| `FAISS_CANDIDATE_K` | 50 | Initial candidates from FAISS |
| `RETURN_K` | 10 | Final chunks returned |
| `UPVOTE_ALPHA` | 0.3 | Weight for upvote boost in reranking |
| `MIN_COSINE_SIM` | 0.25 | Minimum similarity threshold |
| `MAX_CONTEXT_CHARS` | 10,000 | Context window budget (~2,500 tokens) |
| `MAX_CHUNK_TEXT_LEN` | 300 | Truncate each chunk to this length in context |

#### Step 3 — Spike detection (`src/rag/spike_detector.py`)
- Checks if the query period coincides with any known NS events from `ns_events.json`
- Identifies sentiment spikes in the temporal data using z-scores against a 6-month rolling baseline
- Notable threshold: |z| ≥ 1.5; major threshold: |z| ≥ 2.0
- Builds a longitudinal anomaly timeline for multi-year queries

#### Step 4 — Context assembly (`src/rag/context_assembler.py`)

Assembles a bounded context string (≤ `MAX_CONTEXT_CHARS` = 10,000 chars) from:

1. **Statistical facts** from the fact table (never dropped)
2. **NS events** overlapping the queried period (never dropped, max 3–6 events)
3. **Longitudinal timeline** (for multi-year queries — spike/dip timeline)
4. **Topic digest** for matched topic (max 1)
5. **Temporal narratives** for matched months (max 2)
6. **Retrieved chunks** with metadata
7. **User query** (always last, never dropped)

Budget enforcement: fixed parts (stats + events + query) are always included. Optional parts are included in priority order (timeline > digest > narrative > chunks) until the budget is exhausted. Chunks are truncated if partially fitting; everything else is dropped entirely.

If the user asks a methodology question (`filters.ask_methodology = True`), a static methodology explanation block is prepended.

#### Step 5 — Synthesis (`src/rag/synthesizer.py`)

Sends the assembled context + user query to the configured LLM backend. Backend selection (priority order):

| Priority | Backend | Model | Cost |
|---|---|---|---|
| 1 (primary) | Groq | Llama 3.3 70B (`llama-3.3-70b-versatile`) | Free tier (7K requests/month) |
| 2 (fallback) | OpenAI | GPT-4o-mini | ~$0.0004/query |
| 3 (fallback) | Anthropic | Claude Haiku 4.5 | ~$0.001/query |
| 4 (fallback) | FreeLLMAPI | Gemini 2.5 Flash | Local proxy |

Auto-fallback: if Groq hits a rate limit, the synthesizer automatically switches to OpenAI for that request.

The system prompt instructs the model to:
- Never invent statistics — only cite numbers from the provided data
- Distinguish between sentiment and commitment (independent dimensions)
- Use a structured format for longitudinal questions (year-by-year or event-driven)
- Cite sources inline: `[r/singapore · Jan 2019 · 847 upvotes]`
- Keep answers concise: 100–200 words for simple questions, 250–400 for trends, up to 500 for complex multi-part questions

### Known limitations & improvement opportunities

- **Query router is rule-based**: Complex or ambiguous queries may be misclassified. "How has the attitude toward reservist changed and what are the main complaints?" is both quantitative and qualitative. **Improvement**: Use a lightweight LLM call for query classification, or default to a hybrid mode that includes both stats and chunks.
- **Topic keyword map is manually maintained**: The `TOPIC_KEYWORD_MAP` in `query_router.py` has 15–30 keywords per macro topic. If a user asks about "NS MR (military run)" and "MR" isn't in the map, the topic won't be matched. **Improvement**: Use embedding similarity between the query and topic digest text instead of keyword matching.
- **FAISS is brute-force**: `IndexFlatIP` does exact search over all 737K vectors. This is fast enough (~80ms) but won't scale if the corpus grows 10x. **Improvement**: Switch to `IndexIVFFlat` or `IndexHNSW` for approximate nearest-neighbor search if needed.
- **Context window is conservatively sized**: 10,000 chars (~2,500 tokens) is well within Llama 3.3's 128K context limit. More context could improve answer quality for complex questions. **Improvement**: Increase `MAX_CONTEXT_CHARS` to 20,000–30,000 and include more chunks and narratives.
- **No conversation memory**: Each question is independent — the bot doesn't remember previous questions in the session. "What about the same topic in 2023?" after a 2022 question will fail. **Improvement**: Maintain a short conversation history in the Streamlit session state and include it in the context.
- **Chunk text truncated to 300 chars**: Retrieved chunks are truncated to `MAX_CHUNK_TEXT_LEN = 300` characters in the context. Longer chunks lose context. **Improvement**: Increase to 500 chars and reduce the number of chunks from 10 to 7 to stay within budget.

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
Chunk parquets (737K chunks) ────────────────┐
    │                                         │
    ├──► Stage 4: BERTopic ──► chunk_topics.parquet
    │                                         │
    ├──► Stage 5a: SingBERT ──► chunk_sentiment.parquet
    │                                         │
    ├──► Stage 5b: Cascade ──► chunk_commitment_cascade.parquet
    │                                         │
    ▼                                         ▼
Stage 6: doc aggregation ──► doc_sentiment.parquet
    │
    ├──► Stage 7: divergence ──► doc_divergence_v2.parquet
    │                            topic_discourse_intensity.parquet
    │
    ├──► Stage 8: temporal ──► temporal_sentiment.parquet
    │                          temporal_sentiment_overall.parquet
    │
    └──► RAG build ──► chunk_faiss.index
                       chunk_metadata.parquet
                       rag_topic_digests.json
                       rag_temporal_narratives.json
                       rag_fact_table.parquet
                           │
                           ▼
                    Stage 9: Dashboard
                    (app/dashboard.py)
```

---

## File manifest

### Source code

| Path | Stage | Description |
|---|---|---|
| `config.py` | All | Global paths and constants |
| `src/data/loader.py` | 1 | ZST decompression and ingestion |
| `src/data/cleaner.py` | 2 | Bot removal, NS filtering, text cleaning |
| `src/features/chunker.py` | 3 | Semantic chunking with NS re-filter |
| `src/features/prompts/commitment_v2_prompt.py` | 5b | Gemma 4 labelling prompt for commitment |
| `src/models/topic_labels.py` | 4 | Manual 4-level topic taxonomy |
| `src/analysis/stage6_doc_aggregation.py` | 6 | Chunk → doc rollup |
| `src/analysis/stage7_divergence_v2.py` | 7 | Post-vs-comment divergence |
| `src/analysis/stage8_temporal.py` | 8 | Monthly temporal aggregation |
| `src/rag/config.py` | RAG | RAG constants and paths |
| `src/rag/query_router.py` | RAG | Rule-based intent + filter extraction |
| `src/rag/retriever.py` | RAG | FAISS search + upvote reranking |
| `src/rag/spike_detector.py` | RAG | Temporal anomaly detection |
| `src/rag/context_assembler.py` | RAG | Context window assembly |
| `src/rag/synthesizer.py` | RAG | LLM backend selection + streaming |
| `app/dashboard.py` | 9 | Streamlit dashboard |

### Data files (in `data/processed/new/`)

| File | Stage | Rows | Key columns |
|---|---|---|---|
| `submissions_chunks.parquet` | 3 | ~57K | chunk_id, text, embedding, subreddit, score |
| `comments_chunks.parquet` | 3 | ~680K | chunk_id, text, embedding, post_id, score |
| `chunk_topics.parquet` | 4 | ~737K | chunk_id, topic_id_fine, topic_prob_fine |
| `chunk_sentiment.parquet` | 5a | ~737K | chunk_id, sent_neg, sent_neu, sent_pos |
| `chunk_commitment_cascade.parquet` | 5b | ~737K | chunk_id, buyin_label, stance_label, prob_* |
| `doc_sentiment.parquet` | 6 | ~550K | doc_id, sent_label, commit_net, topic_macro |
| `doc_divergence_v2.parquet` | 7 | varies | post_id, upvote_weighted_div, topic_macro |
| `topic_discourse_intensity.parquet` | 7 | 17 | topic_macro, discourse_intensity |
| `temporal_sentiment.parquet` | 8 | varies | year_month, subreddit, topic_macro, pct_neg |
| `temporal_sentiment_overall.parquet` | 8 | varies | year_month, subreddit, net_buyin, net_stance |
| `chunk_faiss.index` | RAG | 737K vectors | 768-dim embeddings |
| `chunk_metadata.parquet` | RAG | ~737K | chunk_id, text_snippet, topic_macro, upvotes |
| `rag_topic_digests.json` | RAG | 17 entries | Per-topic summaries |
| `rag_temporal_narratives.json` | RAG | varies | Per-month narratives |
| `rag_fact_table.parquet` | RAG | varies | Pre-computed stats at multiple granularities |

### Human-labelled evaluation data

| File | Rows | Purpose |
|---|---|---|
| `gold_eval_v2.csv` | 54 | Commitment gold set (subset) |
| `gold_eval_v2_balanced.csv` | 180 | Commitment gold set (class-balanced) |
| `commitment_manual_annotations.csv` | 947 | Manual buyin + stance annotations |
| `cascade/stage{1a,1b,2a,2b}_train.csv` | varies | Cascade classifier training sets |
| `cascade/stage{2a,2b}_silver_v2.csv` | 6K / 8K | Gemma 4 silver labels |
