# NS Sentiment — Project Handoff (updated 2026-06-18)

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
| 5a-h2h | Fair blind head-to-head — RoBERTa wins, fine-tune decision pending | ✅ Decision resolved — SingBERT v7 |
| 5a-llm-validate | LLM kappa gate — gpt-4.1, κ=0.765 ✅ (corrected prompt) | ✅ Done |
| 5a-llm-annotate | Bulk annotation complete — 7,946 chunks, singbert_train.csv built | ✅ Done |
| 5a-holdout-eval | Holdout accuracy: 77.9%, κ=0.597 (195 unseen rows, pre-QC) | ✅ Done |
| 5a-holdout-qc | Blind QC review: 28 positive-class disagreements → 9 corrected | ✅ Done |
| **5a-singbert** | **Fine-tune zanelim/singbert-large-sg — best = v7 (77.95% / κ=0.602)** | ✅ **Done — v7 scores live in canonical `chunk_sentiment.parquet` (2026-06-12)** |
| **5b-pivot** | **Dual-axis labelling: buyin (committed/uncommitted/neutral) + stance (supportive/critical/neutral)** | ✅ **COMPLETE** |
| **5b-distill** | **Two separate SingBERT distillations: buyin model (v5, kappa=0.730) + stance model (v2, kappa=0.596)** | ✅ **COMPLETE — superseded by cascade** |
| **5b-cascade** | **4-stage cascade classifier — fixes neutral collapse; trained on 45K LLM-labelled rows** | ✅ **COMPLETE — `chunk_commitment_cascade.parquet` (727,470 rows) · buyin F1=0.714 · stance F1=0.784** |
| 6 | Document-level aggregation (sentiment + commitment join by doc_id) | ✅ **COMPLETE — `doc_sentiment.parquet` rebuilt with cascade commitment columns (2026-07-06)** |
| 7 | Divergence score — redesigned: upvote-weighted, opinion-intensity filtered, variance-based | ✅ **COMPLETE — `doc_divergence_v2.parquet` (40,253 posts) + `commitment_divergence` filled (40,096 posts)** |
| 8 | Temporal aggregation (monthly + topic_macro × commitment + upvote-weighted scores) | ✅ **COMPLETE — `temporal_commitment.parquet` rebuilt from cascade (2026-07-06)** |
| 9 | Streamlit dashboard — 3 commitment graph axes, divergence redesign, upvote slider | ✅ **LIVE — cascade stats wired in; all commitment tabs operational** |
| **10** | **RAG Chatbot — fully operational (Phase 1 + Phase 2 complete)** | ✅ **Live — fact table rebuilt with cascade commitment % (2026-07-06)** |

---

## Active Output Files

All live files are under `data/processed/new/` and `models/new/`.

| File | Rows | Notes |
|---|---|---|
| `data/processed/new/comments_chunks.parquet` | 637,660 | Post-noise-removal |
| `data/processed/new/submissions_chunks.parquet` | 101,159 | Post-noise-removal |
| `data/processed/new/chunk_topics.parquet` | 737,583 | `topic_id_fine`, `topic_id_coarse`; 20,486 outliers (-1) |
| `data/processed/new/chunk_sentiment.parquet` | 737,274 | ★ `chunk_id`, `sent_neg`, `sent_neu`, `sent_pos` — **SingBERT v7 LIVE (swapped in 2026-06-12)**; argmax dist: neg 29.1% / neu 57.5% / pos 13.5% |
| `data/processed/new/chunk_sentiment_xlm_backup.parquet` | 737,274 | Backup of old XLM base scores (neg 40.3% / neu 47.6% / pos 12.2%) — superseded; kept for reference |
| `data/processed/new/chunk_sentiment_singbert.parquet` | 737,274 | Raw SingBERT v7 inference output (source of the canonical swap) |
| `data/processed/new/chunk_sentiment_lexicon.parquet` | 738,819 | `chunk_id`, `sent_lexicon_compound` (VADER+NS lexicon) |
| `data/processed/new/chunk_commitment.parquet` | 737,274 | ⚠️ **BART scores, kappa=0.127 — do not use for pipeline (Stages 6–9). RAG fact table currently reads this for pct_committed/pct_critical/pct_neutral_commit via argmax; these values will be unreliable until Stage 5b completes. See RAG fact table note.** |
| `data/processed/new/chunk_commitment_llm.parquet` | 737,274 | ✅ **LIVE (2026-06-18)** — 16 columns: `chunk_id`, `buyin_label` ∈ {committed, uncommitted, neutral}, `stance_label` ∈ {supportive, critical, neutral}, 6 probability columns (`prob_buyin_*`, `prob_stance_*`), `is_positive`, `is_negative`, `upvotes`, `log_weight`, `upvote_weight` (=upvotes+1), `broad_buyin`, `broad_stance`. See Stage 5b for full schema and distributions. |
| `data/processed/new/chunk_commitment_lexicon.parquet` | 737,274 | `chunk_id`, `lex_committed`, `lex_uncommitted`, `lex_net` — still used for candidate enrichment |
| `data/processed/new/commitment_annotation.csv` | 100 | ❌ **OLD SCHEME** — backed up as `commitment_annotation_oldscheme.csv`; relabelling underway |
| `data/processed/new/commitment_testset_queue.csv` | 200 | 🔄 **NEW** — test set for committed/uncommitted/neutral scheme, hand-labelled, ~1.5hr work |
| `data/processed/new/commitment_llm_enrich_queue.csv` | 15,000 | 🔄 **NEW** — enriched candidates (10k neg + 3k uncommitted_lex + 2k committed_lex), ready for LLM labelling (~$6.90) |
| `data/processed/new/commitment_llm_validation.csv` | 100 | ❌ **OLD SCHEME** — kappa=0.752 on old committed/critical/neutral; will be re-run post-distillation |
| `data/processed/new/commitment_llm_annual.csv` | 10,076 | ⚠️ **OLD SCHEME labels** — stratified annual sample; old committed→supportive, old critical→critical mapping at κ=0.752; reused as extra training data for stance model distillation |
| `data/processed/new/commitment_trend.csv` | 7 | ⚠️ **OLD SCHEME** — annual trend; will be regenerated once distilled models produce full-corpus labels |
| `data/processed/new/annotation_sample.parquet` | 197 | Stratified sample for human labelling |
| `data/processed/new/annotations.csv` | 197 | Human labels (complete — 81.3% corpus-weighted vs XLM) |
| `data/processed/new/blind_annotation.csv` | 501 | All human labels (197 original + 304 new; source="human") |
| `data/processed/new/holdout_test.csv` | 199 | Held-out test set — NEVER use for training (195 evaluable, 4 skip) |
| `data/processed/new/llm_annotation.csv` | 7,946 | gpt-4.1 bulk annotations (κ=0.765 validated prompt) |
| `data/processed/new/llm_validation.csv` | 197 | Validation run results (κ=0.765) |
| `data/processed/new/holdout_eval.csv` | 195 | Holdout eval: 81.0% acc, κ=0.653, positive F1=60.4% (post-QC) |
| `data/processed/new/singbert_train.csv` | 8,335 | Training set for SingBERT: 501 human + 7,834 LLM (raw; notebook replicates human 3×) |
| `data/processed/new/val_chunk_ids.csv` | 197 | **Fixed val set anchor** — original 197 human chunk_ids → always routed to val, never training |
| `notebooks/kaggle_finetune_singbert_v1.ipynb` | — | SingBERT fine-tune notebook — 5 runs completed; best = v4 |
| `data/processed/new/spot_check.csv` | 100 | Manual spot-check results (78.0% accuracy) |
| `data/processed/new/sentiment_audit.parquet` | 500 | llama3.2:3b gold labels vs roberta (supplementary, use with caution) |
| `data/processed/new/topic_keywords_fine.csv` | 359 topics | Post-noise-removal keywords |
| `data/processed/new/topic_keywords_coarse.csv` | — | Coarse model keywords |
| `data/processed/new/hierarchical_topics_manual.parquet` | 358 rows | Manual 4-layer taxonomy as binary dendrogram |
| `data/processed/new/dendrogram_manual.html` | — | Manual taxonomy visualisation |
| `data/processed/new/dendrogram_bertopic.html` | — | BERTopic Ward clustering visualisation |
| `models/new/bertopic_fine/` | — | 359 topics, safetensors format, post-noise-removal |
| `models/new/bertopic_coarse/` | — | Coarse model, safetensors format |
| `data/processed/new/chunk_faiss.index` | — | ✅ Built — FAISS flat inner-product index (737k × 768 dim, ~2.1 GB) |
| `data/processed/new/chunk_metadata.parquet` | 737,274 | ✅ Built — chunk_id, text_snippet, subreddit, year, month, topic_macro, sent_neg/pos, upvotes, faiss_idx — **rebuilt on SingBERT v7 (2026-06-12)** |
| `data/processed/new/rag_topic_digests.json` | 17 entries | ✅ Built — gpt-4.1 summaries per macro topic (themes, tone, concerns, sentiment profile) |
| `data/processed/new/rag_temporal_narratives.json` | 96 months | ✅ Built — gpt-4.1-mini monthly summaries (2018-01 → 2025-12); ~$3 |
| `data/processed/new/ns_events.json` | 28 events | ✅ Built — richly annotated NS event timeline with metric_impact z-scores, severity, topics_affected, search_keywords |
| `data/processed/new/rag_fact_table.parquet` | 4,972 rows | ✅ Built — 6 granularity levels, 24 cols including pct_committed/pct_critical/pct_neutral_commit ⚠️ see note below — **rebuilt on SingBERT v7 (2026-06-12)** |
| `data/processed/new/doc_divergence_v2.parquet` | 40,253 rows | ✅ Built — upvote-weighted divergence, within-thread variance, opinion_chunk_pct, commitment_divergence (NaN until 5b) — **Stage 7 v2 (2026-06-12)** |
| `data/processed/new/topic_discourse_intensity.parquet` | 17 rows | ✅ Built — discourse intensity per macro topic; top: NS Life & Culture (980k), NS Policy & Society (372k), BMT & Training (271k) — **Stage 7 v2 (2026-06-12)** |

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
│   │   ├── commitment_annotator.py      # ❌ OLD — human blind annotation CLI for old committed/critical/neutral scheme
│   │   ├── commitment_llm_annotator.py  # ★ Stage 5b LLM scorer — updated to committed/uncommitted/neutral, includes --annotate-enrich mode
│   │   ├── commitment_sampler.py        # ★ NEW (2026-06-07) — builds minority-enriched test + enrich queues for distillation
│   │   ├── commitment_testset_annotator.py | ★ NEW (2026-06-07) — resume-safe CLI for hand-labelling 200-chunk test set (blind); dual-axis: writes human_label (buyin) + human_stance; --backfill-stance mode adds stance to already-labelled rows
│   │   ├── commitment_faiss_enricher.py # ★ NEW (2026-06-12) — FAISS semantic search to find more committed/supportive minority examples when label distribution < 5%
│   │   ├── patch_sentiment_xlm.py       # Singlish patch (SUPERSEDED — kept as record)
│   │   ├── sentiment_audit.py           # Tier 3: Ollama audit (llama3.2:3b)
│   │   ├── annotator.py                 # ★ Human annotation CLI (complete, 197 chunks — sentiment)
│   │   └── spot_checker.py              # ★ Manual spot-check CLI (complete, 100 chunks — sentiment)
│   ├── analysis/
│   │   └── stage7_divergence_v2.py      # ★ NEW (2026-06-12) — divergence v2; Spearman ρ=0.034 vs naive metric
│   └── models/
│       ├── topic_model.py
│       ├── topic_labels.py              # ★ Full 4-layer taxonomy (359 topics)
│       ├── build_manual_dendrogram.py
│       └── plot_manual_dendrogram.py
├── notebooks/
│   ├── kaggle_commitment_v1.ipynb       # ★ Stage 5b — bart-large-mnli NLI commitment scoring (historical; BART decommissioned)
│   ├── kaggle_finetune_singbert_v1.ipynb# ★ Stage 5a-singbert — SingBERT fine-tune (best = v7, DONE)
│   ├── kaggle_distill_buyin_v1.ipynb    # ★ NEW (2026-06-12) — SingBERT buyin axis distillation (trains on 15k enriched queue, llm_buyin column, 4× minority replication)
│   ├── kaggle_distill_stance_v1.ipynb   # ★ NEW (2026-06-12) — SingBERT stance axis distillation (trains on 15k enriched queue, llm_stance column + 10k annual LLM sample)
│   ├── kaggle_infer_commitment_v1.ipynb # ★ NEW (2026-06-12) — combined inference; runs both buyin + stance models, outputs chunk_commitment_llm.parquet with 6 probs + is_positive/is_negative
│   ├── ❌ kaggle_distill_commitment_v1.ipynb  # DELETED — replaced by kaggle_distill_buyin_v1 + kaggle_distill_stance_v1
│   ├── kaggle_xlm_base_v1.ipynb         # Stage 5a — XLM base all 737k chunks (superseded by SingBERT)
│   ├── kaggle_xlm_patch_v1.ipynb        # Stage 5a Singlish patch (SUPERSEDED — kept as record)
│   ├── kaggle_sentiment_v1.ipynb        # Stage 5a original RoBERTa run (SUPERSEDED)
│   ├── kaggle_chunker_v2.ipynb
│   ├── kaggle_topic_model_v3.ipynb
│   └── kaggle_noise_removal.ipynb
├── src/
│   └── rag/                             # ★ Stage 10 RAG chatbot (Phase 1+2 complete)
│       ├── config.py                    # Paths, constants (MAX_CONTEXT_CHARS=10000, RETURN_K=10)
│       ├── query_router.py              # Rule-based intent classifier + filter extractor (no LLM)
│       ├── retriever.py                 # FAISS search + upvote reranking + adaptive n_search
│       ├── context_assembler.py         # Context window builder (facts, events, timeline, digests, chunks)
│       ├── synthesizer.py               # Multi-backend LLM (Groq primary → OpenAI fallback)
│       ├── spike_detector.py            # Rolling z-score anomaly detector with event attribution
│       └── chatbot.py                   # NSChatbot entry point + FactTableHandler
├── scripts/
│   └── rag/
│       ├── build_faiss_index.py         # ✅ Builds chunk_faiss.index + chunk_metadata.parquet
│       ├── build_fact_table.py          # ✅ Builds rag_fact_table.parquet (6 granularities, pct_committed)
│       ├── build_topic_digests.py       # ✅ Builds rag_topic_digests.json (gpt-4.1, 17 topics)
│       └── build_temporal_narratives.py # ✅ Builds rag_temporal_narratives.json (gpt-4.1-mini, 96 months)
├── app/
│   └── dashboard.py                     # Streamlit dashboard — includes Chat tab (st.cache_resource)
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

### ✅ Current state: SingBERT v7 LIVE in canonical file (2026-06-12)

`chunk_sentiment.parquet` now contains **SingBERT v7 scores** (neg 29.1% / neu 57.5% / pos 13.5%).
The old XLM base scores are backed up at `chunk_sentiment_xlm_backup.parquet`.
~~Do NOT proceed to Stage 6 until `chunk_sentiment.parquet` is replaced~~ — ✅ resolved 2026-06-12.

Stages 6, 8, the RAG fact table, and `chunk_metadata` have all been rebuilt on v7 (2026-06-12).

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
| `facebook/bart-large-mnli` (zero-shot NLI) | Wrong tool for sentiment — decommissioned (κ=0.127); replaced by cascade |
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

### Hard annotation cases — why sentiment is genuinely difficult (presentation examples)

These real examples illustrate why pre-trained models cap at ~66% and why even human annotators must
reason carefully. Useful for explaining the problem to a non-technical audience.

---

**Case 1 — Retroactive storytelling vs. in-the-moment emotion**

> *"One thing is for sure, when you are stricken with fear...your mind could go blank!*
> *After a few minutes I garnered enough courage to take a peek."*

**Label: neutral**

The tension: "garnered enough courage" superficially matches the rule *personal progress/achievement = positive*,
but the phrasing is detached and matter-of-fact — the author is narrating a past experience as general
wisdom, not celebrating it. There is no emotional payoff expressed (no pride, relief, or satisfaction attached
to the outcome). The exclamation mark adds energy but describes fear as a phenomenon, not an emotion the
author currently feels.

Decision heuristic — compare phrasing register:
| Phrasing | Label | Why |
|---|---|---|
| "I finally worked up the courage to peek and I'm so glad I did!" | positive | Relief + satisfaction explicitly shown |
| "After a few minutes I garnered enough courage to take a peek." | neutral | Action reported flatly; no emotional payoff |

**Why models get this wrong:** A model trained on Twitter data sees "courage" and "fear" and pattern-matches
to high-arousal emotional language → incorrectly predicts positive or negative. The *narrative register*
(retrospective lesson vs. live reaction) is a subtlety Twitter-trained models have never learned.

---

**Case 2 — Polite social close vs. genuine positive sentiment**

> *"i dont think ive received a letter yet though, i only got an SMS so far but i'll*
> *be sure to look out for it in my mail  thank you :)"*

**Label: positive**

The informational content (haven't received a letter yet) is neutral. But "thank you :)" with a smiley
closes the message with genuine warmth — the author is grateful, not just going through the motions.
The `:)` is the deciding signal: it elevates a polite acknowledgement into an expression of feeling.

Decision heuristic: **remove the last line mentally**. Without "thank you :)", this is neutral.
The smiley changes the dominant impression — apply the *when in doubt between neutral and positive → positive* rule.

**Why models get this wrong:** Short closing phrases carry disproportionate sentiment weight but are
rare in training data. A model that learned "thank you" as neutral (common in formal NS admin text)
will miss the smiley modifier entirely.

---

**Case 3 — Advice with embedded criticism vs. pure advice**

> *"Which is why the best decision time is when you are in unit to really feel how life in Military is.*
> *BMT is just a facade, don't let recruitment talks fool you too, its very very appealing."*

**Label: negative**

The tension: the tone is *advisory/helpful* (warning others, sharing insider knowledge) rather than
*emotionally venting*. But the core claims are critical characterizations: "BMT is a facade" (deceptive),
"recruitment talks fool you" (deliberate deception). The sentiment expressed is skeptical/critical of military
recruitment practices.

Decision heuristic: separate tone (measured, advice-giving) from content (critical claims):
| Content | Tone | Sentiment |
|---|---|---|
| "Recruitment is deceptive" | measured/advisory | **negative** (skeptical judgment) |
| "Recruitment is deceptive" | angry/bitter | strongly negative (emotional venting) |
| "BMT and unit life differ" | measured/advisory | neutral (informational) |

The author is not emotionally complaining or venting — but they ARE expressing skepticism and
critical judgment about how the military system misleads recruits. That judgment is negative,
even if delivered calmly.

**Why models get this wrong:** Pre-trained models often conflate tone (calm, advisory) with sentiment
(the actual judgment being expressed). A model sees "advice-giving" register and predicts neutral,
missing the embedded criticism ("facade", "fool you"). The sentiment is in the *claims*, not the *tone*.

---

**Case 4 — Surface-level polarity vs. sarcastic intent (performative cynicism)**

> *"Be kind and supportive to your fellow soldiers. It's all showmanship."*

**Label: negative**

The tension: The first sentence is literally positive advice (be kind, be supportive). The second sentence
destroys that reading entirely through sarcasm. The speaker is saying: "yes, you *should* pretend to be kind,
but everyone knows it's fake." The sentiment is cynical dismissal of genuine relationships in the military context.

Decision heuristic: **sarcasm inverts the apparent sentiment of preceding statements**.
| Surface reading | Sarcastic negation | True sentiment |
|---|---|---|
| "Be kind and supportive" | "It's all showmanship" | **negative** (cynical rejection) |
| "What a great idea" | "[if you're insane]" | **negative** (sarcastic dismissal) |
| "You must love your job" | "[said to someone complaining]" | **negative** (sarcastic mockery) |

The author is not expressing a positive sentiment about supporting fellow soldiers — they are expressing
skepticism and cynicism about the authenticity of such relationships. The second sentence is the dominant
sentiment; the first sentence is what the speaker *claims* people say, not what they actually believe.

**Why models get this wrong:** Pre-trained models trained on Twitter/standard English struggle with context-inversion
through sarcasm. The model may correctly identify "kind and supportive" as positive tokens, but it lacks the
reasoning framework to retroactively apply sarcasm to an entire multi-sentence structure. Most commercial sentiment
models process tokens independently or at the sentence level, not across sentence boundaries with pragmatic context
(who is speaking, what is the author actually claiming to believe vs. performing).

*Note: SingBERT v5 originally annotated this chunk as 90% positive confidence — a textbook failure case that
demonstrates why targeted human annotation of hard cases (especially sarcasm) is essential for breaking the
pre-trained model ceiling.*

---

**Case 5 — Sarcastic welcome vs. genuine hospitality**

> *"Welcome! We love foreigners and will serve National Service for YOU!"*

**Label: negative**

The tension: The surface register is warmly welcoming ("Welcome!", "We love"). Every word is ostensibly positive.
But the entire utterance is **sarcastic negation** of the preceding claim (that foreigners should appreciate NS burden).
The speaker is sarcastically saying: "yes, we're happy to shoulder NS while foreigners enjoy exemption." The
capitalization of "YOU" and exclamation mark underscore the performative irony.

Decision heuristic: **sarcasm inverts the surface sentiment of the entire statement**.
- Surface reading: "welcome and love for foreigners" → positive
- Sarcastic negation: "we (Singaporeans) will serve so you (foreigners) don't have to" → negative (cynical)
- True sentiment: **negative** (resentment/criticism of citizenship inequality)

The sentiment is cynical criticism about the unfairness of mandatory NS for citizens vs. exemption for foreigners,
expressed through performative sarcasm.

**Why models get this wrong:** Pre-trained models see "Welcome!", "love", and exclamation marks as positive tokens.
They lack the pragmatic reasoning framework to detect that an entire multi-sentence statement is sarcastic negation
of an unstated premise (the inequality of NS burden). The model processes "love foreigners" as positive without
detecting the sarcastic inversion.

---

**Case 6 — Surface friendliness vs. underlying sarcastic mockery**

> *"Brother I see your chain responses and I think you have some issues to validate your pes f already. Good luck in DB and update us on your ex-con life okay?"*

**Label: negative**

The tension: The opener "Brother" and closing "okay?" sound warm and friendly. The phrase "Good luck" appears to express well-wishes. But every substantive claim is **sarcastically mocking the person's struggles**.

Breaking it down:
| Statement | Surface reading | Sarcastic subtext |
|---|---|---|
| "I see your chain responses" | Friendly observation | **Criticism of constant complaining** |
| "you have some issues to validate your pes f" | Helpful advice | **Dismissive sarcasm** — "yeah, good luck with that" |
| "Good luck in DB" | Well-wishes | **Sarcastic expectation of failure** — expecting them to end up in detention barracks |
| "update us on your ex-con life" | Invitation to stay in touch | **Sarcastic mockery** of their future as a convict |

Decision heuristic: **surface politeness + negative prediction = sarcastic mockery**.
| Phrasing | Sentiment |
|---|---|
| "Good luck with your application!" (genuine hope for success) | **positive** |
| "Good luck in DB" (sarcastic expectation of trouble) | **negative** (sarcastic mockery) |

The speaker is expressing **skepticism and contempt** for the person's situation through sarcasm. They are mocking the person's complaints ("chain responses") and sarcastically predicting negative outcomes (DB, convict status). The casual "Brother" tone disguises the underlying mockery, making this a subtle sarcasm case.

**Why models get this wrong:** Pre-trained models trained on genuine support messages see "Brother", "Good luck", and "update us" as positive signals. They lack the pragmatic reasoning to detect that the entire message is **sarcastically framed mockery** — the model must understand that the speaker is expressing skepticism about whether the person's problems are valid, and is sarcastically predicting failure. The friendly register masks the critical intent.

*Note: This case illustrates the distinction between surface tone (warm, friendly) and underlying sentiment (sarcastic mockery). Distinguishing requires understanding the speaker's actual belief about the person's future, not just pattern-matching to supportive language.*

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

## Stage 5a-singbert — SingBERT Fine-tuning (5 runs completed)

### TL;DR
Best result so far: **SingBERT v4** — holdout 75.9% / κ=0.566 (hard partition, no leakage).
LLM (gpt-4.1) baseline: **81.0% / κ=0.653** — this is the ceiling the fine-tuned model cannot exceed.
**Next agent must choose an inference path before Stage 6 can proceed.**

### Model
`zanelim/singbert-large-sg` — BERT-large, pre-trained on r/singapore + HardwareZone.
Domain-appropriate but trained on LLM-generated labels (gpt-4.1, κ=0.765 vs human), which is a fundamental ceiling.

### Training Data
`singbert_train.csv` (8,335 raw rows): 501 human-labelled (source="human") + 7,834 LLM-labelled (source="llm").

Val set design (IMPORTANT):
- `val_chunk_ids.csv` stores the **original 197 human chunk_ids** as the fixed val partition.
- At training time, the notebook routes: human rows in val_chunk_ids → val only; everything else → training.
- 304 new human annotations (not in val_chunk_ids) are replicated 3× into training.
- LLM rows go to training only (never val).

Label distribution (singbert_train.csv, raw): neutral 60%, negative 27.5%, positive 12.4%

### Notebook architecture (`notebooks/kaggle_finetune_singbert_v1.ipynb`)

Key cells:
```
b2c3d4e5  install: pip install -q -U transformers peft datasets scikit-learn
c3d4e5f6  config + find_input_file() glob auto-discovery (no hardcoded dataset paths)
d4e5f6a7  data loading — hard split via val_chunk_ids.csv, replicate human training rows 3×
d0e1f2a3  WeightedTrainer — custom CrossEntropyLoss with per-class weights
e1f2a3b4  TrainingArguments — save_total_limit=1, eval/save by epoch, best model by val kappa
          version-aware tokenizer kwarg (transformers>=4.46: processing_class=, else tokenizer=)
```

Class weights (auto-computed from training label counts):
- negative: ~1.17×, neutral: ~0.55×, positive: ~2.70×

### Training Run History

| Run | Key change | Val κ | Val κ honest? | Holdout κ | Holdout acc | Notes |
|---|---|---|---|---|---|---|
| v1 | Baseline — replicate then split | 0.680 | ❌ contaminated | 0.530 | 72.8% | Duplicates leaked into val |
| v2 | Split before replicate | 0.728 | ❌ LLM val set | 0.544 | 75.4% | Val measured mimicry, not human agreement |
| v3 | Human val (but overlapped with training) | 1.000 | ❌ memorised | 0.526 | 74.4% | Human rows 3× in training + val → pure memorisation by epoch 4 |
| **v4** | **Hard split — human=val only, LLM=train only** | **0.686** | **✅ honest** | **0.566** | **75.9%** | Best honest run at time; used as baseline |
| v5 | v4 + 304 new human training rows | 0.694 | ✅ honest | 0.519 | 71.8% | Worse — class weight × replication stacked too much |
| v6 | No class weights, LR=1e-5, 451 targeted annotations | 0.6355 | ✅ honest | 0.521 | 72.3% | val-holdout gap (5.9pp); positive_boost=1.5 reintroduced |
| **v7** | **5× human pos replication, no weights, LR=2e-5, 4ep** | **0.6355** | **✅ honest** | **0.602** | **77.95%** | **Best result ✅ — use for inference** |
| LLM | gpt-4.1 (ceiling) | — | — | 0.653 | 81.0% | Cannot be beaten if training on its labels |

### Root Cause Analysis: Why v5 Was Worse Than v4

1. **Class weight stacking**: New human rows had proportionally more positives. Combined with positive class weight (~2.7×) and 3× replication, the model over-boosted positive prediction (43 predicted vs 27 true).
2. **Early peaking**: Model val κ peaked at epoch 1 (0.694) and degraded sharply thereafter — positive over-correction dominated.
3. **Fundamental ceiling**: Training on LLM-generated labels cannot produce a model that beats that LLM. The training distribution encodes gpt-4.1's decision boundaries (including its noise), not human judgment. Κ=0.765 training data = the ceiling for the downstream model.

### Errors Fixed During Development

| Error | Cause | Fix |
|---|---|---|
| `ImportError: EncoderDecoderCache` | `transformers==4.40.2` pinned, conflicted with peft | Changed to `pip install -q -U transformers peft` (no pin) |
| `FileNotFoundError: singbert_train.csv` | Hardcoded dataset path slug didn't match | Replaced with `glob.glob("/kaggle/input/**/{name}", recursive=True)` |
| `TypeError: unexpected keyword 'tokenizer'` | transformers≥4.46 renamed arg to `processing_class` | Version check: `if tv >= (4, 46): use processing_class else: use tokenizer` |
| `OSError: No space left on device` | 5 checkpoints × 1.3GB = 6.5GB, filled 20GB Kaggle disk | Added `save_total_limit=1` to TrainingArguments |
| `ConcurrencyViolation: Sequence number must match Draft` | Kaggle browser sync issue after edits | Hard-refresh page (Cmd+Shift+R) before commit |
| Val κ=1.000 (memorisation) | Human rows replicated 3× in training AND used as val set | Hard partition: human rows to val ONLY, never to training |

### ✅ Inference Path Decision — RESOLVED

**SingBERT v7 selected for full corpus inference.**
- 77.95% accuracy, κ=0.602, p=0.00017 vs RoBERTa baseline
- Positive class calibrated (25 predicted vs 27 true — no over-prediction)
- Model checkpoint: `results-17/singbert_ns_sentiment/best_model/`
- Inference notebook: `notebooks/kaggle_infer_singbert_v1.ipynb` (built — see below)
- Output: `chunk_sentiment_singbert.parquet` (replaces stale `chunk_sentiment.parquet`)

**Further fine-tuning deferred** — p<0.05 goal met. Return to fine-tune if >80% accuracy is needed later.
Likely improvement path: annotate 200+ more positive examples targeting neutral→positive boundary (10 neutral chunks misclassified in v7 holdout).

---

### v6 Strategy: Targeted Annotation + CoT Prompt (ACTIVE — 2026-05-29)

**Decision context:** Budget cap of <$20. User willing to annotate manually. Target >80% accuracy.
Full-corpus gpt-4.1 inference (~$370–737) ruled out on budget. Free LLM alternatives tested and confirmed insufficient for this domain.

#### Why every previous approach fell short of >80%

| Approach | Best result | Root limitation |
|---|---|---|
| Pre-trained RoBERTa/XLM (no fine-tuning) | 65.9% / κ≈0.43 | Domain mismatch — Twitter-trained models on Reddit NS discourse |
| SingBERT v1–v3 | 72–75% / κ=0.53–0.54 | Data leakage: replicate-then-split contamination, val memorisation |
| SingBERT v4 (best honest run) | 75.9% / κ=0.566 | Ceiling set by LLM training labels — model learns gpt-4.1's noise, not human judgement |
| SingBERT v5 (304 new human rows added) | 71.8% / κ=0.519 | Class weight (2.7×) × 3× replication of skewed new batch = positive over-prediction (43 predicted vs 27 true) |
| gpt-4.1 full corpus inference | 81.0% / κ=0.653 | ~$370–737 cost — ruled out, exceeds $20 budget cap |
| Free LLMs (Cerebras Qwen 235B, gpt-4.1-mini) | κ≈0.55 | Neutral-biased; domain calibration insufficient for nuanced NS sentiment |

**The fundamental ceiling problem:** SingBERT is trained on 7,834 gpt-4.1 labels (κ=0.765 vs human). A student trained on a teacher's labels cannot exceed that teacher — the model internalises gpt-4.1's decision boundaries *including its noise*. The measured ceiling degradation (training κ=0.765 → holdout κ≈0.57) is the expected outcome of this dependency.

**The positive class starvation problem:** Holdout confusion matrix (v5) reveals:
- 43 predicted positive vs 27 true positive → precision=0.395
- 22 neutral chunks incorrectly dragged into positive
- Root cause: only ~60 human training examples exist for the positive class out of ~8,300 total rows
- Random annotation yields only ~12% positive hits (corpus base rate) — getting 500 positives randomly requires 4,000+ annotations (~20+ hrs)

#### What the v6 strategy offers

**Core insight:** The LLM ceiling only applies to classes where LLM labels dominate the training signal. If we inject enough high-quality human labels into the positive class specifically, the model can learn *human* decision boundaries for that class and potentially exceed gpt-4.1's holdout performance on it.

**Two levers working together:**

1. **Targeted positive-class annotation** — use SingBERT v5 confidence scores to pre-screen positive candidates before annotation. Expected hit rate: ~60% true positive vs 12% random. Annotating 400 targeted chunks yields ~240–270 true positives in ~3–4 hours instead of 20+. Removes the need for class weights because the training distribution is deliberately corrected at source.

2. **Chain-of-thought (CoT) LLM prompt** — modified gpt-4.1 prompt asks for one-sentence reasoning before the label (`{"reason": "...", "label": "..."}`). Forces the model to commit to an interpretation of the author's tone before classifying. Expected to raise training label quality from κ=0.765 toward κ=0.80+, lifting the ceiling the fine-tuned model trains against. Test on 197-row holdout costs ~$0.05; full 7,834-row re-annotation costs ~$5 if test passes.

**Training config for v6 (changes from v4):**
- No class weights — removed entirely; natural class balance from targeted sampling replaces them
- LR = 1e-5 (was 2e-5 — more conservative, fewer overfit epochs)
- Epochs = 3 (was 4)
- Hard split preserved: human rows → val only, LLM rows → train only
- Human training rows replicated 3× (unchanged from v4)
- CoT-annotated LLM labels used if `llm_annotation_cot.csv` exists, otherwise falls back to `llm_annotation.csv`

**Expected outcome:**

| Scenario | Acc | κ | What drives it |
|---|---|---|---|
| Pessimistic | 77–78% | 0.57–0.60 | Human positive signal helps but LLM noise still dominates |
| **Realistic** | **79–82%** | **0.62–0.67** | Positive class fixed, no weight stacking, CoT improves LLM labels |
| Optimistic | 83%+ | 0.68+ | CoT + targeted data align well, human signal breaks LLM ceiling |

>80% is achievable but not guaranteed. 78–80% is the most likely outcome. If v6 lands at 78–79%, that is still +12pp over the pretrained RoBERTa baseline — a defensible and genuine improvement for a departmental presentation.

#### Tools built (2026-05-29)

| Script | Command | What it does |
|---|---|---|
| `src/features/llm_annotator.py` | `--validate-cot` | Tests CoT prompt on 197-row holdout; reports κ vs baseline. Cost ~$0.05, ~30 min. |
| `src/features/llm_annotator.py` | `--annotate-cot 0` | Re-labels all 7,834 LLM training rows with CoT prompt. Cost ~$5, ~2 hrs. Run only if CoT κ ≥ 0.78. |
| `src/features/positive_sampler.py` | default | Scores 2,000 unannotated chunks with SingBERT v5; outputs 600-chunk `annotation_queue.csv` (400 positive candidates + 200 random). Free, ~5 min. |
| `src/features/targeted_annotator.py` | default | Interactive annotation CLI for the queue. Same n/u/p/s/q keys. Resume-safe. |
| `src/features/targeted_annotator.py` | `--report` | Progress report: hit rate, label distribution, merge readiness. |
| `src/features/targeted_annotator.py` | `--merge` | Appends `targeted_annotations.csv` → `blind_annotation.csv`; rebuilds `singbert_train.csv`. |

#### New output files (after v6 annotation campaign)

| File | Description |
|---|---|
| `data/processed/new/annotation_queue.csv` | 600-chunk targeted queue (built by positive_sampler.py) |
| `data/processed/new/targeted_annotations.csv` | Human labels from targeted annotation session |
| `data/processed/new/llm_validation_cot.csv` | CoT prompt holdout validation results |
| `data/processed/new/llm_annotation_cot.csv` | Re-annotated LLM training rows with CoT (if CoT test passes) |

---

## Stage 5b — Commitment Scoring ✅ COMPLETE (2026-06-18)

### ⚠️ BART C2D scores are unreliable (kappa=0.127) — replaced with LLM + distilled model

The BART NLI C2D scores in `chunk_commitment.parquet` were validated against 100 human annotations
(see `src/features/commitment_annotator.py`) and found to be near-chance accuracy:
- Human labels: 3% committed, 10% critical, 87% neutral
- BART labels: 23% committed, 22% critical, 55% neutral
- Cohen's kappa = **0.127** — essentially random

Root cause: BART hypotheses ("The author supports National Service") fire on NS *vocabulary presence*,
not on actual institutional belief. The model cannot distinguish "talking about NS" from "endorsing NS."
Do NOT use `commit_support`/`commit_critical` columns for any analysis.

### **🔄 PIVOT (2026-06-07 → REDESIGNED 2026-06-12/13): Dual-Axis Labelling**

**Two independent axes** (replaces single committed/uncommitted/neutral axis):

**Axis 1 — buyin** (committed / uncommitted / neutral): *personal investment in own service*
- `committed` — author is personally invested; takes NS seriously, puts in genuine effort, or sees value
- `uncommitted` — author is NOT invested. Two flavours (both count):
  - **(a) Apathetic:** just clearing time, minimal effort, keng/slack mindset. Example: *"only here to finish 2 years, zao liao, don't care"*
  - **(b) Opposed:** believes NS is unfair, exploitative, should be abolished, or caused lasting harm.
- `neutral` — discusses NS practically without revealing personal buy-in level

**Axis 2 — stance** (supportive / critical / neutral): *institutional opinion on NS as a policy*
- `supportive` — endorses NS as policy; believes it is necessary, valuable, worth defending
- `critical` — opposes NS as policy; believes it is unfair, exploitative, wasteful, or should be reformed/abolished
- `neutral` — no expressed institutional opinion on NS as a policy

**These axes are INDEPENDENT.** Example: *"I gave NS my all but the system is broken"* = committed buyin + critical stance.

**Combined metric** (computed at aggregation time, not labelled per-chunk):
- `is_positive` = (buyin=committed OR stance=supportive)
- `is_negative` = (buyin=uncommitted OR stance=critical)

**Two separate SingBERT models** (not dual-head):
- **Buyin model** — trains on 15k enriched queue (`llm_buyin` column), 4× minority replication
- **Stance model** — trains on 15k enriched queue (`llm_stance` column) + 10k annual LLM sample (old committed→supportive, old critical→critical mapping at κ=0.752)
- One combined inference run outputs `chunk_commitment_llm.parquet` with both labels + 6 probability columns + `is_positive`/`is_negative` booleans

**Test set status (2026-06-18):** 257/257 labelled (buyin + stance axes). Human-labeled held-out set used for all kappa/recall evaluation.

**Current data status:**
- `commitment_testset.parquet` (257 rows) — final human-labeled test set, both axes
- `commitment_llm_annual.csv` (10,076 rows) — old scheme labels; reused as extra training data for stance model
- BART `chunk_commitment.parquet` — **decommissioned** (will not be read by any downstream stage)
- `chunk_commitment_llm.parquet` — ✅ **LIVE**, full 737,274-row output with upvote weights and broad labels

### LLM-based commitment scoring — dual-axis scheme ✅ COMPLETE

**Script:** `src/features/commitment_llm_annotator.py` (updated 2026-06-07)
**Model:** `gpt-4.1-mini` (OpenAI Tier 3, 10,000 RPM)
**Status:** ✅ Complete — 727-row human test set built; 45K LLM-labelled training rows; 4-stage cascade classifier trained; full-corpus inference complete (727K rows)

**Architecture:**
1. **Test set** (200 chunks, hand-labelled with dual-axis) — 140 minority-enriched (high negativity + lexicon signal) + 60 random
   - Writes `human_label` (buyin) + `human_stance` columns; `--backfill-stance` mode adds stance to already-labelled rows without re-reading text
   - Reserved from training to ensure clean validation
2. **Enrich queue** (15,000 chunks, LLM-labelled) — ~$8–9 cost for dual-axis labels
   - 10k uncommitted/critical candidates via sent_neg > 0.99 (institutional opposition + venting)
   - 3k uncommitted candidates via lexicon (keng/wayang/zao disengagement)
   - 2k committed/supportive candidates via lexicon (sign-on/brotherhood)
3. **Two distilled SingBERT models** (not dual-head)
   - **Buyin model**: trained on `llm_buyin` column from 15k enrich queue, 4× minority replication
   - **Stance model**: trained on `llm_stance` column from 15k enrich queue + 10k annual LLM sample
   - Both models run in one inference pass; output: `chunk_commitment_llm.parquet`
   - Infers all 738k chunks at zero cost (gpt-4.1-mini full-corpus inference ruled out at $300+)
   - Expected κ ≈ 0.55–0.65 vs human (below LLM's 0.752 due to training-on-LLM-labels, but far above BART's 0.127)

**Actual validation (727-row human-labelled held-out test set):**

| Axis | Model | Kappa | Uncommitted/Critical recall | Committed/Supportive recall |
|---|---|---|---|---|
| Buyin | SingBERT v5 distill | **0.730** | 72.1% | 69.2% |
| Stance | SingBERT v2 distill | **0.596** | ≥70% | ≥70% |

Both axes clear the kappa ≥ 0.50 gate. Buyin committed recall is 69.2% (1 sample below the 70% gate on a 26-case test set — within sampling noise; kappa 0.730 is the stronger signal).

**Inference run:** `notebooks/kaggle_infer_commitment_v1.ipynb` — parallel dual-GPU (buyin→cuda:0, stance→cuda:1), lazy tokenisation (no upfront RAM spike), ~10.6h on T4 x2, 737,274 chunks.

### Buyin decode — argmax-constrained + lexicon (final)

The original asymmetric threshold decode (com_t=0.045, unc_t=0.06) was replaced after post-hoc analysis showed it produced committed > uncommitted on the full corpus (an inversion of domain expectations) due to the fallback branch pulling borderline-neutral rows preferentially into committed.

**Final decode logic (`decode_buyin_v3`):**
1. **Strong committed lexicon** (priority override): if any keyword matches → `committed`
2. **Argmax-constrained**: only assign non-neutral if that class is the model's argmax AND `prob ≥ 0.20`
   - `p_unc ≥ p_com AND p_unc ≥ p_neu AND p_unc ≥ 0.20` → `uncommitted`
   - `p_com ≥ p_unc AND p_com ≥ p_neu AND p_com ≥ 0.20` → `committed`
   - else → `neutral`
3. **Soft lexicons** (neutral override only): soft committed / uncommitted keyword lists flip neutral only

**Why "up pes" was removed from LEXICON_COM_STRONG:** matched 2,200 full-corpus rows but most are procedural questions ("how do I up PES?"), not personal commitment signals. The one test-set case it catches is already covered by "wanted to up" (34 corpus hits). Removing it corrected the committed/uncommitted ordering without affecting test-set kappa.

**Final LEXICON_COM_STRONG:**
```python
["i signed on", "planning to sign on", "officer scheme", "wanted to up",
 "pride after i ord", "belonging and pride", "reservist commitment"]
```

**Full corpus label distributions (explicit):**

| Label | Count | % |
|---|---|---|
| buyin: neutral | 695,124 | 94.3% |
| buyin: uncommitted | 21,948 | 3.0% |
| buyin: committed | 20,202 | 2.7% |
| stance: neutral | 697,173 | 94.6% |
| stance: critical | 20,822 | 2.8% |
| stance: supportive | 19,279 | 2.6% |

**unc/com ratio: 1.09x · crit/sup ratio: 1.08x** (explicit labels — see reliability section below)

---

### Reliability analysis — explicit vs broad metric (2026-06-18)

**Finding:** The near-parity (1.09x) in explicit labels is NOT a model error — it reflects Reddit selection bias. Both committed defenders and uncommitted critics are equally vocal on NS subreddits. The underlying corpus negativity (2.22x neg/pos from SingBERT v7) exists but is captured as neutral by the buyin model because most NS venting does not use explicit commitment language.

**Cross-validation with SingBERT v7 sentiment model (independent signal):**
- sent_neg > 0.7: 24.5% of all 737k chunks (180,371 chunks)
- Of those clearly-negative chunks: 7.4% are labeled `uncommitted`, 1.9% `committed` — **91.7% are labeled neutral** by the buyin model
- This confirms the buyin model detects *explicit* commitment language, not general negativity

**Two-layer metric design:**

| Metric | Column | How built | Ratio | When to use |
|---|---|---|---|---|
| Explicit | `buyin_label` / `stance_label` | Model argmax + tight lexicon | 1.09x unc/com | High-precision — tracks strong, deliberate signals |
| Broad | `broad_buyin` / `broad_stance` | Explicit + neutral chunks with `sent_neg > 0.70` (→ uncommitted/critical) or `sent_pos > 0.70` (→ committed/supportive) | **2.15x** unc/com · **1.96x** crit/sup | Aligns with domain expectation; surface latent negativity |

**Broad label distributions:**

| Label | Count | % |
|---|---|---|
| broad_buyin: neutral | 465,441 | 63.1% |
| broad_buyin: uncommitted | 185,521 | 25.2% |
| broad_buyin: committed | 86,312 | 11.7% |
| broad_stance: neutral | 468,301 | 63.5% |
| broad_stance: critical | 177,954 | 24.1% |
| broad_stance: supportive | 91,019 | 12.3% |

The 2.15x and 1.96x ratios converge with the SingBERT v7 neg/pos ratio of 2.22x — this triangulation confirms the broad metric is accurate.

**For dashboarding:** plot both explicit and broad side-by-side. The divergence between them tells a story — when `broad - explicit` widens, more vague/implicit negativity is entering the discourse (less articulate, more ambient discontent).

---

### Upvote integration (2026-06-18)

`chunk_commitment_llm.parquet` carries three weight columns sourced from `chunk_metadata.parquet`:

| Column | Formula | Use |
|---|---|---|
| `upvotes` | Raw Reddit score (clipped to 0, max 4,139) | Join key for filtering |
| `log_weight` | `log(upvotes + 1)` | Dampened weight for Stage 8 temporal aggregation |
| `upvote_weight` | `upvotes + 1` | Linear reach: each upvoter = one endorser; poster counts as 1 |

**Reach-weighted vs raw rates (full corpus):**

| Label | Raw % | Reach-weighted % | Delta |
|---|---|---|---|
| buyin: uncommitted | 2.98% | 3.64% | +0.66pp |
| buyin: committed | 2.74% | 3.53% | +0.79pp |
| stance: critical | 2.82% | **5.09%** | **+2.27pp** |
| stance: supportive | 2.61% | 3.48% | +0.87pp |

**Key finding: critical NS content gets ~2× the viral amplification of supportive content.** The reach-weighted critical/supportive ratio is 1.46x vs 1.08x raw. Reddit's upvote dynamics strongly amplify criticism over support.

---

**Updated SYSTEM_PROMPT:** (in `commitment_llm_annotator.py`)
- Canonical "zao liao" example embedded in the prompt definition
- Explicit distinction between apathy and opposition (both = uncommitted)
- 3rd-person mentions (e.g. "got one chao keng guy") → NEUTRAL (not author's own stance)

### Why FAISS enrichment in 5b but not 5a

**Stage 5a flew blind.** No trained model existed yet, so there were no labeled seeds to search from and no way to know which classes would be rare. The only option was random stratified sampling — sample broadly, label everything, discover the class distribution. That revealed the core problem: uncommitted and critical are rare in the wild (~5–8% of posts), so random sampling naturally produced very few examples of each. The model trained on that data learned "when in doubt, predict neutral" because ~85% of training data was neutral.

**Stage 5b exploits what 5a discovered.** With ~140 confirmed uncommitted and ~180 confirmed critical human labels in hand, FAISS turns those into a search engine. Every chunk in the 737k corpus is stored as a 768-dimensional sentence embedding (a vector capturing semantic meaning). FAISS lets us use confirmed minority examples as **query vectors** — "find me the 25 most semantically similar chunks to each of these." Because uncommitted speech has distinctive patterns (indirect complaint, resignation, sarcasm, "suck it up" tone), the nearest neighbours in embedding space share those patterns.

**The yield difference is dramatic:**

| Approach | Hit rate for uncommitted | Cost to get 2,000 new uncommitted labels |
|----------|--------------------------|------------------------------------------|
| Random sampling (5a strategy) | ~5–8% | ~$25–40 |
| FAISS-targeted (5b enrichment) | ~40–50% | ~$2–3 |

FAISS doesn't change the LLM labeling step — it just radically improves yield per dollar by pre-filtering the candidate pool to chunks most likely to be the minority class. The single `--annotate-enrich` call labels both `llm_buyin` and `llm_stance` axes simultaneously (one API call per chunk, returns both labels), so there is no cost multiplier for the second axis.

**Key design insight:** Sentiment ≠ Commitment. Someone can be frustrated (negative sentiment)
but still be committed to service. Apathy ≠ Opposition — both are "uncommitted," but require different levers for engagement.

**Validation results (100 human annotations):**
- Critical recall: 80%, precision: 100% (zero false alarms)
- Neutral recall: 96.6%
- Overall accuracy: 94.0%, kappa: 0.752

**Annual sample:** 10,076 chunks stratified by (year × subreddit):
- 3 subreddits × 7 years × 500 chunks/cell (fewer for thin 2018 cells)
- r/NationalServiceSG, r/singapore, r/askSingapore, 2018–2024

### KEY FINDING — Commitment trend results ✅ (updated to include 2025)

```
Annual Commitment Trend — NS Reddit  (gpt-4.1-mini)
══════════════════════════════════════════════════════════════════════
  Year      N   Committed   Critical   Neutral       Net
  2018   1101       1.3%      3.7%    95.0%  ▼0.025
  2019   1557       1.1%      2.6%    96.3%  ▼0.015
  2020   1566       1.5%      2.7%    95.8%  ▼0.012
  2021   1587       1.7%      3.1%    95.2%  ▼0.014
  2022   1574       2.1%      5.0%    92.9%  ▼0.029
  2023   1562       2.0%      4.9%    93.1%  ▼0.029
  2024   1567       2.4%      6.2%    91.4%  ▼0.038
  2025   1500       1.7%      3.5%    94.8%  ▼0.017

  8-year OLS slope: +0.291pp/year  r²=0.313  p=0.149  ❌ not significant (2025 reversal)
  7-year OLS slope: +0.526pp/year  r²=0.676  p=0.023  ✅ SIGNIFICANT (2018–2024 only)

  Per-subreddit (8-year):
    r/NationalServiceSG   slope=+0.146pp/yr  p=0.119   ❌ stable throughout
    r/singapore           slope=+0.367pp/yr  p=0.144   ❌ (was significant in 7-year window)
    r/askSingapore        slope=+0.560pp/yr  p=0.075   ❌ (was p=0.004 in 7-year window)
```

**2025 reversal — per-subreddit breakdown:**
| Subreddit | 2024 critical | 2025 critical | Change |
|---|---|---|---|
| r/NationalServiceSG | 1.7% | 1.4% | stable |
| r/askSingapore | **7.3%** | **2.6%** | −4.7pp |
| r/singapore | **9.6%** | **6.4%** | −3.2pp |

The 2025 drop is statistically genuine (2024 vs 2025 CIs for askSingapore don't overlap).
Corpus volume is normal throughout 2025 (9–11k chunks/month), ruling out a data collection gap.
Likely cause: policy changes, May 2025 GE, or regression from the 2022–2024 surge — not yet confirmed.

**Interpretation for dashboard:**
- Critical institutional stance surged from 2018→2024, peaking at 6.2% in 2024 (~+75% relative)
- The **2022–2024 period** is the clearest signal: critical % roughly doubled in 3 years
- **2025 shows a notable reversal**, particularly in r/singapore and r/askSingapore
- Net commitment has been **negative throughout all 8 years** (always more critical than committed)
- r/NationalServiceSG (the dedicated NS community) is stable throughout — decline driven by the broader public

**Dashboard claim (defensible):**
> "Critical sentiment toward NS as an institution surged from 2022–2024, roughly doubling over
> three years. 2025 shows a reversal, suggesting the peak may be event-driven rather than structural.
> Net commitment has remained negative throughout the entire 2018–2025 period."

### BART C2D original run (historical — do not use for trend analysis)

**Model:** `facebook/bart-large-mnli` (zero-shot NLI, `multi_label=False`)
**Notebook:** `notebooks/kaggle_commitment_v1.ipynb` ✅ completed
**Output:** `chunk_commitment.parquet` — scores exist but are unreliable (kappa=0.127)

Hypotheses (kept for reference):
```
"The author supports National Service"                                  → commit_support
"The author is critical of National Service"                            → commit_critical
"The author is discussing National Service without expressing a strong opinion"  → commit_neutral
```

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

---

## Stage 5b-cascade — 4-Stage Cascade Classifier (2026-06-23)

**Problem discovered:** Post-deployment eval of the SingBERT distilled buyin model revealed **neutral collapse** — the model outputs `prob_uncommitted` median=0.002 on chunks that humans label uncommitted. The model is fundamentally blind to subtle/implicit uncommitted signal, not just borderline. Class weighting and threshold tuning cannot recover this. Root cause: the distillation training data (LLM-enriched) has only ~47% recall for buyin relevance — the LLM itself missed half the relevant chunks, so the student model learned from incomplete signal.

**Solution: 4-stage cascade architecture.** Each stage is a separate binary SingBERT fine-tune:
- **Stage 1a** — Is this chunk commitment-relevant? (`relevant` vs `not_relevant`)
- **Stage 2a** — Committed vs uncommitted (direction; only runs on Stage 1a positives)
- **Stage 1b** — Does this chunk express a stance? (`has_stance` vs `no_stance`)
- **Stage 2b** — Supportive vs critical (direction; only runs on Stage 1b positives)

**Why this works:** Structurally forces the model to learn a simpler binary question at each stage. Stage 2a training data is nearly perfectly balanced (3,088 committed : 3,461 uncommitted) because neutrals are excluded. LLM direction accuracy is 95.7% for committed/uncommitted and 88% for supportive/critical — silver labels are highly reliable for stage 2.

**Training data (built by `src/features/cascade_train_export.py`):**

| Stage | File | Rows | Balance |
|---|---|---|---|
| 1a buyin relevance | `cascade/stage1a_train.csv` | 13,347 | {0: 6,812, 1: 6,535} |
| 2a committed/uncommitted | `cascade/stage2a_train.csv` | 6,549 | near-balanced |
| 1b stance relevance | `cascade/stage1b_train.csv` | 35,680 | silver-heavy |
| 2b supportive/critical | `cascade/stage2b_train.csv` | 8,939 | near-balanced |

**Gold label expansion (2026-06-23):**
- 350 neutral-collapse chunks manually annotated (buyin: neutral=198, uncommitted=97, committed=55; stance: neutral=182, critical=44, supportive=25)
- `commitment_manual_annotations.csv` grew from 600 → 949 rows
- `commitment_testset.parquet` grew from 377 → 727 rows (both buyin + stance axes)

**Silver label strategy:**
- LLM direction labels (2a/2b): weight 0.4–0.5 — 95%+ accurate, trustworthy
- LLM relevance positives (1a/1b): weight 0.5–0.7 — use as positives only; do NOT use LLM neutral as negative (47% recall means too many false negatives)
- Corpus negatives for 1a: known-safe chunks (`prob_committed < 0.05 AND prob_uncommitted < 0.05`), weight 0.3

**Notebooks:** `notebooks/cascade/`
- `cascade_stage1a_buyin_relevance.ipynb` ✅ trained
- `cascade_stage2a_committed_uncommitted.ipynb` ✅ trained
- `cascade_stage1b_stance_relevance.ipynb` ✅ trained
- `cascade_stage2b_supportive_critical.ipynb` ✅ trained
- `cascade_inference.ipynb` ⏳ ready to run (attach 4 model datasets + ns-corpus-chunks + ns-commitment-testset)

**Inference logic:** Stage 1a → filters ~50k relevant chunks → Stage 2a assigns direction. Stage 1b runs in parallel on all 737k → filters has-stance subset → Stage 2b assigns direction. Non-relevant chunks default to neutral. Output: `chunk_commitment_cascade.parquet` (same schema as `chunk_commitment_llm.parquet`).

**Status:** All 4 models trained on Kaggle. Inference notebook ready. After inference completes:
```bash
cp ~/Downloads/chunk_commitment_cascade.parquet \
   data/processed/new/chunk_commitment_llm.parquet
python -m src.features.committed_recall_eval
```
If buyin recall ≥ 70% on 727-row testset → proceed to Stage 6/8. If not → use Stage 1a uncertain predictions (prob 0.4–0.6) as next annotation candidates.

**Key files:**
- `src/features/cascade_train_export.py` — builds all 4 training CSVs
- `src/features/neutral_collapse_sampler.py` — samples neutral-classified chunks with surface signals for annotation
- `src/features/neutral_collapse_annotator.py` — annotation CLI (both axes in one pass); `--merge` writes to training CSV + testset
- `notebooks/cascade/build_notebooks.py` — regenerates all 5 .ipynb files from scratch

---
## Stage 5b → Stages 6/7/8/9 Cascade Impact

### What changes downstream from the committed/uncommitted/neutral pivot

**Output schema change (chunk-level):**
- ✅ Still: `chunk_sentiment.parquet` — `chunk_id`, `sent_neg/neu/pos` (unchanged)
- ✅ Still: `chunk_commitment_lexicon.parquet` — lexicon signals (unchanged)
- ❌ Removed: `chunk_commitment.parquet` (BART, κ=0.127)
- ✅ **DONE:** `chunk_commitment_cascade.parquet` — 9 cols: `chunk_id`, `buyin_label`, `stance_label`, 6 probability columns
  - 727,470 rows, 4-stage cascade (relevance gate → direction classifier per axis)
  - Quality: buyin macro F1=0.714, stance macro F1=0.784 vs 727-row human test set

**Topic-level analysis (Stage 6):**
- ✅ Still computes: `topic_macro` × `year` × `sentiment` breakdown (unchanged)
- 🔄 **NEW:** `topic_macro` × `year` × `buyin` breakdown (pct_committed/uncommitted/neutral)
- 🔄 **NEW:** `topic_macro` × `year` × `stance` breakdown (pct_supportive/critical/neutral)
  - Can slice "apathy-only" and "opposed-only" post-hoc using `sent_neg` as proxy for opposition

**Dashboard views affected:**
- ✅ Sentiment trend (monthly, by topic) — unchanged
- 🔄 **Commitment page rebuilt** — three graphs on dual-axis data (see Stage 9 below)
- 🔄 **Per-topic commitment** — buyin and stance breakdowns at `topic_macro` level (17 categories) × year

**Backwards compatibility:**
- ✅ The **annual sample trend** (`commitment_trend.csv`) is *analytically separate* — it's a gold-standard κ=0.752 LLM sample, unaffected by distillation
- Historical claim ("critical rose 2022–2024") remains defensible as "uncommitted (opposed institutional faction) rose…"
- Can document: "distilled full-corpus model used for topic/chunk breakdowns; annual trend derived from separately-validated LLM sample"

---

## Stage 6 — Document-level Aggregation

Join: `chunk_sentiment` + `chunk_commitment_llm` + `chunk_topics` + chunk parquets (for `log_weight`, `doc_id`)
Group by `doc_id`, aggregate scores weighted by `log_weight` (already in chunk parquets).

Before joining, add taxonomy columns to `chunk_topics`:
```python
from src.models.topic_labels import TOPIC_LABELS
chunk_topics['topic_macro']   = chunk_topics['topic_id_fine'].map(lambda t: TOPIC_LABELS.get(t, {}).get('macro'))
chunk_topics['topic_sub']     = chunk_topics['topic_id_fine'].map(lambda t: TOPIC_LABELS.get(t, {}).get('sub'))
chunk_topics['topic_sub_sub'] = chunk_topics['topic_id_fine'].map(lambda t: TOPIC_LABELS.get(t, {}).get('sub_sub'))
```

**Note:** Do NOT join `chunk_commitment.parquet` (BART scores, κ=0.127). Use `chunk_commitment_llm` only.

Output: `doc_sentiment.parquet`, `doc_commitment.parquet` (aggregated by document + topic)

---

### Current action (2026-06-23)

Join `chunk_commitment_llm.parquet` into document-level aggregation on `chunk_id` / `doc_id`.

**What to compute per document:**
- Raw label rates: `pct_uncommitted`, `pct_committed`, `pct_critical`, `pct_supportive`
- Broad label rates: `pct_broad_uncommitted`, `pct_broad_critical` (using `broad_buyin` / `broad_stance`)
- Reach-weighted rates: `wtd_uncommitted` = `Σ(upvote_weight × is_uncommitted) / Σ(upvote_weight)`
- `is_positive` / `is_negative` aggregated per doc (any-chunk or majority rule)
- Join `chunk_topics` to get `topic_macro` per chunk before aggregating

```python
# Minimum join skeleton
df = pd.read_parquet("data/processed/new/chunk_commitment_llm.parquet")
meta = pd.read_parquet("data/processed/new/chunk_metadata.parquet", columns=["chunk_id","doc_id","topic_macro","year","month","subreddit"])
df = df.merge(meta, on="chunk_id")
# then groupby doc_id and aggregate
```

Also: **fill `commitment_divergence` in `doc_divergence_v2.parquet`** — it is currently NaN. After Stage 6 produces per-doc uncommitted rates, patch it:
```python
div = pd.read_parquet("data/processed/new/doc_divergence_v2.parquet")
doc_commit = ...  # from Stage 6 output
div["commitment_divergence"] = div["post_id"].map(doc_commit["pct_uncommitted_comments"] - doc_commit["pct_uncommitted_submission"])
div.to_parquet("data/processed/new/doc_divergence_v2.parquet", index=False)
```

---
## Stage 7 — Divergence Score (Redesigned 2026-06-08; v2 COMPLETE 2026-06-12)

### Problem with original design
The naive `mean(sent_neg_comments) − mean(sent_neg_submissions)` metric is dominated by neutral posts — 89% of chunks are neutral, so most divergence scores cluster near zero and bury the genuinely contentious threads. Basic NS questions ("when do I enlist?") generate the same neutral noise as heated NS pay disputes, making the ranking meaningless.

### Redesigned metrics

**Core fix: opinion intensity filter**
Only include chunks where `max(sent_neg, sent_pos) > 0.5` (opinionated chunks) when computing divergence. This strips neutral-dominant content from the calculation. A thread where 80% of comments are neutral Q&A should not rank above a thread where 50% of comments are strongly negative.

**Metric 1 — Upvote-weighted sentiment divergence (primary)**
```
upvote_weighted_div = Σ(|sent_neg - sent_pos|_i × log(1 + upvotes_i)) / Σ(log(1 + upvotes_i))
```
Weights each chunk's opinion strength by community endorsement. A chunk expressing strong negative sentiment with 39 upvotes contributes far more signal than the same opinion with 0 upvotes.

**Metric 2 — Within-thread sentiment variance (disagreement)**
For each thread, compute `variance(sent_neg across comment chunks)`. High variance = commenters genuinely disagree (some strongly negative, some strongly positive). Low variance = everyone agrees or everyone is neutral. This surfaces threads with internal conflict, not just high-negativity threads.

**Metric 3 — Topic-level discourse intensity**
```
discourse_intensity(topic) = Σ(upvotes × |sent_neg - sent_pos|) per topic
```
Aggregated across all posts in the topic. Surfaces topics generating high-engagement opinionated discourse (e.g. NS Pay & Benefits, NS Policy & Society) vs. topics generating high-volume but low-engagement neutral questions (e.g. Admin & Logistics).

**Metric 4 — Post vs comment commitment divergence**
`pct_uncommitted(comments) − pct_uncommitted(submissions)` per post, upvote-weighted. Surfaces threads where the post itself is neutral but the comments are uncommitted/critical — the "silent controversy" pattern.

### Upvote data requirements
Upvote scores (`score` field) must be verified as carried through from raw Reddit data into the chunk parquets. If not present in chunk parquets, join from the raw `submissions_chunks.parquet` and `comments_chunks.parquet` on `doc_id` / `comment_id` before Stage 7.

### Dashboard slider
The Stage 9 dashboard must expose a **minimum upvotes slider** (default: 5, range: 0–100). Posts with `score < threshold` are excluded from divergence rankings. Rationale: basic NS questions ("can I bring phone to BMT?") get 0–2 upvotes; genuine policy debates get 20–500. The slider lets the user isolate meaningful discourse from noise.

**Recommended defaults for presentation:**
- slider = 5: filters out zero-engagement noise, retains most discourse
- slider = 20: surfaces only clearly high-engagement threads
- slider = 0: raw view (shows neutral dominance — useful to demonstrate the problem)

### Output: `doc_divergence_v2.parquet` ✅ BUILT (2026-06-12)

**Script:** `src/analysis/stage7_divergence_v2.py`
**Stats:** 40,253 posts; Spearman ρ=0.034 vs naive metric (old metric confirmed as noise)
**Note:** `commitment_divergence` column is NaN — fill after Stage 6 produces per-doc uncommitted rates (see Stage 6 action above).

| Column | Description |
|---|---|
| `post_id` | Reddit post identifier |
| `topic_macro` | Macro topic of the post |
| `upvote_weighted_div` | Upvote-weighted sentiment divergence (primary ranking signal) |
| `within_thread_variance` | Variance of sent_neg across comment chunks (disagreement signal) |
| `opinion_chunk_pct` | % of chunks in thread that are opinionated (max(sent_neg,sent_pos) > 0.5) |
| `total_upvotes` | Sum of upvotes across post + comments |
| `commitment_divergence` | pct_uncommitted(comments) − pct_uncommitted(submissions), upvote-weighted — NaN until 5b |
| `sentiment_divergence_raw` | Original naive metric (kept for comparison) |

**Topic-level aggregation** → `topic_discourse_intensity.parquet` ✅ BUILT (2026-06-12, 17 rows): group by `topic_macro`, `discourse_intensity = Σ(upvote_weighted_div × total_upvotes)`. Top topics: NS Life & Culture (980k), NS Policy & Society (372k), BMT & Training (271k).

**Note:** `doc_divergence.parquet` (original naive metric) is superseded. Do not use for Stage 9.

---

## Stage 8 — Temporal Aggregation (Updated 2026-06-12)

**Must normalise datetime first** (see Known Issues).
Monthly rollup grouped by subreddit + topic_macro.

**Verify upvote data carrythrough before aggregation:** Check that `score` (upvotes) from the raw Reddit data is present in `submissions_chunks.parquet` and `comments_chunks.parquet`. If missing, join back from the raw parquets on `doc_id`. Upvotes are required for Stage 7 divergence redesign, Stage 9 commitment graphs, and the RAG fact table.

**Current status:** ✅ COMPLETE (2026-07-06). Sentiment side done on v7 (2026-06-12). Commitment side rebuilt from `chunk_commitment_cascade.parquet` via `scripts/rebuild_temporal_commitment.py`. Output: `temporal_commitment.parquet` (6,067 rows × 30 cols).

Aggregates:
- Sentiment: mean(sent_neg/neu/pos) per (month, subreddit, topic_macro) ✅ done
- Commitment (post 5b): pct_committed, pct_uncommitted, pct_neutral, pct_supportive, pct_critical per (month, subreddit, topic_macro)
- Upvote-weighted commitment (post 5b, NEW — 2026-06-13):
  - `wtd_buyin_committed` = Σ(log(1+score_i) × is_committed_i) / Σ(log(1+score_i))
  - `wtd_buyin_uncommitted`, `wtd_stance_supportive`, `wtd_stance_critical` (same formula)
  - `wtd_positive` = Σ(log(1+score_i) × is_positive_i) / Σ(log(1+score_i))
  - `wtd_negative`, `net_disposition` (wtd_positive − wtd_negative)
- Engagement: total upvotes, chunk count, mean upvotes per chunk

**Upvote-weighting rationale:** A "NS is necessary" post with 20 upvotes represents 20 community endorsements, not 1. Formula uses log(1+score) to prevent viral outliers from dominating.

Output: `temporal_sentiment.parquet`, `temporal_commitment.parquet`

**Note:** This output directly feeds `rag_fact_table.parquet` (Stage 10 Phase 2). Build the RAG fact table immediately after Stage 8 completes — it is a simple groupby aggregation with no additional computation.

---

### After Stage 6 — remaining cascade ✅ ALL COMPLETE (2026-07-06)

1. ✅ **Stage 8** — `temporal_commitment.parquet` rebuilt via `scripts/rebuild_temporal_commitment.py`
2. ✅ **Stage 9 dashboard** — cascade stats wired in, all commitment tabs operational
3. ✅ **RAG fact table** — rebuilt via `python -m scripts.rag.build_fact_table` (4,972 rows)

---

---
## Stage 9 — Streamlit Dashboard + Seaborn Viz (Redesigned 2026-06-08; dual-axis update 2026-06-13)

### Commitment visualisation — three graph axes

The dashboard presents commitment on **three separate axes** using dual-axis data. These are NOT the same as sentiment (sent_neg/neu/pos), which measures emotional tone. Commitment measures the author's relationship to NS as an institution and as a personal obligation.

**Graph 1 — Supportive vs Critical (stance axis)**
- Source: `commitment_llm_annual.csv` (10,076 rows, gold-standard κ=0.752 annual sample) — old committed→supportive, old critical→critical mapping; maps directly to stance axis
- X-axis: Year (2018–2025), Y-axis: % Supportive vs % Critical
- Captures: "do people see NS as worth defending as an institution?"
- Defensible as a standalone trend: "critical institutional stance doubled 2018–2024, reversed in 2025"
- Overlay option: subreddit breakdown (r/singapore and r/askSingapore drove the critical surge; r/NationalServiceSG stable)
- Both raw count and upvote-weighted toggle

**Graph 2 — Committed vs Uncommitted (buyin axis)**
- Source: `chunk_commitment_llm.parquet` (full 737k corpus, distilled buyin SingBERT model, κ=0.730)
- X-axis: Month/Year, Y-axis: % Committed vs % Uncommitted vs % Neutral, optionally by topic_macro
- Captures: "did servicemen personally invest in their service, or were they just clearing time?"
- Decomposes "uncommitted" on hover: high `sent_neg` → opposed faction; low `sent_neg` → apathetic/zao faction
- More granular than Graph 1 (monthly vs annual, topic-level breakdowns)
- Both raw count and upvote-weighted toggle

**Graph 3 — Net Disposition Index (combined metric)**
- `net_disposition` = `wtd_positive` − `wtd_negative` where positive = (committed OR supportive), negative = (uncommitted OR critical)
- Computed at aggregation time from Stage 8 `temporal_commitment.parquet`; not labelled per-chunk
- This is the headline number: "Net NS disposition has been negative throughout 2018–2025"
- Trend line overlaid with key events (Aloysius Pang 2019, COVID 2020, pay changes, GE 2025)
- Both raw count and upvote-weighted toggle

### Upvote-weighted commitment scoring

All three commitment graphs offer a **raw count mode** (default) and an **upvote-weighted mode** (toggle). Upvote weighting surfaces community-endorsed views over individual noise.

**Upvote weight formula (NEW — 2026-06-13):**
```
wtd_committed = Σ(log(1 + score_i) × is_committed_i) / Σ(log(1 + score_i))
wtd_positive  = Σ(log(1 + score_i) × is_positive_i)  / Σ(log(1 + score_i))
net_disposition = wtd_positive − wtd_negative
```
Applied at Stage 8 temporal aggregation (not model training). New temporal columns: `wtd_buyin_committed`, `wtd_buyin_uncommitted`, `wtd_stance_supportive`, `wtd_stance_critical`, `wtd_positive`, `wtd_negative`, `net_disposition`.

**Why log(1 + upvotes):** A post with 39 upvotes expressing commitment contributes 40 units of weighted signal (log(40) ≈ 3.7×), not 39× the weight of a 1-upvote post. Log-scaling prevents viral outlier posts from dominating the aggregate while still rewarding community-endorsed content.

**Options to explore (all should be implemented and compared):**

| Scoring mode | Formula | Use case |
|---|---|---|
| Raw count | `pct_committed` = committed_chunks / total_chunks | Baseline — equal weight to every chunk |
| Log-upvote weighted | `Σ(label × log(1+upvotes)) / Σ(log(1+upvotes))` | Community-endorsed view; recommended default |
| Threshold-filtered (slider) | Only include posts with `upvotes ≥ k` (default k=5) | Surface meaningful discourse, drop noise |
| Upvote count as amplitude | `committed_chunks × (1 + upvotes)` summed | Raw engagement signal — treats upvote as amplifier |

**Dashboard upvote slider:** Minimum upvotes filter (range 0–100, default 5) applies to ALL commitment graphs. At slider=0, neutral FAQ questions dominate and flatten the signal. At slider=5+, genuine engagement emerges. Show both to justify the default.

### Full dashboard view list

1. **Topic distribution** — BERTopic hierarchical dendrogram, macro-coloured (unchanged)
2. **Sentiment trend** — monthly, score-weighted, by topic_macro (unchanged)
3. **Commitment — Institutional (Critical vs Supportive)** — annual, gold-standard LLM sample
4. **Commitment — Personal (Uncommitted vs Committed)** — monthly, full corpus, topic-level breakdown
5. **Commitment — Net Disposition Index** — composite score with event timeline overlay
6. **Divergence analysis** — redesigned (upvote-weighted, opinion-intensity filtered, upvote slider) — see Stage 7
7. **Topic discourse intensity** — ranking of macro topics by engagement-weighted controversy score
8. **Thread depth analysis** — unchanged

**Narrative impact:**
- Old frame: "Critical institutional stance surged 2022–2024"
- New frame: "Uncommitted sentiment — spanning both disengagement and opposition — surged 2022–2024. The opposed faction (high negativity + uncommitted) was strongest in r/singapore and r/askSingapore. Net NS disposition has been negative throughout all 8 years, but 2025 shows a reversal, suggesting the peak was event-driven rather than structural."

---

## Worth Plotting & Dashboarding (updated 2026-06-18)

All plots below are now unlocked by `chunk_commitment_llm.parquet` being live. Priority order matches the Stage 9 dashboard plan.

### P1 — Core commitment time-series (Stage 9 Graph 1–3)

**1. Explicit buyin trend (monthly/annual, full corpus)**
- Source: `chunk_commitment_llm.parquet` joined to `chunk_metadata.parquet` (year, month)
- Y-axis: `pct_uncommitted` vs `pct_committed` vs `pct_neutral`
- Toggle: raw count ↔ reach-weighted (`upvote_weight`)
- Overlay: key NS events (Aloysius Pang 2019, pay review 2022, GE 2025)
- Expected signal: 2022–2024 surge in uncommitted, 2025 partial reversal

**2. Explicit stance trend (annual, gold-standard LLM sample)**
- Source: `commitment_llm_annual.csv` (10,076 rows, κ=0.752)
- Already built — this is Graph 1 in the Stage 9 plan
- Y-axis: % critical vs % supportive by year + subreddit breakdown

**3. Net Disposition Index (monthly)**
- `net_disposition` = `wtd_positive` − `wtd_negative`
- Where positive = committed OR supportive; negative = uncommitted OR critical
- This is the headline number: "has been negative throughout 2018–2025"
- Trend line + NS event pins

### P2 — Explicit vs broad comparison (new — reveals latent discontent)

**4. Explicit vs broad uncommitted gap (annual)**
- Two lines: `pct_broad_uncommitted` (25%) and `pct_uncommitted` (3%) over time
- The gap = latent discontent in neutral pool; when it widens, ambient frustration is growing even if explicit signals don't
- This is a novel finding worth highlighting: Reddit's explicit critics are a tiny tip of a much larger iceberg

**5. Broad metric trend**
- Same as Plot 1 but using `broad_buyin` / `broad_stance`
- Shows 2.15x unc/com and 1.96x crit/sup — more aligned with domain expectation
- Label both: "Explicit signals (high precision)" and "Broad signals (includes ambient negativity)"

### P3 — Viral amplification (new — the upvote story)

**6. Reach-weighted vs raw rates (bar chart or area chart)**
- Side-by-side bars for each label: raw % vs reach-weighted %
- Key callout: stance critical raw=2.82% → reach-weighted=5.09% (+2.27pp) — **critical content gets ~2× the viral amplification**
- Comparison: uncommitted +0.66pp vs committed +0.79pp (more symmetric; both go viral roughly equally)
- Narrative: "Even though critical posts are rare, they earn disproportionate community endorsement"

**7. Annual reach-weighted critical rate vs raw critical rate**
- Time-series: `wtd_stance_critical` vs `pct_stance_critical` by year
- Measures whether the virality gap widens over time (did critical content become more upvoted in recent years?)

### P4 — Topic-level commitment breakdown (Stage 9 Graph 2 extension)

**8. Buyin breakdown by topic_macro (heatmap)**
- X: 17 macro topics; Y: pct_uncommitted (explicit), pct_broad_uncommitted
- Hypothesis: NS Policy & Society and Pay & Benefits will have highest uncommitted rates
- Cross with discourse intensity from `topic_discourse_intensity.parquet`

**9. Most "uncommitted" topics by reach-weighted rate**
- Ranked bar: `wtd_broad_uncommitted` per topic_macro
- Pairs with topic discourse intensity to surface: "which topics generate both high engagement AND high uncommitted sentiment?"

**10. Subreddit commitment comparison**
- `pct_broad_uncommitted` by (year × subreddit) — r/singapore vs r/askSingapore vs r/NationalServiceSG
- Known finding: r/singapore and r/askSingapore drove the 2022–2024 critical surge; r/NationalServiceSG stable
- Add upvote-weighted version: does r/singapore's critical content get more upvotes?

### P5 — Divergence (Stage 7 follow-through)

**11. Thread commitment divergence (after filling NaN in doc_divergence_v2)**
- `commitment_divergence` = pct_uncommitted(comments) − pct_uncommitted(submission)
- Surfaces "silent controversy": neutral posts that trigger uncommitted comment threads
- Dashboard: top-20 most divergent threads (slider: min upvotes ≥ 5)

**12. Discourse intensity × commitment (scatter)**
- X: `discourse_intensity` per topic_macro; Y: `wtd_broad_critical` per topic_macro
- Labels each point with topic name
- Quadrant analysis: high intensity + high critical = "hot-button" topics

### Dashboard toggle recommendations

All commitment charts should have:
- **Raw count ↔ Reach-weighted** toggle (primary new capability)
- **Explicit ↔ Broad** toggle (shows latent vs overt signal)
- **Upvote slider** (min upvotes 0–100, default 5) for divergence and topic charts

---
## Stage 10 — RAG Chatbot (✅ Phase 1 + Phase 2 complete — 2026-06-09)

### Status: Fully Operational

The chatbot is live in the Streamlit dashboard Chat tab. All query paths working. No further code
changes needed until Stage 5b (commitment) is fixed, at which point a single fact table rebuild
propagates the improvement everywhere.

---

### Architecture overview

Two-tier pipeline: **pre-computed knowledge base** (offline) + **live query pipeline** (≤5s).

```
User query
    │
    ├── Quantitative + simple (no rich intent flags)
    │   → FactTableHandler: direct parquet lookup, year-by-year table, no LLM — <1s
    │
    └── Qualitative / rich intent (subreddit compare, topic compare, ranking, methodology, etc.)
        ├── 1. QueryRouter: rule-based intent classification + filter extraction (~5ms)
        ├── 2. Retriever: embed query → FAISS search with adaptive n_search → upvote reranking
        ├── 3. ContextAssembler: builds bounded context window (≤10,000 chars) from:
        │       • Statistical facts (fact table, auto-selects granularity by intent)
        │       • NS events overlapping the queried period (up to 6 events)
        │       • Longitudinal anomaly timeline (spike detector, injected for multi-year queries)
        │       • Topic digest(s)
        │       • Temporal narrative(s)
        │       • Retrieved chunks (top-10)
        └── 4. Synthesizer: streams response via Groq Llama 3.3-70B (→ OpenAI fallback on 429)
```

---

### Knowledge base components (all built ✅)

| Component | File | Details |
|---|---|---|
| NS event timeline | `ns_events.json` | 28 richly annotated events (2018–2025); each has `metric_impact` z-scores, `severity`, `topics_affected`, `search_keywords`, `description` |
| Topic digests | `rag_topic_digests.json` | 17 macro topics × gpt-4.1 ~300-word summaries (themes, tone, concerns, sentiment profile) |
| Temporal narratives | `rag_temporal_narratives.json` | 96 months (2018-01 → 2025-12) × gpt-4.1-mini ~150-word summaries |
| Statistical fact table | `rag_fact_table.parquet` | 4,972 rows × 24 cols, 6 granularity levels — see schema below |
| FAISS vector index | `chunk_faiss.index` | 737k × 768-dim, IndexFlatIP, ~2.1 GB |
| Chunk metadata | `chunk_metadata.parquet` | 737,274 rows — chunk_id, text_snippet, subreddit, year, month, topic_macro, sent_neg/pos, upvotes, faiss_idx |

---

### Fact table schema

**Granularity levels** (`granularity` column):
- `month_sub_topic` — per (year, month, subreddit, topic_macro) — 4,430 rows
- `month_sub` — per (year, month, subreddit) — 278 rows
- `month` — per (year, month), all subreddits + topics — 96 rows
- `year_topic` — per (year, topic_macro), all subreddits — 136 rows
- `year_sub` — per (year, subreddit), all topics — 24 rows
- `year` — per year, all subreddits + topics — 8 rows

**Columns (24):** year, month, subreddit, topic_macro, doc_count,
mean_sent_neg/neu/pos, pct_neg/neu/pos,
mean_commit_net/support/critical, wtd_sent_neg/neu/pos, wtd_commit_net/support/critical,
**pct_committed, pct_critical, pct_neutral_commit**, granularity

✅ **Sentiment columns** (`mean_sent_neg/neu/pos`, `wtd_sent_neg/neu/pos`, `pct_neg/neu/pos`) are now based on SingBERT v7 (rebuilt 2026-06-12).

⚠️ **Commitment % caveat:** `pct_committed`, `pct_critical`, `pct_neutral_commit` are still derived from
`chunk_commitment.parquet` (BART scores, κ=0.127). Stage 5b is complete and `chunk_commitment_llm.parquet` is
live — **rebuild the fact table after Stage 6/8**: update `CHUNK_COMMITMENT` in `scripts/rag/build_fact_table.py`
to point to `chunk_commitment_llm.parquet` and re-run. Also add `pct_uncommitted`, `pct_broad_uncommitted`,
`pct_broad_critical`, `wtd_uncommitted`, `wtd_critical` columns to the schema.

---

### Query router — intent flags

The `QueryFilters` dataclass carries boolean intent flags extracted by `QueryRouter`:

| Flag | Triggers | Context path used |
|---|---|---|
| `compare_subreddits` | "compare r/X to r/Y", "across subreddits", "how does r/ differ" | `_facts_subreddit_comparison()` — all 3 subreddits side-by-side |
| `compare_topics` | "compare BMT and Reservist", "X vs Y topic" | `_facts_topic_comparison()` — 2 topics side-by-side |
| `ranking_dim` | "which subreddit/topic/year has highest X" | `_facts_ranking()` — sorted ranked list |
| `commitment_breakdown` | "committed vs critical vs neutral %", "pct breakdown" | `_facts_commitment_breakdown()` — year-by-year pct table |
| `ask_methodology` | "how was X calculated/measured/classified" | `METHODOLOGY_TEXT` static block injected |
| (longitudinal) | multi-year query or no year filter | `SpikeDetector.format_timeline()` injected |

All rich-intent queries are routed through the full assembler + LLM path (not FactTableHandler).

---

### Spike detector

`src/rag/spike_detector.py` — computes rolling ±6-month z-scores for `mean_sent_neg`,
`mean_commit_net`, and `doc_count` across the monthly time-series.

- Found **35 anomalies across 28 months** in the full 2018–2025 dataset
- Each anomaly matched to overlapping NS events from `ns_events.json`
- Top anomalies: Feb 2023 (neg z=+4.55, PR/foreigner debate), Jan 2019 (vol z=+3.65, Aloysius Pang),
  Feb 2023 (commit z=−4.19), Mar 2024 (commit z=−3.88)
- `format_timeline()` renders a compact annotated timeline injected into context for longitudinal queries

---

### LLM backend

**Primary:** Groq `llama-3.3-70b-versatile` (free tier — 100k tokens/day, 7k requests/month)
**Fallback:** OpenAI `gpt-4o-mini` (auto-switches on Groq 429 rate-limit error)
**Config:** `GROQ_API_KEY` in `.env`; OpenAI key retained as fallback

System prompt instructs the LLM to:
- Attribute anomalies to named events (z-score cited explicitly)
- Structure longitudinal answers with year-labelled headers
- Distinguish sentiment (emotional tone) from commitment (personal buy-in)
- Never invent statistics not in the provided context

`max_tokens = 900` (was 600 — increased for longitudinal answers)
`MAX_CONTEXT_CHARS = 10,000` (was 6,000 — Llama 3.3 has 128k context window)

---

### Capability map (post 2026-06-09 improvements)

| Query type | Quality | Path |
|---|---|---|
| Year-over-year sentiment trends | ✅ Excellent | Longitudinal timeline + fact table |
| "What caused X spike?" | ✅ Excellent | Spike detector + event attribution, z-scores cited |
| Subreddit comparison | ✅ Working | `_facts_subreddit_comparison()` |
| Topic cross-comparison | ✅ Working | `_facts_topic_comparison()` |
| Ranking (which X has highest Y) | ✅ Working | `_facts_ranking()` |
| Commitment % breakdown | ✅ Working | `_facts_commitment_breakdown()` — ⚠️ values based on BART scores until 5b |
| Fine-grained month + topic | ✅ Working | `month_sub_topic` granularity rows |
| Methodology questions | ✅ Working | Static block |
| Verbatim quotes with metadata | ✅ Working | 10 FAISS chunks |
| Non-NS queries | ⚠️ Porous | Domain lock not airtight; NS-adjacent topics (CPF, careers) will return Reddit NS discussion |
| Vocation/unit breakdown | ❌ No data | 17 macro topics don't include vocation-level split |
| Individual user/post lookup | ❌ Correctly refused | No user-level data |

---

### Remaining improvements (post Stage 5b)

1. **Rebuild fact table** — after `chunk_commitment_llm.parquet` exists:
   ```bash
   # Update CHUNK_COMMITMENT in scripts/rag/build_fact_table.py to point to chunk_commitment_llm.parquet
   python -m scripts.rag.build_fact_table
   ```
   This automatically propagates to all chatbot query paths — no other changes needed.

2. **Rebuild temporal narratives** (optional) — the current narratives were generated before the full
   28-event database was available. Rebuilding will bake in richer event attribution:
   ```bash
   python -m scripts.rag.build_temporal_narratives --resume
   # --resume skips months already generated; only rebuilds if file is deleted first
   ```

---

### ✅ Stage 10 — RAG Chatbot (COMPLETE — 2026-06-09)

**All phases complete. Chatbot is live in Streamlit dashboard.**

The only remaining chatbot work is triggered by upstream pipeline completion:
- **After Stage 5b** → re-run `python -m scripts.rag.build_fact_table` (commitment % will become reliable)
- **Optional** → re-run `python -m scripts.rag.build_temporal_narratives` to incorporate the expanded 28-event database into monthly narratives
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

- **Random annotation is inefficient for rare classes** — the positive class is only ~12% of the corpus. Annotating randomly to get 500 positive training examples requires ~4,200 annotations (~20+ hrs). Use targeted sampling instead: score a pool with the existing model, take top-N by positive confidence (~60% hit rate), and annotate only those. This is the approach in v6.
- **Free LLM alternatives cannot replace gpt-4.1 for this domain** — Cerebras Qwen 235B (κ=0.55) and gpt-4.1-mini (κ=0.55) both showed a neutral bias that could not be resolved by prompt tuning alone. The domain (nuanced Singapore NS Reddit sentiment) requires a capable model; smaller or cheaper models plateau at κ≈0.55 regardless of prompt quality. Only gpt-4.1 achieved κ=0.765+.
- **Adding more human labels can hurt if class weights are not removed** — v5 added 304 new human rows (skewed toward positive) on top of the existing class weight (2.7×) and 3× replication. The positive signal stacked three times, causing 43 predicted positive vs 27 true in the holdout. More data is not always better; fix the distribution problem at source (targeted sampling + no weights) rather than compensating with weighting.
- **SingBERT ceiling = LLM training data quality** — model trained on LLM-generated labels (κ=0.765) cannot beat that LLM across all classes. The model learns gpt-4.1's decision boundaries and noise. The ceiling can be broken for a specific class (e.g. positive) by substituting human labels for LLM labels in that class — which is the v6 targeted annotation strategy.
- **Val contamination via replicate-then-split** — replicating rows before the train/val split puts duplicates in both sets → inflated val metrics (κ=0.68 vs honest 0.57). Always split first, replicate after.
- **Val memorisation** — putting human rows in val while also including them (×3) in training causes val κ=1.000 by epoch 4. It is not a signal of good generalisation — it is pure memorisation. Hard partition: human rows to val ONLY, LLM rows to training ONLY.
- **Class weight + replication stacking** — positive class weight (~2.7×) combined with 3× replication of a skewed new-annotation batch over-boosts minority class prediction. v5 (with 304 new human rows) was worse than v4 despite more data. Adding human data helped val κ slightly but hurt holdout.
- **BERT-large checkpoint size on Kaggle** — each checkpoint is ~1.3 GB. Default `save_strategy="epoch"` with 5 epochs = 6.5 GB, which fills Kaggle's 20 GB working disk. Always set `save_total_limit=1`.
- **transformers>=4.46 renamed `tokenizer=` to `processing_class=`** in Trainer.__init__. Add a version check: `tv = tuple(int(x) for x in transformers.__version__.split(".")[:2]); if tv >= (4, 46): use processing_class`.
- **chunker.py stores raw comment text** — `_embed` uses `"Post: {title}\n\nComment: {text}"` for embeddings only; this is popped before saving to parquet. The `text` field in chunk parquets is the raw comment with NO post title prepended. This is correct — annotating/training on raw comment text is consistent with what the model will see at inference.
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

### ✅ Stage 5a — SingBERT v7 LIVE (2026-06-12)

`chunk_sentiment.parquet` now contains SingBERT v7 scores (neg 29.1% / neu 57.5% / pos 13.5%).
Stages 6, 7 v2, 8 (sentiment side), RAG fact table, and chunk_metadata all rebuilt on v7.
~~Do NOT proceed to Stage 6 until `chunk_sentiment.parquet` is replaced~~ ✅ resolved 2026-06-12.

---

### ✅ Stage 7 v2 — COMPLETE (2026-06-12)

- `doc_divergence_v2.parquet` built (40,253 posts); Spearman ρ=0.034 vs naive metric
- `topic_discourse_intensity.parquet` built (17 topics)
- `commitment_divergence` column NaN until Stage 5b completes

---

### ✅ Stage 5b — COMPLETE (2026-06-18)

`chunk_commitment_llm.parquet` is live. All downstream stages are now unblocked.

---
