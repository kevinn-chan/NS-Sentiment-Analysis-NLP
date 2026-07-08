# NS Sentiment — RAG Chatbot: Full Architecture Design
*Written 2026-06-08. Companion to HANDOFF.md Stage 10.*

---

## 1. Design Mandate

The chatbot must answer two fundamentally different types of questions:

**Type A — Quantitative:**
> "How much did public sentiment towards the conscription debate decrease by in 2023?"

The answer is a number. It requires pre-aggregated statistics, a delta calculation, and formatted output. **No LLM needed.** An LLM would only add latency and hallucination risk.

**Type B — Qualitative / Contextual:**
> "Explain the sharp drop in public sentiment in January 2019."

The answer requires: (a) knowing *that* sentiment dropped (statistics), (b) knowing *what* people were saying (chunk retrieval), and (c) knowing *why* it happened (Aloysius Pang's death — an external event the corpus cannot self-explain). Three different knowledge sources, assembled before the LLM sees anything.

**The core mistake to avoid:** feeding the raw user query into a vector search and asking an LLM to answer from chunks alone. This fails on quantitative questions entirely, fails on contextual questions because the corpus has no external event knowledge, and is too slow when the LLM must reason cold over raw text.

**The solution:** pre-compute everything possible. The LLM at query time only *synthesises* a bounded, pre-assembled context window. It does not compute, retrieve, or reason about the corpus structure.

**Hard constraint:** ≤5 seconds perceived response time. Achieved via: streaming output, pre-loaded in-memory assets, and routing quantitative queries away from the LLM entirely.

---

## 2. System Overview

```
╔══════════════════════════════════════════════════════════════════════╗
║                    OFFLINE BUILD PIPELINE                            ║
║                    (run once; incremental updates when data changes)  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  [chunk parquets + embeddings]                                       ║
║         │                                                            ║
║         ├──→ indexer.py ──────────────→ chunk_faiss.index            ║
║         │                               chunk_metadata.parquet       ║
║         │                                                            ║
║         └──→ knowledge_builder.py                                    ║
║                    │                                                 ║
║                    ├── top chunks per topic ──→ gpt-4.1              ║
║                    │                               │                 ║
║                    │                               ▼                 ║
║                    │                    rag_topic_digests.json       ║
║                    │                                                 ║
║                    └── temporal_sentiment.parquet + ns_events.json   ║
║                                  │                                   ║
║                                  ├──→ gpt-4.1-mini ──→ rag_temporal_ ║
║                                  │                      narratives   ║
║                                  │                                   ║
║                                  └──→ aggregator.py ──→ rag_fact_    ║
║                                                          table.parquet║
║                                                                      ║
║  [manual curation] ──────────────────────────→ ns_events.json        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                    RUNTIME QUERY PIPELINE (≤5 seconds)               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  User Query                                                          ║
║      │                                                               ║
║      ▼  ~5ms                                                         ║
║  [Intent Classifier]  ←── rule-based, no LLM                        ║
║      │                                                               ║
║      ├── QUANTITATIVE ──→ [Fact Table Query] ──→ [Formatter]         ║
║      │                        ~50ms                  ~5ms            ║
║      │                    (pandas on parquet)     (<1s total)        ║
║      │                                                               ║
║      └── QUALITATIVE ──→ [Filter Extractor]  ~5ms                   ║
║                                │                                     ║
║                    ┌───────────┴──────────────┐                      ║
║                    ▼                          ▼                      ║
║             [Context Loader]          [Query Embedder]               ║
║               ~10ms                      ~50ms                       ║
║             (JSON in memory)                │                        ║
║              - topic digest(s)             ▼                        ║
║              - temporal narrative    [FAISS Search]                  ║
║              - ns events               ~100ms                        ║
║              - fact stats           (metadata pre-filtered)          ║
║                    │                      │                          ║
║                    └──────────┬───────────┘                          ║
║                               ▼  ~5ms                               ║
║                    [Context Assembler]                               ║
║                    ≤1500 tokens total                                 ║
║                               │                                      ║
║                               ▼  ~2-4s perceived                    ║
║                    [LLM Synthesis — streaming]                       ║
║                    Claude Haiku / gpt-4.1-mini                       ║
║                               │                                      ║
║                    ┌──────────┴──────────────┐                       ║
║                    ▼                         ▼                       ║
║             Streamed answer          Expandable source panel         ║
║             (with inline citations)  (chunk text + metadata)         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 3. Pre-computed Knowledge Base — All Five Components

### 3.1 `ns_events.json` — External Event Timeline

**Why it exists:** The corpus shows *that* sentiment dropped in January 2019 and *what* people said, but cannot explain *why*. That requires external knowledge (Aloysius Pang's death on 23 Jan 2019). This file is the only way to answer "why" questions reliably. It is manually curated — there is no automated way to derive this.

**Schema:**
```json
{
  "events": [
    {
      "id": "aloysius_pang_2019",
      "title": "Corporal Aloysius Pang training death",
      "date_start": "2019-01-23",
      "date_end": "2019-03-31",
      "description": "NSF Aloysius Pang, a 28-year-old medic, died during a Singapore Armed Forces exercise in New Zealand on 23 January 2019 after being injured by a self-propelled howitzer. The incident triggered a national debate about NS training safety standards, adequacy of medical coverage during overseas exercises, SAF accountability, and whether servicemen are adequately protected. It became the single largest driver of NS discourse volume in the 2018–2019 period, with sentiment turning sharply negative across all three subreddits.",
      "topic_tags": ["NS Policy & Society", "BMT & Training", "Medical & Health"],
      "subreddit_focus": ["r/singapore", "r/askSingapore"],
      "sentiment_impact": "sharp_negative_spike",
      "volume_impact": "high",
      "search_keywords": ["aloysius", "pang", "new zealand", "training death", "siac", "howitzer", "safety", "medic"]
    },
    {
      "id": "covid_ns_disruptions_2020",
      "title": "COVID-19 NS disruptions and delayed enlistment",
      "date_start": "2020-03-01",
      "date_end": "2020-09-30",
      "description": "The COVID-19 pandemic caused widespread disruption to NS enlistment schedules, training programmes, and ICT cycles. MINDEF announced delayed enlistments, curtailed training exercises, and modified operational requirements. Discourse centred on uncertainty about enlistment timing, concerns about BMT safety under COVID, and questions about NS allowance during disruption periods.",
      "topic_tags": ["Enlistment & Pre-NS", "BMT & Training", "Admin & Logistics", "NS Policy & Society"],
      "subreddit_focus": ["r/singapore", "r/NationalServiceSG", "r/askSingapore"],
      "sentiment_impact": "moderate_negative",
      "volume_impact": "high",
      "search_keywords": ["covid", "pandemic", "delay", "enlistment postponed", "bmt closure", "ict cancelled"]
    },
    {
      "id": "ns_allowance_increase_2022",
      "title": "NS allowance increase announcement",
      "date_start": "2022-01-01",
      "date_end": "2022-06-30",
      "description": "MINDEF announced increases to NS allowances in 2022 following long-standing complaints about low pay relative to civilian salaries and rising cost of living. The increase was welcomed but also generated continued discourse about whether NS pay remains inadequate and whether it reflects the true economic cost to servicemen and their families.",
      "topic_tags": ["Pay & Benefits", "NS Policy & Society"],
      "subreddit_focus": ["r/singapore", "r/askSingapore"],
      "sentiment_impact": "mixed",
      "volume_impact": "moderate",
      "search_keywords": ["allowance increase", "pay raise", "ns pay", "mindef announcement", "salary"]
    },
    {
      "id": "ge2025",
      "title": "Singapore General Election 2025",
      "date_start": "2025-04-01",
      "date_end": "2025-06-30",
      "description": "The Singapore General Election took place in May 2025. Preceding the election, political discourse around NS policy, NS equity, and the fairness of mandatory conscription featured in campaign debates. The 2025 reversal in critical sentiment (notable drops in r/singapore and r/askSingapore) may be related to political environment changes post-election.",
      "topic_tags": ["NS Policy & Society", "Gender & Diversity"],
      "subreddit_focus": ["r/singapore"],
      "sentiment_impact": "possible_reversal_driver",
      "volume_impact": "moderate",
      "search_keywords": ["election", "ge2025", "policy", "vote", "parliament", "opposition"]
    }
  ]
}
```

**Notes for curation:**
- Every event date range should overlap generously — discourse about an event lasts weeks after it occurs
- The `search_keywords` field is used to boost FAISS retrieval for event-related queries
- Add: any NS fatalities or serious injuries surfaced in corpus volume spikes; high-profile misconduct cases; NS policy announcements; major NS-related parliamentary debates
- Target 15–25 events minimum for 2018–2025 coverage

---

### 3.2 `rag_topic_digests.json` — Per-Topic Narrative Summaries

**Why it exists:** When a user asks "what do NSmen say about mental health?", the LLM should not be retrieving cold chunks and reasoning about the topic from scratch. The topic digest pre-bakes the answer: themes, tone, representative quotes, sentiment/commitment profile. The LLM at synthesis time only needs to *personalise* this to the specific query.

**Generation process (run `knowledge_builder.py --topic-digests`):**
1. For each of 17 macro topics, pull the top 50 chunks by upvotes from `chunk_metadata.parquet`
2. Include sentiment and commitment labels, chunk date, subreddit
3. Run gpt-4.1 with the prompt below
4. Store output as JSON keyed by `topic_macro`

**Generation prompt:**
```
You are summarizing Reddit discourse about National Service (NS) in Singapore for an
analytical database. This summary will be used to answer questions about this topic area.

Topic area: {topic_macro}
Total chunks in corpus: {N}
Sentiment profile: {neg_pct}% negative / {neu_pct}% neutral / {pos_pct}% positive
Commitment profile: {committed_pct}% committed / {uncommitted_pct}% uncommitted / {neutral_pct}% neutral
Date range: {earliest} to {latest}

Top 50 chunks by upvotes (authentic Reddit text):
{chunks_formatted}

Write a structured analytical summary with these exact sections:

DOMINANT_THEMES: (3-5 bullet points of what people most discuss in this topic)
COMMON_CONCERNS: (the frustrations, complaints, or anxieties most expressed)
COMMUNITY_TONE: (describe the emotional register — go beyond the statistics)
UNCOMMITTED_EXAMPLES: (what does "uncommitted" look like here? concrete examples from the quotes)
NOTABLE_PATTERNS: (any shifts over time, subreddit differences, or controversies)
REPRESENTATIVE_QUOTES: (3-5 direct quotes that best capture this topic's discourse)

Be specific and analytical. Write for a government department analyst. Do not invent data.
```

**Output schema:**
```json
{
  "NS Policy & Society": {
    "dominant_themes": [
      "Fairness of mandatory conscription vs PR/foreigner exemptions",
      "Whether NS produces genuine national defence value",
      "NS as a social contract that is perceived as one-sided"
    ],
    "common_concerns": "The dominant frustration is perceived inequity...",
    "community_tone": "Predominantly adversarial and skeptical...",
    "uncommitted_examples": "In this topic, uncommitted manifests as institutional opposition rather than apathy...",
    "notable_patterns": "Critical discourse roughly doubled from 2021 to 2024...",
    "representative_quotes": [
      {"text": "...", "subreddit": "r/singapore", "year": 2023, "upvotes": 847},
      {"text": "...", "subreddit": "r/askSingapore", "year": 2022, "upvotes": 412}
    ],
    "sentiment_profile": {"neg": 0.62, "neu": 0.29, "pos": 0.09},
    "commitment_profile": {"committed": 0.04, "uncommitted": 0.18, "neutral": 0.78},
    "chunk_count": 42381,
    "last_updated": "2026-06-08"
  }
}
```

**Estimated cost:** ~$2 total (17 topics × ~300 words, gpt-4.1 pricing)

---

### 3.3 `rag_temporal_narratives.json` — Monthly Discourse Summaries

**Why it exists:** When a user asks about a specific time period, the system needs pre-baked context about what was happening that month — discourse volume, dominant topics, sentiment pattern, and which NS events occurred. This prevents the LLM from seeing raw chunk data and having to infer the narrative cold.

**Dependency:** Requires `temporal_sentiment.parquet` and `temporal_commitment.parquet` (Stage 8). Do not generate before Stage 8 is complete.

**Generation process (run `knowledge_builder.py --temporal-narratives`):**
1. For each month in the corpus (2018-01 to 2025-12, ~96 months):
   - Pull stats from `rag_fact_table.parquet` for that month
   - Pull the top 10 chunks by upvotes from that month
   - Cross-reference `ns_events.json` for events overlapping that month
2. Run gpt-4.1-mini with the prompt below (cheaper — 96 calls, lower stakes than topic digests)
3. Store output as JSON keyed by `"YYYY-MM"`

**Generation prompt:**
```
You are writing a brief analytical note about NS (National Service) Reddit discourse
for a specific month. This note will be used to answer questions about what was
happening in NS public discourse during this period.

Month: {year}-{month}
Corpus volume: {chunk_count} chunks ({volume_vs_baseline}% vs monthly baseline of {baseline})
Sentiment: {neg_pct}% negative / {neu_pct}% neutral / {pos_pct}% positive
Top topics this month: {topic_distribution}
Subreddit breakdown: {subreddit_distribution}

Known NS events in this period:
{events_text}  [or "No recorded NS events for this period" if none]

Top 10 chunks by upvotes:
{chunks_formatted}

Write a 120–150 word analytical note covering:
1. Was this month notable? (volume spike, sentiment shift, specific controversy)
2. What dominated the discourse?
3. If a specific event drove discourse, what was its impact?
4. What is the overall public mood toward NS this month?

If the month was unremarkable, say so in 2-3 sentences and move on.
Write factually. Do not invent data not present above.
```

**Output schema:**
```json
{
  "2019-01": {
    "volume": 4821,
    "volume_vs_baseline": "+340%",
    "sentiment": {"neg": 0.71, "neu": 0.23, "pos": 0.06},
    "dominant_topics": ["NS Policy & Society", "BMT & Training", "Medical & Health"],
    "events": ["aloysius_pang_2019"],
    "narrative": "January 2019 was the single most negative-sentiment month in the 2018–2025 dataset. Discourse volume spiked 340% above baseline following the death of Corporal Aloysius Pang during an overseas SAF exercise on 23 January. The dominant themes were training safety accountability, adequacy of medical support during overseas operations, and institutional responsibility. r/singapore and r/askSingapore drove the spike; r/NationalServiceSG showed a more muted response. Sentiment reached 71% negative — the corpus peak. The discourse remained elevated through February before returning to baseline in March.",
    "notable": true,
    "last_updated": "2026-06-08"
  },
  "2020-06": {
    "volume": 3102,
    "volume_vs_baseline": "+18%",
    "sentiment": {"neg": 0.38, "neu": 0.52, "pos": 0.10},
    "dominant_topics": ["Enlistment & Pre-NS", "Admin & Logistics"],
    "events": ["covid_ns_disruptions_2020"],
    "narrative": "June 2020 saw elevated volume driven by COVID-19 enlistment uncertainty. The dominant question was when delayed enlistments would resume and what the modified BMT programme would look like. Sentiment was moderately negative but not acutely — more anxious and uncertain than outraged. Admin questions dominated across all three subreddits.",
    "notable": true,
    "last_updated": "2026-06-08"
  }
}
```

**Estimated cost:** ~$3 total (96 months × ~150 words, gpt-4.1-mini pricing)

---

### 3.4 `rag_fact_table.parquet` — Pre-aggregated Statistical Database

**Why it exists:** Every quantitative query is answered by a direct pandas lookup on this table. No LLM, no vector search. The table is designed so that any reasonable quantitative question can be answered by a single `groupby` + `filter` + `agg` operation in <50ms.

**Dependency:** Requires `temporal_sentiment.parquet` + `temporal_commitment.parquet` (Stage 8). Also requires upvote data carried through from raw parquets.

**Schema (one row per unique combination of granularity keys):**

```
Granularity keys:
    year            int    (2018–2025)
    month           int    (1–12)
    subreddit       str    ("r/singapore", "r/askSingapore", "r/NationalServiceSG", "all")
    topic_macro     str    (17 macro topics + "all")
    doc_type        str    ("submission", "comment", "all")

Metrics:
    chunk_count             int    total chunks in this cell
    total_upvotes           int    sum of upvotes
    mean_upvotes            float  mean upvotes per chunk

    mean_sent_neg           float  mean negative sentiment score
    mean_sent_pos           float  mean positive sentiment score
    mean_sent_neu           float  mean neutral sentiment score
    pct_neg_dominant        float  % chunks where sent_neg is highest class
    pct_pos_dominant        float  % chunks where sent_pos is highest class
    pct_neu_dominant        float  % chunks where sent_neu is highest class

    upvote_weighted_sent_neg  float  Σ(sent_neg × log(1+upvotes)) / Σ(log(1+upvotes))
    upvote_weighted_sent_pos  float  Σ(sent_pos × log(1+upvotes)) / Σ(log(1+upvotes))

    pct_committed           float  % chunks labelled committed
    pct_uncommitted         float  % chunks labelled uncommitted
    pct_neutral_commit      float  % chunks labelled neutral (commitment)
    pct_critical            float  % chunks from critical sub-faction (sent_neg > 0.6 + uncommitted)
    pct_supportive          float  % chunks from supportive sub-faction (sent_pos > 0.4 + committed)
    net_disposition         float  (pct_committed + pct_supportive) - (pct_uncommitted + pct_critical)
    upvote_weighted_disposition  float  same but log(1+upvotes) weighted

    discourse_intensity     float  Σ(upvotes × |sent_neg - sent_pos|) — engagement-weighted controversy
```

**Pre-built aggregations (all combinations pre-computed at build time):**
- Full corpus × year (8 rows)
- Full corpus × year × month (96 rows)
- Full corpus × year × subreddit (24 rows)
- Full corpus × year × topic_macro (136 rows)
- Full corpus × year × month × subreddit (~288 rows)
- Full corpus × year × topic_macro × subreddit (~408 rows)
- Full corpus × year × month × topic_macro (~1,632 rows)

Total: ~2,600 pre-computed rows. Tiny. Loads entirely into memory at dashboard start.

---

### 3.5 `chunk_faiss.index` + `chunk_metadata.parquet` — Vector Search Layer

**Why it exists:** For qualitative questions, the system needs to retrieve actual Reddit quotes that are semantically relevant to the query. FAISS enables this over 737k chunks in ~100ms.

**Index type:** `faiss.IndexFlatIP` (inner product = cosine similarity on L2-normalised vectors). Flat = exact search, no approximation. 737k vectors fit in ~2.1 GB RAM — acceptable for a local dashboard.

**Build process:**
1. Load embeddings from chunk parquets (verify `chunker.py` saves them; if not, re-embed with `all-mpnet-base-v2`)
2. Stack into numpy array shape `(737274, 768)`, dtype float32
3. L2-normalise all vectors: `faiss.normalize_L2(embeddings)`
4. Build index: `index = faiss.IndexFlatIP(768); index.add(embeddings)`
5. Save: `faiss.write_index(index, "chunk_faiss.index")`
6. Build metadata companion: lightweight parquet with only the columns needed for filtering and display

**`chunk_metadata.parquet` schema (no embeddings, stays small):**
```
chunk_id        str     unique chunk identifier
faiss_idx       int     row position in FAISS index (= metadata row index)
text_snippet    str     first 300 characters of chunk text
subreddit       str
year            int
month           int
topic_macro     str
topic_fine_id   int
sent_neg        float
sent_pos        float
sent_neu        float
commit_label    str     (committed / uncommitted / neutral — available after 5b)
upvotes         int     score field from Reddit
doc_type        str     (submission / comment)
doc_id          str     parent post ID
```

**Upvote reranking formula:**
After FAISS returns top-50 by cosine similarity, rerank by:
```
combined_score = cosine_similarity × (1 + α × log(1 + upvotes))
α = 0.3  (tunable — higher = more weight to upvotes, lower = purer semantic match)
```
Return top-10 after reranking. This ensures community-endorsed relevant chunks surface above zero-upvote exact-match noise.

**Can be built immediately** — does not depend on Stages 6–8. Build now.

---

## 4. Query Processing — Full Step-by-Step

### Step 1: Intent Classification (rule-based, ~5ms)

```python
QUANTITATIVE_PATTERNS = [
    r'\bhow much\b',
    r'\bwhat (is|was|were) the (percentage|proportion|rate|share|%)\b',
    r'\b(increase|decrease|change|shift|trend|rise|fall|drop|grow)\b.*\b(year|month|20\d\d)\b',
    r'\bwhich (topic|subreddit|year|month|community)\b.*\b(most|highest|lowest|least|biggest|largest)\b',
    r'\bcompare\b',
    r'\byear.over.year\b',
    r'\bover time\b',
    r'\bby (topic|subreddit|year|month|community)\b',
    r'\b(20\d\d) (vs|versus|compared to) (20\d\d)\b',
]

QUALITATIVE_PATTERNS = [
    r'\bexplain\b',
    r'\bwhy\b',
    r'\bwhat (do|did|does) (people|nsm[ae]n|singaporeans?|redditors?|they) (say|think|feel|believe)\b',
    r'\bdescribe\b',
    r'\btell me about\b',
    r'\bwhat happened\b',
    r'\bgive me (examples?|instances?|quotes?)\b',
    r'\bwhat.s the (mood|feeling|sentiment) (about|toward|on)\b',
]
```

If both patterns match → classify as qualitative (contextual synthesis handles numbers too).
If neither matches → classify as qualitative (safer default; qualitative path handles everything).

---

### Step 2: Filter Extraction (regex + keyword dict, ~5ms)

```python
def extract_filters(query: str) -> dict:
    filters = {}

    # Year
    year_match = re.findall(r'\b(201[89]|202[0-5])\b', query)
    if year_match:
        filters['years'] = [int(y) for y in year_match]

    # Month
    MONTHS = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
        'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
        'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
    }
    for name, num in MONTHS.items():
        if re.search(r'\b' + name + r'\b', query.lower()):
            filters['months'] = filters.get('months', []) + [num]

    # Subreddit
    SUBREDDIT_MAP = {
        'r/singapore': ['r/singapore', 'singapore subreddit'],
        'r/askSingapore': ['r/asksingapore', 'asksingapore'],
        'r/NationalServiceSG': ['r/nationalservicesg', 'nationalservicesg', 'ns subreddit']
    }
    for canonical, aliases in SUBREDDIT_MAP.items():
        if any(a in query.lower() for a in aliases):
            filters['subreddits'] = filters.get('subreddits', []) + [canonical]

    # Topic
    TOPIC_KEYWORDS = {
        'NS Policy & Society': [
            'conscription', 'policy', 'mandatory', 'abolish', 'prs', 'foreigners',
            'exemption', 'defend', 'national service', 'institution', 'obligation'
        ],
        'Pay & Benefits': [
            'pay', 'allowance', 'salary', 'cpf', 'benefits', 'income', 'wage',
            'compensation', 'token', 'stipend'
        ],
        'BMT & Training': [
            'bmt', 'basic military training', 'recruit', 'training', 'exercise',
            'field camp', 'outfield', 'safety'
        ],
        'Mental Health': [
            'mental health', 'depression', 'suicide', 'stress', 'anxiety',
            'wellbeing', 'psychological', 'burnout'
        ],
        'Physical Fitness & IPPT': [
            'ippt', 'fitness', 'rt', 'remedial training', 'run', 'physical',
            'napfa', 'gold', 'silver'
        ],
        'Vocations & Units': [
            'vocation', 'unit', 'ocs', 'scs', 'officer', 'specialist', 'combat',
            'infantry', 'armour', 'artillery', 'navy', 'air force', 'rsaf', 'rsa'
        ],
        'Medical & Health': [
            'pes', 'medical', 'pes status', 'downpes', 'injury', 'health', 'medical board'
        ],
        'Enlistment & Pre-NS': [
            'enlist', 'enlistment', 'camo', 'pre-ns', 'disruption', 'deferment'
        ],
        'Reservist & ICT': [
            'reservist', 'ict', 'in-camp', 'ippt', 'orns', 'call up', 'reservist duties'
        ],
        'Relationships & Social': [
            'relationship', 'girlfriend', 'breakup', 'friends', 'social', 'ord leave'
        ],
        'Gender & Diversity': [
            'women', 'female', 'gender', 'diversity', 'ns for women', 'equality', 'sons'
        ],
    }
    topic_scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query.lower())
        if score > 0:
            topic_scores[topic] = score
    if topic_scores:
        # Top 1-2 matching topics
        top_topics = sorted(topic_scores, key=topic_scores.get, reverse=True)[:2]
        filters['topics'] = top_topics

    # Metric type
    if any(w in query.lower() for w in ['commitment', 'committed', 'uncommitted', 'invested', 'buy-in']):
        filters['metric'] = 'commitment'
    elif any(w in query.lower() for w in ['sentiment', 'feel', 'opinion', 'negative', 'positive']):
        filters['metric'] = 'sentiment'

    return filters
```

---

### Step 3A: Quantitative Path (no LLM, <1 second)

```python
def handle_quantitative(query: str, filters: dict, fact_table: pd.DataFrame) -> QuantitativeResult:
    df = fact_table.copy()

    # Apply filters
    if 'years' in filters:
        df = df[df['year'].isin(filters['years'])]
    if 'topics' in filters:
        df = df[df['topic_macro'].isin(filters['topics'])]
    if 'subreddits' in filters:
        df = df[df['subreddit'].isin(filters['subreddits'])]

    # Determine metric
    metric_col = 'mean_sent_neg'
    if filters.get('metric') == 'commitment':
        metric_col = 'net_disposition'
    elif 'positive' in query.lower():
        metric_col = 'mean_sent_pos'

    # Year-over-year delta
    yearly = df.groupby('year')[metric_col].mean().reset_index()

    result_lines = []
    for i in range(1, len(yearly)):
        yr = int(yearly.iloc[i]['year'])
        curr = yearly.iloc[i][metric_col]
        prev = yearly.iloc[i-1][metric_col]
        delta = curr - prev
        pct = (delta / prev * 100) if prev != 0 else 0
        direction = "increased" if delta > 0 else "decreased"
        result_lines.append(
            f"{yr}: {metric_col} = {curr:.3f} ({direction} by {abs(delta):.3f} / {abs(pct):.1f}% from {yr-1})"
        )

    # Format
    topic_str = filters.get('topics', ['all topics'])[0]
    answer = f"**{metric_col.replace('_', ' ').title()} — {topic_str}**\n\n"
    answer += "\n".join(result_lines)
    answer += f"\n\n*Based on {int(df['chunk_count'].sum()):,} chunks across the filtered dataset.*"

    return QuantitativeResult(answer=answer, data=yearly)
```

**Examples of what this handles directly:**
- "How much did negative sentiment toward NS pay change in 2022?" → delta on `mean_sent_neg` for `Pay & Benefits`, year=2022
- "Which topic had the highest negative sentiment in 2024?" → `argmax(mean_sent_neg)` across `topic_macro` for year=2024
- "Compare r/singapore vs r/askSingapore commitment levels in 2023" → filter by subreddit, year=2023, metric=net_disposition
- "What was the trend in uncommitted sentiment from 2018 to 2025?" → yearly `pct_uncommitted` across all subreddits

---

### Step 3B: Qualitative Path — Context Assembly (~170ms before LLM)

**Sub-step 3B-1: Load pre-computed context from memory (~10ms)**

```python
def load_context(filters: dict, events: list, digests: dict, narratives: dict) -> ContextBundle:
    bundle = ContextBundle()

    # Topic digests
    for topic in filters.get('topics', []):
        if topic in digests:
            bundle.add_digest(digests[topic])

    # Temporal narratives
    if 'years' in filters:
        for year in filters['years']:
            months = filters.get('months', list(range(1, 13)))
            for month in months:
                key = f"{year}-{month:02d}"
                if key in narratives:
                    bundle.add_narrative(narratives[key])

    # NS events overlapping the queried period
    if 'years' in filters:
        for event in events:
            event_start = datetime.fromisoformat(event['date_start'])
            event_end = datetime.fromisoformat(event['date_end'])
            for year in filters['years']:
                months = filters.get('months', list(range(1, 13)))
                for month in months:
                    period_start = datetime(year, month, 1)
                    period_end = datetime(year, month, calendar.monthrange(year, month)[1])
                    if event_start <= period_end and event_end >= period_start:
                        bundle.add_event(event)

    # Relevant fact stats for the period
    bundle.add_stats(
        query_fact_table(filters, fact_table)
    )

    return bundle
```

**Sub-step 3B-2: FAISS retrieval (~150ms)**

```python
def retrieve(query: str, filters: dict, index, metadata: pd.DataFrame,
             events: list, top_k: int = 50, return_k: int = 10) -> list[ChunkResult]:

    # Embed query
    q_emb = embedder.encode([query], normalize_embeddings=True).astype('float32')

    # Build metadata mask
    mask = pd.Series([True] * len(metadata))
    if 'years' in filters:
        mask &= metadata['year'].isin(filters['years'])
    if 'months' in filters:
        mask &= metadata['month'].isin(filters['months'])
    if 'subreddits' in filters:
        mask &= metadata['subreddit'].isin(filters['subreddits'])
    if 'topics' in filters:
        mask &= metadata['topic_macro'].isin(filters['topics'])

    candidate_idxs = mask[mask].index.tolist()

    # FAISS search within candidates
    selector = faiss.IDSelectorBatch(np.array(candidate_idxs, dtype=np.int64))
    search_params = faiss.SearchParameters()
    search_params.sel = selector
    distances, indices = index.search(q_emb, top_k, params=search_params)

    # Rerank: cosine_sim × (1 + 0.3 × log(1 + upvotes))
    ALPHA = 0.3
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        row = metadata.iloc[idx]
        upvote_boost = np.log1p(float(row['upvotes']))
        combined = float(dist) * (1 + ALPHA * upvote_boost)
        results.append(ChunkResult(
            text=row['text_snippet'],
            subreddit=row['subreddit'],
            year=int(row['year']),
            month=int(row['month']),
            topic=row['topic_macro'],
            upvotes=int(row['upvotes']),
            sent_neg=float(row['sent_neg']),
            sent_pos=float(row['sent_pos']),
            score=combined,
            faiss_sim=float(dist)
        ))

    results.sort(key=lambda x: x.score, reverse=True)

    # Minimum similarity gate: drop results below threshold
    results = [r for r in results if r.faiss_sim >= 0.25]

    # Boost results whose text matches ns_events search_keywords
    active_events = [e for e in events if e['id'] in context_bundle.event_ids]
    for r in results:
        for event in active_events:
            if any(kw in r.text.lower() for kw in event['search_keywords']):
                r.score *= 1.2

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:return_k]
```

**Sub-step 3B-3: Context window assembly (~5ms)**

Priority order — drop from the bottom if total exceeds 1500 tokens:

```
1. System prompt                     ~200 tokens   [NEVER dropped]
2. Statistical facts for period       ~100 tokens   [NEVER dropped]
3. NS events for period               ~150 tokens each, max 3 events
4. Topic digest(s)                    ~400 tokens, max 1 digest (most relevant)
5. Temporal narrative(s)              ~200 tokens, max 2 months
6. Top chunks                         ~150 tokens each (120-char text + metadata), max 8
7. User query                          ~50 tokens   [NEVER dropped]

Total target: ≤1500 tokens
```

```python
def assemble_context(query: str, filters: dict, bundle: ContextBundle,
                     chunks: list[ChunkResult]) -> str:

    parts = []

    # Statistical facts (always first — grounds the LLM in numbers)
    if bundle.stats:
        parts.append(f"## STATISTICAL DATA FOR THIS QUERY\n{bundle.stats_formatted()}")

    # NS events
    for event in bundle.events[:3]:
        parts.append(
            f"## NS EVENT: {event['title']} ({event['date_start']} to {event['date_end']})\n"
            f"{event['description']}"
        )

    # Topic digest (most relevant only)
    if bundle.digests:
        digest = bundle.digests[0]
        parts.append(
            f"## TOPIC CONTEXT: {digest['topic']}\n"
            f"Themes: {'; '.join(digest['dominant_themes'])}\n"
            f"Tone: {digest['community_tone']}\n"
            f"Sentiment profile: {digest['sentiment_profile']['neg']*100:.0f}% negative"
        )

    # Temporal narratives (max 2)
    for narrative in bundle.narratives[:2]:
        parts.append(f"## PERIOD: {narrative['period']}\n{narrative['narrative']}")

    # Retrieved chunks
    chunk_texts = []
    for i, chunk in enumerate(chunks, 1):
        date_str = f"{calendar.month_abbr[chunk.month]} {chunk.year}"
        sentiment_str = "negative" if chunk.sent_neg > 0.5 else ("positive" if chunk.sent_pos > 0.5 else "neutral")
        chunk_texts.append(
            f"[{i}] {chunk.subreddit} · {date_str} · {chunk.upvotes} upvotes · "
            f"{chunk.topic} · {sentiment_str}\n\"{chunk.text}\""
        )
    if chunk_texts:
        parts.append("## RETRIEVED QUOTES FROM CORPUS\n" + "\n\n".join(chunk_texts))

    # User query (last)
    parts.append(f"## USER QUESTION\n{query}")

    full_context = "\n\n---\n\n".join(parts)

    # Truncate if over limit (rough token estimate: 1 token ≈ 4 chars)
    while len(full_context) > 6000 and parts:
        # Drop second-to-last part (keep user query last)
        if len(parts) > 2:
            parts.pop(-2)
            full_context = "\n\n---\n\n".join(parts)
        else:
            break

    return full_context
```

---

### Step 4: LLM Synthesis with Streaming

**Model selection:**
- **Claude Haiku (claude-haiku-4-5)** — preferred. Fast, cheap ($0.25/1M input tokens), streams well.
- **gpt-4.1-mini** — alternative. Similar speed/cost profile.
- Do NOT use Claude Sonnet/Opus or gpt-4.1 — unnecessary for synthesis-only role; adds latency.

**System prompt (loaded once at startup):**

```
You are an analytical assistant for the NS Sentiment Research Project — a study of
Singaporean public attitudes toward National Service (NS) based on 737,000 Reddit posts
and comments from r/singapore, r/askSingapore, and r/NationalServiceSG, spanning 2018–2025.

Your role is to answer questions about this dataset using ONLY the statistical data,
event context, topic summaries, and quotes provided to you. Never invent statistics.
Never speculate beyond what the data shows.

CRITICAL DISTINCTIONS — get these right:
• "Sentiment" = emotional tone of a post (negative/neutral/positive). Measures HOW someone says something.
• "Commitment" = personal buy-in to NS (committed/uncommitted/neutral). Measures WHERE someone stands on NS.
• These are INDEPENDENT. A serviceman can be negative in sentiment (frustrated about pay) but still committed.
• "Uncommitted" has two flavours: APATHETIC (just clearing time, zao liao) and OPPOSED (institutional criticism).
• "Critical" is a subset of uncommitted — institutional opposition, NOT mere complaint.
• "Net disposition" = combined (committed + supportive) minus (uncommitted + critical).

ANSWER FORMAT:
1. Lead with a direct 1–2 sentence answer to the question
2. Support with specific statistics from the provided data (cite exact numbers)
3. Cite quotes inline: [r/singapore · Jan 2019 · 847 upvotes]
4. If an NS event explains the pattern, name it and explain its impact
5. Note any caveats or data gaps honestly

LENGTH: 150–300 words for most answers. Expand for complex multi-part questions.
TONE: Analytical. You are reporting to a supervisor or government department.

DO NOT:
• Invent any number not present in the provided statistical data
• Mix up sentiment and commitment
• Reference general knowledge about Singapore that is not in the provided context
• Recommend policies or make normative judgements
```

**Streaming call:**

```python
import anthropic

client = anthropic.Anthropic()

def synthesize_streaming(context: str, query: str):
    """Yields text chunks for streaming output."""
    messages = [
        {
            "role": "user",
            "content": context  # context already includes the query at the end
        }
    ]

    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=messages
    ) as stream:
        for text in stream.text_stream:
            yield text
```

---

## 5. Streamlit Chat Tab Implementation

```python
# In app/dashboard.py — add "💬 Ask NS Data" tab

def render_chat_tab(chatbot: NSChatbot):
    st.markdown("## Ask about NS Sentiment Data")
    st.markdown(
        "*Ask quantitative questions (e.g. 'How much did negative sentiment increase in 2023?') "
        "or contextual ones (e.g. 'Explain the sentiment spike in January 2019')*"
    )

    # Initialize session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} sources"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**{src.subreddit}** · "
                            f"{calendar.month_abbr[src.month]} {src.year} · "
                            f"{src.upvotes} upvotes · {src.topic}"
                        )
                        st.markdown(f"> {src.text}")
                        st.divider()

    # Input
    if prompt := st.chat_input("Ask about NS sentiment, commitment, or public discourse..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_box = st.empty()
            sources_box = st.empty()

            result = chatbot.route(prompt)

            if result.type == "quantitative":
                # No streaming needed — instant answer
                response_box.markdown(result.answer)
                full_response = result.answer
                sources = []

            else:
                # Streaming synthesis
                full_response = ""
                for chunk in chatbot.stream(result):
                    full_response += chunk
                    response_box.markdown(full_response + "▌")
                response_box.markdown(full_response)
                sources = result.chunks

                # Show sources
                if sources:
                    with sources_box.expander(f"📎 {len(sources)} sources from corpus"):
                        for src in sources:
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.caption(f"**{src.subreddit}**")
                                st.caption(f"{calendar.month_abbr[src.month]} {src.year}")
                                st.caption(f"↑ {src.upvotes} upvotes")
                                st.caption(f"{src.topic}")
                            with col2:
                                sentiment_tag = (
                                    "🔴 negative" if src.sent_neg > 0.5 else
                                    "🟢 positive" if src.sent_pos > 0.5 else
                                    "⚪ neutral"
                                )
                                st.caption(sentiment_tag)
                                st.markdown(f'> "{src.text}"')
                            st.divider()

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })
```

---

## 6. Code Module Structure

```
src/
└── rag/
    ├── __init__.py
    ├── config.py               # Paths, model names, constants (ALPHA, MIN_SIM, MAX_TOKENS, etc.)
    ├── chatbot.py              # NSChatbot orchestrator — main entry point
    ├── query_router.py         # Intent classification + filter extraction
    ├── fact_table.py           # Quantitative query handler (pandas operations)
    ├── retriever.py            # FAISS search + upvote reranking
    ├── context_assembler.py    # Assembles bounded context window from components
    ├── synthesizer.py          # LLM streaming call (Anthropic SDK)
    ├── events.py               # Load + date-range-match ns_events.json
    ├── prompts.py              # SYSTEM_PROMPT + generation prompts
    └── indexer.py              # (Build script, not runtime) — builds FAISS index + metadata

scripts/
└── rag/
    ├── build_index.py          # Run once: chunk parquets → chunk_faiss.index + chunk_metadata.parquet
    ├── build_topic_digests.py  # Run once: top chunks per topic → gpt-4.1 → rag_topic_digests.json
    ├── build_temporal_narratives.py  # Run after Stage 8: temporal_sentiment → gpt-4.1-mini → narratives
    └── build_fact_table.py     # Run after Stage 8: temporal parquets → rag_fact_table.parquet
```

**`chatbot.py` — main orchestrator:**

```python
class NSChatbot:
    def __init__(self):
        # Load everything into memory at startup
        self.index = faiss.read_index(config.FAISS_INDEX_PATH)
        self.metadata = pd.read_parquet(config.CHUNK_METADATA_PATH)
        self.fact_table = pd.read_parquet(config.FACT_TABLE_PATH)
        self.digests = json.load(open(config.TOPIC_DIGESTS_PATH))
        self.narratives = json.load(open(config.TEMPORAL_NARRATIVES_PATH))
        self.events = json.load(open(config.NS_EVENTS_PATH))['events']
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.router = QueryRouter()
        self.retriever = Retriever(self.index, self.metadata, self.embedder)
        self.assembler = ContextAssembler(self.digests, self.narratives, self.events, self.fact_table)
        self.synthesizer = Synthesizer()

    def route(self, query: str) -> QueryResult:
        intent = self.router.classify(query)
        filters = self.router.extract_filters(query)

        if intent == "quantitative":
            answer = FactTableHandler(self.fact_table).handle(query, filters)
            return QueryResult(type="quantitative", answer=answer)
        else:
            chunks = self.retriever.retrieve(query, filters)
            context = self.assembler.assemble(query, filters, chunks)
            return QueryResult(type="qualitative", context=context, chunks=chunks)

    def stream(self, result: QueryResult):
        """Generator for streaming synthesis."""
        yield from self.synthesizer.stream(result.context)
```

---

## 7. Latency Budget

| Step | Time | Notes |
|---|---|---|
| Dashboard startup (load all assets) | ~8–12s | One-time cost; FAISS index load dominates |
| Intent classification | ~2ms | Pure regex |
| Filter extraction | ~3ms | Pure regex + dict lookup |
| **Quantitative path total** | **<100ms** | Pandas groupby on ~2,600 row table |
| Context loader (JSON in memory) | ~10ms | Already loaded at startup |
| Query embedding | ~45ms | SentenceTransformer on CPU |
| FAISS search (50k candidates) | ~80ms | Flat index, inner product |
| Reranking + filtering | ~5ms | Pure numpy |
| Context assembly | ~5ms | String concatenation |
| **Qualitative path (pre-LLM)** | **~150ms** | |
| LLM first token (streaming) | ~600ms | Claude Haiku cold start |
| LLM full generation (600 tokens) | ~2–3s | Streaming — user sees text immediately |
| **Qualitative path total perceived** | **~2–3s** | User sees first words at ~750ms |

---

## 8. Edge Cases and Failure Modes

| Scenario | Handling |
|---|---|
| No time filter in query | Use full corpus; no temporal narrative loaded; fact table aggregated across all years |
| Query about a month with no recorded NS event | Temporal narrative still answers; event field in context says "No recorded events" |
| Query about an event not in `ns_events.json` | System answers from chunks only; response notes "no event annotation available for this period" |
| FAISS returns chunks with cosine sim < 0.25 (irrelevant) | Drop those chunks; rely on digests/narratives alone; note in response that direct quotes were sparse |
| Quantitative query with no matching data in fact table | Return: "No data found for [filters]. The corpus covers [date range] and [topics]." |
| Ambiguous topic (e.g. "training" maps to both BMT & Training and Physical Fitness) | Take top 2 topic matches; load both digests; FAISS searches across both topics |
| Query completely outside NS domain | System prompt redirects: "This chatbot covers NS sentiment data from Reddit (2018–2025). Your question appears to be outside this scope." |
| Quantitative + qualitative hybrid ("explain why NS pay sentiment changed in 2022") | Route as qualitative; include fact table stats in context assembly; LLM gets both the numbers and the context |
| User asks for something that requires commitment data but 5b distillation is incomplete | Gracefully note: "Commitment-level analysis is based on the 10,076-chunk annual sample (κ=0.752). Full-corpus commitment labels are in progress." |

---

## 9. Testing Plan

**Phase 1 build validation (before connecting LLM):**

| Test | Pass criterion |
|---|---|
| FAISS index loads correctly | `index.ntotal == 737274` |
| Metadata row count matches | `len(metadata) == 737274` |
| Query embedding shape correct | `(1, 768)` float32, L2-normalised |
| FAISS retrieval returns results | Top-10 for 5 test queries all have cosine sim > 0.25 |
| Upvote reranking changes order | At least 3 of 10 results swap positions after reranking |
| Filter extraction | 20 manually crafted queries → verify extracted filters match intent |

**Phase 2 end-to-end validation (20 representative queries):**

Quantitative (answer from fact table, no LLM):
1. "What was the negative sentiment rate for NS Policy & Society in 2024?"
2. "Which macro topic had the highest negative sentiment across 2018–2025?"
3. "How did uncommitted sentiment change year-over-year from 2022 to 2024?"
4. "Compare negative sentiment in r/singapore vs r/askSingapore in 2023."
5. "What was the net disposition score in 2025?"
6. "Which year had the highest discourse intensity for Pay & Benefits?"
7. "What percentage of NS Policy & Society chunks were negative in 2022 vs 2021?"
8. "How much did committed sentiment change between 2018 and 2024?"
9. "Which subreddit had the highest upvote-weighted negative sentiment in 2024?"
10. "What was the month with the highest negative sentiment in 2019?"

Qualitative (RAG synthesis):
11. "Explain the sharp drop in public sentiment in January 2019." → must mention Aloysius Pang
12. "What do NSmen say about NS pay?" → must cite Pay & Benefits digest + quotes
13. "Why did critical sentiment spike in 2022–2024?" → must surface trend + possible policy drivers
14. "What does 'zao liao' culture look like in the data?" → must retrieve uncommitted examples
15. "Explain the 2025 sentiment reversal." → must mention GE2025 if in ns_events.json
16. "What are the most common concerns about BMT?" → must use BMT & Training digest
17. "How did COVID affect NS sentiment?" → must retrieve COVID-era chunks + event annotation
18. "What do people in r/NationalServiceSG vs r/singapore feel differently about?"
19. "Are NSmen more committed or uncommitted in the Mental Health topic?"
20. "Show me examples of sarcasm or cynicism about NS." → FAISS retrieval of high-negativity + uncommitted chunks

**Pass criteria for qualitative:**
- Correctly cites at least 2 specific chunks with metadata
- Does not invent a statistic not in the context window
- Correctly distinguishes sentiment from commitment when both are relevant
- Mentions Aloysius Pang for query 11 (event annotation test)
- Response length 150–350 words
- First streaming token appears within 1 second

---

## 10. Build Sequence (Chronological)

### Now (Phase 1 — independent of pipeline)

**Day 1:**
1. Inspect `src/features/chunker.py` — confirm whether embeddings are saved as a column in chunk parquets. If yes, load directly. If no, run re-embedding locally (~3 hrs CPU).
2. Run `scripts/rag/build_index.py`:
   - Load all chunk parquets
   - Stack embeddings → shape `(737274, 768)`
   - L2-normalise, build `IndexFlatIP`, write `chunk_faiss.index`
   - Extract metadata columns, write `chunk_metadata.parquet`
   - Verify: `index.ntotal == 737274`, spot-check 5 queries

**Day 1–2:**
3. Manually build `ns_events.json`:
   - Research key NS events 2018–2025 (start with corpus volume spikes — months where chunk count is >200% of baseline)
   - Minimum: Aloysius Pang (Jan 2019), COVID disruptions (Mar 2020), NS pay changes (2022), GE2025 (May 2025)
   - Target 15–25 events
4. Run `scripts/rag/build_topic_digests.py`:
   - For each of 17 macro topics, pull top-50 chunks by upvotes from chunk_metadata.parquet
   - Run gpt-4.1 with topic digest prompt
   - Write `rag_topic_digests.json`
   - Cost: ~$2, ~45 min

**Day 2–3:**
5. Build `src/rag/` module skeleton — all files with stub implementations
6. Implement `query_router.py` — intent classifier + filter extractor
7. Implement `retriever.py` — FAISS search + upvote reranking
8. Unit test retrieval: 20 sample queries, verify filter + retrieval correctness

### After Stage 8 completes (Phase 2)

9. Run `scripts/rag/build_fact_table.py`:
   - Load `temporal_sentiment.parquet` + `temporal_commitment.parquet`
   - Build all pre-aggregated combinations
   - Write `rag_fact_table.parquet`
10. Run `scripts/rag/build_temporal_narratives.py`:
    - For each month 2018-01 → 2025-12
    - Cross-reference ns_events.json
    - Run gpt-4.1-mini with temporal narrative prompt
    - Write `rag_temporal_narratives.json`
    - Cost: ~$3, ~1 hr
11. Implement `fact_table.py` — quantitative query handler
12. Implement `context_assembler.py` — context window builder with token budget
13. Implement `synthesizer.py` — Anthropic streaming call
14. Implement `chatbot.py` — full orchestrator
15. Run 20-query test suite (see Section 9)
16. Integrate chat tab into Streamlit dashboard

### Phase 3 — Calibration

17. Tune `ALPHA = 0.3` (upvote reranking weight) — adjust if community-endorsed results feel over/under-represented
18. Tune `MIN_SIM = 0.25` (cosine similarity floor) — adjust if irrelevant chunks appear
19. Tune context token budget — check if 1500 tokens is too tight or too generous for typical queries
20. Final validation: re-run all 20 test queries, check all pass criteria

---

## 11. File Manifest

| File | Size estimate | Built when | Cost |
|---|---|---|---|
| `data/processed/new/chunk_faiss.index` | ~2.1 GB | Phase 1 | Free |
| `data/processed/new/chunk_metadata.parquet` | ~150 MB | Phase 1 | Free |
| `data/processed/new/ns_events.json` | ~50 KB | Phase 1 | Free (manual) |
| `data/processed/new/rag_topic_digests.json` | ~200 KB | Phase 1 | ~$2 |
| `data/processed/new/rag_fact_table.parquet` | ~500 KB | Phase 2 (after Stage 8) | Free |
| `data/processed/new/rag_temporal_narratives.json` | ~500 KB | Phase 2 (after Stage 8) | ~$3 |
| `src/rag/` (all modules) | ~800 lines | Phase 1–2 | Free |
| `scripts/rag/` (all build scripts) | ~400 lines | Phase 1–2 | Free |

**Total external API cost:** ~$5. All runtime inference is free (FAISS + pandas). LLM synthesis costs ~$0.001 per query (Haiku pricing).
