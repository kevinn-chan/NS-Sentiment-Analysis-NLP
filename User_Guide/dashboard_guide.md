# NS Sentinel — Dashboard User Guide

Complete guide to the NS Sentinel Streamlit dashboard. Written for a data scientist who hasn't seen the project before — every page, control, chart, and feature is documented, including what the numbers mean, how to interpret the visualisations, and where there are known limitations or improvement opportunities.

---

## Table of contents

1. [Getting started](#getting-started)
2. [Global controls](#global-controls)
3. [Page 1 — Overview](#page-1--overview)
4. [Page 2 — Sentiment Trends](#page-2--sentiment-trends)
5. [Page 3 — Topic Analysis](#page-3--topic-analysis)
6. [Page 4 — Divergence](#page-4--divergence)
7. [Page 5 — Commitment](#page-5--commitment)
8. [Page 6 — Population](#page-6--population)
9. [Page 7 — Sentinel Bot](#page-7--sentinel-bot)
10. [Keyboard shortcuts and tips](#keyboard-shortcuts-and-tips)
11. [Data refresh workflow](#data-refresh-workflow)
12. [Known issues and improvement opportunities](#known-issues-and-improvement-opportunities)

---

## Getting started

### Launch the dashboard

```bash
cd ns_sentiment
source .venv/bin/activate
streamlit run app/dashboard.py
```

The dashboard opens at `http://localhost:8501` in your default browser. If port 8501 is in use, Streamlit auto-increments to 8502, 8503, etc.

### Requirements

| Requirement | Details |
|---|---|
| **Python** | 3.10+ |
| **Packages** | All packages in `requirements.txt` installed in the virtual environment |
| **Data files** | Pre-computed parquets in `data/processed/new/`. Most are included in the repo. Three large files (>100 MB, gitignored) must be present locally — see below |
| **LLM API key** | For the Sentinel Bot (Page 7) only. Set in `.env` at the project root |

### Large files (gitignored, must be on disk)

These files exceed GitHub's 100 MB per-file limit and are not in the repository. They must be present in `data/processed/new/` for the dashboard to function fully:

| File | Size | Purpose | What breaks without it |
|---|---|---|---|
| `submissions_chunks.parquet` | 460 MB | Chunk text + embeddings (submissions) | Sentinel Bot cannot retrieve chunks |
| `comments_chunks.parquet` | 2.9 GB | Chunk text + embeddings (comments) | Sentinel Bot cannot retrieve chunks |
| `chunk_faiss.index` | 2.1 GB | FAISS vector index (737K embeddings) | Sentinel Bot cannot search |

If you don't have these files, run `scripts/rag/build_index.py` after generating chunks (Stage 3) to rebuild them.

### LLM API key setup

Create a `.env` file in the project root:

```bash
# Primary (recommended — free tier, Llama 3.3 70B)
GROQ_API_KEY=gsk_...

# Fallback options (any one is sufficient)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

The synthesizer tries backends in order: Groq → OpenAI → Anthropic → FreeLLMAPI. Only one key is needed. Groq is recommended because it's free (7,000 requests/month on the free tier).

---

## Global controls

### Sidebar

The left sidebar is visible on every page and contains:

1. **Theme toggle**: "Light mode" checkbox at the top. Dark mode is the default and is optimised for presentations. All charts, KPI cards, backgrounds, and text colours adapt.

2. **Navigation buttons**: 7 page names arranged vertically. Click to navigate. The current page is highlighted.

3. **Filters** (below navigation):

   | Filter | Options | Default | Notes |
   |---|---|---|---|
   | **Subreddit** | r/singapore, r/askSingapore, r/NationalServiceSG, All | All | Multi-select. "All" combines all 3 subreddits. |
   | **Year range** | Slider from 2018 to 2025 | 2019–2025 | Data exists before 2019 but is very sparse (r/askSingapore and r/NationalServiceSG barely existed). Setting the range to start at 2018 may show erratic patterns from low sample sizes. |
   | **Show low-volume months** | Checkbox | Unchecked (hidden) | Months with fewer than 30 documents are hidden by default. Toggle on to include them — useful for niche subreddits or very early periods where every data point matters. |
   | **Apply Filters** | Button | — | Filters only take effect when you click this button. A green "FILTERS APPLIED" confirmation banner appears. If you change filters without clicking Apply, the dashboard continues to show the old data. |

### How filters interact

- Filters apply globally across all pages within the session. If you filter to r/NationalServiceSG and then navigate from Overview to Commitment, the Commitment page also shows r/NationalServiceSG only.
- Changing pages does NOT reset filters.
- Refreshing the browser DOES reset filters to defaults.
- The year range slider controls which documents are included in all calculations. This includes KPI counts, trend charts, topic distributions, and the Sentinel Bot's fact table lookups.

---

## Page 1 — Overview

The landing page. Provides a high-level summary of the full corpus within the current filter range.

### Headline KPIs (top row)

Five metric cards arranged horizontally:

| KPI | What it shows | Typical value (full corpus) |
|---|---|---|
| **Documents** | Total Reddit posts + comments in the filtered corpus | ~549,671 |
| **Analysed Passages** | Total text chunks after semantic chunking. Each document produces ~1–3 chunks. | ~737,274 |
| **Fine-Grain Topics** | Number of leaf-level BERTopic topics in the taxonomy | 359 (fixed) |
| **Neg : Pos Ratio** | For every 1 positive post, how many negative posts exist. Neutral posts are excluded from this ratio. | ~2.0 : 1 |
| **Net Sentiment** | `(% positive docs − % negative docs)` across the filtered corpus. Negative means more negative than positive content. | Varies with filters |

**Interpretation**: The neg:pos ratio of ~2:1 means negative NS discourse outweighs positive by about 2×. This is consistent across years and subreddits — people are more likely to post when frustrated or venting than when happy.

### NS Sentiment Over Time (line chart)

The primary chart on the page. Monthly net sentiment with contextual overlays.

- **Blue solid line**: Monthly net sentiment `(% pos − % neg)`. Always negative in this corpus (net sentiment has never been positive in any month across the full dataset).
- **Amber dashed line**: Rolling average (smoothing window selectable — see below). Smooths out month-to-month noise to show the underlying trend.
- **Grey vertical bands**: Notable NS events with labels at the top. These are pre-defined in `data/ns_events.json` and include:
  - CFC Aloysius Pang training death (Jan 2019)
  - COVID-19 Circuit Breaker (Apr 2020)
  - Russia-Ukraine War (Feb 2022)
  - NS55 enhancements announcement
  - And others — check `ns_events.json` for the full list

**Interactive controls**:
- **Toggle**: "Overall" (all subreddits combined) vs "By subreddit" (separate lines for each subreddit, colour-coded)
- **Trend line dropdown**: Smoothing window — None, 3-month, 6-month, 12-month rolling average
- **Hover tooltip**: Hovering over any data point shows:
  - The exact net sentiment value
  - A plain-language summary of what drove sentiment that month (pre-computed)
  - Top 3 contributing topics with their percentage share, e.g., "Safety & Incidents (34%) · Conscription & National Duty (18%) · BMT Enlistment & Life (12%)"

### Sentiment Distribution (stacked bar chart)

Monthly stacked bars showing the proportion of negative (red), neutral (grey), and positive (green) documents. Useful for answering: "Is sentiment getting worse because there are more negative posts, or because positive posts are disappearing?"

### Most Negative Topics (horizontal bar chart)

Top 5 macro topics ranked by percentage of negative documents. Quick identification of the most negatively discussed NS themes.

---

## Page 2 — Sentiment Trends

Deep-dive into sentiment over time with five analytical views.

### Sub-tabs

#### Net Sentiment

Monthly `(% pos − % neg)` line chart. Same as the Overview chart but with more room and controls. Toggle "By subreddit" / "Overall". Each data point has the hover tooltip with sentiment drivers and top 3 contributing topics.

#### % Negative

Monthly percentage of documents classified as negative. Tracks absolute negativity rather than the balance between positive and negative. Useful when you want to know "is the volume of complaining increasing?" independent of positive content.

#### % Positive

Monthly percentage of positive documents. Typically 5–15% of all NS discourse. Look for sustained increases (indicating improving attitudes) vs temporary spikes (usually triggered by a specific positive event like NS55 enhancements).

#### Full Stack

Stacked area chart with all three sentiment classes (negative, neutral, positive) as proportions over time. The three areas always sum to 100%. Shows the composition of discourse — most months are 60–75% neutral, 15–25% negative, 5–15% positive.

#### Grievance Amplification

Compares sentiment in **original posts** vs **their comment sections**. Identifies periods where:
- The community **amplifies negativity** (comments more negative than the post → commenters pile on)
- The community **dampens negativity** (comments less negative than the post → commenters push back or provide consolation)

This uses the divergence metrics from Stage 7. High amplification periods often coincide with viral negative incidents (e.g., training deaths).

### Reading the charts

- All sentiment line charts include event annotation bands (grey vertical areas with labels at top)
- **Hover over any data point** on net sentiment, % negative, or % positive charts to see a natural-language summary of what drove sentiment that month and the top 3 contributing topics with percentage shares
- The dashed trend line smooths out monthly noise — look at this for the overall direction of sentiment
- When "By subreddit" is selected: r/NationalServiceSG (amber) is typically more negative than r/singapore (blue) or r/askSingapore (green). This is expected — r/NationalServiceSG is where active NSFs vent. r/singapore covers NS as one of many topics and tends to be less emotionally charged.
- Low-volume months (< 30 docs) are hidden by default. Toggle "Show low-volume months" in the sidebar to see them, but treat those data points with caution — a single viral post can swing the entire month's reading.

---

## Page 3 — Topic Analysis

The most feature-rich page. Explore the full topic taxonomy (17 macro → 52 sub → 112 cluster → 359 leaf topics).

### Controls

- **Min upvotes per document**: Slider (default 0). Set to 5+ to focus on community-validated content and filter out low-engagement posts. This is useful for filtering out posts that nobody read or agreed with.
- Subreddit and year range come from the sidebar filters.

### Sub-tabs

#### Landscape (treemap)

An interactive treemap showing the full topic hierarchy.

- **Tile size** = document volume (more documents → larger tile)
- **Tile colour** = net sentiment (red → negative, green → positive, grey/white → neutral)
- **Click any tile** to drill into that level's children. A breadcrumb trail at the top shows your current depth — click any breadcrumb to zoom back out.
- **Hover** to see exact stats: document count, sentiment breakdown (% neg / neu / pos), dominant stance label

The hierarchy (from broadest to narrowest):
```
All NS Discourse (root)
 └── 17 macro topics (e.g., "BMT & Training")
      └── 52 sub topics (e.g., "Basic Military Training")
           └── 112 cluster topics (e.g., "BMT Enlisted Life")
                └── 359 leaf topics (e.g., "Recruit Platoon Dynamics")
```

**How to read the treemap**: Large red tiles represent high-volume, high-negativity topics — these are the areas of NS discourse generating the most unhappiness. Large green tiles are positive areas (rare in NS discourse). Small tiles are niche topics with low discussion volume.

#### Drill Down

Select a specific macro topic from the dropdown. Shows:
- **Nested treemap** of sub/cluster/leaf topics within that macro topic
- **Topic-specific sentiment trend** (line chart of that topic's net sentiment over time)
- **Top keywords** per sub-topic
- **Document count breakdown** showing how discussion volume is distributed within the macro topic

#### Over Time

Stacked area chart showing how each macro topic's discussion volume changes over time. Identify:
- Topics that are growing in volume (e.g., "Mental Health" has been trending upward)
- Topics that are declining (e.g., some equipment topics decline as NS modernises)
- Seasonal patterns (e.g., "BMT & Training" spikes during enlistment months)

#### Topic Rankings

Sortable table ranking all 17 macro topics by various metrics. Click any column header to sort.

| Column | What it measures |
|---|---|
| Topic | Macro topic name |
| Docs | Total document count |
| % Corpus | Percentage of total corpus this topic represents |
| Net Sent. | Net sentiment `(% pos − % neg)` |
| % Neg | Percentage of documents classified negative |
| % Pos | Percentage of documents classified positive |
| Net Stance | Net stance `(% supportive − % critical)` |

This is the quickest way to compare topics numerically. Sort by "% Neg" to find the most negative topics, or by "Net Stance" to find the most criticised topics.

#### Volume by Topic

Horizontal bar chart comparing document volume across all 17 macro topics. At a glance: "NS Life & Culture" and "BMT & Training" dominate the discourse; niche topics like "Technology & Digital NS" or "Gear & Equipment" have much less discussion.

#### Sentiment by Topic

Diverging horizontal bar chart showing net sentiment per macro topic. Bars extending left = net negative (more negative than positive discussion). Bars extending right = net positive (rare). Most NS topics are net negative, but the magnitude varies significantly.

#### Stance by Topic

Same format as Sentiment by Topic, but showing net stance `(% supportive − % critical)` per macro topic. Topics extending left are more criticised (critical > supportive). Topics extending right have more defenders. "NS Policy & Society" tends to be the most criticised; "Vocations & Units" tends to be more balanced.

---

## Page 4 — Divergence

Analyses how comment sections differ from original posts in sentiment — do communities amplify, challenge, or agree with posters?

### Headline KPIs

| KPI | What it shows |
|---|---|
| **Threaded Posts** | Number of submissions that have at least 1 analysed comment chunk |
| **Top Reach + Divergence** | The macro topic with the highest *discourse intensity* (`Σ(upvotes × |neg − pos|)` per topic). This identifies topics where opinionated, high-visibility discussion concentrates. |
| **Most Opinionated** | Topic with the highest proportion of opinionated chunks (chunks where `max(sent_neg, sent_pos) > 0.5`) |
| **Tone Shift Rate** | Percentage of threads where the dominant sentiment of comments differs from the dominant sentiment of the original post |

### Sub-tabs

#### Tone Shift Matrix

A 3×3 heatmap showing transition probabilities from post sentiment → comment sentiment:

```
                     Comments
                 Neg    Neu    Pos
Posts  Neg    [  ■  ] [     ] [     ]
       Neu    [     ] [  ■  ] [     ]
       Pos    [     ] [     ] [  ■  ]
```

- **Diagonal cells** (e.g., Neg→Neg): Community agrees with the post's tone. High diagonal = community echoes posters.
- **Off-diagonal cells** (e.g., Pos→Neg): Community shifts tone. The Pos→Neg cell is especially interesting — it means someone posted something positive and the community responded negatively.
- Accompanied by a "Key Patterns" callout box highlighting the most notable pattern (e.g., "Negative posts receive more negative comments 73% of the time — an echo chamber effect").

**What to look for**: A high Neg→Neg value with a low Pos→Pos value suggests an asymmetric echo chamber — negativity is amplified, but positivity is not reinforced.

#### Community Battlegrounds

A sortable table of the most divergent threads — posts where comment section sentiment differs the most from the post itself. Ranked by `upvote_weighted_div` (the primary divergence metric).

Each row shows:
- Post title (truncated), subreddit, date
- Post sentiment label and comment sentiment label
- Divergence score
- Number of comment chunks
- Total upvotes

**How to use**: Click on high-divergence threads to understand what triggers community pushback. Common patterns: controversial policy opinions, personal stories that the community disagrees with, factual claims the community corrects.

#### Opinion Density

Chart showing which topics generate the most opinionated (non-neutral) discussion. Topics with high opinion density are emotionally charged — people feel strongly about them regardless of whether they're positive or negative.

---

## Page 5 — Commitment

Tracks commitment (personal investment in NS) and stance (support/criticism of NS policy) over time using the 4-stage cascade classifier.

### Understanding the two axes

This page uses two independent dimensions (see [pipeline.md](pipeline.md) Stage 5b for full details):

1. **Buyin** (personal investment):
   - **Committed**: Personally invested in NS — pride, growth, voluntary engagement
   - **Uncommitted**: Personally disengaged — "waste of time", counting down to ORD, reluctant compliance
   - **Neutral**: No buyin signal expressed

2. **Stance** (policy opinion):
   - **Supportive**: Endorses NS as institution/policy
   - **Critical**: Opposes NS as institution/policy
   - **Neutral**: No stance expressed

These are **independent**. A person can be committed to NS but critical of its policies, or uncommitted but supportive of the principle. This 2D framework captures more nuance than a single positive/negative axis.

### Model Details (expandable panel)

Click to expand a panel explaining:
- The cascade architecture (Stage 1a/1b → Stage 2a/2b)
- The two-axis framework with examples
- Evaluation metrics (Stage 2a F1=0.714, Stage 2b F1=0.784)

### Headline KPIs

| KPI | What it shows | Typical value |
|---|---|---|
| **Total Chunks** | Number of chunks classified by the cascade model | ~737K |
| **Uncommitted / Committed** | Percentage of buyin-relevant chunks classified as uncommitted vs committed. Only counts chunks where buyin is non-neutral. | ~60% / 40% |
| **Critical / Supportive** | Percentage of stance-relevant chunks classified as critical vs supportive. Only counts chunks where stance is non-neutral. | ~55% / 45% |

### Sub-tabs

#### Commitment Decline

The primary analytical view. Contains:

1. **Thesis statement**: A data-driven narrative summarising the commitment trend. Auto-generated from the data. Example: *"Net commitment has been negative every year on record — uncommitted sentiment consistently outweighs committed."*

2. **Three KPI cards**:
   - **Peak Decline**: The largest drop in net commitment between consecutive years within the filter range
   - **Recovery Since Trough**: The recovery from the lowest point to the most recent year
   - **Net vs Baseline**: Current commitment compared to the first year in the filter range. Green arrow = improvement, red arrow = decline.

3. **Monthly Net Commitment chart**: Line chart of `net_buyin = (% committed − % uncommitted)` per month.
   - **Blue line**: Monthly net buyin
   - **Dashed amber line**: 6-month rolling average
   - **Red shaded region**: Area below zero (net uncommitted exceeds committed — which is most of the time)
   - **Event annotation bands**: Same NS events as the sentiment charts

**How to interpret**: Net commitment is almost always negative in this corpus, meaning more people express disengagement from NS than personal investment. The key trends to look for are:
- Is the negative gap widening (worsening commitment) or narrowing (improving)?
- Do specific events (e.g., NS55 enhancements, training deaths) cause visible shifts?
- Does the rolling average show a structural trend or just noise?

#### Signal Detail

Breakdown of individual commitment signals over time:
- Monthly `% committed`, `% uncommitted`, `% supportive`, `% critical` as separate line charts
- Helps diagnose whether changes in net commitment are driven by:
  - More uncommitted posts (people venting more), or
  - Fewer committed posts (people stopping to defend NS), or
  - Both

#### By Topic

Commitment and stance breakdown per macro topic. Shows which topics are associated with:
- The highest commitment (e.g., "Vocations & Units" — people talking about their vocation tend to express more personal investment)
- The highest criticism (e.g., "NS Policy & Society" — policy discussions draw the most critical stance)
- The most neutral (e.g., "Admin & Logistics" — procedural questions rarely express commitment or criticism)

---

## Page 6 — Population

Demographics and posting patterns of the Reddit user base in the NS corpus.

### Headline KPIs

| KPI | What it shows |
|---|---|
| **Unique Authors** | Total distinct Reddit usernames who authored at least one post or comment in the filtered corpus |
| **Unique Posts** | Total submissions (not counting comments) |
| **Unique Comments** | Total comments |
| **Returning Authors** | Percentage of authors who posted more than once in the corpus |
| **Single-Sub Authors** | Percentage of authors who only posted in one subreddit (never cross-posted between the 3 subreddits) |

### Visualisations

**Top 20 User Flairs** (horizontal bar chart):
- Shows the most common Reddit user flairs. Flairs are self-selected identity labels that users assign to themselves in each subreddit (e.g., "Infantry", "Military Police", "Pre-Enlistee", "NSman").
- **Caveat**: Flairs are self-reported and optional. Most users don't set a flair. The distribution represents the subset who actively self-identify, not the full population.

**Posting Activity by Year** (dual-axis chart):
- Bars = total posts per year
- Line = unique authors per year
- Shows the growth of NS discourse on Reddit over time. Both metrics typically increase from 2018 to 2022, then plateau.

**Subreddit Breakdown** (pie/donut chart):
- Shows how posts are distributed across the 3 subreddits. r/singapore dominates in volume but r/NationalServiceSG has the highest NS-post density.

**Top Authors** (table):
- Most prolific contributors ranked by total post + comment count
- Columns: author, post count, comment count, primary subreddit, dominant sentiment

**Author Engagement Distribution** (histogram):
- Shows the long-tail distribution of posts per author
- Most authors post once (drive-by venting or questions). A small core group (< 5% of authors) generates > 30% of all content.
- The x-axis is typically log-scaled to show the tail clearly.

---

## Page 7 — Sentinel Bot

An AI chatbot that answers questions about NS sentiment using RAG (Retrieval-Augmented Generation). It searches through the full 737K chunk corpus and synthesises data-grounded answers.

### Requirements

| Requirement | Status check |
|---|---|
| Large data files on disk | Check that `submissions_chunks.parquet`, `comments_chunks.parquet`, and `chunk_faiss.index` exist in `data/processed/new/` |
| LLM API key | Check `.env` file in project root. At least one of: `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| First load time | ~15 seconds to load FAISS index and chunk metadata into memory. Subsequent queries are fast (< 3 seconds). |

### How to use

Type a question in the chat input at the bottom of the page. The bot can handle:

**Quantitative questions** (answered from pre-computed fact table — fast, precise):
- "What percentage of posts about BMT are negative?"
- "How has sentiment about IPPT changed from 2020 to 2024?"
- "Which subreddit is most negative about NS policy?"
- "What's the commitment breakdown for r/NationalServiceSG?"
- "Compare sentiment between r/singapore and r/askSingapore on mental health"

**Qualitative questions** (answered from retrieved chunks — richer, with quotes):
- "What do people complain about most regarding BMT?"
- "Why did sentiment about NS pay improve in 2024?"
- "What are the main arguments against conscription?"
- "Give me examples of positive NS experiences"
- "What do people say about reservist obligations?"

**Methodology questions**:
- "How does the sentiment model work?"
- "What data is this analysis based on?"
- "How many topics are there?"

### How it works internally

1. **Query routing**: Your question is classified as quantitative or qualitative by regex pattern matching (no LLM call). The router also extracts structured filters: years, months, subreddits, topics, metrics.
2. **Retrieval** (qualitative queries): Your question is embedded using the same model that embedded all chunks (`all-mpnet-base-v2`). The 10 most semantically similar chunks are retrieved from the FAISS index, filtered by your query's implied filters (year, subreddit, topic), and reranked by `cosine_similarity × (1 + 0.3 × log(1 + upvotes))`.
3. **Fact lookup** (quantitative queries): Pre-computed statistics are looked up from `rag_fact_table.parquet` — no FAISS search needed. Includes sentiment breakdowns, commitment metrics, document counts.
4. **Spike detection**: The query period is checked against known NS events (`ns_events.json`) and temporal anomalies (z-score spikes in sentiment).
5. **Context assembly**: All sources (facts, events, topic digests, temporal narratives, retrieved chunks) are packed into a 10,000-character context window, prioritised by relevance.
6. **Synthesis**: Groq Llama 3.3 70B (or fallback LLM) generates a response. The system prompt instructs it to never invent statistics, cite data inline, and keep answers concise.

### Tips for best results

- **Be specific**: "What do people say about BMT food in r/NationalServiceSG in 2023?" gets a much better answer than "Tell me about BMT"
- **Include time periods**: The bot can compare across years ("How has sentiment about IPPT changed from 2019 to 2024?") — this triggers longitudinal timeline context
- **Name the subreddit**: If you're interested in a specific community, name it. Different subreddits have very different tones.
- **Quantitative > qualitative for stats**: If you want a number, phrase it as a quantitative question. "What percentage..." triggers the fact table and gives precise, pre-computed answers.
- **The bot doesn't remember context**: Each question is independent. "What about that in 2023?" after asking about BMT in 2022 will not carry over the BMT context. Rephrase as a standalone question.
- **Source citations**: The bot cites sources inline, e.g., `[r/singapore · Jan 2019 · 847 upvotes]`. These refer to real Reddit chunks in the corpus.

### Limitations

- **No conversation memory**: Each question is independent (see above)
- **Rate limits**: Groq free tier allows ~7,000 requests/month. Heavy use may hit the limit, triggering automatic fallback to OpenAI (requires `OPENAI_API_KEY`).
- **Topic matching is keyword-based**: If you ask about a topic using unusual phrasing, the bot may not match it to the right macro topic. Use terms from the 17 macro topic names (listed in Page 3 — Topic Rankings) for best matching.
- **10K char context limit**: For very broad queries, the bot may not be able to include all relevant data. Narrower queries get better answers.

---

## Keyboard shortcuts and tips

| Action | How |
|---|---|
| **Collapse sidebar** | Click the "X" or use Streamlit's sidebar toggle. Good for presentations. |
| **Chart zoom** | Click and drag on any Plotly chart to zoom into a region |
| **Chart pan** | Shift + click-drag |
| **Chart reset** | Double-click the chart to reset zoom |
| **Download chart** | Hover over any chart → click the camera icon (top-right of chart) to download as PNG |
| **URL state** | Each page has a URL parameter (e.g., `?page=Commitment`). Bookmark specific pages for quick access. |
| **Filter persistence** | Filters persist across page navigation within the same session. Refreshing the browser resets to defaults. |

---

## Data refresh workflow

When new Reddit data is added to the corpus (e.g., new months of data from an alternative to Pushshift), the full pipeline must be re-run to update the dashboard. Here's the sequence:

```
1. Add new ZST dumps to data/raw/
2. Re-run Stage 1 (loader.py) — produces updated interim parquets
3. Re-run Stage 2 (cleaner.py) — filters to NS-relevant content
4. Re-run Stage 3 (chunker.py) — produces new chunks + embeddings
5. Re-run Stage 4 on Kaggle — topic modelling on new chunks
   (or use BERTopic's transform() to assign existing topics to new chunks)
6. Re-run Stage 5a on Kaggle — SingBERT sentiment inference
7. Re-run Stage 5b on Kaggle — Cascade commitment inference
8. Re-run Stage 6 (stage6_doc_aggregation.py)
9. Re-run Stage 7 (stage7_divergence_v2.py)
10. Re-run Stage 8 (stage8_temporal.py)
11. Re-run RAG builds: build_index.py, build_topic_digests.py,
    build_temporal_narratives.py, build_fact_table.py
12. Restart the dashboard
```

**Stages 4, 5a, 5b run on Kaggle** (T4 GPU required). Everything else runs locally.

**Shortcut for small updates**: If you only need to update the RAG knowledge base without re-running the full pipeline, you can run just steps 8–11 to refresh the pre-computed stats, narratives, and fact tables. The FAISS index will still reflect the old chunk set, so new data won't appear in retrieved chunks.

---

## Known issues and improvement opportunities

### Dashboard UX

- **No export functionality**: The dashboard doesn't have a "download data" button for the underlying data behind each chart. Users who need the raw data must access the parquet files directly with pandas. **Improvement**: Add CSV download buttons to key tables and charts.
- **No saved filter presets**: Frequently used filter combinations (e.g., "r/NationalServiceSG, 2022–2024, no low-volume months") must be manually re-applied each session. **Improvement**: Add named filter presets that persist across sessions via `st.query_params`.
- **Slow initial load**: The dashboard loads all parquets into memory on startup (~5–10 seconds). This is acceptable but could be improved with lazy loading (load data only when a page is visited). **Improvement**: Use `@st.cache_resource` more aggressively and load page-specific data on demand.
- **Mobile responsiveness**: The dashboard is designed for desktop-width screens. On mobile or narrow windows, some charts and KPI cards overlap or truncate. **Improvement**: Add responsive CSS or mobile-specific layouts.

### Data quality

- **r/NationalServiceSG dominates commitment signals**: This subreddit has a much higher proportion of committed/uncommitted posts than r/singapore (because all its posts are NS-focused). When viewing commitment metrics for "All" subreddits, r/NationalServiceSG's signal may dominate. **Mitigation**: Use the subreddit filter to compare subreddits individually.
- **Deleted/removed posts**: Some Reddit posts and comments have `[deleted]` or `[removed]` content. These are cleaned to empty strings in Stage 2 but may still contribute a chunk with just the title text. Such chunks have limited value for sentiment/commitment analysis.
- **Temporal bias**: Reddit's user base skews young, male, and English-speaking. The sentiment captured here is from a self-selected, non-representative sample of NS-related opinions. The dashboard measures *online discourse*, not population-level attitudes.

### Models

- **Sentiment model (SingBERT)**: Macro F1 of 0.78. Main weakness is positive recall (0.78) — some genuinely positive posts are classified as neutral. See [pipeline.md](pipeline.md) Stage 5a for improvement ideas.
- **Commitment model (cascade)**: Stage 2a F1 of 0.714 is the weakest link. Ambiguous texts with mixed committed/uncommitted signals are the primary source of error. More human-labelled training data (currently 1,267 gold rows) would help the most. See [pipeline.md](pipeline.md) Stage 5b for details.
- **Topic model (BERTopic)**: 5–8% of chunks are outliers (no topic assigned). These appear in sentiment/commitment aggregation but are invisible in topic-level analysis. The static model doesn't adapt to new topics.

### Sentinel Bot

- **No conversation memory** (described above)
- **Query router keyword gaps**: Topic matching may miss unusual phrasings. E.g., asking about "tekong food" may not match to "Food & Cookhouse" if "tekong" isn't in that topic's keyword list. **Improvement**: Use embedding similarity for topic matching instead of keywords.
- **Fixed context budget**: The 10K char limit may be insufficient for complex multi-topic queries. See [pipeline.md](pipeline.md) RAG section for details.
