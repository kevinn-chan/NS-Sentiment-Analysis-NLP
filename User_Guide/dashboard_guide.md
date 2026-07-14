# NS Sentinel — Dashboard User Guide

This guide walks through every page and feature of the NS Sentinel dashboard.

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

---

## Getting started

### Launch the dashboard

```bash
cd ns_sentiment
source .venv/bin/activate
streamlit run app/dashboard.py
```

The dashboard opens at `http://localhost:8501` in your default browser.

### Requirements

- Python 3.10+
- All packages in `requirements.txt` installed
- Pre-computed data files in `data/processed/new/` (included in the repository except for the 3 large files listed in README)
- For the Sentinel Bot page: a `.env` file with an LLM API key (`GROQ_API_KEY` recommended — free tier Llama 3.3 70B; `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` also supported)

---

## Global controls

### Sidebar

The left sidebar is visible on every page and contains:

- **Theme toggle**: Switch between dark mode (default) and light mode using the "Light mode" checkbox at the top
- **Navigation buttons**: Click any of the 7 page names to navigate
- **Filters** (below navigation):
  - **Subreddit filter**: Select one or more of the 3 subreddits (r/singapore, r/askSingapore, r/NationalServiceSG), or "All"
  - **Year range slider**: Filter data to a specific year range (default: 2019–2025). Data before 2019 exists but is low-volume
  - **Show low-volume months**: Toggle whether to include months with fewer than 30 documents in trend charts
  - **Apply Filters button**: Click to apply filter changes. A green "FILTERS APPLIED" confirmation appears

### Theme

The dashboard supports full dark and light mode theming. All charts, KPI cards, and text adapt to the selected theme. Dark mode is the default and is optimised for presentation use.

---

## Page 1 — Overview

The landing page provides a high-level summary of the entire corpus.

### Headline KPIs (top row)

Five metric cards across the top:

| KPI | What it means |
|-----|---------------|
| **Documents** | Total Reddit posts + comments in the corpus (549,671) |
| **Analysed Passages** | Total text chunks after semantic chunking (737,274). Each document is split into ~1–3 chunks. |
| **Fine-Grain Topics** | Number of leaf topics in the BERTopic taxonomy (359) |
| **Neg : Pos Ratio** | For every 1 positive post, how many negative posts exist (2.0 : 1). Neutral posts are excluded from this ratio. |
| **Net Sentiment** | Overall net sentiment: (% positive − % negative) across all documents. A negative value means more negative than positive content. |

### NS Sentiment Over Time (chart)

A line chart showing monthly net sentiment (% positive docs − % negative docs).

- **Blue solid line**: Monthly net sentiment
- **Amber dashed line**: 6-month rolling average (smoothed trend)
- **Grey vertical bands**: Notable NS events (e.g., "CFC Aloysius Pang training death", "COVID-19 Circuit Breaker", "Russia-Ukraine War")
- **Hover on any data point**: Displays a tooltip with a plain-language summary of what drove sentiment that month, plus the top 3 topics contributing to the sentiment shift (e.g., "Safety & Incidents (34%) · Conscription & National Duty (18%) · BMT Enlistment & Life (12%)"). This is powered by pre-computed monthly sentiment driver analysis.
- **Toggle**: Switch between "Overall" (all subreddits combined) and "By subreddit" (separate lines per subreddit)
- **Trend line dropdown**: Choose smoothing window (3-month, 6-month, 12-month, or none)

### Sentiment Distribution (bar chart)

Stacked bar chart showing the proportion of negative, neutral, and positive documents per month. Helps visualise whether sentiment shifts are driven by more negative content appearing or positive content disappearing.

### Most Negative Topics (bar chart)

Horizontal bar chart of the top 5 macro topics by percentage of negative documents. Quick identification of which NS topics generate the most negative sentiment.

---

## Page 2 — Sentiment Trends

Deep-dive into sentiment over time with multiple views.

### Sub-tabs

| Tab | What it shows |
|-----|---------------|
| **Net Sentiment** | Monthly net sentiment (% pos − % neg) as a line chart. Toggle between "By subreddit" and "Overall". |
| **% Negative** | Monthly percentage of negative documents. Useful for tracking absolute negativity rather than relative balance. |
| **% Positive** | Monthly percentage of positive documents. |
| **Full Stack** | Stacked area chart showing all three sentiment classes (negative, neutral, positive) as proportions over time. Shows how the sentiment composition changes. |
| **Grievance Amplification** | Compares sentiment in original posts vs their comment sections. Identifies periods where the community amplifies or dampens negative sentiment. |

### Reading the charts

- Each chart includes event annotation bands (grey vertical areas with labels)
- **Hover over any data point** to see a natural-language summary of what drove sentiment that month and the top 3 contributing topics with their percentage share. This hover detail is available on all sentiment line charts (net sentiment, % negative, % positive).
- The trend line (dashed) smooths out monthly noise — look at this for the overall direction
- When "By subreddit" is selected, r/NationalServiceSG (amber) is typically more negative than r/singapore (blue) or r/askSingapore (green)
- Low-volume months (< 30 docs) are hidden by default to prevent noisy spikes. Toggle "Show low-volume months" in the sidebar to include them.

---

## Page 3 — Topic Analysis

The most feature-rich page. Explore all 359 topics across the full taxonomy hierarchy.

### Controls

- **Min upvotes per document**: Slider to filter out low-engagement content. Set to 5+ to focus on community-validated content.
- **Subreddit filter**: Filter topics by subreddit

### Sub-tabs

#### Landscape (treemap)

An interactive treemap showing the full topic hierarchy:
- **Size** of each tile = document volume (more documents → larger tile)
- **Colour** = net sentiment (red → negative, green → positive, grey → neutral)
- **Click** a tile to zoom into that level. Click the breadcrumb at the top to zoom back out.
- **Hover** over any tile to see exact statistics: document count, sentiment breakdown, stance label

The hierarchy levels are:
- All NS Discourse (root)
  - 17 macro topics (e.g., "BMT & Training", "NS Policy & Society")
    - 52 sub topics
      - 112 cluster topics
        - 359 leaf topics

#### Drill Down

Select a specific macro topic from the dropdown. Shows:
- Nested treemap of sub/cluster/leaf topics within that macro
- Topic-specific sentiment trend chart
- Top keywords per sub-topic
- Document count breakdown

#### Over Time

Stacked area chart showing how topic volume changes over time. Identify which topics are growing or shrinking in discussion volume.

#### Topic Rankings

Table ranking all macro topics by various metrics:
- Net sentiment, % negative, % positive
- Document count, % of total corpus
- Stance (net supportive/critical)
- Sortable by clicking column headers

#### Volume by Topic

Bar chart comparing document volume across all 17 macro topics. See at a glance which topics dominate NS discourse.

#### Sentiment by Topic

Diverging bar chart showing net sentiment per macro topic. Topics to the left are net negative; topics to the right are net positive.

#### Stance by Topic

Similar to Sentiment by Topic, but showing net stance (supportive − critical) per macro topic. Identifies which topics draw the most criticism vs support.

---

## Page 4 — Divergence

Analyses how comment sections differ from original posts in sentiment.

### Headline KPIs

| KPI | What it means |
|-----|---------------|
| **Threaded Posts** | Number of submissions with at least 1 comment chunk analysed |
| **Top Reach + Divergence** | The macro topic with the highest discourse intensity (upvotes × sentiment polarisation) |
| **Most Opinionated** | Topic with the highest proportion of opinionated chunks (non-neutral) |
| **Tone Shift Rate** | Percentage of threads where comment sentiment differs from post sentiment |

### Sub-tabs

#### Tone Shift Matrix

A 3×3 heatmap showing how often comments match or diverge from the original post's sentiment:
- **Diagonal** (e.g., Negative → Negative): community agrees with the post's tone
- **Off-diagonal** (e.g., Positive → Negative): community shifts tone
- Accompanied by "Key Patterns" callout box highlighting the most notable patterns

#### Community Battlegrounds

Table of the most divergent threads — posts where the comment section's sentiment differs the most from the original post. Each row shows:
- Post title, subreddit, date
- Post sentiment vs comment sentiment
- Divergence score
- Number of comments

#### Opinion Density

Chart showing which topics generate the most opinionated (non-neutral) discussion. Topics with high opinion density are those where people feel strongly, regardless of direction.

---

## Page 5 — Commitment

Tracks commitment to NS over time using the 4-stage cascade classifier.

### Model Details (expandable)

Click to expand a panel explaining the cascade model architecture, the two-axis framework (buyin vs stance), and evaluation metrics.

### Headline KPIs

| KPI | What it means |
|-----|---------------|
| **Total Chunks** | Number of chunks classified by the cascade model |
| **Uncommitted / Committed** | Percentage of buyin-relevant chunks classified as uncommitted vs committed |
| **Critical / Supportive** | Percentage of stance-relevant chunks classified as critical vs supportive |

### Sub-tabs

#### Commitment Decline

The main analytical view. Shows:

- **Thesis statement**: A data-driven narrative summarising the commitment trend (e.g., "Net commitment has been negative every year on record — uncommitted sentiment consistently outweighs committed")
- **Three KPI cards**:
  - **Peak Decline**: The largest drop in net commitment (e.g., 2019→2022, −0.019)
  - **Recovery Since Trough**: The recovery from the lowest point (e.g., 2022→2025, +0.026)
  - **Net vs Baseline**: Current commitment compared to the earliest year in the filter range. Green = improvement, red = decline.
- **Monthly Net Commitment chart**: Line chart of `net_buyin = (% committed − % uncommitted)` per month, with 6-month rolling average and shaded "below zero" region

#### Signal Detail

Breakdown of all commitment signals:
- Monthly time series of each individual metric (% committed, % uncommitted, % supportive, % critical)
- Helps identify whether changes in net commitment are driven by more people becoming uncommitted or fewer people expressing commitment

#### By Topic

Commitment breakdown per macro topic. Shows which topics are associated with the strongest commitment signals and which topics people are most critical of.

---

## Page 6 — Population

Demographics and posting patterns of the user base.

### Headline KPIs

| KPI | What it means |
|-----|---------------|
| **Unique Authors** | Total distinct Reddit usernames in the corpus |
| **Unique Posts** | Total submissions (excluding comments) |
| **Unique Comments** | Total comments |
| **Returning Authors** | Percentage of authors who posted more than once |
| **Single-Sub Authors** | Percentage of authors who only posted in one subreddit |

### Visualisations

- **Top 20 User Flairs**: Bar chart of the most common Reddit user flairs. Flairs are self-selected identity labels (e.g., "Intelligence", "Military Police", "Pre-Enlistee"). Provides insight into the self-identified demographics of the posting population.

- **Posting Activity by Year**: Dual-axis chart showing total posts (bars) and unique authors (line) per year. Shows growth in NS discourse over time.

- **Subreddit Breakdown**: How posts are distributed across the 3 subreddits.

- **Top Authors**: Table of the most prolific contributors, with their post counts and primary subreddit.

- **Author Engagement Distribution**: Histogram of posts-per-author, showing the long-tail distribution (most authors post once; a small group posts frequently).

---

## Page 7 — Sentinel Bot

An AI-powered chatbot that answers questions about NS sentiment using RAG (Retrieval-Augmented Generation).

### Requirements

- The 3 large files must be present: `comments_chunks.parquet`, `submissions_chunks.parquet`, `chunk_faiss.index`
- A `.env` file with `ANTHROPIC_API_KEY=sk-ant-...`
- On first visit, the knowledge base takes ~15 seconds to load into memory

### How to use

Type a question in the chat input at the bottom. The bot can answer:

**Quantitative questions**:
- "What percentage of posts about BMT are negative?"
- "How has sentiment about IPPT changed over time?"
- "Which subreddit is most negative about NS policy?"

**Qualitative questions**:
- "What do people complain about most regarding BMT?"
- "Why did sentiment about NS pay improve in 2024?"
- "What are the main arguments against conscription?"

**Comparative questions**:
- "How does sentiment about NS differ between r/singapore and r/NationalServiceSG?"
- "Which topics saw the biggest sentiment change after COVID?"

### How it works

1. Your question is classified as quantitative or qualitative
2. For quantitative: stats are looked up from a pre-computed fact table
3. For qualitative: your question is embedded and the 10 most relevant text chunks are retrieved via FAISS
4. Topic digests, temporal narratives, and relevant events are assembled as context
5. Groq Llama 3.3 70B (or fallback LLM) synthesises a data-grounded answer from all the context
6. The answer appears in the chat, with citations to specific data points

### Tips for best results

- Be specific about the topic, time period, or subreddit you're interested in
- Quantitative questions get faster, more precise answers
- The bot has access to all 737K chunks, so it can surface specific quotes and examples
- For broad questions ("how do people feel about NS?"), the bot will draw from topic digests rather than individual chunks

---

## Keyboard shortcuts and tips

- **Sidebar collapse**: Click the "X" or use Streamlit's sidebar toggle to hide the sidebar for a wider view during presentations
- **Chart zoom**: All Plotly charts support zoom (click-drag), pan (shift-drag), and reset (double-click)
- **Chart download**: Hover over any chart and click the camera icon to download as PNG
- **URL state**: Each page has a URL parameter (e.g., `?page=Commitment`). You can bookmark specific pages.
- **Filter persistence**: Filters persist across page navigation within the same session. Changing filters requires clicking "Apply Filters".
