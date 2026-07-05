"""
NS Sentinel Dashboard — Stage 9
================================
Run: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NS Sentinel",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "new"

# ── Design tokens ─────────────────────────────────────────────────────────────
# Initialise session state for light mode before any widget is rendered
if "light_mode" not in st.session_state:
    st.session_state["light_mode"] = False

_LIGHT = st.session_state.get("light_mode", False)

if _LIGHT:
    BG     = "#F0F2F5"
    SURF   = "#FFFFFF"
    SURF2  = "#F5F5F5"
    BORDER = "#E0E0E0"
    TXT    = "#212121"
    MUTED  = "#757575"
    ACCENT = "#B71C1C"
    CYAN   = "#0277BD"
    GREEN  = "#1B5E20"
    AMBER  = "#E65100"
    PAPER  = "#FAFAFA"
    NEU    = "#90A4AE"
    FILL_POS = "rgba(27,94,32,0.12)"
    FILL_NEG = "rgba(183,28,28,0.12)"
    FILL_BAR = "rgba(0,0,0,0.08)"
else:
    BG     = "#070B0F"
    SURF   = "#0C1420"
    SURF2  = "#111D2E"
    BORDER = "#1B2D44"
    TXT    = "#C8D8E8"
    MUTED  = "#4D6A87"
    ACCENT = "#FF3820"   # Singapore red-orange
    CYAN   = "#00C4FF"
    GREEN  = "#00E896"
    AMBER  = "#FFB800"
    PAPER  = "#0C1420"
    NEU    = "#2A4A6A"
    FILL_POS = "rgba(0,232,150,0.15)"
    FILL_NEG = "rgba(255,56,32,0.15)"
    FILL_BAR = "rgba(255,255,255,0.25)"

SENT_COLORS = {"neg": ACCENT, "neu": NEU, "pos": GREEN}
SUB_COLORS  = {
    "NationalServiceSG": CYAN,
    "singapore":         AMBER,
    "askSingapore":      GREEN,
}

# ── Plotly dark template ──────────────────────────────────────────────────────
_tpl = go.layout.Template(layout=go.Layout(
    paper_bgcolor=SURF,
    plot_bgcolor=SURF,
    font=dict(family="'Barlow Semi Condensed', sans-serif", color=MUTED, size=11),
    xaxis=dict(
        gridcolor=BORDER, gridwidth=1, linecolor=BORDER, linewidth=1,
        showline=True, tickcolor=BORDER, tickfont=dict(size=10, color=MUTED),
        zeroline=False, showgrid=False,
    ),
    yaxis=dict(
        gridcolor=BORDER, gridwidth=1, linecolor=BORDER, showline=False,
        tickcolor=BORDER, tickfont=dict(size=10, color=MUTED),
        zeroline=False, showgrid=True,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
        font=dict(size=10, color=MUTED),
        orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
    ),
    margin=dict(l=0, r=0, t=36, b=0),
    colorway=[CYAN, ACCENT, GREEN, AMBER, "#B44FFF", "#FF7B54"],
))
_tpl_name = "ns_light" if _LIGHT else "ns_dark"
pio.templates[_tpl_name] = _tpl
# ponytail: pio.templates.default is process-global and races across concurrent
# Streamlit sessions; apply the template per-figure via lay()/go.Figure(layout=...) instead.

# ── Global CSS ────────────────────────────────────────────────────────────────
# Inject a blocking <style> tag via st.html to paint the background before
# Streamlit's own skeleton renders — eliminates the white flash on navigation.
_cs = "light" if _LIGHT else "dark"
st.html(f"""
<style>
/* Applied before any other stylesheet — paints background immediately on load */
html {{ background: {BG} !important; color-scheme: {_cs}; }}
body {{ background: {BG} !important; }}
* {{ box-sizing: border-box; }}
</style>
<script>
/* Set background on documentElement the instant JS runs (before first paint) */
document.documentElement.style.backgroundColor = '{BG}';
document.documentElement.style.colorScheme = '{_cs}';
document.documentElement.setAttribute('data-theme', '{_cs}');
</script>
""")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow+Semi+Condensed:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;600&display=swap');

/* ─ Base ─────────────────────────────────────────────────────────────────── */
html, body, .stApp {{
    font-family: 'Barlow Semi Condensed', sans-serif !important;
    color: {TXT} !important;
}}
.stApp {{
    background-color: {BG} !important;
    background-image:
        radial-gradient(circle, {BORDER}8C 1px, transparent 1px) !important;
    background-size: 28px 28px !important;
}}

/* ─ Chrome cleanup ────────────────────────────────────────────────────────── */
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{
    display: none !important;
    visibility: hidden !important;
}}
[data-testid="stHeader"] {{
    background: transparent !important;
    border-bottom: none !important;
    height: 0 !important;
    min-height: 0 !important;
}}

/* ─ Sidebar always visible ────────────────────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {{
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 240px !important;
    min-width: 240px !important;
    transform: none !important;
}}

/* ─ Sidebar collapsed toggle — make it obvious ────────────────────────────── */
[data-testid="stSidebarCollapsedControl"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: {ACCENT} !important;
    border-radius: 0 4px 4px 0 !important;
    width: 28px !important;
}}
[data-testid="stSidebarCollapsedControl"] button {{
    color: #FFFFFF !important;
}}

/* ─ Main content ──────────────────────────────────────────────────────────── */
.block-container {{
    padding: 2rem 2.5rem 3rem 2.5rem !important;
    max-width: 1440px !important;
    animation: fadeUp 0.4s ease-out both;
}}
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0);    }}
}}

/* ─ Sidebar shell ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {SURF} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] section[data-testid="stSidebarContent"] {{
    padding: 0 !important;
}}
[data-testid="stSidebar"] .block-container {{
    padding: 0 !important;
    animation: none !important;
}}

/* ─ Sidebar brand ─────────────────────────────────────────────────────────── */
.nav-brand {{
    padding: 1.5rem 1.25rem 1.25rem;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 1rem;
}}
.nav-brand-glyph {{
    font-size: 1.3rem;
    color: {ACCENT};
    line-height: 1;
    display: block;
    margin-bottom: 0.35rem;
}}
.nav-brand-name {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.45rem;
    letter-spacing: 0.1em;
    color: {TXT};
    line-height: 1;
    display: block;
}}
.nav-brand-sub {{
    font-size: 0.66rem;
    font-weight: 400;
    color: {MUTED};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
    display: block;
}}

/* ─ Sidebar section label ─────────────────────────────────────────────────── */
.nav-section {{
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    color: {MUTED};
    text-transform: uppercase;
    padding: 0 1.25rem 0.4rem;
    display: block;
}}

/* ─ Nav links ─────────────────────────────────────────────────────────────── */
.nav-link {{
    display: block;
    padding: 8px 1.25rem;
    color: {MUTED} !important;
    text-decoration: none !important;
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-weight: 500;
    font-size: 0.82rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    border-left: 2px solid transparent;
    margin-bottom: 1px;
}}
.nav-link:hover {{
    color: {TXT} !important;
    background: {SURF2};
    border-left-color: {ACCENT}59;
    text-decoration: none !important;
}}
.nav-link.active {{
    color: {ACCENT} !important;
    background: {SURF2};
    border-left-color: {ACCENT};
    text-decoration: none !important;
}}
.nav-divider {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 1rem 1.25rem;
}}

/* ─ Sidebar filter labels ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {{
    color: {MUTED} !important;
    font-family: 'Barlow Semi Condensed', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
}}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] span,
[data-testid="stSidebar"] [data-testid="stSelectbox"] span {{
    color: {TXT} !important;
}}

/* ─ Metrics ───────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {SURF} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 5px !important;
    padding: 1rem 1.25rem 0.9rem !important;
    position: relative;
    overflow: hidden;
}}
[data-testid="stMetric"]::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {ACCENT} 0%, {ACCENT}00 70%);
}}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div {{
    font-family: 'Barlow Semi Condensed', sans-serif !important;
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: {MUTED} !important;
}}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: {CYAN} !important;
    letter-spacing: -0.02em !important;
}}
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] div {{
    font-family: 'Barlow Semi Condensed', sans-serif !important;
    font-size: 0.75rem !important;
    color: {GREEN} !important;
}}

/* ─ Headings ──────────────────────────────────────────────────────────────── */
h1 {{
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2.8rem !important;
    letter-spacing: 0.06em !important;
    color: {TXT} !important;
    line-height: 1 !important;
    margin-bottom: 0.15rem !important;
}}
h2 {{
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.5rem !important;
    letter-spacing: 0.06em !important;
    color: {TXT} !important;
    margin-bottom: 0.15rem !important;
}}
h3, h4 {{
    font-family: 'Barlow Semi Condensed', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: {MUTED} !important;
    margin-top: 1.4rem !important;
    margin-bottom: 0.5rem !important;
}}
p {{ color: {MUTED}; font-size: 0.85rem; }}

/* ─ Divider ───────────────────────────────────────────────────────────────── */
hr {{
    border: none !important;
    border-top: 1px solid {BORDER} !important;
    margin: 1.5rem 0 !important;
}}

/* ─ Page title block ──────────────────────────────────────────────────────── */
.page-header {{
    border-left: 3px solid {ACCENT};
    padding-left: 1rem;
    margin-bottom: 1.75rem;
    animation: fadeUp 0.35s ease-out both;
}}
.page-header-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    letter-spacing: 0.07em;
    color: {TXT};
    line-height: 1;
    display: block;
}}
.page-header-sub {{
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-size: 0.72rem;
    font-weight: 400;
    color: {MUTED};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 5px;
    display: block;
}}

/* ─ Tabs ──────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] button[role="tab"] {{
    font-family: 'Barlow Semi Condensed', sans-serif !important;
    font-size: 0.77rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: {MUTED} !important;
    padding: 6px 16px !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom-color: {ACCENT} !important;
}}

/* ─ Form controls ─────────────────────────────────────────────────────────── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stMultiSelect"] > div > div {{
    background: {SURF2} !important;
    border-color: {BORDER} !important;
    color: {TXT} !important;
}}
[data-baseweb="tag"] {{
    background: {CYAN}1F !important;
    border-color: {CYAN}4D !important;
}}
[data-baseweb="tag"] span {{ color: {CYAN} !important; }}

/* ─ Slider ────────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {{
    background: {ACCENT} !important;
    border-color: {ACCENT} !important;
}}
[data-testid="stSlider"] [data-testid="stSliderTrack"] > div:nth-child(2) {{
    background: {ACCENT}80 !important;
}}

/* ─ Radio ─────────────────────────────────────────────────────────────────── */
[data-testid="stRadio"] label p {{
    color: {MUTED} !important;
    font-family: 'Barlow Semi Condensed', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}}
[data-testid="stRadio"] [aria-checked="true"] ~ div p {{
    color: {ACCENT} !important;
}}

/* ─ Toggle ────────────────────────────────────────────────────────────────── */
[data-testid="stToggle"] label p {{
    color: {MUTED} !important;
    font-family: 'Barlow Semi Condensed', sans-serif !important;
    font-size: 0.78rem !important;
}}

/* ─ Selectbox / multiselect text ──────────────────────────────────────────── */
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stMultiSelect"] div[data-baseweb="select"] span {{
    color: {TXT} !important;
}}

/* ─ Nav buttons (secondary kind = all 6 page nav buttons) ─────────────────── */
[data-testid="stSidebar"] button[kind="secondary"] {{
    background: transparent !important;
    border: none !important;
    border-left: 2px solid transparent !important;
    border-radius: 0 !important;
    color: {MUTED} !important;
    font-family: 'Barlow Semi Condensed', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    text-align: left !important;
    padding: 8px 1.25rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: color 0.1s ease, border-left-color 0.1s ease, background 0.1s ease !important;
    margin: 0 !important;
    box-shadow: none !important;
}}
[data-testid="stSidebar"] button[kind="secondary"]:hover {{
    color: {TXT} !important;
    background: {SURF2} !important;
    border-left-color: {ACCENT}59 !important;
}}
/* Active nav button — injected via data-active attribute set by JS */
[data-testid="stSidebar"] button[kind="secondary"][data-active="true"],
[data-testid="stSidebar"] button[kind="secondary"].nav-active {{
    color: {ACCENT} !important;
    background: {SURF2} !important;
    border-left-color: {ACCENT} !important;
}}

/* ─ Apply Filters button (primary kind) ───────────────────────────────────── */
[data-testid="stSidebar"] button[kind="primary"] {{
    background: {ACCENT}1A !important;
    border: 1px solid {ACCENT}59 !important;
    color: {ACCENT} !important;
    font-family: 'Barlow Semi Condensed', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    padding: 7px 12px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
    margin-top: 0.5rem !important;
}}
[data-testid="stSidebar"] button[kind="primary"]:hover {{
    background: {ACCENT}38 !important;
    border-color: {ACCENT} !important;
}}
[data-testid="stSidebar"] button[kind="primary"]:active {{
    background: {ACCENT}59 !important;
}}
.filter-pending {{
    font-size: 0.62rem;
    color: {AMBER};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    text-align: center;
    padding: 3px 0 0;
    display: block;
}}
.filter-applied {{
    font-size: 0.62rem;
    color: {GREEN};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    text-align: center;
    padding: 3px 0 0;
    display: block;
}}

/* ─ Staggered reveal helpers ──────────────────────────────────────────────── */
.reveal {{ animation: fadeUp 0.45s ease-out both; }}
.reveal-1 {{ animation-delay: 0.05s; }}
.reveal-2 {{ animation-delay: 0.12s; }}
.reveal-3 {{ animation-delay: 0.20s; }}
.reveal-4 {{ animation-delay: 0.30s; }}

/* ─ Anti-flash: keep background consistent during every Streamlit state transition ─ */
/* These selectors cover the skeleton loader, the app-view wrapper, stale iframes,
   and the transient "stale" overlay that goes white before content is ready. */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="stSkeleton"],
[data-testid="stSkeletonText"],
[data-stale="true"],
.st-emotion-cache-z5fcl4,
.st-emotion-cache-1y4p8pa,
.st-emotion-cache-13ln4jf,
section[data-testid="stSidebar"] ~ div {{
    background-color: {BG} !important;
    transition: none !important;
}}
/* Prevent the brief overlay flash when Streamlit re-runs */
[data-testid="stApp"]::before,
[data-testid="stApp"]::after {{
    background-color: {BG} !important;
}}

/* ─ Chat page: expander, sample-question buttons, chat input/messages ─ */
[data-testid="stExpander"] {{
    background-color: {SURF} !important;
    border: 1px solid {BORDER} !important;
}}
[data-testid="stExpander"] summary {{
    background-color: {SURF} !important;
    color: {TXT} !important;
}}
[data-testid="stExpander"] p {{
    color: {TXT} !important;
}}
[data-testid="stExpander"] [data-testid="stButton"] button {{
    background-color: {SURF2} !important;
    border: 1px solid {BORDER} !important;
    color: {TXT} !important;
}}
[data-testid="stChatInput"],
[data-testid="stChatInput"] div,
[data-testid="stChatInput"] textarea,
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"],
[data-testid="stBottom"] > div {{
    background-color: {SURF} !important;
    color: {TXT} !important;
}}
[data-testid="stChatInput"] {{
    border: 1px solid {BORDER} !important;
}}
[data-testid="stChatMessage"] {{
    background-color: {SURF2} !important;
    color: {TXT} !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    doc     = pd.read_parquet(DATA_DIR / "doc_sentiment.parquet")
    topic_s = pd.read_parquet(DATA_DIR / "topic_summary.parquet")
    div_raw = pd.read_parquet(DATA_DIR / "doc_divergence.parquet")
    div_v2  = pd.read_parquet(DATA_DIR / "doc_divergence_v2.parquet")
    div     = div_raw.merge(div_v2, on="post_id", how="left", suffixes=("", "_v2"))
    div["total_upvotes"] = div["total_upvotes"].fillna(0).astype(int)
    # Join leaf topic (topic_sub_sub) from doc_sentiment submissions
    _ds_leaf = pd.read_parquet(
        DATA_DIR / "doc_sentiment.parquet",
        columns=["post_id", "topic_sub_sub"],
    ).drop_duplicates("post_id")
    div = div.merge(_ds_leaf, on="post_id", how="left")
    t_ov    = pd.read_parquet(DATA_DIR / "temporal_sentiment_overall.parquet")
    t_top   = pd.read_parquet(DATA_DIR / "temporal_sentiment.parquet")
    # Compute proper upvote-weighted mean negativity from chunk_metadata
    # wtd_sent_neg = Σ((upvotes+1) × sent_neg) / Σ(upvotes+1) per subreddit × month
    # Both this and mean_sent_neg use the same continuous score so the gap is interpretable:
    # wtd > mean → high-upvote posts are more negative → community upvoting negativity
    _cm = pd.read_parquet(DATA_DIR / "chunk_metadata.parquet",
                          columns=["subreddit","year","month","sent_neg","upvotes"])
    _cm = _cm[_cm["year"] >= 2018].copy()
    _cm["year_month"] = _cm["year"].astype(str) + "-" + _cm["month"].astype(str).str.zfill(2)
    _cm["upvote_w"]   = _cm["upvotes"] + 1
    _cm["wxs"]        = _cm["upvote_w"] * _cm["sent_neg"]
    wtd_neg_ts = (
        _cm.groupby(["year_month","subreddit"])
        .agg(wxs_sum=("wxs","sum"), w_sum=("upvote_w","sum"))
        .assign(wtd_sent_neg=lambda d: d["wxs_sum"] / d["w_sum"])
        [["wtd_sent_neg"]]
        .reset_index()
    )
    t_dist  = pd.read_parquet(DATA_DIR / "temporal_topic_dist.parquet")
    _da_monthly        = pd.read_parquet(DATA_DIR / "doc_author_monthly.parquet")        # unique authors per topic_sub×month
    _da_monthly_global = pd.read_parquet(DATA_DIR / "doc_author_monthly_global.parquet") # truly unique authors per month (no double-count)
    _da_monthly_macro  = pd.read_parquet(DATA_DIR / "doc_author_monthly_macro.parquet")  # deduplicated per topic_macro×month
    _da_monthly_leaf   = pd.read_parquet(DATA_DIR / "doc_author_monthly_leaf.parquet")   # deduplicated per topic_sub_sub×month
    # Stage 5b — SingBERT dual-axis commitment (commitment κ=0.750, stance κ=0.596)
    tc      = pd.read_parquet(DATA_DIR / "temporal_commitment.parquet")
    tdi     = pd.read_parquet(DATA_DIR / "topic_discourse_intensity.parquet")
    # Submissions text — join titles for representative-posts panel in Topic Analysis
    _subs_raw = pd.read_parquet(
        DATA_DIR.parent.parent / "interim" / "submissions_raw.parquet",
        columns=["id", "title", "score", "permalink"],
    ).rename(columns={"id": "post_id"})
    doc_posts = (
        doc[doc["doc_type"] == "submission"][
            ["doc_id", "post_id", "topic_macro", "topic_sub", "topic_sub_sub",
             "sent_neg", "sent_pos", "total_upvotes_commit", "created_utc"]
        ]
        .merge(_subs_raw, on="post_id", how="left")
        .dropna(subset=["title"])
    )
    return doc, topic_s, div, t_ov, t_top, t_dist, tc, tdi, wtd_neg_ts, doc_posts, _da_monthly, _da_monthly_macro, _da_monthly_leaf

doc, topic_s, div, t_ov, t_top, t_dist, tc, tdi, wtd_neg_ts, doc_posts, _da_monthly, _da_monthly_macro, _da_monthly_leaf = load_all()


# ── Monthly sentiment drivers (event annotations + topic context) ────────────
@st.cache_data
def load_monthly_drivers():
    import json as _json
    _path = DATA_DIR / "monthly_sentiment_drivers.json"
    if _path.exists():
        with open(_path) as f:
            _raw = _json.load(f)
        return {m["month"]: m for m in _raw}
    return {}

_monthly_drivers = load_monthly_drivers()


def _driver_hover(ym: str) -> str:
    """Build hover text for a given year-month from the drivers data.
    Wraps long lines at ~80 chars to prevent tooltip overflow."""
    m = _monthly_drivers.get(ym)
    if not m:
        return ""
    summary = m.get("summary", "")
    # Wrap long summaries into ~80-char lines
    words, lines, cur = summary.split(), [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > 80:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    wrapped = "<br>".join(lines)
    drivers = m.get("top_drivers", [])[:3]
    if drivers:
        tops = " · ".join(f"{d['topic']} ({d['share_pct']}%)" for d in drivers)
        return f"{wrapped}<br>Top: {tops}"
    return wrapped


@st.cache_data
def _load_events_json():
    import json as _json
    _path = DATA_DIR / "ns_events.json"
    if _path.exists():
        with open(_path) as f:
            return _json.load(f)
    return []

_ns_events = _load_events_json()


_SKIP_CATS = {"structural_social", "recurring_event", "platform_effect"}


def _topic_event_hover(topic_macro: str, ym: str) -> str:
    """Return hover text for a topic×month: matching discrete events by topic and date range."""
    from datetime import date
    try:
        ym_date = date.fromisoformat(ym + "-01")
    except ValueError:
        return ""
    ym_end = date(ym_date.year, ym_date.month, 28)
    hits = []
    for e in _ns_events:
        if e.get("category") in _SKIP_CATS:
            continue
        if topic_macro not in e.get("topics_affected", []):
            continue
        try:
            ds = date.fromisoformat(e["date_start"])
            de = date.fromisoformat(e["date_end"])
        except (ValueError, KeyError):
            continue
        if ds <= ym_end and de >= ym_date:
            hits.append(e["title"])
    if not hits:
        return ""
    return "Events: " + " · ".join(hits)


# ── Population data (pre-aggregated, loads in <1s) ───────────────────────────
@st.cache_data
def load_population_data():
    """Returns dict of pre-aggregated population tables built by scripts/build_population_stats.py"""
    P = DATA_DIR
    return {
        "author_stats":   pd.read_parquet(P / "pop_author_stats.parquet"),
        "flair_dist":     pd.read_parquet(P / "pop_flair_dist.parquet"),       # author flairs from zst
        "monthly":        pd.read_parquet(P / "pop_monthly.parquet"),
        "yearly_authors": pd.read_parquet(P / "pop_yearly_authors.parquet"),
        "engagement":     pd.read_parquet(P / "pop_engagement.parquet"),
        "top_authors":    pd.read_parquet(P / "pop_top_authors.parquet"),
        "sub_breakdown":  pd.read_parquet(P / "pop_sub_breakdown.parquet"),
    }


# ── RAG Chatbot (lazy-loaded only on Chat page) ───────────────────────────────
@st.cache_resource(show_spinner=False)
def load_chatbot():
    """
    Load NSChatbot once per session. Cached across Streamlit reruns.
    Takes ~12s on first load (FAISS 2.26 GB + SentenceTransformer + metadata).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.rag.chatbot import NSChatbot
    return NSChatbot()


# ── Navigation ────────────────────────────────────────────────────────────────
PAGES = ["Overview", "Sentiment Trends", "Topic Analysis", "Divergence", "Commitment", "Population", "Sentinel Bot"]
_raw_page = st.query_params.get("page", "Overview")
current_page = _raw_page if _raw_page in PAGES else "Overview"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.sidebar.toggle("☀️ Light mode", key="light_mode")

    st.markdown(
        '<div class="nav-brand">'
        '  <span class="nav-brand-glyph">◈</span>'
        '  <span class="nav-brand-name">NS SENTINEL</span>'
        '  <span class="nav-brand-sub">Singapore Reddit · 2018–2025</span>'
        '</div>'
        '<span class="nav-section">Navigation</span>',
        unsafe_allow_html=True,
    )

    for _pg in PAGES:
        if st.button(_pg, key=f"_nav_{_pg}", use_container_width=True):
            st.query_params["page"] = _pg
            st.rerun()

    # Highlight the active nav button via JS (skip Apply Filters = last .stButton)
    _nav_pages_js = repr(list(PAGES))
    st.components.v1.html(f"""<script>
    (function highlight() {{
        const target = {repr(current_page)};
        const pages  = {_nav_pages_js};
        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (!sidebar) {{ setTimeout(highlight, 80); return; }}
        sidebar.querySelectorAll('.stButton button').forEach(b => {{
            const txt = b.textContent.trim();
            if (!pages.includes(txt)) return;  // skip Apply Filters & others
            if (txt === target) {{
                b.setAttribute('data-active', 'true');
                b.classList.add('nav-active');
            }} else {{
                b.removeAttribute('data-active');
                b.classList.remove('nav-active');
            }}
        }});
    }})();
    </script>""", height=0)

    st.markdown('<hr class="nav-divider">', unsafe_allow_html=True)
    st.markdown('<span class="nav-section">Filters</span>', unsafe_allow_html=True)

    # ── Applied filter state (used by charts) ──────────────────────────────
    _ALL_SUBS = ["NationalServiceSG", "singapore", "askSingapore"]
    if "af" not in st.session_state:
        st.session_state.af = {
            "subreddits": _ALL_SUBS,
            "yr": (2019, 2025),
            "show_lv": False,
        }

    # ── Staged widgets (not yet applied) ──────────────────────────────────
    _sub_staged = st.multiselect(
        "Subreddits",
        options=_ALL_SUBS,
        default=st.session_state.af["subreddits"],
        key="_sub_stage",
    )
    _yr_staged  = st.slider(
        "Year range", 2019, 2025,
        value=st.session_state.af["yr"],
        key="_yr_stage",
    )
    _lv_staged  = st.toggle(
        "Show low-volume months",
        value=st.session_state.af["show_lv"],
        key="_lv_stage",
    )

    # Detect pending changes
    _pending = (
        set(_sub_staged or _ALL_SUBS) != set(st.session_state.af["subreddits"])
        or _yr_staged != st.session_state.af["yr"]
        or _lv_staged != st.session_state.af["show_lv"]
    )

    if st.button("Apply Filters", use_container_width=True, type="primary"):
        st.session_state.af = {
            "subreddits": _sub_staged or _ALL_SUBS,
            "yr": _yr_staged,
            "show_lv": _lv_staged,
        }
        st.rerun()

    if _pending:
        st.markdown('<span class="filter-pending">◆ unapplied changes</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="filter-applied">✓ filters applied</span>', unsafe_allow_html=True)

# ── Read active filter values ─────────────────────────────────────────────────
_af       = st.session_state.get("af", {"subreddits": ["NationalServiceSG", "singapore", "askSingapore"], "yr": (2019, 2025), "show_lv": False})
subreddits = _af["subreddits"]
yr_start   = str(_af["yr"][0])
yr_end     = str(_af["yr"][1]) + "-12"
show_lv    = _af["show_lv"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def page_header(title: str, sub: str):
    st.markdown(
        f'<div class="page-header">'
        f'  <span class="page-header-title">{title}</span>'
        f'  <span class="page-header-sub">{sub}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def slabel(s: str) -> str:
    return {"neg": "Negative", "neu": "Neutral", "pos": "Positive"}.get(s, s)


def filter_temporal(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["subreddit"].isin(subreddits)].copy()
    out = out[(out["year_month"] >= yr_start) & (out["year_month"] <= yr_end)]
    if not show_lv:
        out = out[~out["low_volume"]]
    return out


def lay(fig, h=400, title=None, yt=None, xt=None, yf=None):
    # ponytail: Streamlit's frontend fills in paper/plot bgcolor and font.color
    # from its own (static, config.toml) theme whenever they aren't set directly
    # on the top-level layout — nesting them inside `template` alone isn't enough.
    fig.update_layout(
        template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF,
        font=dict(color=MUTED, family="'Barlow Semi Condensed', sans-serif", size=11),
    )
    kw: dict = dict(height=h, margin=dict(l=0, r=0, t=36 if title else 12, b=0))
    if title:
        kw["title"] = dict(
            text=title,
            font=dict(size=11, color=MUTED, family="'Barlow Semi Condensed',sans-serif"),
            x=0, pad=dict(b=8),
        )
    if yt:
        kw["yaxis_title"] = yt
    if xt:
        kw["xaxis_title"] = xt
    if yf:
        kw["yaxis_tickformat"] = yf
    fig.update_layout(**kw)
    return fig


cfg = {"displayModeBar": False}

t_ov_f   = filter_temporal(t_ov)
t_top_f  = filter_temporal(t_top)
t_dist_f = t_dist[
    t_dist["subreddit"].isin(subreddits) &
    (t_dist["year_month"] >= yr_start) &
    (t_dist["year_month"] <= yr_end)
].copy()


def net_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Doc-count-weighted pct_pos / pct_neg collapsed across subreddits → net_sent."""
    rows = []
    for ym, g in df.groupby("year_month"):
        total = g["doc_count"].sum()
        if total > 0:
            pct_pos = float((g["pct_pos"] * g["doc_count"]).sum() / total)
            pct_neg = float((g["pct_neg"] * g["doc_count"]).sum() / total)
        else:
            pct_pos = pct_neg = 0.0
        rows.append({"year_month": ym, "pct_pos": pct_pos, "pct_neg": pct_neg, "doc_count": int(total)})
    out = pd.DataFrame(rows)
    out["net_sent"] = out["pct_pos"] - out["pct_neg"]
    return out.sort_values("year_month")


def build_net_sent_fig(split: bool, height: int = 420, smooth: int = 0) -> go.Figure:
    """Net sentiment chart.
    split=False → combined area line; split=True → per-subreddit lines.
    smooth > 0 → add N-month rolling average overlay.
    """
    fig = go.Figure()

    def _smooth(series, n):
        return series.rolling(n, min_periods=max(1, n//2), center=True).mean() if n > 0 else None

    if split:
        for sub in subreddits:
            d = t_ov_f[t_ov_f["subreddit"] == sub].sort_values("year_month").copy()
            d["net_sent"] = d["pct_pos"] - d["pct_neg"]
            col = SUB_COLORS.get(sub, CYAN)
            _hovers = [_driver_hover(ym) for ym in d["year_month"]]
            fig.add_trace(go.Scatter(
                x=d["year_month"], y=d["net_sent"],
                mode="lines", name=sub,
                line=dict(color=col, width=2),
                customdata=_hovers,
                hovertemplate="%{x}<br>net: %{y:+.1%}<br>%{customdata}<extra>" + sub + "</extra>",
            ))
            if smooth > 0:
                sr = _smooth(d["net_sent"], smooth)
                fig.add_trace(go.Scatter(
                    x=d["year_month"], y=sr, mode="lines",
                    name=f"{sub} {smooth}m avg",
                    line=dict(color=col, width=2.5, dash="dot"),
                    hovertemplate="%{x}<br>" + f"{smooth}m avg: %{{y:+.1%}}<extra>{sub} trend</extra>",
                ))
    else:
        d = net_overall(t_ov_f)
        pos_y = d["net_sent"].clip(lower=0)
        neg_y = d["net_sent"].clip(upper=0)
        fig.add_trace(go.Scatter(
            x=d["year_month"], y=pos_y, mode="none", fill="tozeroy",
            fillcolor=FILL_POS, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=d["year_month"], y=neg_y, mode="none", fill="tozeroy",
            fillcolor=FILL_NEG, showlegend=False, hoverinfo="skip",
        ))
        _hovers = [_driver_hover(ym) for ym in d["year_month"]]
        fig.add_trace(go.Scatter(
            x=d["year_month"], y=d["net_sent"],
            mode="lines", name="Net sentiment",
            line=dict(color=CYAN, width=2.5),
            customdata=_hovers,
            hovertemplate="%{x}<br>net: %{y:+.1%}<br>%{customdata}<extra></extra>",
        ))
        if smooth > 0:
            sr = _smooth(d["net_sent"], smooth)
            fig.add_trace(go.Scatter(
                x=d["year_month"], y=sr, mode="lines",
                name=f"{smooth}m trend",
                line=dict(color=AMBER, width=2, dash="dot"),
                hovertemplate="%{x}<br>" + f"{smooth}m avg: %{{y:+.1%}}<extra>Trend</extra>",
            ))
    fig.add_hline(y=0, line_color=BORDER, line_width=1, line_dash="dot")
    # Event pin annotations — short horizontal labels staggered to avoid overlap
    _pin_events = []
    if _ns_events:
        for ev in _ns_events:
            if ev.get("severity") not in ("critical", "high"):
                continue
            if ev.get("category") in ("structural_social", "platform_effect", "recurring_event", "social_trend", "ongoing_debate"):
                continue
            ev_ym = ev["date_start"][:7]
            if ev_ym < yr_start or ev_ym > yr_end:
                continue
            _pin_events.append((ev_ym, ev))
        for i, (ev_ym, ev) in enumerate(_pin_events):
            fig.add_vline(x=ev_ym, line_color=BORDER, line_width=1, line_dash="dot")
            _short = ev["title"].split("—")[0].strip()
            if len(_short) > 25:
                _short = _short[:23] + "…"
            _y_frac = 0.97 - (i % 3) * 0.04
            fig.add_annotation(
                x=ev_ym, y=_y_frac, yref="paper",
                text=f"<b>{_short}</b>", font=dict(size=8, color=MUTED),
                showarrow=False, yanchor="top", xanchor="left",
            )
    lay(fig, h=height, yt="Net sentiment  (positive − negative share)", yf="+.0%")
    fig.update_layout(
        hovermode="closest",
        hoverlabel=dict(namelength=-1, font_size=11, bgcolor=SURF,
                        bordercolor=BORDER),
    )
    return fig


@st.cache_data
def build_hierarchy():
    """
    Pre-compute sentiment + SingBERT commitment stats at all 3 topic levels.
    Returns (macro_df, sub_df, leaf_df).
    """
    base = doc.dropna(subset=["topic_macro"]).copy()

    def _agg(df, groupby_cols):
        g = df.groupby(groupby_cols, observed=True)
        out = g.agg(
            doc_count        = ("doc_id",           "count"),
            pct_neg          = ("sent_label",        lambda x: (x == "neg").mean()),
            pct_neu          = ("sent_label",        lambda x: (x == "neu").mean()),
            pct_pos          = ("sent_label",        lambda x: (x == "pos").mean()),
            mean_unc         = ("pct_uncommitted",   "mean"),
            mean_com         = ("pct_committed",     "mean"),
            mean_crit        = ("pct_critical",      "mean"),
            mean_sup         = ("pct_supportive",    "mean"),
            mean_net_disp    = ("net_disposition",   "mean"),
        ).reset_index()
        out["net_sent"]        = out["pct_pos"] - out["pct_neg"]
        out["mean_commit_net"] = out["mean_com"] - out["mean_unc"]  # compat alias for downstream charts
        return out

    macro_df = _agg(base, ["topic_macro"])
    sub_df   = _agg(base.dropna(subset=["topic_sub"]),     ["topic_macro", "topic_sub"])
    leaf_df  = _agg(base.dropna(subset=["topic_sub_sub"]), ["topic_macro", "topic_sub", "topic_sub_sub"])
    return macro_df, sub_df, leaf_df


@st.cache_data
def build_treemap_df():
    """
    Build explicit node table for go.Treemap so EVERY node (root, macro, sub, leaf)
    has computed stats — eliminating NaN in hover tooltips.
    """
    h_m, h_s, h_l = build_hierarchy()
    h_l_clean = h_l.dropna(subset=["topic_macro", "topic_sub", "topic_sub_sub"]).copy()

    nodes = []

    # Root
    total = int(h_m["doc_count"].sum())
    w = h_m["doc_count"]
    nodes.append(dict(
        id="ALL", label="All NS Discourse", parent="", value=total,
        net_sent=float((h_m["net_sent"] * w).sum() / total),
        pct_neg=float((h_m["pct_neg"] * w).sum() / total),
        pct_neu=float((h_m["pct_neu"] * w).sum() / total),
        pct_pos=float((h_m["pct_pos"] * w).sum() / total),
        commit=float((h_m["mean_commit_net"] * w).sum() / total),
    ))

    # Macro
    for _, r in h_m.iterrows():
        nodes.append(dict(
            id=r["topic_macro"], label=r["topic_macro"], parent="ALL",
            value=int(r["doc_count"]), net_sent=r["net_sent"],
            pct_neg=r["pct_neg"], pct_neu=r["pct_neu"], pct_pos=r["pct_pos"],
            commit=r["mean_commit_net"],
        ))

    # Sub (unique id = macro||sub to avoid name collisions)
    for _, r in h_s.iterrows():
        nid = f"{r['topic_macro']}||{r['topic_sub']}"
        nodes.append(dict(
            id=nid, label=r["topic_sub"], parent=r["topic_macro"],
            value=int(r["doc_count"]), net_sent=r["net_sent"],
            pct_neg=r["pct_neg"], pct_neu=r["pct_neu"], pct_pos=r["pct_pos"],
            commit=r["mean_commit_net"],
        ))

    # Leaf (unique id = macro||sub||leaf)
    for _, r in h_l_clean.iterrows():
        parent_id = f"{r['topic_macro']}||{r['topic_sub']}"
        nid = f"{r['topic_macro']}||{r['topic_sub']}||{r['topic_sub_sub']}"
        nodes.append(dict(
            id=nid, label=r["topic_sub_sub"], parent=parent_id,
            value=int(r["doc_count"]), net_sent=r["net_sent"],
            pct_neg=r["pct_neg"], pct_neu=r["pct_neu"], pct_pos=r["pct_pos"],
            commit=r["mean_commit_net"],
        ))

    return pd.DataFrame(nodes)


@st.cache_data
def _topic_time_series(filter_col: str, filter_val: str, base_df=None) -> pd.DataFrame:
    """Shared monthly aggregation for macro / sub / leaf topic levels."""
    _src = base_df if base_df is not None else doc
    sub = _src[
        (_src[filter_col] == filter_val) &
        (_src["created_utc"] >= "2019-01-01")
    ].copy()
    sub["year_month"] = sub["created_utc"].dt.to_period("M").astype(str)
    m = sub.groupby("year_month").agg(
        doc_count=("doc_id", "count"),
        unique_authors=("author", "nunique"),
        pct_neg=("sent_label", lambda x: (x == "neg").mean()),
        pct_pos=("sent_label", lambda x: (x == "pos").mean()),
        pct_neu=("sent_label", lambda x: (x == "neu").mean()),
        mean_commit_net=("net_disposition", "mean"),
    ).reset_index().sort_values("year_month")
    m["net_sent"] = m["pct_pos"] - m["pct_neg"]
    return m


def macro_time_series(topic_macro_name: str, base_df=None) -> pd.DataFrame:
    return _topic_time_series("topic_macro", topic_macro_name, base_df)


def sub_time_series(topic_sub_name: str, base_df=None) -> pd.DataFrame:
    return _topic_time_series("topic_sub", topic_sub_name, base_df)


def leaf_time_series(topic_sub_sub_name: str, base_df=None) -> pd.DataFrame:
    return _topic_time_series("topic_sub_sub", topic_sub_sub_name, base_df)


def _top_posts(filter_col: str, filter_val: str, sentiment: str = "neg", n: int = 4) -> pd.DataFrame:
    """Return top-n most negative (or positive) submission titles for a topic filter."""
    mask = doc_posts[filter_col] == filter_val
    sort_col = "sent_neg" if sentiment == "neg" else "sent_pos"
    return (
        doc_posts[mask]
        .nlargest(n, sort_col)[["title", sort_col, "score", "permalink", "created_utc"]]
        .reset_index(drop=True)
    )


def _render_posts_panel(filter_col: str, filter_val: str, dominant_sentiment: str) -> None:
    """Render representative posts widget (top neg + top pos)."""
    top_neg = _top_posts(filter_col, filter_val, "neg", 3)
    top_pos = _top_posts(filter_col, filter_val, "pos", 3)
    st.markdown(
        f'<p style="color:{MUTED};font-size:0.62rem;letter-spacing:0.14em;'
        f'text-transform:uppercase;margin:0.6rem 0 0.3rem;">Representative Posts</p>',
        unsafe_allow_html=True,
    )
    if not top_neg.empty:
        st.markdown(
            f'<p style="color:{ACCENT};font-size:0.62rem;letter-spacing:0.1em;'
            f'text-transform:uppercase;margin:0.4rem 0 0.2rem;">▼ Most negative</p>',
            unsafe_allow_html=True,
        )
        for _, row in top_neg.iterrows():
            url = f"https://reddit.com{row['permalink']}" if pd.notna(row.get("permalink")) else "#"
            score_lbl = f"↑{int(row['score']):,}" if pd.notna(row.get("score")) else ""
            st.markdown(
                f'<div style="border-left:2px solid {ACCENT};padding:0.2rem 0.5rem;'
                f'margin-bottom:0.35rem;">'
                f'<a href="{url}" target="_blank" style="color:{TXT};font-size:0.74rem;'
                f'text-decoration:none;line-height:1.3;">{row["title"][:120]}</a>'
                f'<span style="color:{MUTED};font-size:0.62rem;margin-left:0.4rem;">{score_lbl}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    if not top_pos.empty:
        st.markdown(
            f'<p style="color:{GREEN};font-size:0.62rem;letter-spacing:0.1em;'
            f'text-transform:uppercase;margin:0.6rem 0 0.2rem;">▲ Most positive</p>',
            unsafe_allow_html=True,
        )
        for _, row in top_pos.iterrows():
            url = f"https://reddit.com{row['permalink']}" if pd.notna(row.get("permalink")) else "#"
            score_lbl = f"↑{int(row['score']):,}" if pd.notna(row.get("score")) else ""
            st.markdown(
                f'<div style="border-left:2px solid {GREEN};padding:0.2rem 0.5rem;'
                f'margin-bottom:0.35rem;">'
                f'<a href="{url}" target="_blank" style="color:{TXT};font-size:0.74rem;'
                f'text-decoration:none;line-height:1.3;">{row["title"][:120]}</a>'
                f'<span style="color:{MUTED};font-size:0.62rem;margin-left:0.4rem;">{score_lbl}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


h_macro, h_sub, h_leaf = build_hierarchy()
tm_df = build_treemap_df()

# Diverging net_sent colour scale (red → grey → green). Fixed regardless of
# theme — tile fills are always dark/saturated enough for white overlay text,
# unlike SURF2 which goes near-white in light mode and breaks contrast.
NET_SCALE = [
    [0.0,  ACCENT],
    [0.35, "#2A3F55"],
    [0.5,  "#5C6B7A"],
    [0.65, "#1A4A3A"],
    [1.0,  GREEN],
]


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def net_color(val: float, lo: float = -0.6, hi: float = 0.15) -> str:
    """Map a net_sent scalar to a hex colour using the diverging scale (theme-aware)."""
    t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    lo_c, mid_c, hi_c = _hex_to_rgb(ACCENT), _hex_to_rgb("#5C6B7A"), _hex_to_rgb(GREEN)
    # interpolate: below 0.5 = accent→grey, above 0.5 = grey→green
    if t < 0.5:
        frac = t / 0.5
        r = int(lo_c[0] + frac * (mid_c[0] - lo_c[0]))
        g = int(lo_c[1] + frac * (mid_c[1] - lo_c[1]))
        b = int(lo_c[2] + frac * (mid_c[2] - lo_c[2]))
    else:
        frac = (t - 0.5) / 0.5
        r = int(mid_c[0] + frac * (hi_c[0] - mid_c[0]))
        g = int(mid_c[1] + frac * (hi_c[1] - mid_c[1]))
        b = int(mid_c[2] + frac * (hi_c[2] - mid_c[2]))
    return f"#{r:02X}{g:02X}{b:02X}"


def sentiment_bar(row, width=180):
    """Inline HTML stacked sentiment bar (neg | neu | pos)."""
    _get = lambda k: float(getattr(row, k) if hasattr(row, k) else row[k])
    neg_w = int(_get("pct_neg") * width)
    neu_w = int(_get("pct_neu") * width)
    pos_w = width - neg_w - neu_w
    return (
        f'<div style="display:flex;height:8px;width:{width}px;border-radius:2px;overflow:hidden;">'
        f'  <div style="width:{neg_w}px;background:{ACCENT};"></div>'
        f'  <div style="width:{neu_w}px;background:#2A4A6A;"></div>'
        f'  <div style="width:{pos_w}px;background:{GREEN};"></div>'
        f'</div>'
    )


def commit_label(v):
    """Plain-language commitment label from raw score."""
    if v > 0.05: return "strongly committed"
    if v > 0.02: return "slightly committed"
    if v > -0.02: return "neutral"
    if v > -0.05: return "slightly critical"
    return "strongly critical"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if current_page == "Overview":
    page_header(
        "NS SENTINEL",
        "National Service Discourse · Singapore Reddit · SingBERT NLP Analysis",
    )

    # Compute polarised net sentiment (excludes neutrals)
    _ov_opinionated = doc[doc["sent_label"] != "neu"]["sent_label"]
    _ov_neg_count = (_ov_opinionated == "neg").sum()
    _ov_pos_count = (_ov_opinionated == "pos").sum()
    _ov_neg_ratio = _ov_neg_count / (_ov_neg_count + _ov_pos_count) if (_ov_neg_count + _ov_pos_count) > 0 else 0
    _ov_neg_pos_ratio = _ov_neg_count / _ov_pos_count if _ov_pos_count > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Documents", "549,671")
    with c2: st.metric("Analysed Passages", "737,274", delta="docs split into text segments")
    with c3: st.metric("Fine-grain Topics", "359")
    with c4: st.metric("Neg : Pos Ratio", f"{_ov_neg_pos_ratio:.1f} : 1", delta="neutrals excluded", delta_color="off")
    with c5: st.metric("Net Sentiment", f"{(_ov_pos_count - _ov_neg_count) / len(doc):.1%}", delta="all-time, net negative", delta_color="inverse")
    st.markdown(
        f'<p style="color:{MUTED};font-size:0.74rem;margin-top:-0.25rem;">'
        f'<b style="color:{CYAN};">Documents</b> = individual Reddit posts + comments (549K). &nbsp;'
        f'<b style="color:{CYAN};">Analysed Passages</b> = each document split into ~1–3 text segments for '
        f'the dual-axis commitment model (737K). '
        f'<b style="color:{CYAN};">Neg : Pos Ratio</b> = for every 1 positive post, there are ~2 negative posts (neutral posts excluded).</p>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Net sentiment time series ───────────────────────────────────────────
    st.markdown("#### NS sentiment over time")
    _ov_c1, _ov_c2 = st.columns([2, 2])
    with _ov_c1:
        _net_split = st.radio("View", ["Overall", "By subreddit"],
                              horizontal=True, key="ov_net_split")
    with _ov_c2:
        _ov_smooth = st.selectbox("Trend line", ["6-month", "3-month", "12-month", "None"],
                                  index=0, key="ov_smooth")
    _smooth_n = {"None": 0, "3-month": 3, "6-month": 6, "12-month": 12}[_ov_smooth]
    st.plotly_chart(
        build_net_sent_fig(split=(_net_split == "By subreddit"), height=380,
                           smooth=_smooth_n),
        use_container_width=True, config=cfg, theme=None,
    )
    st.markdown(
        f'<p style="color:{MUTED};font-size:0.76rem;margin-top:-0.5rem;">'
        f'Net = % positive docs − % negative docs per month. '
        f'Above zero = more positive than negative. '
        f'<span style="color:{AMBER};">Amber dashed</span> = smoothed trend line.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("#### Corpus sentiment distribution")
        sc = doc["sent_label"].value_counts().reset_index()
        sc.columns = ["label", "count"]
        sc["label_full"] = sc["label"].map(slabel)
        fig = px.pie(
            sc, values="count", names="label_full",
            color="label",
            color_discrete_map={slabel(k): v for k, v in SENT_COLORS.items()},
            hole=0.66,
        )
        fig.update_traces(
            textinfo="percent",
            textfont=dict(size=11, family="JetBrains Mono", color=TXT),
            marker=dict(line=dict(color=SURF, width=3)),
        )
        lay(fig, h=300)
        st.plotly_chart(fig, use_container_width=True, config=cfg, theme=None)

    with col_r:
        st.markdown("#### Volume by subreddit")
        sub_c = doc["subreddit"].value_counts().reset_index()
        sub_c.columns = ["subreddit", "count"]
        fig2 = px.bar(
            sub_c, x="count", y="subreddit", orientation="h",
            color="subreddit", color_discrete_map=SUB_COLORS,
        )
        fig2.update_traces(showlegend=False, marker_line_width=0)
        lay(fig2, h=300, xt="Documents")
        fig2.update_layout(yaxis=dict(categoryorder="total ascending", showgrid=False))
        st.plotly_chart(fig2, use_container_width=True, config=cfg, theme=None)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Most negative topics")

    ts = topic_s.sort_values("pct_neg", ascending=False).head(5).sort_values("pct_neg", ascending=True)
    fig3 = go.Figure()
    for col_key, color, lbl in [
        ("pct_pos", GREEN, "Positive"),
        ("pct_neu", NEU, "Neutral"),
        ("pct_neg", ACCENT, "Negative"),
    ]:
        fig3.add_trace(go.Bar(
            y=ts["topic_macro"], x=ts[col_key],
            name=lbl, orientation="h",
            marker_color=color, marker_line_width=0,
        ))
    fig3.update_layout(barmode="stack")
    lay(fig3, h=260, xt="Share of documents")
    fig3.update_layout(
        xaxis=dict(tickformat=".0%", range=[0, 1], showgrid=False),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig3, use_container_width=True, config=cfg, theme=None)
    st.markdown(
        f'<p style="color:{MUTED};font-size:0.74rem;">Top 5 by % negative. '
        f'Full 17-topic breakdown in Topic Analysis tab.</p>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Sentiment Trends":
    _yr = _af["yr"]
    page_header(
        "SENTIMENT TRENDS",
        f"Monthly sentiment scores by subreddit · {_yr[0]}–{_yr[1]}",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    tab0, tab1, tab2, tab3, tab4 = st.tabs(["Net Sentiment", "% Negative", "% Positive", "Full Stack", "Grievance Amplification"])

    with tab0:
        _st_c1, _st_c2 = st.columns([2, 2])
        with _st_c1:
            _net_split = st.radio("View", ["By subreddit", "Overall"],
                                  horizontal=True, key="st_net_split")
        with _st_c2:
            _st_smooth = st.selectbox("Trend line", ["6-month","3-month","12-month","None"],
                                      index=0, key="st_smooth")
        _st_smooth_n = {"None":0,"3-month":3,"6-month":6,"12-month":12}[_st_smooth]
        st.plotly_chart(
            build_net_sent_fig(split=(_net_split == "By subreddit"), height=400,
                               smooth=_st_smooth_n),
            use_container_width=True, config=cfg, theme=None,
        )
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.76rem;">'
            f'Net = % positive docs − % negative docs. '
            f'"Overall" aggregates all selected subreddits weighted by monthly doc count.</p>',
            unsafe_allow_html=True,
        )

    with tab1:
        _t1_sm = st.selectbox("Trend line", ["6-month","3-month","12-month","None"],
                              index=0, key="tab1_smooth")
        _t1_n  = {"None":0,"3-month":3,"6-month":6,"12-month":12}[_t1_sm]
        fig = go.Figure()
        for sub in subreddits:
            d = t_ov_f[t_ov_f["subreddit"] == sub].sort_values("year_month")
            col = SUB_COLORS.get(sub, CYAN)
            _hovers = [_driver_hover(ym) for ym in d["year_month"]]
            fig.add_trace(go.Scatter(
                x=d["year_month"], y=d["pct_neg"],
                mode="lines", name=sub,
                line=dict(color=col, width=2),
                customdata=_hovers,
                hovertemplate="%{x}<br>%{y:.1%}<br>%{customdata}<extra>" + sub + "</extra>",
            ))
            if _t1_n > 0:
                sr = d["pct_neg"].rolling(_t1_n, min_periods=max(1,_t1_n//2), center=True).mean()
                fig.add_trace(go.Scatter(
                    x=d["year_month"], y=sr, mode="lines",
                    name=f"{sub} {_t1_n}m avg",
                    line=dict(color=col, width=2.5, dash="dot"),
                    hovertemplate="%{x}<br>" + f"{_t1_n}m avg: %{{y:.1%}}<extra>{sub} trend</extra>",
                ))
        lay(fig, h=380, yt="% docs with negative dominant label", yf=".0%")
        st.plotly_chart(fig, use_container_width=True, config=cfg, theme=None)
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.76rem;">'
            f'r/NationalServiceSG (the dedicated NS community) stays structurally lower than the general-public subreddits. '
            f'r/askSingapore\'s negativity has converged toward r/singapore by 2024 — the broader public is increasingly aligned in negativity.</p>',
            unsafe_allow_html=True,
        )

    with tab2:
        _t2_sm = st.selectbox("Trend line", ["6-month","3-month","12-month","None"],
                              index=0, key="tab2_smooth")
        _t2_n  = {"None":0,"3-month":3,"6-month":6,"12-month":12}[_t2_sm]
        fig = go.Figure()
        for sub in subreddits:
            d = t_ov_f[t_ov_f["subreddit"] == sub].sort_values("year_month")
            # ponytail: suppress months with <10 docs — 0% artifacts from thin months
            d = d[d["doc_count"] >= 10].copy()
            col = SUB_COLORS.get(sub, CYAN)
            _hovers = [_driver_hover(ym) for ym in d["year_month"]]
            fig.add_trace(go.Scatter(
                x=d["year_month"], y=d["pct_pos"],
                mode="lines", name=sub,
                line=dict(color=col, width=2),
                customdata=_hovers,
                hovertemplate="%{x}<br>%{y:.1%}<br>%{customdata}<extra>" + sub + "</extra>",
            ))
            if _t2_n > 0:
                sr = d["pct_pos"].rolling(_t2_n, min_periods=max(1,_t2_n//2), center=True).mean()
                fig.add_trace(go.Scatter(
                    x=d["year_month"], y=sr, mode="lines",
                    name=f"{sub} {_t2_n}m avg",
                    line=dict(color=col, width=2.5, dash="dot"),
                    hovertemplate="%{x}<br>" + f"{_t2_n}m avg: %{{y:.1%}}<extra>{sub} trend</extra>",
                ))
        lay(fig, h=380, yt="% docs with positive dominant label", yf=".0%")
        st.plotly_chart(fig, use_container_width=True, config=cfg, theme=None)
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.76rem;">'
            f'Months with fewer than 10 documents are excluded to suppress low-volume artifacts. '
            f'Positive sentiment is less differentiated across communities than negative sentiment.</p>',
            unsafe_allow_html=True,
        )

    with tab3:
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.8rem;">'
            f'Left: aggregate sentiment composition (all subreddits combined). '
            f'Right: per-subreddit breakdown side-by-side. '
            f'Stacked bands sum to 100% each month.</p>',
            unsafe_allow_html=True,
        )

        # Build aggregate data
        agg_rows = []
        for ym, g in t_ov_f.groupby("year_month"):
            total = g["doc_count"].sum()
            if total > 0:
                agg_rows.append({
                    "year_month": ym,
                    "pct_pos": float((g["pct_pos"] * g["doc_count"]).sum() / total),
                    "pct_neu": float((g["pct_neu"] * g["doc_count"]).sum() / total),
                    "pct_neg": float((g["pct_neg"] * g["doc_count"]).sum() / total),
                })
        agg = pd.DataFrame(agg_rows).sort_values("year_month")

        n_sub = len(subreddits)
        # Left panel = aggregate, Right panels = one per subreddit
        panels = [("Overall", agg)] + [
            (sub, t_ov_f[t_ov_f["subreddit"] == sub].sort_values("year_month"))
            for sub in subreddits
        ]
        ncols = 1 + n_sub
        col_widths = [2] + [1] * n_sub
        fs_cols = st.columns(col_widths, gap="small")

        for idx, (panel_label, panel_df) in enumerate(panels):
            with fs_cols[idx]:
                st.markdown(
                    f'<p style="color:{TXT};font-size:0.72rem;font-weight:700;'
                    f'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">'
                    f'{panel_label}</p>',
                    unsafe_allow_html=True,
                )
                fig_p = go.Figure()
                for s_key, color, lbl in [
                    ("pct_neg", ACCENT, "Negative"),
                    ("pct_neu", NEU, "Neutral"),
                    ("pct_pos", GREEN, "Positive"),
                ]:
                    fig_p.add_trace(go.Scatter(
                        x=panel_df["year_month"], y=panel_df[s_key],
                        name=lbl, stackgroup="one", mode="lines",
                        line=dict(width=0), fillcolor=color,
                        showlegend=(idx == 0),
                        hovertemplate=f"%{{x}}: %{{y:.1%}}<extra>{lbl}</extra>",
                    ))
                h_p = 320 if idx == 0 else 320
                fig_p.update_layout(
                    height=h_p, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
                    margin=dict(l=0, r=0, t=8, b=40),
                    yaxis=dict(tickformat=".0%", showgrid=True, gridcolor=BORDER,
                               tickfont=dict(size=9, color=MUTED), range=[0, 1]),
                    xaxis=dict(showgrid=False, tickfont=dict(size=8, color=MUTED),
                               tickangle=-45),
                    legend=dict(orientation="h", y=-0.18, x=0,
                                font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_p, use_container_width=True, config=cfg, theme=None)

    with tab4:
        _mns_ctrl_c1, _mns_ctrl_c2 = st.columns([3, 1])
        with _mns_ctrl_c1:
            st.markdown("#### Grievance amplification — do communities upvote negativity?")
        with _mns_ctrl_c2:
            _mns_sm = st.selectbox("Trend line", ["None","3-month","6-month","12-month"],
                                   index=0, key="mns_smooth")
        _mns_n = {"None":0,"3-month":3,"6-month":6,"12-month":12}[_mns_sm]
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.76rem;">'
            f'<span style="color:{CYAN};">━━</span> Simple mean — all posts count equally. &nbsp;·&nbsp; '
            f'<span style="color:{AMBER};">━━</span> Upvote-weighted mean — posts with more upvotes count more. '
            f'Both use the same continuous negativity score (0–1). '
            f'<b style="color:{AMBER};">Amber above cyan</b> = the community is upvoting negative posts more than average — shared grievance. '
            f'<b style="color:{CYAN};">Cyan above amber</b> = high-upvote posts are actually less negative than average.</p>',
            unsafe_allow_html=True,
        )
        _yr_range = st.session_state.af.get("yr", (2019, 2025))
        _wtd_f = wtd_neg_ts[
            (wtd_neg_ts["subreddit"].isin(subreddits)) &
            (wtd_neg_ts["year_month"] >= f"{_yr_range[0]}-01") &
            (wtd_neg_ts["year_month"] <= f"{_yr_range[1]}-12")
        ]
        # Compute shared y-axis range across all subreddits
        _all_y_vals = []
        _panel_data = []
        for sub in subreddits:
            d  = t_ov_f[t_ov_f["subreddit"] == sub].sort_values("year_month")
            dw = _wtd_f[_wtd_f["subreddit"] == sub].sort_values("year_month")
            shared_ym = set(d["year_month"]) & set(dw["year_month"])
            d  = d[d["year_month"].isin(shared_ym)]
            dw = dw[dw["year_month"].isin(shared_ym)]
            _all_y_vals.extend(d["mean_sent_neg"].tolist())
            _all_y_vals.extend(dw["wtd_sent_neg"].tolist())
            _panel_data.append((sub, d, dw))
        _y_min = min(_all_y_vals, default=0) - 0.02
        _y_max = max(_all_y_vals, default=1) + 0.02

        ms_cols = st.columns(len(subreddits), gap="medium")
        for col, (sub, d, dw) in zip(ms_cols, _panel_data):
            with col:
                st.markdown(
                    f'<p style="color:{SUB_COLORS.get(sub, TXT)};font-size:0.72rem;'
                    f'font-weight:700;letter-spacing:0.1em;text-transform:uppercase;'
                    f'margin-bottom:2px;">{sub}</p>',
                    unsafe_allow_html=True,
                )
                _avg_gap = (dw["wtd_sent_neg"].values - d["mean_sent_neg"].values).mean() if len(d) and len(dw) else 0
                _n_months = len(d)
                _gap_label = (
                    f'Avg gap: <b style="color:{ACCENT};">+{_avg_gap:.3f}</b> — high-upvote posts are more negative ({_n_months} months)'
                    if _avg_gap > 0.005 else
                    f'Avg gap: <b style="color:{GREEN};">{_avg_gap:+.3f}</b> — high-upvote posts are less negative ({_n_months} months)'
                    if _avg_gap < -0.005 else
                    f'Avg gap: <b style="color:{MUTED};">{_avg_gap:+.3f}</b> — no clear upvote bias ({_n_months} months)'
                )
                st.markdown(f'<p style="color:{MUTED};font-size:0.69rem;margin-bottom:2px;">{_gap_label}</p>',
                            unsafe_allow_html=True)
                fig_ms = go.Figure()
                fig_ms.add_trace(go.Scatter(
                    x=list(d["year_month"]) + list(d["year_month"])[::-1],
                    y=list(d["mean_sent_neg"]) + list(dw["wtd_sent_neg"])[::-1],
                    fill="toself", fillcolor="rgba(255,184,0,0.08)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False, hoverinfo="skip",
                ))
                _hovers = [_driver_hover(ym) for ym in d["year_month"]]
                fig_ms.add_trace(go.Scatter(
                    x=d["year_month"], y=d["mean_sent_neg"],
                    mode="lines", name="Unweighted mean",
                    line=dict(color=CYAN, width=2),
                    customdata=_hovers,
                    hovertemplate="%{x}<br>unweighted: %{y:.3f}<br>%{customdata}<extra></extra>",
                ))
                fig_ms.add_trace(go.Scatter(
                    x=dw["year_month"], y=dw["wtd_sent_neg"],
                    mode="lines", name="Upvote-weighted mean",
                    line=dict(color=AMBER, width=1.5),
                    hovertemplate="%{x}<br>upvote-weighted: %{y:.3f}<extra>Upvote-weighted</extra>",
                ))
                if _mns_n > 0 and not d.empty:
                    _r = d["mean_sent_neg"].rolling(_mns_n, min_periods=max(1,_mns_n//2), center=True).mean()
                    fig_ms.add_trace(go.Scatter(
                        x=d["year_month"], y=_r, mode="lines",
                        name=f"{_mns_n}m avg", showlegend=False,
                        line=dict(color=CYAN, width=2, dash="dot"),
                        hovertemplate=f"%{{x}}<br>{_mns_n}m avg: %{{y:.3f}}<extra></extra>",
                    ))
                fig_ms.update_layout(
                    height=280, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
                    margin=dict(l=0, r=0, t=8, b=0),
                    yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED),
                               range=[_y_min, _y_max]),
                    xaxis=dict(showgrid=False, tickfont=dict(size=8, color=MUTED), tickangle=-45),
                    legend=dict(orientation="h", y=1.12, x=0,
                                font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_ms, use_container_width=True, config=cfg, theme=None)
        st.markdown(
            f'<p style="color:{AMBER};font-size:0.82rem;font-weight:600;margin-top:8px;">'
            f'⚡ Despite lower overall negativity (% Negative tab), r/NationalServiceSG shows the strongest '
            f'grievance-amplification effect — NS servicemen disproportionately upvote negative posts as '
            f'validation ("this happened to me too"), even though fewer posts are negative overall.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.72rem;">'
            f'All three panels share the same y-axis for direct visual comparison. '
            f'Month count shown per subreddit — smaller samples may produce noisier gaps.</p>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TOPIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Topic Analysis":
    page_header(
        "TOPIC ANALYSIS",
        "17 macro · 52 sub · 112 leaf topic clusters · sentiment + commitment at every level",
    )

    # ── Upvote filter ─────────────────────────────────────────────────────────
    _uv_col, _uv_ctx = st.columns([2, 5])
    with _uv_col:
        _min_uv = st.slider(
            "Min upvotes per document",
            min_value=0, max_value=15, value=0, step=1,
            key="ta_min_upvotes",
        )
    # ── Subreddit filter ────────────────────────────────────────────────────
    _ALL_SUBS_TT = ["All", "NationalServiceSG", "singapore", "askSingapore"]
    with _uv_ctx:
        _tt_sub = st.selectbox("Subreddit filter", _ALL_SUBS_TT, index=0, key="tt_sub")

    # Apply upvote + subreddit filter to doc for this page's derived computations
    doc_ta = doc if _min_uv == 0 else doc[doc["total_upvotes_commit"] >= _min_uv].copy()
    if _tt_sub != "All":
        doc_ta = doc_ta[doc_ta["subreddit"] == _tt_sub].copy()
    _uv_count = len(doc_ta)
    _uv_pct = _uv_count / len(doc)
    _filter_parts = []
    if _min_uv > 0:
        _filter_parts.append(f"{'15+' if _min_uv == 15 else f'≥{_min_uv}'} upvotes")
    if _tt_sub != "All":
        _filter_parts.append(f"r/{_tt_sub}")
    _filter_desc = " · ".join(_filter_parts) if _filter_parts else "all documents"
    st.markdown(
        f'<p style="color:{MUTED};font-size:0.76rem;">'
        f'Showing <span style="color:{CYAN};">{_uv_count:,}</span> docs '
        f'({_uv_pct:.1%} of corpus) — {_filter_desc}.</p>',
        unsafe_allow_html=True,
    )

    @st.cache_data
    def build_hierarchy_filtered(min_uv: int, sub_filter: str = "All"):
        _base = doc_ta.dropna(subset=["topic_macro"]).copy()
        def _agg(df, groupby_cols):
            g = df.groupby(groupby_cols, observed=True)
            out = g.agg(
                doc_count     = ("doc_id",           "count"),
                pct_neg       = ("sent_label",        lambda x: (x == "neg").mean()),
                pct_neu       = ("sent_label",        lambda x: (x == "neu").mean()),
                pct_pos       = ("sent_label",        lambda x: (x == "pos").mean()),
                mean_unc      = ("pct_uncommitted",   "mean"),
                mean_com      = ("pct_committed",     "mean"),
                mean_crit     = ("pct_critical",      "mean"),
                mean_sup      = ("pct_supportive",    "mean"),
                mean_net_disp = ("net_disposition",   "mean"),
            ).reset_index()
            out["net_sent"]        = out["pct_pos"] - out["pct_neg"]
            out["mean_commit_net"] = out["mean_com"] - out["mean_unc"]
            return out
        m = _agg(_base, ["topic_macro"])
        s = _agg(_base.dropna(subset=["topic_sub"]),     ["topic_macro", "topic_sub"])
        l = _agg(_base.dropna(subset=["topic_sub_sub"]), ["topic_macro", "topic_sub", "topic_sub_sub"])
        return m, s, l

    h_macro_ta, h_sub_ta, h_leaf_ta = build_hierarchy_filtered(_min_uv, _tt_sub)

    @st.cache_data
    def build_treemap_filtered(min_uv: int, sub_filter: str = "All"):
        h_m, h_s, h_l = build_hierarchy_filtered(min_uv, sub_filter)
        h_l_clean = h_l.dropna(subset=["topic_macro","topic_sub","topic_sub_sub"]).copy()
        nodes = []
        total = int(h_m["doc_count"].sum())
        w = h_m["doc_count"]
        nodes.append(dict(
            id="ALL", label="All NS Discourse", parent="", value=total,
            net_sent=float((h_m["net_sent"]*w).sum()/total),
            pct_neg=float((h_m["pct_neg"]*w).sum()/total),
            pct_neu=float((h_m["pct_neu"]*w).sum()/total),
            pct_pos=float((h_m["pct_pos"]*w).sum()/total),
            commit=float((h_m["mean_commit_net"]*w).sum()/total),
        ))
        for _, r in h_m.iterrows():
            nodes.append(dict(
                id=r["topic_macro"], label=r["topic_macro"], parent="ALL",
                value=int(r["doc_count"]), net_sent=r["net_sent"],
                pct_neg=r["pct_neg"], pct_neu=r["pct_neu"], pct_pos=r["pct_pos"],
                commit=r["mean_commit_net"],
            ))
        for _, r in h_s.iterrows():
            nid = f"{r['topic_macro']}||{r['topic_sub']}"
            nodes.append(dict(
                id=nid, label=r["topic_sub"], parent=r["topic_macro"],
                value=int(r["doc_count"]), net_sent=r["net_sent"],
                pct_neg=r["pct_neg"], pct_neu=r["pct_neu"], pct_pos=r["pct_pos"],
                commit=r["mean_commit_net"],
            ))
        for _, r in h_l_clean.iterrows():
            parent_id = f"{r['topic_macro']}||{r['topic_sub']}"
            nid = f"{r['topic_macro']}||{r['topic_sub']}||{r['topic_sub_sub']}"
            nodes.append(dict(
                id=nid, label=r["topic_sub_sub"], parent=parent_id,
                value=int(r["doc_count"]), net_sent=r["net_sent"],
                pct_neg=r["pct_neg"], pct_neu=r["pct_neu"], pct_pos=r["pct_pos"],
                commit=r["mean_commit_net"],
            ))
        return pd.DataFrame(nodes)

    tm_df_ta = build_treemap_filtered(_min_uv, _tt_sub)

    # ── Build trend datasets (from doc_ta so upvote + subreddit filters apply) ─
    _ta_trend = doc_ta[doc_ta["created_utc"] >= "2019-01-01"].dropna(subset=["topic_macro"]).copy()
    _ta_trend["year_month"] = _ta_trend["created_utc"].dt.to_period("M").astype(str)
    _tt_dist = _ta_trend.groupby(["year_month", "topic_macro"]).agg(
        doc_count=("doc_id", "count"),
        unique_authors=("author", "nunique"),
    ).reset_index()
    _tt_sent = _ta_trend.groupby(["year_month", "topic_macro"]).agg(
        doc_count=("doc_id", "count"),
        pct_pos=("sent_label", lambda x: (x == "pos").mean()),
        pct_neg=("sent_label", lambda x: (x == "neg").mean()),
    ).reset_index()
    _tt_sent["net_sentiment"] = _tt_sent["pct_pos"] - _tt_sent["pct_neg"]

    _tt_dist["quarter"] = pd.PeriodIndex(_tt_dist["year_month"], freq="M").asfreq("Q").astype(str)
    _tt_qtr = _tt_dist.groupby(["quarter", "topic_macro"]).agg(doc_count=("doc_count","sum"), unique_authors=("unique_authors","sum")).reset_index()
    _tt_qtr["rank"] = _tt_qtr.groupby("quarter")["doc_count"].rank(method="min", ascending=False).astype(int)

    _topics = sorted(_tt_dist["topic_macro"].dropna().unique())
    _PALETTE = [
        "#00C4FF","#FF3820","#00E896","#FFB800","#C77DFF","#FF6B6B","#4ECDC4","#FFE66D",
        "#A8DADC","#F4A261","#2EC4B6","#E71D36","#FF9F1C","#CBFF8C","#8338EC","#FB5607",
        "#06D6A0","#EF476F","#118AB2","#FBD87F",
    ]
    _topic_color = {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(_topics)}

    tab_tm, tab_dd, tab_ot, tt_bump, tt_bar, tt_sent_tab, tt_drill = st.tabs([
        "Landscape", "Drill Down", "Over Time",
        "Topic Rankings", "Volume by Topic", "Sentiment by Topic", "Topic Drill-down",
    ])

    # ── Tab: Landscape (treemap) ───────────────────────────────────────────────
    with tab_tm:
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.8rem;">Size = document volume · '
            f'Colour = net sentiment (red → negative, green → positive). '
            f'Click a tile to zoom in · hover for exact stats · '
            f'click a leaf cluster to see its longitudinal sentiment trend below.</p>',
            unsafe_allow_html=True,
        )

        # Build go.Treemap with explicit nodes — every node has computed stats, no NaN
        # ── Enrich tm_df_ta with derived display columns ───────────────────────
        _tm = tm_df_ta.copy()
        # Unicode sentiment bar: █ neg · ░ neu · ▓ pos (24 chars)
        def _bar(neg, neu, pos, n=24):
            nn = max(0, round(neg * n))
            np_ = max(0, round(pos * n))
            nu = max(0, n - nn - np_)
            return "█" * nn + "░" * nu + "▓" * np_
        _tm["bar_str"] = [_bar(r.pct_neg, r.pct_neu, r.pct_pos) for _, r in _tm.iterrows()]
        # Rank all nodes by net_sent (1 = most negative across the whole tree)
        _tm["rank"] = _tm["net_sent"].rank(method="min", ascending=True).astype(int)
        _n_total = len(_tm)
        # Build mixed-type customdata as list of lists
        # idx: 0=pct_neg 1=pct_pos 2=commit 3=value 4=pct_neu 5=net_sent 6=bar_str 7=rank 8=n_total
        def _commit_label(v):
            if v > 0.05: return f"{v:+.3f} (net committed)"
            if v < -0.05: return f"{v:+.3f} (net uncommitted)"
            return f"{v:+.3f} (neutral)"
        _cd = [
            [r.pct_neg, r.pct_pos, _commit_label(r.commit), r.value, r.pct_neu, r.net_sent,
             r.bar_str, int(r["rank"]), _n_total]
            for _, r in _tm.iterrows()
        ]

        fig_tm = go.Figure(go.Treemap(
            ids=_tm["id"],
            labels=_tm["label"],
            parents=_tm["parent"],
            values=_tm["value"],
            branchvalues="total",
            customdata=_cd,
            marker=dict(
                colors=_tm["net_sent"].tolist(),
                colorscale=NET_SCALE,
                cmid=0.0,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Net sent", font=dict(size=10, color=MUTED)),
                    tickformat="+.0%", thickness=12, len=0.55,
                    tickfont=dict(size=9, color=MUTED),
                    bgcolor=SURF, bordercolor=BORDER, borderwidth=1,
                ),
                line=dict(color=BG, width=1.5),
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Docs: %{customdata[3]:,.0f}<br>"
                "Neg: %{customdata[0]:.1%} · Neu: %{customdata[4]:.1%} · Pos: %{customdata[1]:.1%}<br>"
                "Net sentiment: %{customdata[5]:+.1%}"
                "<extra></extra>"
            ),
            texttemplate=(
                "<b>%{label}</b><br>"
                "<span style='font-size:24px'>%{customdata[5]:+.1%}</span><br>"
                "<span style='font-size:9px'>"
                "NEG %{customdata[0]:.0%}  ·  NEU %{customdata[4]:.0%}  ·  POS %{customdata[1]:.0%}"
                "  ·  %{customdata[3]:,.0f} docs</span>"
            ),
            # ponytail: fixed white, not TXT — tile fills (NET_SCALE) are always
            # dark/saturated regardless of theme, so text must stay light.
            textfont=dict(family="'Barlow Semi Condensed', sans-serif", size=12, color="#FFFFFF"),
            textposition="top left",
        ))
        fig_tm.update_layout(
            template=_tpl, paper_bgcolor=SURF, font=dict(color=TXT),
            height=520,
            margin=dict(l=0, r=0, t=10, b=0),
        )

        # ── Side-by-side: treemap left, analytics panel right ─────────────────
        _corpus_avg_net = float(_tm[_tm["id"] == "ALL"]["net_sent"].iloc[0]) if "ALL" in _tm["id"].values else 0.0

        # ── Session-state selection (persists across reruns) ───────────────────
        if "tm_sel_id" not in st.session_state:
            st.session_state["tm_sel_id"] = None

        # ── Shared helper: build the immersive trend figure ───────────────────
        def _build_trend_fig(ts, net_v, vs_avg, display_name, level_label, sel_id, height=520, topic_macro=None):
            from plotly.subplots import make_subplots as _msp
            _bg = "#7f1414" if net_v < -0.3 else "#b71c1c" if net_v < -0.1 else \
                  "#5d1414" if net_v < 0 else "#1b5e20" if net_v > 0.1 else "#263238"
            ts_roll = ts["net_sent"].rolling(6, min_periods=2, center=True).mean() if not ts.empty else ts["net_sent"]
            _worst_ym = ts.loc[ts["net_sent"].idxmin(), "year_month"] if not ts.empty else "—"
            _best_ym  = ts.loc[ts["net_sent"].idxmax(), "year_month"] if not ts.empty else "—"
            _td = (ts.iloc[-12:]["net_sent"].mean() - ts.iloc[:12]["net_sent"].mean()) if len(ts) >= 12 else 0
            _tl = "▲ Sentiment improving" if _td > 0.005 else "▼ Sentiment declining" if _td < -0.005 else "→ Sentiment stable"
            _tc = GREEN if _td > 0.005 else ACCENT if _td < -0.005 else MUTED
            _rank_row = _tm.loc[_tm["id"] == sel_id, "rank"] if sel_id in _tm["id"].values else pd.Series()
            _n_pipes = sel_id.count("||")
            _rank_scope = "leaf topics" if _n_pipes == 2 else "sub-topics" if _n_pipes == 1 else "macro topics"
            # Compute rank within the same level only
            if not _rank_row.empty:
                _same_level = _tm[_tm["id"].str.count(r"\|\|") == _n_pipes]
                _level_rank = int(_same_level["net_sent"].rank(method="min", ascending=True)[_tm["id"] == sel_id].iloc[0])
                _level_n = len(_same_level)
                _rank_str = f"negativity rank: <b>#{_level_rank}</b> of {_level_n} {_rank_scope}<br>"
            else:
                _rank_str = ""
            fig = _msp(rows=2, cols=1, row_heights=[0.72, 0.28], shared_xaxes=True, vertical_spacing=0.04,
                       specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
            if not ts.empty:
                fig.add_trace(go.Scatter(x=ts["year_month"], y=ts["net_sent"].clip(upper=0),
                    mode="none", fill="tozeroy", fillcolor=FILL_NEG,
                    showlegend=False, hoverinfo="skip"), row=1, col=1)
                fig.add_trace(go.Scatter(x=ts["year_month"], y=ts["net_sent"].clip(lower=0),
                    mode="none", fill="tozeroy", fillcolor=FILL_POS,
                    showlegend=False, hoverinfo="skip"), row=1, col=1)
                _evt_hovers = [_topic_event_hover(topic_macro, ym) if topic_macro else "" for ym in ts["year_month"]]
                _evt_cd = [h if h else " " for h in _evt_hovers]
                fig.add_trace(go.Scatter(x=ts["year_month"], y=ts["net_sent"],
                    mode="lines", name="Net sentiment",
                    line=dict(color="rgba(255,255,255,0.9)", width=2.5),
                    customdata=_evt_cd,
                    hovertemplate="%{x}: %{y:+.1%}<br>%{customdata}<extra></extra>"), row=1, col=1)
                fig.add_trace(go.Scatter(x=ts["year_month"], y=ts_roll,
                    mode="lines", name="6m trend",
                    line=dict(color=AMBER, width=2, dash="dot"),
                    hovertemplate="%{x}: %{y:+.1%}<extra>6m avg</extra>"), row=1, col=1)
                fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1, line_dash="dot", row=1, col=1)
                _w_y = ts.loc[ts["net_sent"].idxmin(), "net_sent"]
                _b_y = ts.loc[ts["net_sent"].idxmax(), "net_sent"]
                fig.add_annotation(x=_worst_ym, y=_w_y, text=f"▼ {_worst_ym}",
                    font=dict(size=8, color=ACCENT), showarrow=True,
                    arrowcolor=ACCENT, arrowwidth=1, ax=0, ay=24, row=1, col=1)
                fig.add_annotation(x=_best_ym, y=_b_y, text=f"▲ {_best_ym}",
                    font=dict(size=8, color=GREEN), showarrow=True,
                    arrowcolor=GREEN, arrowwidth=1, ax=0, ay=-24, row=1, col=1)
                _vol_merged = ts[["year_month","doc_count","unique_authors"]].copy()
                _vol_merged["unique_authors"] = _vol_merged["unique_authors"].fillna(0).astype(int)
                _ua_vals = _vol_merged["unique_authors"]
                _vol_cd = list(zip(_ua_vals, _evt_cd))
                fig.add_trace(go.Bar(x=_vol_merged["year_month"], y=_vol_merged["doc_count"],
                    marker_color=FILL_BAR, marker_line_width=0,
                    customdata=_vol_cd,
                    name="Docs", hovertemplate="%{x}: %{y:,} docs · %{customdata[0]:,} authors<br>%{customdata[1]}<extra></extra>"),
                    secondary_y=False, row=2, col=1)
                fig.add_trace(go.Scatter(x=_vol_merged["year_month"], y=_ua_vals,
                    mode="lines", line=dict(color=CYAN, width=1.5),
                    name="Unique authors", hovertemplate="%{x}: %{y:,} authors<extra>authors</extra>"),
                    secondary_y=False, row=2, col=1)
            fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99,
                text=f"<span style='font-size:10px;opacity:0.6'>{level_label} · </span><b>{display_name}</b>",
                font=dict(size=14, color="white"), showarrow=False, xanchor="left", yanchor="top")
            fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.91,
                text=f"<b style='font-size:22px'>{net_v:+.1%}</b>  net sentiment",
                font=dict(size=11, color="rgba(255,255,255,0.85)"), showarrow=False, xanchor="left", yanchor="top")
            fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.99,
                text=(f"vs corpus avg: <b>{vs_avg:+.1%}</b><br>{_rank_str}"
                      f"<span style='color:{_tc};'>{_tl}</span>"),
                font=dict(size=10, color="rgba(255,255,255,0.75)"), showarrow=False,
                xanchor="right", yanchor="top", align="right")
            fig.update_layout(height=height, paper_bgcolor=_bg, plot_bgcolor="rgba(0,0,0,0.15)",
                margin=dict(l=16, r=16, t=16, b=8), showlegend=False,
                yaxis=dict(tickformat="+.0%", showgrid=True, gridcolor="rgba(255,255,255,0.08)",
                           tickfont=dict(size=9, color="rgba(255,255,255,0.6)"),
                           zerolinecolor="rgba(255,255,255,0.2)"),
                yaxis2=dict(showgrid=False, tickfont=dict(size=8, color="rgba(255,255,255,0.4)")),
                xaxis2=dict(showgrid=False, tickfont=dict(size=8, color="rgba(255,255,255,0.5)")),
                xaxis=dict(showgrid=False, showticklabels=False))
            fig.update_yaxes(secondary_y=True, row=2, col=1,
                showgrid=False, showticklabels=False,
                tickfont=dict(size=8, color=CYAN))
            return fig

        # ── Selection state ────────────────────────────────────────────────────
        _sel = st.session_state["tm_sel_id"]
        _sel_parts = _sel.split("||") if _sel else []
        _n_parts = len(_sel_parts)
        _is_leaf = _n_parts == 3

        # ── Resolve display info for whatever is selected ──────────────────────
        def _resolve_sel(parts):
            n = len(parts)
            if n == 3:
                name, ts, lbl = parts[2], leaf_time_series(parts[2], doc_ta), "▸▸▸ CLUSTER"
            elif n == 2:
                name, ts, lbl = parts[1], sub_time_series(parts[1], doc_ta), "▸▸ SUB-TOPIC"
            elif n == 1:
                name, ts, lbl = parts[0], macro_time_series(parts[0], doc_ta), "▸ TOPIC"
            else:
                return None
            sel_id = "||".join(parts)
            row = tm_df_ta[tm_df_ta["id"] == sel_id]
            net_v = float(row["net_sent"].iloc[0]) if not row.empty else (ts["net_sent"].mean() if not ts.empty else 0.0)
            return dict(name=name, ts=ts, lbl=lbl, net_v=net_v, sel_id=sel_id)

        # ── LEAF: full immersive view, treemap hidden ──────────────────────────
        if _is_leaf:
            if st.button("← Back to topic map", key="tm_back"):
                st.session_state["tm_sel_id"] = None
                st.rerun()
            _r = _resolve_sel(_sel_parts)
            fig_imm = _build_trend_fig(_r["ts"], _r["net_v"], _r["net_v"] - _corpus_avg_net,
                                       _r["name"], _r["lbl"], _r["sel_id"], height=520,
                                       topic_macro=_sel_parts[0])
            st.plotly_chart(fig_imm, use_container_width=True, config=cfg, theme=None)

        else:
            # ── MACRO / SUB / NONE: treemap always visible ─────────────────────
            tm_event = st.plotly_chart(fig_tm, use_container_width=True,
                                       config=cfg, theme=None, on_select="rerun", key="treemap_sel")
            if tm_event and tm_event.selection and tm_event.selection.get("points"):
                pt  = tm_event.selection["points"][0]
                sel = pt.get("id") or pt.get("label", "")
                # Only rerun if selection actually changed — prevents flash loop
                if sel and sel not in ("ALL", "", None) and sel != _sel:
                    st.session_state["tm_sel_id"] = sel
                    st.rerun()

            # ── Trend chart below treemap when a macro or sub is selected ──────
            if _n_parts >= 1:
                _r = _resolve_sel(_sel_parts)
                if _r:
                    st.markdown("<hr style='margin:0.5rem 0;border-color:rgba(255,255,255,0.1);'>",
                                unsafe_allow_html=True)
                    _bc1, _bc2 = st.columns([6, 1])
                    with _bc2:
                        if st.button("✕ Clear", key="tm_clear"):
                            st.session_state["tm_sel_id"] = None
                            st.rerun()
                    fig_below = _build_trend_fig(_r["ts"], _r["net_v"],
                                                 _r["net_v"] - _corpus_avg_net,
                                                 _r["name"], _r["lbl"], _r["sel_id"], height=360,
                                                 topic_macro=_sel_parts[0])
                    st.plotly_chart(fig_below, use_container_width=True, config=cfg, theme=None)

    # ── Tab: Drill Down ────────────────────────────────────────────────────────
    with tab_dd:
        # ── Level 1: macro ──────────────────────────────────────────────────
        st.markdown("#### Level 1 — Category")
        macro_sorted = h_macro_ta.sort_values("net_sent", ascending=True)
        m_colors = [net_color(v) for v in macro_sorted["net_sent"]]

        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(
            y=macro_sorted["topic_macro"], x=macro_sorted["net_sent"],
            orientation="h", marker_color=m_colors, marker_line_width=0,
            text=[f'{int(d):,}' for d in macro_sorted["doc_count"]],
            textposition="outside", textfont=dict(size=8, color=MUTED),
            customdata=list(zip(
                macro_sorted["pct_neg"], macro_sorted["pct_pos"],
                macro_sorted["doc_count"],
                [commit_label(v) for v in macro_sorted["mean_commit_net"]]
            )),
            hovertemplate=(
                "<b>%{y}</b><br>Net sent: %{x:+.1%}<br>"
                "Neg: %{customdata[0]:.1%} · Pos: %{customdata[1]:.1%}<br>"
                "Docs: %{customdata[2]:,} · %{customdata[3]}"
                "<extra></extra>"
            ),
        ))
        fig_m.add_vline(x=0, line_color=BORDER, line_width=1)
        lay(fig_m, h=480, xt="Net sentiment (positive − negative share)")
        fig_m.update_layout(yaxis=dict(showgrid=False), xaxis=dict(tickformat="+.0%"))
        st.plotly_chart(fig_m, use_container_width=True, config=cfg, theme=None)

        macro_opts = sorted(h_macro_ta["topic_macro"].unique())
        sel_macro = st.selectbox(
            "Select a category to explore its sub-topics →",
            options=macro_opts,
            key="dd_macro",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Level 2: sub ───────────────────────────────────────────────────
        sub_data = h_sub_ta[h_sub_ta["topic_macro"] == sel_macro].sort_values("net_sent", ascending=True)
        st.markdown(f"#### Level 2 — Sub-categories within *{sel_macro}*")
        c1, c2, c3 = st.columns(3, gap="large")
        _l2h = max(200, len(sub_data) * 48 + 40)

        with c1:
            s_colors = [net_color(v) for v in sub_data["net_sent"]]
            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(
                y=sub_data["topic_sub"], x=sub_data["net_sent"],
                orientation="h", marker_color=s_colors, marker_line_width=0,
                text=[f'{int(d):,}' for d in sub_data["doc_count"]],
                textposition="outside", textfont=dict(size=8, color=MUTED),
                customdata=list(zip(
                    sub_data["pct_neg"], sub_data["pct_pos"],
                    sub_data["doc_count"],
                    [commit_label(v) for v in sub_data["mean_commit_net"]]
                )),
                hovertemplate=(
                    "<b>%{y}</b><br>Net sent: %{x:+.1%}<br>"
                    "Neg: %{customdata[0]:.1%} · Pos: %{customdata[1]:.1%}<br>"
                    "Docs: %{customdata[2]:,} · %{customdata[3]}"
                    "<extra></extra>"
                ),
            ))
            fig_s.add_vline(x=0, line_color=BORDER, line_width=1)
            lay(fig_s, h=_l2h, xt="Net sentiment")
            fig_s.update_layout(yaxis=dict(showgrid=False), xaxis=dict(tickformat="+.0%"))
            st.plotly_chart(fig_s, use_container_width=True, config=cfg, theme=None)

        with c2:
            sc_colors = [ACCENT if v < 0 else GREEN for v in sub_data["mean_commit_net"]]
            _commit_cd = [commit_label(v) for v in sub_data["mean_commit_net"]]
            fig_sc = go.Figure(go.Bar(
                y=sub_data["topic_sub"], x=sub_data["mean_commit_net"],
                orientation="h", marker_color=sc_colors, marker_line_width=0,
                customdata=_commit_cd,
                hovertemplate="<b>%{y}</b><br>%{customdata}<extra></extra>",
            ))
            fig_sc.add_vline(x=0, line_color=BORDER, line_width=1)
            lay(fig_sc, h=_l2h, xt="Commitment (support − critical)")
            fig_sc.update_layout(yaxis=dict(showgrid=False))
            st.plotly_chart(fig_sc, use_container_width=True, config=cfg, theme=None)

        with c3:
            fig_comp = go.Figure()
            for s_key, color, lbl in [
                ("pct_neg", ACCENT, "Neg"),
                ("pct_neu", NEU, "Neu"),
                ("pct_pos", GREEN, "Pos"),
            ]:
                fig_comp.add_trace(go.Bar(
                    y=sub_data["topic_sub"], x=sub_data[s_key],
                    name=lbl, orientation="h",
                    marker_color=color, marker_line_width=0,
                ))
            fig_comp.update_layout(barmode="stack")
            lay(fig_comp, h=_l2h, xt="Sentiment composition")
            fig_comp.update_layout(
                xaxis=dict(tickformat=".0%"),
                yaxis=dict(showgrid=False, showticklabels=False),
            )
            st.plotly_chart(fig_comp, use_container_width=True, config=cfg, theme=None)

        sub_opts = sorted(sub_data["topic_sub"].unique())
        if sub_opts:
            sel_sub = st.selectbox(
                "Select a sub-category to see its fine-grain clusters →",
                options=sub_opts,
                key="dd_sub",
            )

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── Level 3: sub_sub overview ──────────────────────────────────
            leaf_data = h_leaf_ta[
                (h_leaf_ta["topic_macro"] == sel_macro) &
                (h_leaf_ta["topic_sub"] == sel_sub)
            ].sort_values("net_sent", ascending=True)

            st.markdown(f"#### Level 3 — Fine-grain clusters within *{sel_sub}*")
            l_colors = [net_color(v) for v in leaf_data["net_sent"]]

            # Three-column overview: net_sent bars | stacked bars | stat cards
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                fig_l = go.Figure()
                fig_l.add_trace(go.Bar(
                    y=leaf_data["topic_sub_sub"], x=leaf_data["net_sent"],
                    orientation="h", marker_color=l_colors, marker_line_width=0,
                    text=[f'{int(d):,}' for d in leaf_data["doc_count"]],
                    textposition="outside", textfont=dict(size=8, color=MUTED),
                    customdata=list(zip(
                        leaf_data["pct_neg"], leaf_data["pct_pos"],
                        leaf_data["doc_count"],
                        [commit_label(v) for v in leaf_data["mean_commit_net"]]
                    )),
                    hovertemplate=(
                        "<b>%{y}</b><br>Net: %{x:+.1%}<br>"
                        "Neg: %{customdata[0]:.1%} · Pos: %{customdata[1]:.1%}<br>"
                        "Docs: %{customdata[2]:,} · %{customdata[3]}"
                        "<extra></extra>"
                    ),
                ))
                fig_l.add_vline(x=0, line_color=BORDER, line_width=1)
                lay(fig_l, h=max(200, len(leaf_data) * 44 + 40), xt="Net sentiment")
                fig_l.update_layout(yaxis=dict(showgrid=False))
                st.plotly_chart(fig_l, use_container_width=True, config=cfg, theme=None)

            with c2:
                fig_ls = go.Figure()
                for s_key, color, lbl in [
                    ("pct_neg", ACCENT, "Neg"),
                    ("pct_neu", NEU, "Neu"),
                    ("pct_pos", GREEN, "Pos"),
                ]:
                    fig_ls.add_trace(go.Bar(
                        y=leaf_data["topic_sub_sub"], x=leaf_data[s_key],
                        name=lbl, orientation="h",
                        marker_color=color, marker_line_width=0,
                    ))
                fig_ls.update_layout(barmode="stack")
                lay(fig_ls, h=max(200, len(leaf_data) * 44 + 40), xt="Sentiment share")
                fig_ls.update_layout(
                    xaxis=dict(tickformat=".0%"),
                    yaxis=dict(showgrid=False, showticklabels=False),
                )
                st.plotly_chart(fig_ls, use_container_width=True, config=cfg, theme=None)

            with c3:
                st.markdown(
                    f'<p style="color:{MUTED};font-size:0.75rem;margin-top:0.5rem;">'
                    f'<b style="color:{TXT};">{sel_sub}</b><br><br>'
                    + "".join(
                        f'<span style="display:block;margin-bottom:10px;">'
                        f'<span style="color:{TXT};font-size:0.78rem;">{row.topic_sub_sub}</span><br>'
                        f'{sentiment_bar(row)}<br>'
                        f'<span style="color:{MUTED};font-size:0.68rem;">'
                        f'{row.doc_count:,} docs · {commit_label(row.mean_commit_net)}</span>'
                        f'</span>'
                        for row in leaf_data.sort_values("pct_neg", ascending=False).itertuples()
                    )
                    + "</p>",
                    unsafe_allow_html=True,
                )

            # ── Level 3 drill: select leaf for longitudinal trend ──────────
            if not leaf_data.empty:
                st.markdown("<hr>", unsafe_allow_html=True)
                leaf_opts = leaf_data.sort_values("pct_neg", ascending=False)["topic_sub_sub"].tolist()
                sel_leaf = st.selectbox(
                    "Select a cluster to see its sentiment over time →",
                    options=leaf_opts, key="dd_leaf",
                )

                ts_leaf = leaf_time_series(sel_leaf, doc_ta)
                if not ts_leaf.empty:
                    dl1, dl2 = st.columns([3, 1], gap="large")
                    with dl1:
                        from plotly.subplots import make_subplots as _msp_dd
                        fig_dlt = _msp_dd(rows=2, cols=1, row_heights=[0.7, 0.3],
                                          shared_xaxes=True, vertical_spacing=0.04)
                        pos_y = ts_leaf["net_sent"].clip(lower=0)
                        neg_y = ts_leaf["net_sent"].clip(upper=0)
                        fig_dlt.add_trace(go.Scatter(
                            x=ts_leaf["year_month"], y=pos_y, mode="none",
                            fill="tozeroy", fillcolor=FILL_POS,
                            showlegend=False, hoverinfo="skip",
                        ), row=1, col=1)
                        fig_dlt.add_trace(go.Scatter(
                            x=ts_leaf["year_month"], y=neg_y, mode="none",
                            fill="tozeroy", fillcolor=FILL_NEG,
                            showlegend=False, hoverinfo="skip",
                        ), row=1, col=1)
                        fig_dlt.add_trace(go.Scatter(
                            x=ts_leaf["year_month"], y=ts_leaf["net_sent"],
                            mode="lines", name="Net sentiment",
                            line=dict(color=CYAN, width=2.5),
                            hovertemplate="%{x}<br>net: %{y:+.1%}<extra></extra>",
                        ), row=1, col=1)
                        roll6 = ts_leaf["net_sent"].rolling(6, min_periods=2, center=True).mean()
                        fig_dlt.add_trace(go.Scatter(
                            x=ts_leaf["year_month"], y=roll6,
                            mode="lines", name="6-month avg",
                            line=dict(color=AMBER, width=1.5, dash="dot"),
                            hovertemplate="%{x}<br>6m avg: %{y:+.1%}<extra></extra>",
                        ), row=1, col=1)
                        fig_dlt.add_hline(y=0, line_color=BORDER, line_width=1, line_dash="dot", row=1, col=1)
                        _ua = ts_leaf["unique_authors"].fillna(0).astype(int)
                        fig_dlt.add_trace(go.Bar(
                            x=ts_leaf["year_month"], y=ts_leaf["doc_count"],
                            marker_color=FILL_BAR, marker_line_width=0,
                            name="Docs",
                            customdata=_ua,
                            hovertemplate="%{x}: %{y:,} docs · %{customdata:,} authors<extra></extra>",
                        ), row=2, col=1)
                        fig_dlt.add_trace(go.Scatter(
                            x=ts_leaf["year_month"], y=_ua,
                            mode="lines", line=dict(color=CYAN, width=1.5),
                            name="Unique authors",
                            hovertemplate="%{x}: %{y:,} authors<extra></extra>",
                        ), row=2, col=1)
                        fig_dlt.update_layout(
                            height=380, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
                            margin=dict(l=0, r=0, t=36, b=0),
                            title=dict(text=f"LONGITUDINAL TREND · {sel_leaf}",
                                       font=dict(size=11, color=MUTED), x=0),
                            showlegend=False,
                            yaxis=dict(tickformat="+.0%", showgrid=True, gridcolor=BORDER,
                                       tickfont=dict(size=9, color=MUTED)),
                            yaxis2=dict(showgrid=True, gridcolor=BORDER,
                                        tickfont=dict(size=8, color=MUTED)),
                            xaxis=dict(showgrid=False, showticklabels=False),
                            xaxis2=dict(showgrid=False, tickfont=dict(size=9, color=MUTED)),
                        )
                        st.plotly_chart(fig_dlt, use_container_width=True, config=cfg, theme=None)

                    with dl2:
                        # Key stats for this leaf
                        leaf_row = leaf_data[leaf_data["topic_sub_sub"] == sel_leaf].iloc[0]
                        net_v = float(leaf_row["net_sent"])
                        st.markdown(
                            f'<div style="background:{SURF2};border:1px solid {BORDER};'
                            f'border-radius:5px;padding:0.9rem 1rem;margin-top:0.5rem;">'
                            f'<p style="color:{MUTED};font-size:0.6rem;letter-spacing:0.14em;'
                            f'text-transform:uppercase;margin:0 0 4px;">All-time net</p>'
                            f'<p style="color:{ACCENT if net_v<0 else GREEN};'
                            f'font-family:JetBrains Mono;font-size:1.5rem;margin:0;">'
                            f'{net_v:+.1%}</p>'
                            f'<hr style="border-color:{BORDER};margin:0.5rem 0;">'
                            f'{sentiment_bar(leaf_row)}'
                            f'<p style="color:{MUTED};font-size:0.65rem;margin:3px 0 0;">'
                            f'{float(leaf_row["pct_neg"]):.1%} neg · {float(leaf_row["pct_pos"]):.1%} pos</p>'
                            f'<hr style="border-color:{BORDER};margin:0.5rem 0;">'
                            f'<p style="color:{MUTED};font-size:0.6rem;letter-spacing:0.1em;'
                            f'text-transform:uppercase;margin:0 0 2px;">Docs</p>'
                            f'<p style="color:{TXT};font-family:JetBrains Mono;font-size:1rem;margin:0;">'
                            f'{int(leaf_row["doc_count"]):,}</p>'
                            f'<hr style="border-color:{BORDER};margin:0.5rem 0;">'
                            f'<p style="color:{MUTED};font-size:0.6rem;letter-spacing:0.1em;'
                            f'text-transform:uppercase;margin:0 0 2px;">Commitment</p>'
                            f'<p style="color:{GREEN if float(leaf_row["mean_commit_net"])>0 else ACCENT};'
                            f'font-family:JetBrains Mono;font-size:0.85rem;margin:0;">'
                            f'{commit_label(float(leaf_row["mean_commit_net"]))}</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ── Tab: Over Time ─────────────────────────────────────────────────────────
    with tab_ot:
        _ot_base = doc_ta[doc_ta["created_utc"] >= "2019-01-01"].copy()
        _ot_base["year_month"] = _ot_base["created_utc"].dt.to_period("M").astype(str)
        _ot_topic_opts = sorted(_ot_base["topic_macro"].dropna().unique())
        _ot_interesting = [t for t in ["NS Policy & Society", "Vocations & Units", "BMT & Training", "NS Life & Culture"] if t in _ot_topic_opts]
        _ot_default = _ot_interesting[:4] if _ot_interesting else _ot_topic_opts[:4]

        sel_topics = st.multiselect(
            "Topics (max 5)", options=_ot_topic_opts, default=_ot_default, max_selections=5,
        )

        _ot_sel = _ot_base[_ot_base["topic_macro"].isin(sel_topics)]
        _ot_monthly = _ot_sel.groupby(["year_month", "topic_macro"]).size().reset_index(name="doc_count")
        _ot_total = _ot_sel.groupby("year_month").size().reset_index(name="total")
        _ot_monthly = _ot_monthly.merge(_ot_total, on="year_month")
        _ot_monthly["topic_pct"] = _ot_monthly["doc_count"] / _ot_monthly["total"]
        _ot_monthly = _ot_monthly.sort_values("year_month")

        _ot_smooth = st.select_slider("Smoothing", ["Monthly", "3-month", "6-month"],
                                       value="3-month", key="ot_share_smooth")
        _ot_win = {"Monthly": 1, "3-month": 3, "6-month": 6}[_ot_smooth]
        if _ot_win > 1:
            for t in sel_topics:
                mask = _ot_monthly["topic_macro"] == t
                _ot_monthly.loc[mask, "topic_pct"] = (
                    _ot_monthly.loc[mask, "topic_pct"]
                    .rolling(_ot_win, min_periods=1, center=True).mean()
                )

        fig = px.area(_ot_monthly, x="year_month", y="topic_pct",
                      color="topic_macro", groupnorm="fraction")
        fig.update_traces(line_width=0)
        lay(fig, h=440, yt="Share of monthly docs", yf=".0%")
        st.plotly_chart(fig, use_container_width=True, config=cfg, theme=None)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.65rem;letter-spacing:0.12em;'
            f'text-transform:uppercase;margin:0 0 6px;">NET SENTIMENT TREND BY TOPIC</p>',
            unsafe_allow_html=True,
        )
        _ot_def_idx = _ot_topic_opts.index("NS Policy & Society") if "NS Policy & Society" in _ot_topic_opts else 0
        cc1, cc2 = st.columns([1, 3])
        with cc1:
            topic_sel = st.selectbox("Topic", _ot_topic_opts, index=_ot_def_idx, key="topic_trend_sel")
            _ot_s_smooth = st.select_slider("Smoothing", ["Monthly", "3-month", "6-month"],
                                             value="6-month", key="ot_sent_smooth")
        _ot_s_win = {"Monthly": 1, "3-month": 3, "6-month": 6}[_ot_s_smooth]
        with cc2:
            _ot_topic = _ot_base[_ot_base["topic_macro"] == topic_sel]
            _subs_present = sorted(_ot_topic["subreddit"].dropna().unique())
            fig = go.Figure()
            for sub in _subs_present:
                d = _ot_topic[_ot_topic["subreddit"] == sub]
                d_m = d.groupby("year_month").agg(
                    doc_count=("doc_id", "count"),
                    pct_pos=("sent_label", lambda x: (x == "pos").mean()),
                    pct_neg=("sent_label", lambda x: (x == "neg").mean()),
                ).reset_index().sort_values("year_month")
                d_m["net_sent"] = d_m["pct_pos"] - d_m["pct_neg"]
                d_m.loc[d_m["doc_count"] < 5, "net_sent"] = None
                if _ot_s_win > 1:
                    d_m["net_sent"] = d_m["net_sent"].rolling(_ot_s_win, min_periods=2, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=d_m["year_month"], y=d_m["net_sent"],
                    mode="lines", name=sub, connectgaps=False,
                    line=dict(color=SUB_COLORS.get(sub, CYAN), width=2),
                    hovertemplate="%{x}<br>net: %{y:+.1%}<extra>" + sub + "</extra>",
                ))
            fig.add_hline(y=0, line_color=BORDER, line_width=1, line_dash="dot")
            lay(fig, h=300, yt="Net sentiment", yf="+.0%")
            st.plotly_chart(fig, use_container_width=True, config=cfg, theme=None)
            st.markdown(
                f'<p style="color:{MUTED};font-size:0.65rem;font-style:italic;">'
                f'Months with fewer than 5 documents per subreddit are excluded.</p>',
                unsafe_allow_html=True,
            )

    # ── Tab: Topic Rankings (bump chart) ─────────────────────────────────────
    with tt_bump:
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.78rem;line-height:1.55;">'
            f'Rank 1 = most-discussed topic that quarter. Lines crossing = topics overtaking each other.</p>',
            unsafe_allow_html=True,
        )
        _top_n = st.slider("Show top N topics", 5, len(_topics), 7, key="tt_bump_n")
        _top_topics_overall = _tt_qtr.groupby("topic_macro")["doc_count"].sum().nlargest(_top_n).index.tolist()
        _bump_df = _tt_qtr[_tt_qtr["topic_macro"].isin(_top_topics_overall)].copy()
        _quarters_sorted = sorted(_bump_df["quarter"].unique())
        _last_q = _quarters_sorted[-1] if _quarters_sorted else None

        fig_bump = go.Figure()
        for _t in _top_topics_overall:
            _td = _bump_df[_bump_df["topic_macro"] == _t].set_index("quarter")
            _ys = [_td.loc[q, "rank"] if q in _td.index else None for q in _quarters_sorted]
            _cnt = [int(_td.loc[q, "doc_count"]) if q in _td.index else None for q in _quarters_sorted]
            _ua  = [int(_td.loc[q, "unique_authors"]) if q in _td.index else None for q in _quarters_sorted]
            _col = _topic_color.get(_t, CYAN)
            _last_rank = next((r for r in reversed(_ys) if r is not None), None)
            fig_bump.add_trace(go.Scatter(
                x=_quarters_sorted, y=_ys, mode="lines+markers", name=_t,
                line=dict(color=_col, width=2), marker=dict(color=_col, size=7),
                customdata=list(zip(_cnt, [_t]*len(_quarters_sorted), _ua)),
                hovertemplate="<b>%{customdata[1]}</b><br>%{x}<br>Rank: %{y}<br>Docs: %{customdata[0]:,} · Authors: %{customdata[2]:,}<extra></extra>",
                connectgaps=False,
            ))
            if _last_q and _last_rank is not None:
                fig_bump.add_annotation(x=_last_q, y=_last_rank, text=f"  {_t}",
                    xanchor="left", showarrow=False, font=dict(size=9, color=_col))
        fig_bump.update_layout(
            height=520, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
            margin=dict(l=0, r=220, t=10, b=0), showlegend=False,
            yaxis=dict(autorange="reversed", title="Rank", tickvals=list(range(1, _top_n+1)),
                       showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED),
                       title_font=dict(size=10, color=MUTED)),
            xaxis=dict(showgrid=False, tickfont=dict(size=9, color=MUTED), tickangle=-35),
        )
        st.plotly_chart(fig_bump, use_container_width=True, config=cfg, theme=None)

    # ── Tab: Volume by Topic ──────────────────────────────────────────────────
    with tt_bar:
        _vol_topics = st.multiselect(
            "Topics (leave empty for all)", options=_topics, default=[], key="tt_vol_topics",
        )
        _vol_show = _vol_topics if _vol_topics else _topics
        _bar_gran = st.radio("Granularity", ["Quarterly", "Monthly"], horizontal=True, key="tt_bargran")
        if _bar_gran == "Quarterly":
            _bar_df, _bar_x_col = _tt_qtr[["quarter","topic_macro","doc_count","unique_authors"]].copy(), "quarter"
            _bar_xs = sorted(_bar_df["quarter"].unique())
        else:
            _bar_df, _bar_x_col = _tt_dist.copy(), "year_month"
            _bar_xs = sorted(_bar_df["year_month"].unique())
        fig_bar = go.Figure()
        for _t in _vol_show:
            _bd = _bar_df[_bar_df["topic_macro"] == _t].set_index(_bar_x_col)
            _ys = [int(_bd.loc[x, "doc_count"]) if x in _bd.index else 0 for x in _bar_xs]
            _au = [int(_bd.loc[x, "unique_authors"]) if x in _bd.index else 0 for x in _bar_xs]
            fig_bar.add_trace(go.Bar(x=_bar_xs, y=_ys, name=_t,
                marker_color=_topic_color.get(_t, CYAN), marker_line_width=0,
                customdata=_au,
                hovertemplate=f"<b>{_t}</b><br>%{{x}}<br>Docs: %{{y:,}} · Authors: %{{customdata:,}}<extra></extra>"))
        fig_bar.update_layout(
            barmode="stack",
            height=480, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED), margin=dict(l=0,r=0,t=10,b=0),
            yaxis=dict(title="Document count", showgrid=True, gridcolor=BORDER,
                       tickfont=dict(size=9, color=MUTED), title_font=dict(size=10, color=MUTED)),
            xaxis=dict(showgrid=False, tickfont=dict(size=9, color=MUTED), tickangle=-35),
            legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config=cfg, theme=None)

    # ── Tab: Sentiment by Topic ───────────────────────────────────────────────
    with tt_sent_tab:
        # Compute average net sentiment per topic for defaults & annotation
        _topic_avg_net = _tt_sent.groupby("topic_macro").apply(
            lambda g: (g["pct_pos"] - g["pct_neg"]).mean(), include_groups=False,
        ).sort_values()
        _most_neg = _topic_avg_net.head(3).index.tolist()
        _least_neg = _topic_avg_net.tail(2).index.tolist()
        _sbt_default = [t for t in _most_neg + _least_neg if t in _topics]

        st.markdown(
            f'<p style="color:{MUTED};font-size:0.78rem;">Net sentiment = % positive − % negative. '
            f'All 17 topics show net-negative sentiment. Operational topics (BMT, Admin, Vocations) are most negative.</p>',
            unsafe_allow_html=True,
        )
        _sbt_sel = st.multiselect(
            "Topics (max 5)", options=_topics, default=_sbt_default, max_selections=5, key="tt_sent_topics",
        )
        _smooth = st.slider("Smoothing (months)", 1, 12, 6, key="tt_sent_smooth")
        fig_sent = go.Figure()
        for _t in (_sbt_sel if _sbt_sel else _topics[:5]):
            _sd = _tt_sent[_tt_sent["topic_macro"] == _t].sort_values("year_month")
            if _sd.empty or _sd["doc_count"].sum() < 10:
                continue
            _sd = _sd.copy()
            _sd.loc[_sd["doc_count"] < 5, "net_sentiment"] = None
            _roll = _sd["net_sentiment"].rolling(_smooth, min_periods=2, center=True).mean()
            fig_sent.add_trace(go.Scatter(
                x=_sd["year_month"].tolist(), y=_roll.tolist(), mode="lines", name=_t,
                connectgaps=False,
                line=dict(color=_topic_color.get(_t, CYAN), width=2.2),
                hovertemplate=f"<b>{_t}</b><br>%{{x}}<br>Net: %{{y:+.1%}}<extra></extra>"))
        fig_sent.add_hline(y=0, line_color=BORDER, line_width=1, line_dash="dot")
        fig_sent.add_annotation(x=0.01, y=0, xref="paper", yref="y", text="Neutral",
            showarrow=False, font=dict(size=9, color=MUTED), xanchor="left", yshift=10)
        fig_sent.update_layout(
            height=480, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED), margin=dict(l=0,r=0,t=10,b=0),
            yaxis=dict(title="Net sentiment", showgrid=True, gridcolor=BORDER,
                       tickfont=dict(size=9, color=MUTED), title_font=dict(size=10, color=MUTED), tickformat="+.0%"),
            xaxis=dict(showgrid=False, tickfont=dict(size=9, color=MUTED), tickangle=-35),
            legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_sent, use_container_width=True, config=cfg, theme=None)

    # ── Tab: Topic Drill-down ─────────────────────────────────────────────────
    with tt_drill:
        _drill_def = _topics.index("Pay & Benefits") if "Pay & Benefits" in _topics else 0
        _drill_topic = st.selectbox("Select topic", _topics, index=_drill_def, key="tt_drill_topic")
        _drill_dist = _tt_dist[_tt_dist["topic_macro"] == _drill_topic].sort_values("year_month")
        _drill_sent = _tt_sent[_tt_sent["topic_macro"] == _drill_topic].sort_values("year_month")
        if _drill_dist.empty:
            st.info("No data for this topic with current filters.")
        else:
            dc1, dc2 = st.columns(2, gap="large")
            with dc1:
                fig_dvol = go.Figure(go.Bar(
                    x=_drill_dist["year_month"], y=_drill_dist["doc_count"],
                    marker_color=_topic_color.get(_drill_topic, CYAN), marker_line_width=0,
                    customdata=_drill_dist["unique_authors"].fillna(0).astype(int),
                    hovertemplate="%{x}<br>Docs: %{y:,} · Authors: %{customdata:,}<extra></extra>"))
                fig_dvol.update_layout(height=320, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
                    margin=dict(l=0,r=0,t=36,b=0),
                    title=dict(text="MONTHLY VOLUME", font=dict(size=11, color=MUTED), x=0),
                    yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED)),
                    xaxis=dict(showgrid=False, tickfont=dict(size=9, color=MUTED), tickangle=-35))
                st.plotly_chart(fig_dvol, use_container_width=True, config=cfg, theme=None)
            with dc2:
                if not _drill_sent.empty:
                    _ds = _drill_sent.copy()
                    _ds.loc[_ds["doc_count"] < 5, "net_sentiment"] = None
                    _ds_roll = _ds["net_sentiment"].rolling(3, min_periods=2, center=True).mean()
                    fig_dsent = go.Figure()
                    fig_dsent.add_trace(go.Scatter(x=_ds["year_month"].tolist(),
                        y=_ds["net_sentiment"].tolist(), mode="lines", name="Raw", connectgaps=False,
                        line=dict(color=MUTED, width=1), hovertemplate="%{x}<br>Net: %{y:+.1%}<extra>raw</extra>"))
                    fig_dsent.add_trace(go.Scatter(x=_ds["year_month"].tolist(),
                        y=_ds_roll.tolist(), mode="lines", name="3m avg", connectgaps=False,
                        line=dict(color=_topic_color.get(_drill_topic, CYAN), width=2),
                        hovertemplate="%{x}<br>Net: %{y:+.1%}<extra>3m avg</extra>"))
                    fig_dsent.add_hline(y=0, line_color=BORDER, line_width=1, line_dash="dot")
                    fig_dsent.update_layout(height=320, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
                        margin=dict(l=0,r=0,t=36,b=0),
                        title=dict(text="NET SENTIMENT TREND", font=dict(size=11, color=MUTED), x=0),
                        yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED), tickformat="+.0%"),
                        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=MUTED), tickangle=-35),
                        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig_dsent, use_container_width=True, config=cfg, theme=None)

            if _tt_sub == "All":
                st.markdown("<hr>", unsafe_allow_html=True)
                _sub_colors = {"NationalServiceSG": CYAN, "singapore": AMBER, "askSingapore": GREEN}
                # Compute from doc_ta for filter consistency
                _drill_ta = _ta_trend[_ta_trend["topic_macro"] == _drill_topic]
                dc3, dc4 = st.columns(2, gap="large")
                with dc3:
                    _dsub_vol = _drill_ta.groupby(["year_month", "subreddit"]).size().reset_index(name="doc_count")
                    fig_dsub = go.Figure()
                    for _sr in ["NationalServiceSG","singapore","askSingapore"]:
                        _sd2 = _dsub_vol[_dsub_vol["subreddit"]==_sr].sort_values("year_month")
                        if _sd2.empty: continue
                        fig_dsub.add_trace(go.Bar(x=_sd2["year_month"], y=_sd2["doc_count"], name=_sr,
                            marker_color=_sub_colors.get(_sr, MUTED), marker_line_width=0))
                    fig_dsub.update_layout(barmode="stack", height=300, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
                        margin=dict(l=0,r=0,t=36,b=0),
                        title=dict(text="VOLUME BY SUBREDDIT", font=dict(size=11, color=MUTED), x=0),
                        yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED)),
                        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=MUTED), tickangle=-35),
                        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig_dsub, use_container_width=True, config=cfg, theme=None)

                with dc4:
                    fig_ssub = go.Figure()
                    for _sr in ["NationalServiceSG","singapore","askSingapore"]:
                        _ss = _drill_ta[_drill_ta["subreddit"]==_sr].copy()
                        _ss_m = _ss.groupby("year_month").agg(
                            doc_count=("doc_id", "count"),
                            pct_pos=("sent_label", lambda x: (x == "pos").mean()),
                            pct_neg=("sent_label", lambda x: (x == "neg").mean()),
                        ).reset_index().sort_values("year_month")
                        _ss_m["net_sentiment"] = _ss_m["pct_pos"] - _ss_m["pct_neg"]
                        _ss_m.loc[_ss_m["doc_count"] < 5, "net_sentiment"] = None
                        if _ss_m["doc_count"].sum() < 10: continue
                        _sr_roll = _ss_m["net_sentiment"].rolling(3, min_periods=2, center=True).mean()
                        fig_ssub.add_trace(go.Scatter(x=_ss_m["year_month"].tolist(), y=_sr_roll.tolist(),
                            mode="lines", name=_sr, connectgaps=False,
                            line=dict(color=_sub_colors.get(_sr, MUTED), width=2),
                            hovertemplate=f"<b>{_sr}</b><br>%{{x}}<br>Net: %{{y:+.1%}}<extra></extra>"))
                    fig_ssub.add_hline(y=0, line_color=BORDER, line_width=1, line_dash="dot")
                    fig_ssub.update_layout(height=300, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
                        margin=dict(l=0,r=0,t=36,b=0),
                        title=dict(text="SENTIMENT BY SUBREDDIT", font=dict(size=11, color=MUTED), x=0),
                        yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED), tickformat="+.0%"),
                        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=MUTED), tickangle=-35),
                        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig_ssub, use_container_width=True, config=cfg, theme=None)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DIVERGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Divergence":
    page_header(
        "POST vs COMMENT DIVERGENCE",
        "How differently do comment sections feel compared to the original post? · 40,253 threaded posts (of 57,290 total submissions) with ≥1 comment",
    )

    # ── Upvote filter ─────────────────────────────────────────────────────────
    _div_uv_col, _div_uv_ctx = st.columns([2, 5])
    with _div_uv_col:
        _div_min_uv = st.slider(
            "Min post upvotes",
            min_value=0, max_value=15, value=0, step=1,
            key="div_min_upvotes",
        )
    with _div_uv_ctx:
        if _div_min_uv == 0:
            _div_uv_total, _div_uv_pct = len(div), 1.0
            _uv_note = "All posts included. 93% already have ≥1 upvote — try threshold 5+ to isolate community-validated content where negative sentiment is amplified."
        else:
            _div_uv_total = int((div["total_upvotes"] >= _div_min_uv).sum())
            _div_uv_pct   = _div_uv_total / len(div)
            _div_uv_label = "15+" if _div_min_uv == 15 else f"≥{_div_min_uv}"
            _uv_note = (
                f"{_div_uv_total:,} posts ({_div_uv_pct:.1%}) with {_div_uv_label} upvotes. "
                + ("At 5+ upvotes negative content rises ~7pp vs the full corpus — high-reach posts skew more critical." if _div_min_uv >= 5 else "Filter narrows to community-validated posts. Effect on sentiment is mild below 5 upvotes.")
            )
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.76rem;margin-top:1.5rem;">{_uv_note}</p>',
            unsafe_allow_html=True,
        )

    # ── Topic level selector ──────────────────────────────────────────────────
    _tlvl_col, _tmacro_col = st.columns([2, 5])
    with _tlvl_col:
        _div_topic_lvl = st.selectbox(
            "Topic granularity",
            ["Macro (17)", "Sub (52)", "Leaf (112)"],
            key="div_topic_lvl",
        )
    _div_topic_col = {
        "Macro (17)": "topic_macro",
        "Sub (52)":   "topic_sub",
        "Leaf (112)": "topic_sub_sub",
    }[_div_topic_lvl]

    with _tmacro_col:
        if _div_topic_lvl != "Macro (17)":
            _macro_opts = sorted(div["topic_macro"].dropna().unique().tolist())
            _sel_macro  = st.selectbox(
                "Filter by macro topic",
                ["All"] + _macro_opts,
                key="div_macro_filter",
            )
        else:
            _sel_macro = "All"

    # v2 signals: upvote_weighted_div, within_thread_variance, opinion_chunk_pct from doc_divergence_v2
    div_f   = div[div["subreddit"].isin(subreddits)].dropna(subset=["topic_macro"]).copy()
    if _div_min_uv > 0:
        div_f = div_f[div_f["total_upvotes"] >= _div_min_uv]
    if _sel_macro != "All":
        div_f = div_f[div_f["topic_macro"] == _sel_macro]
    div_hc  = div_f[~div_f["low_confidence"]]
    div_v2f = div_f.dropna(subset=["upvote_weighted_div", _div_topic_col]).copy()

    # Community reach × divergence = posts people saw AND that divided them
    div_v2f["reach_x_div"] = div_v2f["total_upvotes"] * div_v2f["upvote_weighted_div"]

    _n_posts = len(div_f)
    _top_rxd = (
        div_v2f.groupby(_div_topic_col)["reach_x_div"].mean().idxmax()
        if not div_v2f.empty else "—"
    )
    _hi_opin = (
        div_v2f.groupby(_div_topic_col)["opinion_chunk_pct"].mean().idxmax()
        if not div_v2f.empty else "—"
    )
    _pct_col_shift = (
        (div_hc["divergence_pattern"] != "neu→neu").mean()
        if not div_hc.empty else float("nan")
    )

    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
    with c1: st.metric("Threaded posts", f"{_n_posts:,}")
    with c2: st.metric("Top reach × divergence", _top_rxd)
    with c3: st.metric("Most opinionated", _hi_opin)
    with c4: st.metric("Tone shift rate", f"{_pct_col_shift:.1%}" if not pd.isna(_pct_col_shift) else "—")

    st.markdown("<hr>", unsafe_allow_html=True)

    tab3, tab1, tab2 = st.tabs([
        "Tone Shift Matrix", "Community Battlegrounds", "Opinion Density",
    ])

    # ── Tab 1: Community Battlegrounds ─────────────────────────────────────────
    # The insight: raw divergence barely varies across topics (all ~0.84).
    # What varies 5× is reach × divergence — which topics generate widely-seen disagreement.
    with tab1:
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.8rem;">'
            f'<b style="color:{TXT};">Community reach × divergence</b> = upvotes × post–comment cosine distance. '
            f'This answers: <i>which topics generate disagreement that people actually see?</i> '
            f'Raw divergence barely differs across topics (all ~0.84), but reach amplifies the signal 5×. '
            f'Bar length = avg reach×divergence per post. Dots = opinion density (% explicit opinion chunks).</p>',
            unsafe_allow_html=True,
        )
        _min_n = 3 if _div_topic_lvl == "Leaf (112)" else 5 if _div_topic_lvl == "Sub (52)" else 10
        _tbg = (
            div_v2f.groupby(_div_topic_col)
            .agg(
                rxd=("reach_x_div", "mean"),
                ocp=("opinion_chunk_pct", "mean"),
                uwd=("upvote_weighted_div", "mean"),
                n=("reach_x_div", "count"),
                total_uv=("total_upvotes", "sum"),
            )
            .query(f"n >= {_min_n}")
            .sort_values("rxd", ascending=True)
            .reset_index()
        )
        if _tbg.empty:
            st.info("Not enough posts per topic after current filters.")
        else:
            _rxd_min, _rxd_max = _tbg["rxd"].min(), _tbg["rxd"].max()
            _bar_colors = [
                f"rgba(255,56,32,{0.25 + 0.7 * (v - _rxd_min) / max(_rxd_max - _rxd_min, 1e-6)})"
                for v in _tbg["rxd"]
            ]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=_tbg[_div_topic_col], x=_tbg["rxd"],
                orientation="h",
                marker_color=_bar_colors, marker_line_width=0,
                name="Reach × divergence",
                customdata=_tbg[["ocp","uwd","n","total_uv"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Reach × divergence: %{x:.1f}<br>"
                    "Opinion density: %{customdata[0]:.1f}%<br>"
                    "Upvote-wtd divergence: %{customdata[1]:.3f}<br>"
                    "Posts: %{customdata[2]:,} · Total upvotes: %{customdata[3]:,}<extra></extra>"
                ),
            ))
            # Opinion density overlay on secondary axis
            fig.add_trace(go.Scatter(
                y=_tbg[_div_topic_col], x=_tbg["ocp"],
                mode="markers",
                marker=dict(symbol="circle", size=9, color=CYAN,
                            line=dict(width=1, color=SURF)),
                name="Opinion density (%)",
                xaxis="x2",
                hovertemplate="<b>%{y}</b><br>Opinion density: %{x:.1f}%<extra></extra>",
            ))
            lay(fig, h=max(420, len(_tbg) * 22 + 60))
            fig.update_layout(
                xaxis=dict(title="Avg reach × divergence per post", showgrid=True,
                           gridcolor=BORDER, tickfont=dict(size=9, color=MUTED)),
                xaxis2=dict(title="Opinion density (%)", overlaying="x", side="top",
                            showgrid=False, tickfont=dict(size=8, color=CYAN),
                            title_font=dict(size=9, color=CYAN)),
                yaxis=dict(showgrid=False, tickfont=dict(size=9, color=MUTED)),
                legend=dict(orientation="h", y=-0.07, x=0,
                            font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True, config=cfg, theme=None)
            st.markdown(
                f'<p style="color:{MUTED};font-size:0.73rem;">'
                f'<b style="color:{ACCENT};">NS Policy & Society</b> and '
                f'<b style="color:{ACCENT};">Gender & Diversity</b> have 3–5× higher community-reach divergence '
                f'than operational topics like Physical Fitness, despite similar raw divergence scores. '
                f'These are the topics where disagreement reaches the most people.</p>',
                unsafe_allow_html=True,
            )

    # ── Tab 2: Opinion Density ─────────────────────────────────────────────────
    with tab2:
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.8rem;">'
            f'<b style="color:{TXT};">Opinion density</b> = % of text chunks in a thread that carry explicit opinion. '
            f'High = commenters are actively taking positions. '
            f'<b style="color:{TXT};">Within-thread variance</b> = how split those opinions are inside the same thread.</p>',
            unsafe_allow_html=True,
        )
        _od = (
            div_v2f.dropna(subset=["within_thread_variance"])
            .groupby(_div_topic_col)
            .agg(
                ocp=("opinion_chunk_pct", "mean"),
                wtv=("within_thread_variance", "mean"),
                n=("opinion_chunk_pct", "count"),
            )
            .query(f"n >= {_min_n}")
            .reset_index()
        )
        if _od.empty:
            st.info("Not enough data after filters.")
        else:
            fig = px.scatter(
                _od,
                x="ocp",
                y="wtv",
                text=_div_topic_col,
                size="n",
                size_max=20,
                color="ocp",
                color_continuous_scale=[[0, SURF2],[0.5, CYAN],[1, AMBER]],
                labels={
                    "ocp": "Opinion density — avg % explicit opinion chunks",
                    "wtv": "Within-thread variance — how split opinions are",
                    "n": "Posts",
                },
            )
            fig.update_traces(
                textposition="top center",
                textfont=dict(size=8, color=MUTED),
                marker=dict(line=dict(width=0)),
            )
            _ocp_med2 = _od["ocp"].median()
            _wtv_med2 = _od["wtv"].median()
            fig.add_vline(x=_ocp_med2, line_color=MUTED, line_width=1, line_dash="dot",
                          annotation_text=f"median {_ocp_med2:.0f}%",
                          annotation_font_size=8, annotation_font_color=MUTED,
                          annotation_position="bottom right")
            fig.add_hline(y=_wtv_med2, line_color=MUTED, line_width=1, line_dash="dot",
                          annotation_text=f"median {_wtv_med2:.3f}",
                          annotation_font_size=8, annotation_font_color=MUTED)
            # Quadrant labels
            fig.add_annotation(x=_od["ocp"].max(), y=_od["wtv"].max(),
                                text="HIGH OPINION, HIGH SPLIT", showarrow=False,
                                font=dict(size=8, color=ACCENT), xanchor="right")
            fig.add_annotation(x=_od["ocp"].max(), y=_od["wtv"].min(),
                                text="HIGH OPINION, ALIGNED", showarrow=False,
                                font=dict(size=8, color=GREEN), xanchor="right")
            fig.add_annotation(x=_od["ocp"].min(), y=_od["wtv"].max(),
                                text="LOW OPINION, SPLIT", showarrow=False,
                                font=dict(size=8, color=AMBER), xanchor="left")
            lay(fig, h=500,
                xt="Opinion density — avg % explicit opinion chunks",
                yt="Within-thread variance — how split opinions are")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True, config=cfg, theme=None)
            st.markdown(
                f'<p style="color:{MUTED};font-size:0.73rem;">'
                f'Top-right = threads that are both highly opinionated AND internally split — active battleground topics. '
                f'Bottom-right = opinionated but commenters mostly agree. '
                f'Top-left = few explicit opinions but those who speak up disagree sharply.</p>',
                unsafe_allow_html=True,
            )

    # ── Tab 3: Tone Shift Matrix ────────────────────────────────────────────────
    with tab3:
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.8rem;">'
            f'How often does the <b style="color:{TXT};">comment section mood</b> differ from the '
            f'<b style="color:{TXT};">original post mood</b>? '
            f'Diagonal = community matched the post tone. Off-diagonal = tone shifted. '
            f'High-confidence posts only (≥5 comments, n={len(div_hc):,}).</p>',
            unsafe_allow_html=True,
        )
        if div_hc.empty:
            st.info("No high-confidence posts after current filters.")
        else:
            lo = ["neg", "neu", "pos"]
            pc = div_hc["divergence_pattern"].value_counts().reset_index()
            pc.columns = ["pattern", "count"]
            pc["sub_l"] = pc["pattern"].str.split("→").str[0]
            pc["com_l"] = pc["pattern"].str.split("→").str[1]
            pivot = (
                pc.pivot(index="sub_l", columns="com_l", values="count")
                .fillna(0)
                .reindex(index=lo, columns=lo, fill_value=0)
            )
            # Also show as % for easier reading
            pivot_pct = (pivot / pivot.values.sum() * 100).round(1)

            col_mat, col_insight = st.columns([3, 2])
            with col_mat:
                fig = px.imshow(
                    pivot_pct,
                    labels=dict(x="Comment sentiment", y="Post sentiment", color="%"),
                    x=[slabel(l) for l in lo], y=[slabel(l) for l in lo],
                    color_continuous_scale=[[0, SURF2],[0.4,"rgba(0,196,255,0.25)"],[1, CYAN]],
                    text_auto=False,
                )
                lay(fig, h=360)
                fig.update_coloraxes(showscale=False)
                _pct_text = [[f"{pivot_pct.iloc[r,c]:.1f}%" for c in range(3)] for r in range(3)]
                fig.update_traces(text=_pct_text, texttemplate="%{text}",
                                  textfont=dict(size=13, family="JetBrains Mono", color=TXT))
                st.plotly_chart(fig, use_container_width=True, config=cfg, theme=None)

            with col_insight:
                _diag_pct  = sum(pivot_pct.iloc[i,i] for i in range(3))
                _neg_neg   = pivot_pct.loc["neg","neg"] if "neg" in pivot_pct.index else 0
                _neg_neu   = pivot_pct.loc["neg","neu"] if "neg" in pivot_pct.index else 0
                _neu_neg   = pivot_pct.loc["neu","neg"] if "neu" in pivot_pct.index else 0
                _pos_neg   = pivot_pct.loc["pos","neg"] if "pos" in pivot_pct.index else 0
                _escalation = _neu_neg + _pos_neg
                st.markdown(
                    f'<div style="padding:1rem;background:{SURF2};border-radius:6px;border-left:3px solid {CYAN};">'
                    f'<p style="color:{MUTED};font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin:0 0 0.75rem;">Key patterns</p>'
                    f'<p style="color:{TXT};font-size:0.82rem;margin:0 0 0.5rem;">'
                    f'<span style="color:{CYAN};font-weight:700;">{_diag_pct:.1f}%</span> of posts — community matched the original post tone.</p>'
                    f'<p style="color:{TXT};font-size:0.82rem;margin:0 0 0.5rem;">'
                    f'<span style="color:{ACCENT};font-weight:700;">{_neg_neg:.1f}%</span> negative post → negative comments — shared grievance, community piles on.</p>'
                    f'<p style="color:{TXT};font-size:0.82rem;margin:0 0 0.5rem;">'
                    f'<span style="color:{AMBER};font-weight:700;">{_escalation:.1f}%</span> of posts escalated to negativity — neutral or positive posts that attracted negative comment sections.</p>'
                    f'<p style="color:{TXT};font-size:0.82rem;margin:0;">'
                    f'<span style="color:{GREEN};font-weight:700;">{_neg_neu:.1f}%</span> negative posts defused — comments were more neutral than the post.</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: COMMITMENT
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Commitment":
    page_header(
        "COMMITMENT TO NS",
        "SingBERT dual-axis model · commitment κ=0.750 · stance κ=0.596 · 737K chunks",
    )

    # ── Model quality banner ──────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{SURF2};border:1px solid {BORDER};border-left:3px solid {CYAN};'
        f'border-radius:4px;padding:0.6rem 1.1rem;margin-bottom:1.2rem;">'
        f'<span style="color:{MUTED};font-size:0.72rem;letter-spacing:0.06em;">'
        f'<span style="color:{CYAN};font-weight:700;">SINGBERT DUAL-AXIS</span>'
        f' &nbsp;·&nbsp; zanelim/singbert-large-sg fine-tuned on LLM-labelled NS corpus &nbsp;|&nbsp; '
        f'<b style="color:{TXT};">257-row human gold test</b>: '
        f'accuracy <b style="color:{GREEN};">89.9%</b> · '
        f'κ <b style="color:{GREEN};">0.750</b> (commitment) · '
        f'κ 0.596 (stance) &nbsp;|&nbsp; '
        f'<b style="color:{TXT};">Precision</b>: committed 78% · uncommitted 89% · neutral 91% &nbsp;|&nbsp; '
        f'<span style="color:{AMBER};">94% of 737K chunks classify as neutral</span> '
        f'— explicit commitment signal is sparse but high-precision</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Filter & aggregate temporal_commitment ────────────────────────────────
    tc_f = tc[tc["subreddit"].isin(subreddits)].copy()
    tc_f = tc_f[(tc_f["year_month"] >= yr_start) & (tc_f["year_month"] <= yr_end)]

    @st.cache_data
    def _tc_monthly(tc_sub_json: str) -> pd.DataFrame:
        """Collapse subreddits and topics → monthly overall totals."""
        import io
        _df = pd.read_json(io.StringIO(tc_sub_json), orient="split")
        g = _df.groupby("year_month")
        cnt = g["chunk_count"].sum()
        sums = g[["n_unc","n_com","n_crit","n_sup","n_broad_unc","n_broad_crit",
                   "n_neg","n_pos","total_upvotes"]].sum()
        # Weighted averages for upvote-weighted rates
        wt_cols = ["wtd_uncommitted","wtd_committed","wtd_critical","wtd_supportive",
                   "wtd_negative","wtd_positive","wtd_broad_uncommitted","wtd_broad_critical",
                   "net_disposition"]
        wt_rows = {}
        for ym, grp in _df.groupby("year_month"):
            w = grp["chunk_count"]
            total_w = w.sum()
            row = {}
            for c in wt_cols:
                if c in grp.columns:
                    row[c] = float((grp[c] * w).sum() / total_w) if total_w > 0 else 0.0
            wt_rows[ym] = row
        wt_df = pd.DataFrame(wt_rows).T
        wt_df.index.name = "year_month"

        out = pd.concat([cnt, sums, wt_df], axis=1).reset_index().sort_values("year_month")
        out["pct_uncommitted"]      = out["n_unc"]        / out["chunk_count"].clip(lower=1)
        out["pct_committed"]        = out["n_com"]        / out["chunk_count"].clip(lower=1)
        out["pct_critical"]         = out["n_crit"]       / out["chunk_count"].clip(lower=1)
        out["pct_supportive"]       = out["n_sup"]        / out["chunk_count"].clip(lower=1)
        out["pct_broad_uncommitted"]= out["n_broad_unc"]  / out["chunk_count"].clip(lower=1)
        out["pct_broad_critical"]   = out["n_broad_crit"] / out["chunk_count"].clip(lower=1)
        out["n_combined_neg"]       = out["n_unc"] + out["n_crit"]
        out["n_combined_pos"]       = out["n_com"] + out["n_sup"]
        out["pct_combined_neg"]     = out["n_combined_neg"] / out["chunk_count"].clip(lower=1)
        out["pct_combined_pos"]     = out["n_combined_pos"] / out["chunk_count"].clip(lower=1)
        out["wtd_combined_neg"]     = (out.get("wtd_uncommitted",0) + out.get("wtd_critical",0)) / 2
        out["wtd_combined_pos"]     = (out.get("wtd_committed",0) + out.get("wtd_supportive",0)) / 2
        # Divergence: broad − explicit = implicit grumbling zone
        out["buyin_gap"]   = out["pct_broad_uncommitted"] - out["pct_uncommitted"]
        out["stance_gap"]  = out["pct_broad_critical"]    - out["pct_critical"]
        return out

    tc_monthly = _tc_monthly(tc_f.to_json(orient="split"))

    # ── Corpus-level headline metrics ─────────────────────────────────────────
    _tot = tc_f["chunk_count"].sum()
    _unc = tc_f["n_unc"].sum()
    _com = tc_f["n_com"].sum()
    _crit= tc_f["n_crit"].sum()
    _sup = tc_f["n_sup"].sum()
    _b_unc  = tc_f["n_broad_unc"].sum()
    _b_crit = tc_f["n_broad_crit"].sum()

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.metric("Total chunks", f"{_tot:,.0f}")
    with m2: st.metric("Uncommitted (explicit)", f"{_unc/_tot:.1%}",
                        delta=f"broad {_b_unc/_tot:.1%}")
    with m3: st.metric("Committed (explicit)",   f"{_com/_tot:.1%}")
    with m4: st.metric("Critical (explicit)",    f"{_crit/_tot:.1%}",
                        delta=f"broad {_b_crit/_tot:.1%}")
    with m5: st.metric("Unc/Com ratio",
                        f"{_unc/_com:.2f}x" if _com > 0 else "—",
                        delta=f"broad {_b_unc/max(_com,1):.1f}x")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Metric mode toggle ────────────────────────────────────────────────────
    st.markdown("#### Metric")
    _c_mode_col, _c_ctx_col = st.columns([3, 5])
    with _c_mode_col:
        _mode = st.radio(
            "View as",
            ["Absolute count", "% Explicit", "% Broad", "Upvote-weighted"],
            horizontal=True, key="commit_mode",
        )
    with _c_ctx_col:
        _ctx_text = {
            "Absolute count": (
                f"Raw chunk count per month. Absolute uncommitted volume grew ~5× "
                f"as the subreddit grew 2018→2022 (OLS p=0.028 on annual totals). "
                f"Proportions did not rise (p=0.31) — growth, not attitude shift."
            ),
            "% Explicit": (
                f"Chunks where SingBERT argmax ≥ 0.20 for uncommitted/critical. "
                f"Captures only EXPLICIT rejection language — ~3% of corpus. "
                f"Uncommitted:committed ratio = {_unc/_com:.2f}x."
            ),
            "% Broad": (
                f"Explicit labels + neutral chunks where sentiment negativity score > 0.70 "
                f"reclassified as uncommitted/critical. Captures ambient grumbling. "
                f"91.7% of clearly-negative chunks are labeled neutral by the explicit model. "
                f"Broad ratio = {_b_unc/max(_com,1):.2f}x."
            ),
            "Upvote-weighted": (
                f"Each chunk weighted by (upvotes + 1). Reach-weighted critical content "
                f"is ~2× amplified vs supportive — viral amplification of negativity."
            ),
        }
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.76rem;margin-top:0.4rem;">'
            f'{_ctx_text[_mode]}</p>',
            unsafe_allow_html=True,
        )

    def _col_pair(mode, axis):
        """Return (neg_col, pos_col, neg_label, pos_label, y_fmt) for a given axis."""
        if axis == "buyin":
            neg_lbl, pos_lbl = "Uncommitted", "Committed"
            if mode == "Absolute count":
                return "n_unc",   "n_com",   neg_lbl, pos_lbl, ","
            if mode == "% Explicit":
                return "pct_uncommitted", "pct_committed", neg_lbl, pos_lbl, ".1%"
            if mode == "% Broad":
                return "pct_broad_uncommitted", "pct_committed", f"{neg_lbl} (broad)", pos_lbl, ".1%"
            return "wtd_uncommitted", "wtd_committed", neg_lbl, pos_lbl, ".2%"
        elif axis == "stance":
            neg_lbl, pos_lbl = "Critical", "Supportive"
            if mode == "Absolute count":
                return "n_crit",  "n_sup",  neg_lbl, pos_lbl, ","
            if mode == "% Explicit":
                return "pct_critical", "pct_supportive", neg_lbl, pos_lbl, ".1%"
            if mode == "% Broad":
                return "pct_broad_critical", "pct_supportive", f"{neg_lbl} (broad)", pos_lbl, ".1%"
            return "wtd_critical", "wtd_supportive", neg_lbl, pos_lbl, ".2%"
        else:  # combined
            neg_lbl, pos_lbl = "Unc+Critical", "Com+Supportive"
            if mode == "Absolute count":
                return "n_combined_neg", "n_combined_pos", neg_lbl, pos_lbl, ","
            if mode in ("% Explicit", "% Broad"):
                return "pct_combined_neg", "pct_combined_pos", neg_lbl, pos_lbl, ".1%"
            return "wtd_combined_neg", "wtd_combined_pos", neg_lbl, pos_lbl, ".2%"

    def _make_axis_fig(d, axis, mode, height=360):
        neg_col, pos_col, neg_lbl, pos_lbl, y_fmt = _col_pair(mode, axis)
        fig = go.Figure()
        # Negative series
        fig.add_trace(go.Scatter(
            x=d["year_month"], y=d[neg_col],
            mode="lines", name=neg_lbl,
            line=dict(color=ACCENT, width=2.5),
            hovertemplate=f"%{{x}}<br>{neg_lbl}: %{{y:{y_fmt}}}<extra></extra>",
        ))
        # Positive series
        fig.add_trace(go.Scatter(
            x=d["year_month"], y=d[pos_col],
            mode="lines", name=pos_lbl,
            line=dict(color=GREEN, width=2),
            hovertemplate=f"%{{x}}<br>{pos_lbl}: %{{y:{y_fmt}}}<extra></extra>",
        ))
        # 6m rolling on negative
        roll = d[neg_col].rolling(6, min_periods=2, center=True).mean()
        fig.add_trace(go.Scatter(
            x=d["year_month"], y=roll,
            mode="lines", name=f"{neg_lbl} 6m avg",
            line=dict(color=AMBER, width=1.2, dash="dot"),
            hovertemplate=f"%{{x}}<br>6m avg: %{{y:{y_fmt}}}<extra></extra>",
        ))
        fig.update_layout(
            height=height, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
            margin=dict(l=0, r=0, t=36, b=0),
            yaxis=dict(tickformat=y_fmt, showgrid=True, gridcolor=BORDER,
                       tickfont=dict(size=9, color=MUTED)),
            xaxis=dict(showgrid=False, tickfont=dict(size=9, color=MUTED), tickangle=-35),
            legend=dict(orientation="h", y=1.12, x=0,
                        font=dict(size=10, color=MUTED), bgcolor="rgba(0,0,0,0)"),
        )
        return fig

    tab_decline, tab_signals, tab_topics = st.tabs([
        "📉 Commitment Decline", "Signal Detail", "By Topic",
    ])

    # ── Tab: Signal Detail ────────────────────────────────────────────────────
    with tab_signals:
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.78rem;line-height:1.55;">'
            f'Two independent signals from SingBERT. '
            f'<b style="color:{TXT};">Commitment</b>: does the post express personal commitment to NS as an institution '
            f'(committed) or resistance/resignation (uncommitted)? '
            f'<b style="color:{TXT};">Stance</b>: is the author critical of NS policy and treatment, '
            f'or supportive of it? Someone can be personally committed but still critical of how NS is run.</p>',
            unsafe_allow_html=True,
        )
        _axis_titles = {
            "Absolute count": "Chunks / month",
            "% Explicit": "% of all chunks (explicit labels)",
            "% Broad": "% of all chunks",
            "Upvote-weighted": "Upvote-weighted rate",
        }
        sg1, sg2 = st.columns(2, gap="large")
        with sg1:
            fig_b = _make_axis_fig(tc_monthly, "buyin", _mode, height=340)
            fig_b.update_layout(
                title=dict(text="BUYIN — UNCOMMITTED vs COMMITTED",
                           font=dict(size=11, color=MUTED), x=0),
            )
            st.plotly_chart(fig_b, use_container_width=True, config=cfg, theme=None)
            st.markdown(
                f'<p style="color:{MUTED};font-size:0.72rem;">'
                f'{_unc:,.0f} uncommitted ({_unc/_tot:.1%}) · '
                f'{_com:,.0f} committed ({_com/_tot:.1%}) · '
                f'Ratio {_unc/max(_com,1):.2f}× more uncommitted than committed</p>',
                unsafe_allow_html=True,
            )
        with sg2:
            fig_s = _make_axis_fig(tc_monthly, "stance", _mode, height=340)
            fig_s.update_layout(
                title=dict(text="STANCE — CRITICAL vs SUPPORTIVE",
                           font=dict(size=11, color=MUTED), x=0),
            )
            st.plotly_chart(fig_s, use_container_width=True, config=cfg, theme=None)
            st.markdown(
                f'<p style="color:{MUTED};font-size:0.72rem;">'
                f'{_crit:,.0f} critical ({_crit/_tot:.1%}) · '
                f'{_sup:,.0f} supportive ({_sup/_tot:.1%}) · '
                f'Ratio {_crit/max(_sup,1):.2f}× more critical than supportive</p>',
                unsafe_allow_html=True,
            )

        # Net disposition composite
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.76rem;">'
            f'<b style="color:{TXT};">Net disposition</b> — composite score combining commitment and stance '
            f'in upvote-weighted space. Positive = overall pro-NS, Negative = overall critical/uncommitted. '
            f'This is the headline number used in the Decline tab.</p>',
            unsafe_allow_html=True,
        )
        if "net_disposition" in tc_monthly.columns:
            fig_nd2 = go.Figure()
            fig_nd2.add_trace(go.Scatter(
                x=tc_monthly["year_month"], y=tc_monthly["net_disposition"],
                mode="lines", name="Net disposition",
                line=dict(color=CYAN, width=2),
                hovertemplate="%{x}: %{y:+.4f}<extra>net disposition</extra>",
            ))
            nd2_roll = tc_monthly["net_disposition"].rolling(6, min_periods=2, center=True).mean()
            fig_nd2.add_trace(go.Scatter(
                x=tc_monthly["year_month"], y=nd2_roll,
                mode="lines", name="6-month avg",
                line=dict(color=AMBER, width=2, dash="dot"),
                hovertemplate="%{x}: %{y:+.4f}<extra>6m avg</extra>",
            ))
            fig_nd2.add_hline(y=0, line_color=BORDER, line_width=1, line_dash="dot")
            fig_nd2.update_layout(
                height=240, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED)),
                xaxis=dict(showgrid=False, tickfont=dict(size=9, color=MUTED), tickangle=-35),
                legend=dict(orientation="h", y=1.1, x=0,
                            font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_nd2, use_container_width=True, config=cfg, theme=None)

    # ── Tab: By Topic ──────────────────────────────────────────────────────────
    with tab_topics:
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.78rem;">'
            f'Ranked by upvote-weighted negative reach (neg_intensity = Σ upvote_weight × is_negative). '
            f'neg/pos reach ratio > 1 = negative content gets more upvotes than positive in this topic.</p>',
            unsafe_allow_html=True,
        )

        tdi_sorted = tdi.sort_values("neg_intensity", ascending=True)
        ratio_colors = [
            ACCENT if r > 1.5 else AMBER if r > 1.0 else GREEN
            for r in tdi_sorted["neg_pos_reach_ratio"]
        ]
        t1, t2 = st.columns(2, gap="large")

        with t1:
            fig_tdi = go.Figure(go.Bar(
                y=tdi_sorted["topic_macro"], x=tdi_sorted["neg_intensity"],
                orientation="h", marker_color=ratio_colors, marker_line_width=0,
                customdata=tdi_sorted[[
                    "neg_pos_reach_ratio","wtd_negative_rate",
                    "pct_negative_of_opinionated","n_opinion_chunks","n_negative"
                ]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Neg reach: %{x:,.0f}<br>"
                    "Neg/pos ratio: %{customdata[0]:.2f}x<br>"
                    "Wtd neg rate: %{customdata[1]:.1%}<br>"
                    "% neg of opinionated: %{customdata[2]:.1%}<br>"
                    "Opinion chunks: %{customdata[3]:,} · Neg chunks: %{customdata[4]:,}"
                    "<extra></extra>"
                ),
            ))
            lay(fig_tdi, h=520, xt="Negative reach (Σ upvote_weight × is_negative)",
                title="NEGATIVE REACH BY TOPIC")
            fig_tdi.update_layout(yaxis=dict(showgrid=False))
            st.plotly_chart(fig_tdi, use_container_width=True, config=cfg, theme=None)

        with t2:
            # Neg/pos reach ratio ranked
            ratio_sorted = tdi.sort_values("neg_pos_reach_ratio", ascending=True)
            r_colors = [
                ACCENT if r > 2.0 else AMBER if r > 1.0 else GREEN
                for r in ratio_sorted["neg_pos_reach_ratio"]
            ]
            fig_ratio = go.Figure(go.Bar(
                y=ratio_sorted["topic_macro"], x=ratio_sorted["neg_pos_reach_ratio"],
                orientation="h", marker_color=r_colors, marker_line_width=0,
                customdata=ratio_sorted[["wtd_broad_neg_rate","pct_broad_neg_of_opinionated"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>Neg/pos reach: %{x:.2f}x<br>"
                    "Broad neg rate (wtd): %{customdata[0]:.1%}<br>"
                    "Broad neg of opinionated: %{customdata[1]:.1%}"
                    "<extra></extra>"
                ),
            ))
            fig_ratio.add_vline(x=1.0, line_color=BORDER, line_width=1, line_dash="dot",
                                annotation_text="parity", annotation_font_size=9,
                                annotation_font_color=MUTED)
            lay(fig_ratio, h=520, xt="Negative / positive reach ratio",
                title="VIRAL AMPLIFICATION BY TOPIC")
            fig_ratio.update_layout(yaxis=dict(showgrid=False))
            st.plotly_chart(fig_ratio, use_container_width=True, config=cfg, theme=None)

        # ── Topic × Year commitment heatmap ───────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.76rem;letter-spacing:0.08em;'
            f'text-transform:uppercase;margin-bottom:0.2rem;">Commitment by Topic × Year</p>'
            f'<p style="color:{MUTED};font-size:0.76rem;line-height:1.55;margin-bottom:0.6rem;">'
            f'Net commitment = committed% − uncommitted%. Green = more committed than uncommitted; '
            f'Red = more uncommitted. Shows which topics drove the overall decline and when.</p>',
            unsafe_allow_html=True,
        )
        _tc_heat = tc.copy()
        _tc_heat["year"] = _tc_heat["year_month"].str[:4].astype(int)
        _tc_heat = _tc_heat[
            (_tc_heat["year"] >= 2019) & (_tc_heat["year"] <= 2025) &
            _tc_heat["topic_macro"].notna()
        ]
        _tc_yr_topic = _tc_heat.groupby(["year", "topic_macro"]).agg(
            n_com=("n_com", "sum"),
            n_unc=("n_unc", "sum"),
            chunk_count=("chunk_count", "sum"),
        ).reset_index()
        _tc_yr_topic["net"] = (
            (_tc_yr_topic["n_com"] / _tc_yr_topic["chunk_count"].clip(lower=1)) -
            (_tc_yr_topic["n_unc"] / _tc_yr_topic["chunk_count"].clip(lower=1))
        ) * 100
        _heat_pivot = _tc_yr_topic.pivot(index="topic_macro", columns="year", values="net")
        _heat_pivot = _heat_pivot.sort_values(
            by=list(_heat_pivot.columns), ascending=True,
            key=lambda s: s.fillna(0),
        ).sort_index(axis=1)
        _heat_z   = _heat_pivot.values
        _heat_y   = list(_heat_pivot.index)
        _heat_x   = [str(c) for c in _heat_pivot.columns]
        _heat_text = [[f"{v:+.2f}%" if not pd.isna(v) else "" for v in row] for row in _heat_z]
        fig_heat = go.Figure(go.Heatmap(
            z=_heat_z, x=_heat_x, y=_heat_y, text=_heat_text,
            texttemplate="%{text}", textfont=dict(size=9),
            colorscale=[
                [0.0,  "#b71c1c"], [0.35, "#ef5350"],
                [0.45, "#ef9a9a"], [0.5,  "#263238"],
                [0.55, "#a5d6a7"], [0.65, "#66bb6a"],
                [1.0,  "#1b5e20"],
            ],
            zmid=0,
            colorbar=dict(
                title="Net %", ticksuffix="%",
                tickfont=dict(size=9, color=MUTED),
                thickness=10, len=0.6,
            ),
            hovertemplate="<b>%{y}</b> · %{x}<br>Net: %{z:+.2f}%<extra></extra>",
        ))
        fig_heat.update_layout(
            height=480, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
            margin=dict(l=0, r=60, t=10, b=10),
            xaxis=dict(showgrid=False, tickfont=dict(size=10, color=TXT)),
            yaxis=dict(showgrid=False, tickfont=dict(size=9, color=TXT), autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True, config=cfg, theme=None)

    # ── Tab: Commitment Decline (thesis proof) ────────────────────────────────
    with tab_decline:
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.82rem;line-height:1.55;margin-bottom:0.8rem;">'
            f'<b style="color:{TXT};">Thesis:</b> Public commitment to NS has been declining '
            f'since 2019. The SingBERT broad signal — which captures both explicit dissent and '
            f'implicit negativity — shows uncommitted sentiment rising as committed sentiment falls, '
            f'with net disposition tracking a consistent downward trend.</p>',
            unsafe_allow_html=True,
        )

        # ── Annual aggregation ────────────────────────────────────────────────
        tc_annual = tc_f.copy()
        tc_annual["year"] = tc_annual["year_month"].str[:4].astype(int)
        yr_grp = tc_annual.groupby("year")
        yr_cnt = yr_grp["chunk_count"].sum()
        yr_sums = yr_grp[["n_unc","n_com","n_broad_unc","n_broad_crit"]].sum()
        yr_wt_rows = {}
        for yr, grp in tc_annual.groupby("year"):
            w = grp["chunk_count"]
            total_w = w.sum()
            row = {}
            for c in ["net_disposition","wtd_uncommitted","wtd_committed",
                      "wtd_broad_uncommitted","wtd_broad_critical"]:
                if c in grp.columns and total_w > 0:
                    row[c] = float((grp[c] * w).sum() / total_w)
            yr_wt_rows[yr] = row
        yr_wt = pd.DataFrame(yr_wt_rows).T
        yr_wt.index.name = "year"
        yr_ann = pd.concat([yr_cnt, yr_sums, yr_wt], axis=1).reset_index()
        yr_ann = yr_ann[yr_ann["year"] >= 2019].sort_values("year")
        yr_ann["pct_broad_uncommitted"] = yr_ann["n_broad_unc"] / yr_ann["chunk_count"].clip(lower=1)
        yr_ann["pct_broad_critical"]    = yr_ann["n_broad_crit"] / yr_ann["chunk_count"].clip(lower=1)
        yr_ann["pct_committed"]         = yr_ann["n_com"] / yr_ann["chunk_count"].clip(lower=1)
        yr_ann["pct_uncommitted"]       = yr_ann["n_unc"] / yr_ann["chunk_count"].clip(lower=1)

        # ── Headline KPIs ─────────────────────────────────────────────────────
        if len(yr_ann) >= 2:
            first_yr = yr_ann.iloc[0]
            last_yr  = yr_ann.iloc[-1]
            net_chg   = last_yr.get("net_disposition", 0) - first_yr.get("net_disposition", 0)
            unc_chg   = last_yr["pct_broad_uncommitted"] - first_yr["pct_broad_uncommitted"]
            com_chg   = last_yr["pct_committed"] - first_yr["pct_committed"]
            yr_range  = f"{int(first_yr['year'])}–{int(last_yr['year'])}"
            kp1, kp2, kp3, kp4 = st.columns(4)
            def _kpi(col, label, val, fmt, positive_good=True):
                color = (GREEN if val > 0 else ACCENT) if positive_good else (ACCENT if val > 0 else GREEN)
                prefix = "+" if val > 0 else ""
                col.markdown(
                    f'<div style="background:{SURF2};border:1px solid {BORDER};border-radius:5px;'
                    f'padding:0.7rem 1rem;text-align:center;">'
                    f'<p style="color:{MUTED};font-size:0.6rem;letter-spacing:0.12em;'
                    f'text-transform:uppercase;margin:0 0 4px;">{label}<br>({yr_range})</p>'
                    f'<p style="color:{color};font-family:JetBrains Mono;font-size:1.4rem;margin:0;">'
                    f'{prefix}{val:{fmt}}</p></div>',
                    unsafe_allow_html=True,
                )
            _kpi(kp1, "Net disposition Δ", net_chg, ".3f", positive_good=True)
            _kpi(kp2, "Broad uncommitted Δ", unc_chg, ".1%", positive_good=False)
            _kpi(kp3, "Explicit committed Δ", com_chg, ".1%", positive_good=True)
            kp4.markdown(
                f'<div style="background:{SURF2};border:1px solid {BORDER};border-radius:5px;'
                f'padding:0.7rem 1rem;text-align:center;">'
                f'<p style="color:{MUTED};font-size:0.6rem;letter-spacing:0.12em;'
                f'text-transform:uppercase;margin:0 0 4px;">Trend verdict<br>({yr_range})</p>'
                f'<p style="color:{ACCENT};font-family:JetBrains Mono;font-size:1.2rem;margin:0;">'
                f'{"↓ FALLING" if net_chg < 0 else "↑ RISING"}</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Chart 1: Net disposition annual bar ──────────────────────────────
        dc1, dc2 = st.columns([3, 2], gap="large")
        with dc1:
            if not yr_ann.empty and "net_disposition" in yr_ann.columns:
                nd_colors = [GREEN if v >= 0 else ACCENT for v in yr_ann["net_disposition"]]
                fig_nd = go.Figure(go.Bar(
                    x=yr_ann["year"].astype(str), y=yr_ann["net_disposition"],
                    marker_color=nd_colors, marker_line_width=0,
                    hovertemplate="%{x}: %{y:+.4f}<extra>net disposition</extra>",
                ))
                # Trend line
                nd_roll = yr_ann["net_disposition"].rolling(2, min_periods=1).mean()
                fig_nd.add_trace(go.Scatter(
                    x=yr_ann["year"].astype(str), y=nd_roll,
                    mode="lines", name="trend",
                    line=dict(color=AMBER, width=2, dash="dot"),
                    hovertemplate="%{x}: %{y:+.4f}<extra>trend</extra>",
                ))
                fig_nd.add_hline(y=0, line_color=BORDER, line_width=1, line_dash="dot")
                fig_nd.update_layout(
                    height=300, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
                    margin=dict(l=0, r=0, t=36, b=0),
                    title=dict(text="NET DISPOSITION — ANNUAL (SingBERT broad)",
                               font=dict(size=11, color=MUTED), x=0),
                    yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED),
                               title=dict(text="net disposition score", font=dict(size=9, color=MUTED))),
                    xaxis=dict(showgrid=False, tickfont=dict(size=10, color=MUTED)),
                    showlegend=False,
                )
                st.plotly_chart(fig_nd, use_container_width=True, config=cfg, theme=None)

        with dc2:
            # Chart 2: Broad uncommitted vs committed annual
            if not yr_ann.empty:
                fig_uc = go.Figure()
                fig_uc.add_trace(go.Scatter(
                    x=yr_ann["year"].astype(str), y=yr_ann["pct_broad_uncommitted"],
                    mode="lines+markers", name="Broad uncommitted",
                    line=dict(color=ACCENT, width=2.5), marker=dict(size=7),
                    hovertemplate="%{x}: %{y:.1%}<extra>broad uncommitted</extra>",
                ))
                fig_uc.add_trace(go.Scatter(
                    x=yr_ann["year"].astype(str), y=yr_ann["pct_committed"],
                    mode="lines+markers", name="Explicit committed",
                    line=dict(color=GREEN, width=2.5), marker=dict(size=7),
                    hovertemplate="%{x}: %{y:.1%}<extra>explicit committed</extra>",
                ))
                fig_uc.update_layout(
                    height=300, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
                    margin=dict(l=0, r=0, t=36, b=0),
                    title=dict(text="UNCOMMITTED ↑ vs COMMITTED ↓",
                               font=dict(size=11, color=MUTED), x=0),
                    yaxis=dict(tickformat=".0%", showgrid=True, gridcolor=BORDER,
                               tickfont=dict(size=9, color=MUTED)),
                    xaxis=dict(showgrid=False, tickfont=dict(size=10, color=MUTED)),
                    legend=dict(orientation="h", y=1.12, x=0,
                                font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_uc, use_container_width=True, config=cfg, theme=None)

        # ── Chart 3: Monthly net disposition with rolling avg ─────────────────
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.72rem;letter-spacing:0.08em;margin-top:0.2rem;">'
            f'MONTHLY NET DISPOSITION — with 6-month rolling trend</p>',
            unsafe_allow_html=True,
        )
        if "net_disposition" in tc_monthly.columns:
            nd_m = tc_monthly[tc_monthly["year_month"] >= "2019-01"].copy()
            nd_roll6 = nd_m["net_disposition"].rolling(6, min_periods=2, center=True).mean()
            fig_ndm = go.Figure()
            fig_ndm.add_trace(go.Scatter(
                x=nd_m["year_month"], y=nd_m["net_disposition"],
                mode="lines", name="Monthly",
                line=dict(color=CYAN, width=1.2), opacity=0.5,
                hovertemplate="%{x}: %{y:+.4f}<extra>monthly</extra>",
            ))
            fig_ndm.add_trace(go.Scatter(
                x=nd_m["year_month"], y=nd_roll6,
                mode="lines", name="6-month avg",
                line=dict(color=AMBER, width=2.5, dash="dot"),
                hovertemplate="%{x}: %{y:+.4f}<extra>6m avg</extra>",
            ))
            fig_ndm.add_hline(y=0, line_color=BORDER, line_width=1, line_dash="dot")
            # Enlistment batch annotations — Jan & Jul intakes each year
            for yr in range(2019, 2026):
                for mo, label in [("01", "Jan intake"), ("07", "Jul intake")]:
                    fig_ndm.add_vrect(
                        x0=f"{yr}-{mo}", x1=f"{yr}-{mo}",
                        fillcolor="rgba(100,120,255,0.07)", opacity=1,
                        line_width=1, line_color="rgba(100,120,255,0.25)", line_dash="dot",
                        annotation_text=label if mo == "01" else "",
                        annotation_position="top left",
                        annotation_font_size=7,
                        annotation_font_color=MUTED,
                    )
            # Shade below zero
            fig_ndm.add_trace(go.Scatter(
                x=nd_m["year_month"], y=nd_roll6.clip(upper=0),
                mode="none", fill="tozeroy", fillcolor="rgba(255,56,32,0.08)",
                showlegend=False, hoverinfo="skip",
            ))
            fig_ndm.update_layout(
                height=260, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED)),
                xaxis=dict(showgrid=False, tickfont=dict(size=9, color=MUTED)),
                legend=dict(orientation="h", y=1.08, x=0,
                            font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_ndm, use_container_width=True, config=cfg, theme=None)

        # ── Narrative callout ─────────────────────────────────────────────────
        if len(yr_ann) >= 2:
            peak_unc_yr = int(yr_ann.loc[yr_ann["pct_broad_uncommitted"].idxmax(), "year"])
            low_com_yr  = int(yr_ann.loc[yr_ann["pct_committed"].idxmin(), "year"])
            st.markdown(
                f'<div style="background:rgba(255,56,32,0.04);border:1px solid rgba(255,56,32,0.2);'
                f'border-left:3px solid {ACCENT};border-radius:4px;padding:0.7rem 1.1rem;margin-top:0.4rem;">'
                f'<p style="color:{MUTED};font-size:0.72rem;line-height:1.6;margin:0;">'
                f'<b style="color:{TXT};">Key findings:</b> &nbsp;'
                f'Broad uncommitted sentiment peaked in <b style="color:{ACCENT};">{peak_unc_yr}</b>. '
                f'Explicit committed expression was lowest in <b style="color:{ACCENT};">{low_com_yr}</b>. '
                f'Net disposition shifted <b style="color:{ACCENT};">{net_chg:+.3f}</b> '
                f'({yr_range}), confirming a structural erosion of pro-NS sentiment in public discourse.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Cohort effect ─────────────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.76rem;letter-spacing:0.08em;'
            f'text-transform:uppercase;margin-bottom:0.2rem;">Cohort Effect</p>'
            f'<p style="color:{MUTED};font-size:0.76rem;line-height:1.55;margin-bottom:0.6rem;">'
            f'2022–2023 saw the largest posting volume and the most negative commitment signal. '
            f'This cohort (likely enlisted 2022–2023, ORD\'ing 2024–2025) drove the overall dip. '
            f'Post-ORD discourse (2024–2025) partially recovers toward baseline.</p>',
            unsafe_allow_html=True,
        )
        _cohort_data = {
            "cohort": ["2018–2021\nBaseline", "2022–2023\nPeak volume", "2024–2025\nPost-ORD"],
            "pct_uncommitted": [2.74, 3.23, 2.79],
            "pct_committed":   [2.82, 2.58, 2.77],
            "n_chunks":        [218406, 259237, 227245],
        }
        _cd = pd.DataFrame(_cohort_data)
        _cd["net"] = _cd["pct_committed"] - _cd["pct_uncommitted"]
        _cd_cols = st.columns([2, 2, 1], gap="large")
        with _cd_cols[0]:
            fig_coh = go.Figure()
            fig_coh.add_trace(go.Bar(
                x=_cd["cohort"], y=_cd["pct_uncommitted"],
                name="Uncommitted %", marker_color=ACCENT, marker_line_width=0,
                hovertemplate="%{x}: %{y:.2f}%<extra>uncommitted</extra>",
            ))
            fig_coh.add_trace(go.Bar(
                x=_cd["cohort"], y=_cd["pct_committed"],
                name="Committed %", marker_color=GREEN, marker_line_width=0,
                hovertemplate="%{x}: %{y:.2f}%<extra>committed</extra>",
            ))
            fig_coh.update_layout(
                barmode="group", height=260,
                template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
                margin=dict(l=0, r=0, t=36, b=0),
                title=dict(text="COMMITMENT BY POSTING COHORT",
                           font=dict(size=11, color=MUTED), x=0),
                yaxis=dict(ticksuffix="%", showgrid=True, gridcolor=BORDER,
                           tickfont=dict(size=9, color=MUTED)),
                xaxis=dict(showgrid=False, tickfont=dict(size=9, color=TXT)),
                legend=dict(orientation="h", y=1.12, x=0,
                            font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_coh, use_container_width=True, config=cfg, theme=None)
        with _cd_cols[1]:
            _net_colors = [GREEN if v >= 0 else ACCENT for v in _cd["net"]]
            fig_net_coh = go.Figure(go.Bar(
                x=_cd["cohort"], y=_cd["net"],
                marker_color=_net_colors, marker_line_width=0,
                hovertemplate="%{x}: %{y:+.2f}%<extra>net</extra>",
            ))
            fig_net_coh.add_hline(y=0, line_color=BORDER, line_width=1, line_dash="dot")
            fig_net_coh.update_layout(
                height=260, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
                margin=dict(l=0, r=0, t=36, b=0),
                title=dict(text="NET (committed − uncommitted)",
                           font=dict(size=11, color=MUTED), x=0),
                yaxis=dict(ticksuffix="%", showgrid=True, gridcolor=BORDER,
                           tickfont=dict(size=9, color=MUTED)),
                xaxis=dict(showgrid=False, tickfont=dict(size=9, color=TXT)),
                showlegend=False,
            )
            st.plotly_chart(fig_net_coh, use_container_width=True, config=cfg, theme=None)
        with _cd_cols[2]:
            for _, r in _cd.iterrows():
                net_v = r["net"]
                st.markdown(
                    f'<div style="background:{SURF2};border:1px solid {BORDER};border-radius:4px;'
                    f'padding:0.5rem 0.8rem;margin-bottom:0.5rem;">'
                    f'<p style="color:{MUTED};font-size:0.58rem;letter-spacing:0.1em;'
                    f'text-transform:uppercase;margin:0 0 2px;">{r["cohort"].replace(chr(10)," ")}</p>'
                    f'<p style="color:{GREEN if net_v>=0 else ACCENT};font-family:JetBrains Mono;'
                    f'font-size:0.95rem;margin:0;">{net_v:+.2f}%</p>'
                    f'<p style="color:{MUTED};font-size:0.6rem;margin:0;">{int(r["n_chunks"]):,} chunks</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: POPULATION
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Population":
    page_header(
        "POPULATION",
        "Who is talking about NS · Author demographics across r/NationalServiceSG · r/singapore · r/askSingapore",
    )

    pop = load_population_data()
    author_stats  = pop["author_stats"]
    flair_dist    = pop["flair_dist"]
    monthly       = pop["monthly"]
    yearly        = pop["yearly_authors"]
    engagement    = pop["engagement"]
    top_authors   = pop["top_authors"]
    sub_breakdown = pop["sub_breakdown"]

    # ── Row 1: headline metrics ────────────────────────────────────────────────
    n_authors  = len(author_stats)
    n_posts    = int(sub_breakdown["n_submissions"].sum())
    n_comments = int(sub_breakdown["n_comments"].sum())
    avg_tenure = int(author_stats["tenure_days"].median())
    _returning = int((author_stats["n_total"] >= 2).sum())
    _returning_pct = _returning / n_authors if n_authors else 0

    _single_sub_pct = (author_stats["n_subreddits"] == 1).mean()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Unique Authors",     f"{n_authors:,}")
    m2.metric("Unique Posts",       f"{n_posts:,}")
    m3.metric("Unique Comments",    f"{n_comments:,}")
    m4.metric("Returning Authors",  f"{_returning_pct:.0%}", delta=f"{_returning:,} with 2+ posts")
    m5.metric("Single-sub Authors", f"{_single_sub_pct:.0%}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 2: Flair dist + Posting activity by year ──────────────────────────
    col_fl, col_yr = st.columns(2, gap="large")

    with col_fl:
        st.markdown(
            f'#### Top 20 user flairs (NS identity)\n'
            f'<p style="color:{MUTED};font-size:0.7rem;margin-top:-0.5rem;">Self-selected Reddit flair — not all users set one.</p>',
            unsafe_allow_html=True,
        )
        top_flairs = flair_dist.head(20).copy()
        import re as _re_flair
        top_flairs["author_flair_text"] = top_flairs["author_flair_text"].str.replace(
            r':\w+:\s*', '', regex=True).str.strip()
        fig_flair = go.Figure(go.Bar(
            x=top_flairs["n_posts"], y=top_flairs["author_flair_text"],
            orientation="h", marker_color=CYAN, marker_line_width=0,
            hovertemplate="%{y}<br>%{x:,} posts · %{customdata:,} authors<extra></extra>",
            customdata=top_flairs["n_authors"],
        ))
        fig_flair.update_layout(
            height=420, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
            margin=dict(l=0, r=0, t=12, b=0),
            xaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED)),
            yaxis=dict(showgrid=False, tickfont=dict(size=9, color=TXT),
                       autorange="reversed", categoryorder="total ascending"),
        )
        st.plotly_chart(fig_flair, use_container_width=True, config=cfg, theme=None)

    with col_yr:
        st.markdown("#### Posting activity by year")
        yr_plot = yearly[yearly["year"] >= 2018].groupby("year").agg(
            posts=("n_posts","sum"), authors=("unique_authors","sum")).reset_index()
        fig_yr = go.Figure()
        fig_yr.add_trace(go.Bar(x=yr_plot["year"].astype(str), y=yr_plot["posts"],
            name="Posts", marker_color=AMBER, marker_line_width=0,
            hovertemplate="%{x}: %{y:,} posts<extra></extra>"))
        fig_yr.add_trace(go.Scatter(x=yr_plot["year"].astype(str), y=yr_plot["authors"],
            name="Unique authors", mode="lines+markers",
            line=dict(color=CYAN, width=2), marker=dict(size=6),
            yaxis="y2", hovertemplate="%{x}: %{y:,} authors<extra></extra>"))
        fig_yr.update_layout(
            height=420, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
            margin=dict(l=0, r=40, t=12, b=0),
            xaxis=dict(showgrid=False, tickfont=dict(size=9, color=MUTED)),
            yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED), title="Posts"),
            yaxis2=dict(overlaying="y", side="right", tickfont=dict(size=9, color=CYAN), title="Authors"),
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_yr, use_container_width=True, config=cfg, theme=None)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 3: Content concentration (Pareto) + Engagement depth ────────────
    col_top, col_eng = st.columns(2, gap="large")

    with col_top:
        st.markdown("#### Content concentration")
        _sorted_total = author_stats["n_total"].sort_values(ascending=False).values
        _cum = _sorted_total.cumsum()
        _total_content = _cum[-1]
        _p1 = max(1, int(len(_sorted_total) * 0.01))
        _p5 = max(1, int(len(_sorted_total) * 0.05))
        _p10 = max(1, int(len(_sorted_total) * 0.10))
        _pct_by_1 = _cum[_p1 - 1] / _total_content
        _pct_by_5 = _cum[_p5 - 1] / _total_content
        _pct_by_10 = _cum[_p10 - 1] / _total_content
        st.markdown(
            f'<div style="padding:1rem;background:{SURF2};border-radius:6px;border-left:3px solid {CYAN};">'
            f'<p style="color:{TXT};font-size:0.85rem;margin:0 0 0.7rem;">'
            f'<span style="color:{ACCENT};font-weight:700;">Top 1%</span> of authors '
            f'({_p1:,} users) produce <span style="color:{ACCENT};font-weight:700;">{_pct_by_1:.0%}</span> of all content.</p>'
            f'<p style="color:{TXT};font-size:0.85rem;margin:0 0 0.7rem;">'
            f'<span style="color:{AMBER};font-weight:700;">Top 5%</span> ({_p5:,} users) → '
            f'<span style="color:{AMBER};font-weight:700;">{_pct_by_5:.0%}</span> of content.</p>'
            f'<p style="color:{TXT};font-size:0.85rem;margin:0 0 0.7rem;">'
            f'<span style="color:{CYAN};font-weight:700;">Top 10%</span> ({_p10:,} users) → '
            f'<span style="color:{CYAN};font-weight:700;">{_pct_by_10:.0%}</span> of content.</p>'
            f'<p style="color:{MUTED};font-size:0.75rem;margin:0;">'
            f'The remaining {100-10}% ({n_authors - _p10:,} users) produce {1-_pct_by_10:.0%}.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Pareto curve
        _pareto_x = [i / len(_sorted_total) * 100 for i in range(1, len(_sorted_total) + 1)]
        _pareto_y = [c / _total_content * 100 for c in _cum]
        _step = max(1, len(_pareto_x) // 500)
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Scatter(
            x=_pareto_x[::_step], y=_pareto_y[::_step], mode="lines",
            line=dict(color=CYAN, width=2), fill="tozeroy",
            fillcolor="rgba(0,196,255,0.08)",
            hovertemplate="Top %{x:.1f}% of authors → %{y:.1f}% of content<extra></extra>",
        ))
        fig_pareto.add_shape(type="line", x0=0, x1=100, y0=0, y1=100,
            line=dict(color=BORDER, width=1, dash="dot"))
        lay(fig_pareto, h=300, xt="% of authors (ranked by activity)", yt="% of content produced")
        fig_pareto.update_layout(xaxis=dict(ticksuffix="%"), yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig_pareto, use_container_width=True, config=cfg, theme=None)

    with col_eng:
        st.markdown("#### Engagement depth")
        fig_eng = go.Figure(go.Bar(
            x=engagement["activity_bucket"].astype(str),
            y=engagement["n_authors"],
            marker_color=[CYAN, GREEN, AMBER, ACCENT, "#FF6B6B"],
            marker_line_width=0,
            hovertemplate="%{x}<br>%{y:,} authors<extra></extra>",
            text=engagement["n_authors"].apply(lambda v: f"{v:,}"),
            textposition="outside", textfont=dict(color=TXT, size=10),
        ))
        fig_eng.update_layout(
            height=280, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
            margin=dict(l=0, r=0, t=12, b=40),
            xaxis=dict(showgrid=False, tickfont=dict(size=10, color=TXT)),
            yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED)),
        )
        st.plotly_chart(fig_eng, use_container_width=True, config=cfg, theme=None)

        st.markdown("#### Unique authors by subreddit")
        fig_sub = go.Figure(go.Bar(
            x=sub_breakdown["subreddit"], y=sub_breakdown["unique_authors"],
            marker_color=[SUB_COLORS.get(s, CYAN) for s in sub_breakdown["subreddit"]],
            marker_line_width=0,
            hovertemplate="%{x}<br>%{y:,} unique authors<extra></extra>",
            text=sub_breakdown["unique_authors"].apply(lambda v: f"{v:,}"),
            textposition="outside", textfont=dict(color=TXT, size=10),
        ))
        fig_sub.update_layout(
            height=240, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
            margin=dict(l=0, r=0, t=12, b=0),
            xaxis=dict(showgrid=False, tickfont=dict(size=10, color=TXT)),
            yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED)),
        )
        st.plotly_chart(fig_sub, use_container_width=True, config=cfg, theme=None)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 4: Author score distribution + tenure distribution ────────────────
    col_sc, col_ten = st.columns(2, gap="large")

    with col_sc:
        st.markdown("#### Population insights")
        _one_post = int((author_stats["n_total"] == 1).sum())
        _one_post_pct = _one_post / n_authors if n_authors else 0
        _long_tenure = int((author_stats["tenure_days"] >= 365).sum())
        _cross_sub = int((author_stats["n_subreddits"] >= 2).sum())
        _cross_pct = _cross_sub / n_authors if n_authors else 0
        st.markdown(
            f'<div style="padding:1rem;background:{SURF2};border-radius:6px;border-left:3px solid {AMBER};margin-bottom:1rem;">'
            f'<p style="color:{MUTED};font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin:0 0 0.75rem;">Transient vs Core</p>'
            f'<p style="color:{TXT};font-size:0.82rem;margin:0 0 0.5rem;">'
            f'<span style="color:{AMBER};font-weight:700;">{_one_post_pct:.0%}</span> of authors ({_one_post:,}) posted only once — '
            f'NS discourse has a large transient population alongside a committed core of '
            f'<span style="color:{CYAN};font-weight:700;">{_long_tenure:,}</span> long-tenure users (1yr+).</p>'
            f'</div>'
            f'<div style="padding:1rem;background:{SURF2};border-radius:6px;border-left:3px solid {ACCENT};margin-bottom:1rem;">'
            f'<p style="color:{MUTED};font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin:0 0 0.75rem;">Subreddit Siloing</p>'
            f'<p style="color:{TXT};font-size:0.82rem;margin:0 0 0.5rem;">'
            f'<span style="color:{ACCENT};font-weight:700;">{_single_sub_pct:.0%}</span> of authors post in only one subreddit — '
            f'the three NS communities are largely separate audiences. Only '
            f'<span style="color:{CYAN};font-weight:700;">{_cross_pct:.0%}</span> ({_cross_sub:,}) cross-post.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_ten:
        st.markdown("#### Author tenure distribution (days between first & last post)")
        ten = author_stats[author_stats["tenure_days"] > 0]["tenure_days"]
        ten_bins = pd.cut(ten, bins=[0,7,30,90,365,730,9999],
            labels=["<1w","1w–1m","1–3m","3m–1y","1–2y","2y+"])
        ten_counts = ten_bins.value_counts().sort_index().reset_index()
        ten_counts.columns = ["bucket","authors"]
        fig_ten = go.Figure(go.Bar(
            x=ten_counts["bucket"].astype(str), y=ten_counts["authors"],
            marker_color=AMBER, marker_line_width=0,
            hovertemplate="%{x}<br>%{y:,} authors<extra></extra>",
            text=ten_counts["authors"].apply(lambda v: f"{v:,}"),
            textposition="outside", textfont=dict(color=TXT, size=10),
        ))
        fig_ten.update_layout(
            height=280, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
            margin=dict(l=0, r=0, t=12, b=40),
            xaxis=dict(showgrid=False, tickfont=dict(size=10, color=TXT)),
            yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED)),
        )
        st.plotly_chart(fig_ten, use_container_width=True, config=cfg, theme=None)

        st.markdown("#### Cross-subreddit authors")
        cross = author_stats["n_subreddits"].value_counts().sort_index().reset_index()
        cross.columns = ["subreddits","authors"]
        cross["label"] = cross["subreddits"].map({1:"1 subreddit only",2:"2 subreddits",3:"All 3"})
        fig_cross = go.Figure(go.Bar(
            x=cross["label"], y=cross["authors"],
            marker_color=[CYAN, AMBER, ACCENT], marker_line_width=0,
            hovertemplate="%{x}<br>%{y:,} authors<extra></extra>",
            text=cross["authors"].apply(lambda v: f"{v:,}"),
            textposition="outside", textfont=dict(color=TXT, size=10),
        ))
        fig_cross.update_layout(
            height=220, template=_tpl, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED),
            margin=dict(l=0, r=0, t=12, b=0),
            xaxis=dict(showgrid=False, tickfont=dict(size=10, color=TXT)),
            yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=MUTED)),
        )
        st.plotly_chart(fig_cross, use_container_width=True, config=cfg, theme=None)


# ── Sentinel Bot page ─────────────────────────────────────────────────────────
elif current_page == "Sentinel Bot":
    st.markdown(
        f'<h1 class="page-title">NS Sentinel <span style="color:{CYAN}">Bot</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{MUTED};font-size:13px;margin-bottom:1.5rem">'
        'Ask quantitative or qualitative questions about NS sentiment across 549,671 Reddit documents '
        '(57K posts · 492K comments · r/singapore · r/askSingapore · r/NationalServiceSG · 2018–2025). '
        'Indexed as 737K semantic chunks. Powered by FAISS retrieval + Groq Llama 3.3 synthesis.'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── Initialise session state for chat history ──────────────────────────
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []   # list of {"role": "user"|"assistant", "content": str, "sources": list}
    if "chatbot_error" not in st.session_state:
        st.session_state.chatbot_error = None
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


    # ── Load chatbot (cached) ──────────────────────────────────────────────
    bot = None
    with st.spinner("Loading knowledge base (first visit takes ~15s) …"):
        try:
            bot = load_chatbot()
        except Exception as e:
            st.session_state.chatbot_error = str(e)

    if st.session_state.chatbot_error:
        st.error(f"Failed to load chatbot: {st.session_state.chatbot_error}")
        st.stop()

    # ── Sidebar additions for Chat page ────────────────────────────────────
    with st.sidebar:
        st.markdown('<hr class="nav-divider">', unsafe_allow_html=True)
        st.markdown('<span class="nav-section">Sentinel Bot</span>', unsafe_allow_html=True)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.pending_prompt = None
            st.rerun()
        if st.button("Reload knowledge base", use_container_width=True,
                     help="Reload after new digests/narratives are built"):
            st.cache_resource.clear()
            st.rerun()
        st.markdown(
            f'<p style="font-size:11px;color:{MUTED};margin-top:0.5rem">'
            f'Sources: FAISS 737k chunks · {len(bot.assembler.events if isinstance(bot.assembler.events, list) else bot.assembler.events.get("events", [])):,} events · '
            f'{"✓ Digests" if bot.assembler.digests else "✗ No digests"} · '
            f'{"✓ Narratives" if bot.assembler.narratives else "✗ No narratives"} · '
            f'{"✓ Fact table" if bot.assembler.fact_table is not None else "✗ No fact table"}'
            '</p>',
            unsafe_allow_html=True,
        )

    # ── Example queries ────────────────────────────────────────────────────
    _EXAMPLES = [
        "Explain the sharp drop in public sentiment in January 2019.",
        "What do NSmen say about NS pay?",
        "How did COVID affect NS sentiment in 2020?",
        "Which macro topic had the highest negative sentiment across 2018 to 2025?",
        "What does zao liao culture look like in the data?",
        "Why did critical sentiment spike in 2022 to 2024?",
    ]
    with st.expander("🪖 Sample questions — click to ask", expanded=not st.session_state.chat_messages):
        _cols = st.columns(2)
        for _i, _ex in enumerate(_EXAMPLES):
            if _cols[_i % 2].button(_ex, key=f"ex_{_i}", use_container_width=True):
                st.session_state.pending_prompt = _ex
                st.rerun()

    # ── Render conversation history ────────────────────────────────────────
    _month_names = ["", "Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    def _render_sources(sources):
        if sources:
            with st.expander(f"📎 {len(sources)} sources used", expanded=False):
                for i, src in enumerate(sources, 1):
                    mo = _month_names[src.month] if 1 <= src.month <= 12 else str(src.month)
                    sentiment = (
                        "🔴 negative" if src.sent_neg > 0.5 else
                        "🟢 positive" if src.sent_pos > 0.5 else
                        "⚪ neutral"
                    )
                    st.markdown(
                        f"**[{i}]** `{src.subreddit}` · {mo} {src.year} · "
                        f"↑{src.upvotes} · {src.topic} · {sentiment}  \n"
                        f"> {src.text[:200]}"
                    )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_sources(msg.get("sources", []))

    # ── Chat input ─────────────────────────────────────────────────────────
    from src.rag.chatbot import QuantitativeResult, QualitativeResult
    _typed_prompt = st.chat_input("Ask about NS sentiment, events, topics …")
    prompt = _typed_prompt or st.session_state.pop("pending_prompt", None)

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            result = bot.route(prompt)

            if isinstance(result, QuantitativeResult):
                st.markdown(result.answer)
                sources = []
                full_response = result.answer
            else:
                response_placeholder = st.empty()
                full_response = ""
                try:
                    for token in bot.stream(result):
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                except Exception as e:
                    response_placeholder.error(f"Error during synthesis: {e}")
                    full_response = f"*Error: {e}*"

                sources = result.chunks if result.chunks else []
                _render_sources(sources)

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        })
