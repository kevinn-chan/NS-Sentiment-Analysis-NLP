"""
Side-by-side dendrogram figure:
  Left  — BERTopic organic:  Ward's linkage on topic embeddings (cosine), macro-coloured
  Right — Manual taxonomy:   fixed level bands from hierarchical_topics_manual.parquet

Both panels share the same macro colour palette.
Output: data/processed/new/dendrogram_manual.html
"""

import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram as sp_dendrogram, linkage
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.topic_labels import TOPIC_LABELS, MACRO_CATEGORIES

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ── Shared config ──────────────────────────────────────────────────────────────
topic_ids  = sorted(TOPIC_LABELS.keys())
n          = len(topic_ids)
tid_to_idx = {tid: i for i, tid in enumerate(topic_ids)}

PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#469990', '#9A6324',
    '#800000', '#aaffc3', '#000075', '#a9a9a9', '#808000',
    '#ffd8b1', '#ffe119',
]
macro_color_map = {m: PALETTE[i] for i, m in enumerate(MACRO_CATEGORIES)}
ROOT_COLOR      = 'rgba(180,180,180,0.5)'

idx_to_macro = {tid_to_idx[tid]: info['macro'] for tid, info in TOPIC_LABELS.items()}

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — BERTopic organic (Ward's linkage on topic embeddings)
# ══════════════════════════════════════════════════════════════════════════════
print('Loading model embeddings …')
from bertopic import BERTopic
topic_model = BERTopic.load(
    os.path.join(ROOT, 'models/new/bertopic_fine'), embedding_model=None
)
# topic_embeddings_: index 0 = outlier (-1), index tid+1 = topic tid
raw_embs = topic_model.topic_embeddings_
embs = np.array([raw_embs[tid + 1] for tid in topic_ids])   # (359, 768)

# Ward's linkage on L2-normalised embeddings (≡ cosine distance for Ward)
print('Computing Ward linkage on embeddings …')
embs_norm = normalize(embs)
Z_bert = linkage(embs_norm, method='ward', metric='euclidean')

# Macro lookup for internal nodes
scipy_to_macro_bert = dict(idx_to_macro)
node_count_bert     = {i: 1 for i in range(n)}
next_s = n
for row in Z_bert:
    l, r = int(row[0]), int(row[1])
    lm, rm = scipy_to_macro_bert.get(l), scipy_to_macro_bert.get(r)
    scipy_to_macro_bert[next_s] = lm if lm == rm else None
    node_count_bert[next_s] = node_count_bert[l] + node_count_bert[r]
    next_s += 1

def link_color_bert(k):
    m = scipy_to_macro_bert.get(k)
    return macro_color_map[m] if m else ROOT_COLOR

labels = [TOPIC_LABELS[tid]['name'] for tid in topic_ids]
print('Running scipy dendrogram (BERTopic) …')
dn_bert = sp_dendrogram(Z_bert, labels=labels, no_plot=True,
                         link_color_func=link_color_bert)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — Manual taxonomy (fixed level bands + embedding distances within band)
# ══════════════════════════════════════════════════════════════════════════════
print('Building manual taxonomy linkage …')
manual_ht        = pd.read_parquet(
    os.path.join(ROOT, 'data/processed/new/hierarchical_topics_manual.parquet')
)
manual_ht_sorted = manual_ht.sort_values('Parent_ID').reset_index(drop=True)

BAND = {1.0: (0.05, 0.92), 2.0: (1.05, 1.92),
        3.0: (2.05, 2.92), 4.0: (3.05, 3.92)}

node_to_scipy_man  = dict(tid_to_idx)
scipy_to_macro_man = dict(idx_to_macro)
node_emb_man       = {tid_to_idx[tid]: raw_embs[tid + 1] for tid in topic_ids}
node_count_man     = {tid_to_idx[tid]: 1 for tid in topic_ids}

raw_by_level = {1.0: [], 2.0: [], 3.0: [], 4.0: []}
stage1       = []
next_s = n

for _, row in manual_ht_sorted.iterrows():
    left_id, right_id = int(row['Child_Left_ID']), int(row['Child_Right_ID'])
    parent_id = int(row['Parent_ID'])
    level     = float(row['Distance'])
    count     = int(len(row['Topics']))

    ls = node_to_scipy_man[left_id]
    rs = node_to_scipy_man[right_id]

    el, er   = node_emb_man[ls], node_emb_man[rs]
    cos_d    = float(max(0.0, 1.0 - cosine_similarity([el], [er])[0, 0]))
    raw_by_level[level].append((next_s, cos_d))
    stage1.append([ls, rs, level, count, next_s, cos_d])

    cl, cr = node_count_man[ls], node_count_man[rs]
    node_emb_man[next_s]   = (el * cl + er * cr) / (cl + cr)
    node_count_man[next_s] = cl + cr

    lm = scipy_to_macro_man.get(ls)
    rm = scipy_to_macro_man.get(rs)
    scipy_to_macro_man[next_s] = lm if lm == rm else None

    node_to_scipy_man[parent_id] = next_s
    next_s += 1

def normalise(v, smin, smax, lo, hi):
    return (lo + hi) / 2 if smax == smin else lo + (v - smin) / (smax - smin) * (hi - lo)

level_stats = {lv: (min(d for _, d in ents), max(d for _, d in ents))
               for lv, ents in raw_by_level.items()}
norm_dist = {}
for lv, ents in raw_by_level.items():
    lo, hi = BAND[lv]
    smin, smax = level_stats[lv]
    for ps, rd in ents:
        norm_dist[ps] = normalise(rd, smin, smax, lo, hi)

linkage_man  = []
child_max    = {}
for row in stage1:
    ls, rs, lv, cnt, ps, _ = row
    d  = norm_dist[ps]
    d  = max(d, child_max.get(ls, 0.0) + 1e-6, child_max.get(rs, 0.0) + 1e-6)
    child_max[ps] = d
    linkage_man.append([ls, rs, d, cnt])

Z_man = np.array(linkage_man)

def link_color_man(k):
    m = scipy_to_macro_man.get(k)
    return macro_color_map[m] if m else ROOT_COLOR

print('Running scipy dendrogram (manual) …')
dn_man = sp_dendrogram(Z_man, labels=labels, no_plot=True,
                        link_color_func=link_color_man)

# ══════════════════════════════════════════════════════════════════════════════
# BUILD FIGURES
# ══════════════════════════════════════════════════════════════════════════════
H = max(6000, n * 20)

def build_fig(dn, title, xaxis_opts):
    traces = []
    for icoord, dcoord, color in zip(dn['icoord'], dn['dcoord'], dn['color_list']):
        traces.append(go.Scatter(
            x=dcoord, y=icoord,
            mode='lines',
            line=dict(color=color, width=1.0),
            hoverinfo='none',
            showlegend=False,
        ))
    for macro in MACRO_CATEGORIES:
        traces.append(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color=macro_color_map[macro], width=4),
            name=macro, showlegend=True,
        ))
    traces.append(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color=ROOT_COLOR, width=4),
        name='Cross-macro', showlegend=True,
    ))
    leaf_y = [5 + 10 * i for i in range(len(dn['ivl']))]
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text=title, x=0.5),
            showlegend=True,
            legend=dict(x=1.01, y=1, font=dict(size=10)),
            width=1800,
            height=H,
            margin=dict(l=320, r=220, t=80, b=50),
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.4)',
                       zeroline=False, **xaxis_opts),
            yaxis=dict(tickmode='array', tickvals=leaf_y, ticktext=dn['ivl'],
                       tickfont=dict(size=8), showgrid=False, zeroline=False),
        )
    )
    return fig

# ── BERTopic dendrogram ────────────────────────────────────────────────────────
print('Saving dendrogram_bertopic.html …')
fig_bert = build_fig(
    dn_bert,
    title='<b>NS Sentiment — BERTopic Embedding Clustering</b>',
    xaxis_opts=dict(title='Cosine distance (Ward)'),
)
fig_bert.write_html(os.path.join(ROOT, 'data/processed/new/dendrogram_bertopic.html'))
print('  done')

# ── Manual taxonomy dendrogram ────────────────────────────────────────────────
print('Saving dendrogram_manual.html …')
fig_man = build_fig(
    dn_man,
    title='<b>NS Sentiment — Manual Hierarchical Taxonomy</b>',
    xaxis_opts=dict(
        title='Taxonomy Level',
        tickmode='array',
        tickvals=[0, 1, 2, 3, 4],
        ticktext=['Fine Topic', 'Sub-sub', 'Sub', 'Macro', 'National Service'],
        range=[-0.05, 4.1],
    ),
)
fig_man.write_html(os.path.join(ROOT, 'data/processed/new/dendrogram_manual.html'))
print('  done')
