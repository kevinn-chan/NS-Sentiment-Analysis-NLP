"""
Build a hierarchical_topics dendrogram that exactly mirrors the manual
4-layer taxonomy defined in topic_labels.py.

Layer structure (bottom → top):
  Layer 1 — fine topics          (leaf nodes, distance 0)
  Layer 2 — sub_sub groups       (merge distance 1.0)
  Layer 3 — sub groups           (merge distance 2.0)
  Layer 4 — macro groups         (merge distance 3.0)
  Root                           (merge distance 4.0)

Groups with more than 2 members are reduced to binary nodes via left-spine
chain merging (t0 + t1 → node A, A + t2 → node B, …), which is the standard
approach used by scipy/BERTopic.  The result is a valid BERTopic-compatible
hierarchical_topics DataFrame that can be passed directly to
  topic_model.visualize_hierarchy(hierarchical_topics=df)

Output:
  data/processed/new/hierarchical_topics_manual.parquet
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

# Allow running from repo root or from src/models/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.topic_labels import TOPIC_LABELS, MACRO_CATEGORIES


def build_manual_dendrogram(topic_labels: dict, macro_order: list = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    topic_labels : dict   — the TOPIC_LABELS dict from topic_labels.py
    macro_order  : list   — ordered list of macro names (uses MACRO_CATEGORIES
                            order by default so the tree matches the taxonomy order)

    Returns
    -------
    pd.DataFrame with columns:
        Parent_ID, Parent_Name, Topics,
        Child_Left_ID, Child_Left_Name,
        Child_Right_ID, Child_Right_Name,
        Distance
    """
    # ── 1. Build 4-level hierarchy dict ───────────────────────────────────────
    # macro → sub → sub_sub → [sorted fine topic ids]
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for tid in sorted(topic_labels.keys()):
        info = topic_labels[tid]
        hierarchy[info['macro']][info['sub']][info['sub_sub']].append(tid)

    # ── 2. Node tracking ──────────────────────────────────────────────────────
    # Internal node IDs start after the highest fine topic id
    counter     = [max(topic_labels.keys()) + 1]
    node_topics = {tid: [tid] for tid in topic_labels.keys()}   # id → leaf list
    node_names  = {tid: topic_labels[tid]['name'] for tid in topic_labels.keys()}
    rows        = []

    def new_node(name: str, left_id: int, right_id: int, distance: float) -> int:
        """Create one binary merge node and append a dendrogram row."""
        nid      = counter[0]
        counter[0] += 1
        combined = sorted(node_topics[left_id] + node_topics[right_id])
        node_topics[nid] = combined
        node_names[nid]  = name
        rows.append({
            'Parent_ID':        nid,
            'Parent_Name':      name,
            'Topics':           np.array(combined, dtype=np.int64),
            'Child_Left_ID':    left_id,
            'Child_Left_Name':  node_names[left_id],
            'Child_Right_ID':   right_id,
            'Child_Right_Name': node_names[right_id],
            'Distance':         distance,
        })
        return nid

    def chain(node_ids: list, group_name: str, distance: float) -> int:
        """
        Left-spine binary chain merge of node_ids at the given distance.
        Returns the root node ID of the resulting subtree.
        If only one node, returns it unchanged (no row created).
        """
        if len(node_ids) == 1:
            return node_ids[0]
        current = node_ids[0]
        for right in node_ids[1:]:
            current = new_node(group_name, current, right, distance)
        return current

    # ── 3. Build tree bottom-up ───────────────────────────────────────────────
    macro_iter = macro_order if macro_order else sorted(hierarchy.keys())

    macro_nodes = []
    for macro_name in macro_iter:
        if macro_name not in hierarchy:
            continue

        sub_nodes = []
        for sub_name in sorted(hierarchy[macro_name].keys()):
            subsub_nodes = []
            for subsub_name in sorted(hierarchy[macro_name][sub_name].keys()):
                fine_ids = sorted(hierarchy[macro_name][sub_name][subsub_name])
                # Layer 1 → 2: fine topics merge into sub_sub node
                subsub_root = chain(fine_ids, subsub_name, 1.0)
                subsub_nodes.append(subsub_root)
            # Layer 2 → 3: sub_sub nodes merge into sub node
            sub_root = chain(subsub_nodes, sub_name, 2.0)
            sub_nodes.append(sub_root)
        # Layer 3 → 4: sub nodes merge into macro node
        macro_root = chain(sub_nodes, macro_name, 3.0)
        macro_nodes.append(macro_root)

    # Layer 4 → root: all macro nodes merge into a single root
    chain(macro_nodes, 'National Service', 4.0)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    print('Building manual dendrogram from topic_labels.py …')
    df = build_manual_dendrogram(TOPIC_LABELS, macro_order=MACRO_CATEGORIES)

    out_path = os.path.join(
        os.path.dirname(__file__), '..', '..',
        'data', 'processed', 'new', 'hierarchical_topics_manual.parquet'
    )
    out_path = os.path.normpath(out_path)
    df.to_parquet(out_path, index=False)

    print(f'Saved  → {out_path}')
    print(f'Rows   : {len(df):,}  (= total binary merge operations)')
    print(f'Node ID range: {df["Parent_ID"].min()} – {df["Parent_ID"].max()}')
    print()

    # Quick sanity: root row should contain all 359 fine topics
    root_row = df.loc[df['Distance'] == 4.0].iloc[-1]
    print(f'Root node covers {len(root_row["Topics"])} fine topics '
          f'(expected {len(TOPIC_LABELS)})')

    # Distribution of rows per level
    level_map = {1.0: 'L1→L2 (fine→sub_sub)',
                 2.0: 'L2→L3 (sub_sub→sub)',
                 3.0: 'L3→L4 (sub→macro)',
                 4.0: 'L4→root (macro→root)'}
    print()
    print('Merge rows per level:')
    for dist, label in level_map.items():
        n = (df['Distance'] == dist).sum()
        print(f'  {n:4d}  {label}')
