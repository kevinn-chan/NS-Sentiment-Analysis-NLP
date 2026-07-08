"""
RAG — Context Assembler.

Assembles the bounded context window (<= MAX_CONTEXT_CHARS) from:
  1. Statistical facts from the fact table (if available)
  2. NS events overlapping the queried period
  3. Topic digest(s) for matched topics
  4. Temporal narratives for matched months
  5. Retrieved chunks with metadata
  6. User query (always last)

Priority: items 1-2 are never dropped; 3-6 are dropped from the bottom
up if the budget is exceeded.
"""

import calendar
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.rag.config import (
    NS_EVENTS, TOPIC_DIGESTS, TEMPORAL_NARRATIVES, FACT_TABLE,
    MAX_CONTEXT_CHARS, MAX_CHUNK_TEXT_LEN,
)
from src.rag.query_router import QueryFilters
from src.rag.retriever import ChunkResult
from src.rag.spike_detector import SpikeDetector

log = logging.getLogger(__name__)

MONTH_ABBR = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class AssembledContext:
    text:       str
    chunks:     list[ChunkResult]
    events:     list[dict]
    has_facts:  bool
    token_est:  int   # rough estimate: len(text) / 4


class ContextAssembler:

    def __init__(self):
        self.events       = self._load_json(NS_EVENTS,           "ns_events")
        self.digests      = self._load_json(TOPIC_DIGESTS,       "topic_digests")
        self.narratives   = self._load_json(TEMPORAL_NARRATIVES, "temporal_narratives")
        self.fact_table   = self._load_parquet(FACT_TABLE,       "fact_table")
        self.spike_detector = SpikeDetector()

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_json(self, path: Path, name: str) -> dict | list:
        if not path.exists():
            log.warning(f"  {name} not found at {path} — skipping")
            return {} if name != "ns_events" else {"events": []}
        data = json.load(open(path, encoding="utf-8"))
        log.info(f"  Loaded {name}: {path.name}")
        return data

    def _load_parquet(self, path: Path, name: str) -> pd.DataFrame | None:
        if not path.exists():
            log.warning(f"  {name} not found at {path} — quantitative queries will be limited")
            return None
        df = pd.read_parquet(path)
        log.info(f"  Loaded {name}: {len(df):,} rows")
        return df

    # ── Event matching ────────────────────────────────────────────────────────

    def _matching_events(self, filters: QueryFilters) -> list[dict]:
        """Return events whose date range overlaps the queried period."""
        events_list = self.events if isinstance(self.events, list) else self.events.get("events", [])
        if not filters.years:
            return []

        matched = []
        seen_ids = set()
        for event in events_list:
            if event["id"] in seen_ids:
                continue
            try:
                e_start = pd.Timestamp(event["date_start"])
                e_end   = pd.Timestamp(event["date_end"])
            except Exception:
                continue

            for year in filters.years:
                months = filters.months if filters.months else list(range(1, 13))
                for month in months:
                    last_day = calendar.monthrange(year, month)[1]
                    p_start  = pd.Timestamp(year=year, month=month, day=1)
                    p_end    = pd.Timestamp(year=year, month=month, day=last_day)
                    if e_start <= p_end and e_end >= p_start:
                        matched.append(event)
                        seen_ids.add(event["id"])
                        break

        # For broad queries (multi-year or no specific month), allow more events
        max_events = 6 if len(filters.years) > 2 else 3
        return matched[:max_events]

    def _longitudinal_timeline(self, filters: QueryFilters) -> str:
        """
        Build a spike/dip timeline for longitudinal queries.
        Only injected when:
          - Query spans 2+ years, OR
          - Query has no year filter (full-corpus question), OR
          - Query explicitly asks about trends/changes over time.

        Returns empty string if conditions not met or no anomalies found.
        """
        years = filters.years
        is_longitudinal = (
            not years                        # no filter → full dataset
            or len(years) >= 2              # multi-year
        )
        if not is_longitudinal:
            return ""

        if years:
            year_from = min(years)
            year_to   = max(years)
        else:
            year_from = 2018
            year_to   = 2025

        # Budget: longitudinal timeline gets up to 4000 chars
        # (MAX_CONTEXT_CHARS is 10000; stats ~3500; events ~600; query ~200; rest for timeline+chunks)
        return self.spike_detector.format_timeline(
            year_from=year_from,
            year_to=year_to,
            max_chars=4000,
        )

    # ── Methodology static block ──────────────────────────────────────────────

    METHODOLOGY_TEXT = """## DATASET METHODOLOGY
Dataset: 737,274 Reddit chunks from r/singapore, r/askSingapore, r/NationalServiceSG (2018–2025).
A "chunk" = one post or comment, split at ~512 tokens if longer.

SENTIMENT SCORING (3 continuous scores per chunk, sum to 1.0):
  • sent_neg / sent_pos / sent_neu — produced by a fine-tuned transformer classifier
    trained on Singapore-English Reddit text (SingBERT-based).
  • pct_neg = % of chunks where sent_neg is the argmax (majority-negative chunks).

COMMITMENT SCORING (3 continuous scores per chunk, sum to 1.0):
  • commit_support — probability chunk expresses personal buy-in / pro-NS stance.
  • commit_critical — probability chunk expresses institutional opposition / anti-NS stance.
  • commit_neutral  — probability chunk is apathetic / uncommitted.
  • mean_commit_net = mean(commit_support) − mean(commit_critical) per group.
    Range: −1.0 (fully critical) to +1.0 (fully committed). Near 0 = neutral/apathetic.
  • pct_committed / pct_critical / pct_neutral_commit — % of chunks where that class is argmax.

IMPORTANT CAVEATS:
  • Scores are model predictions, not ground truth. Validated on annotation sample only.
  • "Critical" ≠ "negative sentiment" — a committed person can complain (negative sentiment, high support).
  • Upvote-weighted scores (wtd_*) weight each chunk by log(1 + upvotes) to surface high-engagement posts.
  • Low-volume cells (< 30 docs) have high variance; treat with caution."""

    # ── Fact table query helpers ──────────────────────────────────────────────

    STAT_MAP = [
        ("mean_sent_neg",          "Mean negative sentiment"),
        ("mean_sent_pos",          "Mean positive sentiment"),
        ("pct_neg",                "% chunks majority-negative"),
        ("pct_pos",                "% chunks majority-positive"),
        ("mean_commit_net",        "Net commitment (support − critical)"),
        ("pct_committed",          "% chunks classified committed"),
        ("pct_critical",           "% chunks classified critical"),
        ("pct_neutral_commit",     "% chunks classified neutral/apathetic"),
        ("mean_commit_support",    "Mean commitment support score"),
        ("mean_commit_critical",   "Mean commitment critical score"),
        ("wtd_sent_neg",           "Upvote-weighted negative sentiment"),
        ("wtd_sent_pos",           "Upvote-weighted positive sentiment"),
        # Upvote-weighted buyin axis (Stage 5b LLM labels)
        ("wtd_buyin_committed",    "Upvote-weighted % committed (buyin)"),
        ("wtd_buyin_uncommitted",  "Upvote-weighted % uncommitted (buyin)"),
        ("net_buyin",              "Net buyin (committed − uncommitted, raw count)"),
        # Upvote-weighted stance axis
        ("wtd_stance_supportive",  "Upvote-weighted % supportive (stance)"),
        ("wtd_stance_critical",    "Upvote-weighted % critical (stance)"),
        ("net_stance",             "Net stance (supportive − critical, raw count)"),
        # Combined disposition
        ("wtd_positive",           "Upvote-weighted % positive (committed OR supportive)"),
        ("wtd_negative",           "Upvote-weighted % negative (uncommitted OR critical)"),
        ("net_disposition",        "Net community disposition (wtd_positive − wtd_negative)"),
    ]

    def _fmt_row(self, row: pd.Series, indent: str = "  ") -> list[str]:
        """Format a fact-table row as label: value lines."""
        lines = []
        for col, label in self.STAT_MAP:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    if col.startswith("pct_"):
                        lines.append(f"{indent}{label}: {val:.1%}")
                    else:
                        lines.append(f"{indent}{label}: {val:.3f}")
        if "doc_count" in row.index and pd.notna(row["doc_count"]):
            lines.append(f"{indent}Total docs: {int(row['doc_count']):,}")
        return lines

    def _query_facts(self, filters: QueryFilters) -> str:
        if self.fact_table is None:
            return ""

        # ── Commitment class breakdown: pct_committed / pct_critical / pct_neutral ──
        if filters.commitment_breakdown:
            return self._facts_commitment_breakdown(filters)

        # ── Subreddit comparison: side-by-side all 3 subreddits ──────────────
        if filters.compare_subreddits:
            return self._facts_subreddit_comparison(filters)

        # ── Topic cross-comparison: side-by-side 2 topics ────────────────────
        if filters.compare_topics and len(filters.topics) >= 2:
            return self._facts_topic_comparison(filters)

        # ── Ranking: sort subreddits/topics/years by a metric ────────────────
        if filters.ranking_dim:
            return self._facts_ranking(filters)

        # ── Fine-grained: specific month + topic → use month_sub_topic rows ──
        if filters.months and filters.topics:
            return self._facts_fine_grained(filters)

        # ── Default: annual aggregate rows ───────────────────────────────────
        return self._facts_default(filters)

    def _facts_commitment_breakdown(self, filters: QueryFilters) -> str:
        """Year-by-year breakdown of pct_committed / pct_critical / pct_neutral_commit."""
        df = self.fact_table.copy()
        df = df[(df["month"] == 0) & (df["subreddit"] == "all") & (df["topic_macro"] == "all")]

        if filters.years:
            df = df[df["year"].isin(filters.years)]

        if df.empty:
            return ""

        years_label = ", ".join(str(y) for y in sorted(filters.years)) if filters.years else "2018–2025"
        lines = [
            f"## STATISTICAL DATA — COMMITMENT CLASS BREAKDOWN ({years_label})",
            "Note: each chunk classified by argmax of (commit_support, commit_critical, commit_neutral).",
            "",
        ]

        df_sorted = df.sort_values("year")
        for _, row in df_sorted.iterrows():
            yr = int(row["year"])
            n  = int(row.get("doc_count", 0))
            pct_c  = row.get("pct_committed",      float("nan"))
            pct_cr = row.get("pct_critical",        float("nan"))
            pct_n  = row.get("pct_neutral_commit",  float("nan"))
            net    = row.get("mean_commit_net",      float("nan"))
            lines.append(
                f"  {yr} [{n:,} docs]:  "
                f"committed={pct_c:.1%}  critical={pct_cr:.1%}  neutral/apathetic={pct_n:.1%}  "
                f"(net={net:.3f})"
            )

        return "\n".join(lines)

    def _facts_default(self, filters: QueryFilters) -> str:
        """Standard year-level stats for the queried period."""
        df = self.fact_table.copy()
        df = df[df["month"] == 0]  # annual aggregates

        if filters.years:
            df = df[df["year"].isin(filters.years)]
        if filters.subreddits:
            df = df[df["subreddit"].isin(filters.subreddits)]
        else:
            df = df[df["subreddit"] == "all"]
        if filters.topics:
            df = df[df["topic_macro"].isin(filters.topics[:1])]
        else:
            df = df[df["topic_macro"] == "all"]

        if df.empty:
            return ""

        years_label = ", ".join(str(y) for y in sorted(filters.years)) if filters.years else "2018–2025"
        topic_label = filters.topics[0] if filters.topics else "all topics"
        sr_label    = filters.subreddits[0] if filters.subreddits else "all subreddits"
        lines = [f"## STATISTICAL DATA\nPre-computed statistics ({years_label} · {topic_label} · {sr_label}):"]

        df_sorted = df.sort_values("year")
        if len(df_sorted) > 1:
            for _, row in df_sorted.iterrows():
                yr = int(row["year"])
                n  = int(row.get("doc_count", 0))
                lines.append(f"\n  {yr} [{n:,} docs]:")
                lines.extend(self._fmt_row(row, indent="    "))
        else:
            row = df_sorted.iloc[0]
            lines.extend(self._fmt_row(row, indent="  "))

        return "\n".join(lines)

    def _facts_subreddit_comparison(self, filters: QueryFilters) -> str:
        """Side-by-side stats for all 3 subreddits."""
        df = self.fact_table.copy()
        df = df[(df["month"] == 0) & (df["topic_macro"] == "all")]

        if filters.years:
            df = df[df["year"].isin(filters.years)]

        # Get rows for each subreddit
        ALL_SUBREDDITS = ["singapore", "askSingapore", "NationalServiceSG"]
        df = df[df["subreddit"].isin(ALL_SUBREDDITS)]
        if df.empty:
            return ""

        years_label = ", ".join(str(y) for y in sorted(filters.years)) if filters.years else "2018–2025"
        lines = [f"## STATISTICAL DATA — SUBREDDIT COMPARISON ({years_label})"]

        for sr in ALL_SUBREDDITS:
            sr_rows = df[df["subreddit"] == sr].sort_values("year")
            if sr_rows.empty:
                continue
            # Weighted aggregate across queried years
            if len(sr_rows) > 1:
                total_docs = sr_rows["doc_count"].sum()
                # weighted mean for each stat
                agg = {}
                for col, _ in self.STAT_MAP:
                    if col in sr_rows.columns and col != "doc_count":
                        w = sr_rows["doc_count"].fillna(0)
                        agg[col] = (sr_rows[col].fillna(0) * w).sum() / total_docs if total_docs > 0 else float("nan")
                agg["doc_count"] = total_docs
                row = pd.Series(agg)
            else:
                row = sr_rows.iloc[0]

            lines.append(f"\n  r/{sr} [{int(row.get('doc_count',0)):,} docs]:")
            lines.extend(self._fmt_row(row, indent="    "))

        return "\n".join(lines)

    def _facts_topic_comparison(self, filters: QueryFilters) -> str:
        """Side-by-side stats for 2 topics."""
        df = self.fact_table.copy()

        # Use year_topic granularity
        df = df[(df["granularity"] == "year_topic") & (df["subreddit"] == "all")]
        if filters.years:
            df = df[df["year"].isin(filters.years)]

        years_label = ", ".join(str(y) for y in sorted(filters.years)) if filters.years else "2018–2025"
        lines = [f"## STATISTICAL DATA — TOPIC COMPARISON ({years_label})"]

        for topic in filters.topics[:2]:
            t_rows = df[df["topic_macro"] == topic].sort_values("year")
            if t_rows.empty:
                continue
            if len(t_rows) > 1:
                total_docs = t_rows["doc_count"].sum()
                agg = {}
                for col, _ in self.STAT_MAP:
                    if col in t_rows.columns and col != "doc_count":
                        w = t_rows["doc_count"].fillna(0)
                        agg[col] = (t_rows[col].fillna(0) * w).sum() / total_docs if total_docs > 0 else float("nan")
                agg["doc_count"] = total_docs
                row = pd.Series(agg)
            else:
                row = t_rows.iloc[0]

            lines.append(f"\n  {topic} [{int(row.get('doc_count',0)):,} docs]:")
            lines.extend(self._fmt_row(row, indent="    "))

        return "\n".join(lines)

    def _facts_ranking(self, filters: QueryFilters) -> str:
        """Rank subreddits / topics / years by requested metric."""
        df = self.fact_table.copy()
        dim = filters.ranking_dim
        direction = filters.ranking_dir  # "highest" | "lowest"

        # Pick metric column to sort by
        metric = filters.metric
        col_map = {
            "negative":    "mean_sent_neg",
            "positive":    "mean_sent_pos",
            "commitment":  "mean_commit_net",
            "sentiment":   "mean_sent_neg",  # default to neg for ranking
            "discourse":   "doc_count",
        }
        sort_col = col_map.get(metric, "mean_sent_neg")
        ascending = (direction == "lowest")

        if dim == "subreddit":
            df = df[(df["month"] == 0) & (df["topic_macro"] == "all") &
                    (df["subreddit"] != "all")]
            if filters.years:
                df = df[df["year"].isin(filters.years)]
            # Aggregate across years per subreddit
            group_col = "subreddit"
        elif dim == "topic":
            df = df[(df["granularity"] == "year_topic") & (df["subreddit"] == "all")]
            if filters.years:
                df = df[df["year"].isin(filters.years)]
            group_col = "topic_macro"
        else:  # year
            df = df[(df["month"] == 0) & (df["subreddit"] == "all") &
                    (df["topic_macro"] == "all")]
            group_col = "year"

        if df.empty:
            return ""

        # Weighted aggregate per dimension value
        ranked = []
        for key, grp in df.groupby(group_col):
            total_docs = grp["doc_count"].sum()
            if sort_col in grp.columns:
                w = grp["doc_count"].fillna(0)
                val = (grp[sort_col].fillna(0) * w).sum() / total_docs if total_docs > 0 else float("nan")
            else:
                val = float("nan")
            ranked.append({"key": key, sort_col: val, "doc_count": total_docs})

        ranked_df = pd.DataFrame(ranked).dropna(subset=[sort_col])
        ranked_df = ranked_df.sort_values(sort_col, ascending=ascending)

        years_label = ", ".join(str(y) for y in sorted(filters.years)) if filters.years else "2018–2025"
        dir_label = "lowest → highest" if ascending else "highest → lowest"
        lines = [
            f"## STATISTICAL DATA — {dim.upper()} RANKING by {sort_col} ({years_label}, {dir_label})"
        ]
        for i, (_, row) in enumerate(ranked_df.iterrows(), 1):
            lines.append(f"  #{i} {row['key']}: {sort_col}={row[sort_col]:.3f}  [{int(row['doc_count']):,} docs]")

        return "\n".join(lines)

    def _facts_fine_grained(self, filters: QueryFilters) -> str:
        """Month + topic specific lookup using month_sub_topic granularity."""
        df = self.fact_table.copy()
        df = df[df["granularity"] == "month_sub_topic"]

        if filters.years:
            df = df[df["year"].isin(filters.years)]
        if filters.months:
            df = df[df["month"].isin(filters.months)]
        if filters.topics:
            df = df[df["topic_macro"].isin(filters.topics[:1])]
        if filters.subreddits:
            df = df[df["subreddit"].isin(filters.subreddits)]
        else:
            # Use 'all' subreddit rows if available, else whatever is there
            df_all = df[df["subreddit"] == "all"]
            if not df_all.empty:
                df = df_all

        if df.empty:
            return self._facts_default(filters)  # fallback

        years_label = ", ".join(str(y) for y in sorted(filters.years)) if filters.years else "2018–2025"
        month_names = [MONTH_ABBR[m] for m in filters.months if 1 <= m <= 12]
        month_label = ", ".join(month_names) if month_names else "all months"
        topic_label = filters.topics[0] if filters.topics else "all topics"

        lines = [f"## STATISTICAL DATA — FINE-GRAINED ({month_label} {years_label} · {topic_label}):"]
        df_sorted = df.sort_values(["year", "month"])
        for _, row in df_sorted.iterrows():
            yr  = int(row["year"])
            mo  = int(row["month"])
            mo_str = MONTH_ABBR[mo] if 1 <= mo <= 12 else str(mo)
            sr  = row.get("subreddit", "all")
            n   = int(row.get("doc_count", 0))
            lines.append(f"\n  {mo_str} {yr} · {sr} [{n:,} docs]:")
            lines.extend(self._fmt_row(row, indent="    "))

        return "\n".join(lines)

    # ── Assemble ──────────────────────────────────────────────────────────────

    def assemble(
        self,
        query: str,
        filters: QueryFilters,
        chunks: list[ChunkResult],
    ) -> AssembledContext:
        """
        Build the bounded context string. Drops parts from the bottom
        (preserving query and facts) if total exceeds MAX_CONTEXT_CHARS.
        """
        parts: list[tuple[str, str]] = []  # (label, text)

        # 0. Methodology block (injected first if user is asking about methodology)
        if filters.ask_methodology:
            parts.append(("METHODOLOGY", self.METHODOLOGY_TEXT))

        # 1. Statistical facts (always included if available)
        # Note: _query_facts() now includes the ## header internally
        facts_text = self._query_facts(filters)
        if facts_text:
            parts.append(("STATS", facts_text))

        # 2. NS events
        matched_events = self._matching_events(filters)
        for event in matched_events:
            desc = event.get("description", "")
            mi   = event.get("metric_impact", {})
            mi_text = ""
            if mi:
                mi_lines = []
                for k, v in mi.items():
                    if isinstance(v, dict):
                        vals = ", ".join(f"{kk}={vv}" for kk, vv in v.items())
                        mi_lines.append(f"  {k}: {vals}")
                    else:
                        mi_lines.append(f"  {k}: {v}")
                mi_text = "\nMetric impact:\n" + "\n".join(mi_lines)
            parts.append((
                "EVENT",
                f"## NS EVENT: {event['title']} ({event['date_start']} → {event['date_end']})\n"
                f"Category: {event.get('category','')}\n"
                f"{desc}{mi_text}"
            ))

        # 2b. Longitudinal timeline (only for multi-year / trend queries)
        timeline_text = self._longitudinal_timeline(filters)
        if timeline_text:
            parts.append(("TIMELINE", timeline_text))

        # 3. Topic digests
        digest_data = self.digests if isinstance(self.digests, dict) else {}
        for topic in filters.topics[:1]:  # max 1 digest
            if topic in digest_data:
                d = digest_data[topic]
                digest_text = (
                    f"## TOPIC CONTEXT: {topic}\n"
                    f"Themes: {d.get('dominant_themes', '')}\n"
                    f"Tone: {d.get('community_tone', '')}\n"
                    f"Concerns: {d.get('common_concerns', '')[:300]}\n"
                    f"Sentiment: {d.get('sentiment_profile', {})}"
                )
                parts.append(("DIGEST", digest_text))

        # 4. Temporal narratives
        narrative_data = self.narratives if isinstance(self.narratives, dict) else {}
        shown_narratives = 0
        if filters.years:
            for year in filters.years:
                months = filters.months if filters.months else [None]
                for month in months:
                    if month is None:
                        # Find most notable month for the year
                        year_keys = [k for k in narrative_data if k.startswith(str(year))]
                        notable = [k for k in year_keys if narrative_data[k].get("notable")]
                        keys_to_show = (notable or year_keys)[:1]
                    else:
                        keys_to_show = [f"{year}-{month:02d}"]

                    for key in keys_to_show:
                        if key in narrative_data and shown_narratives < 2:
                            n = narrative_data[key]
                            parts.append((
                                "NARRATIVE",
                                f"## PERIOD: {key}\n{n.get('narrative', '')}"
                            ))
                            shown_narratives += 1

        # 5. Retrieved chunks
        if chunks:
            chunk_lines = []
            for i, c in enumerate(chunks, 1):
                sentiment_tag = (
                    "negative" if c.sent_neg > 0.5 else
                    "positive" if c.sent_pos > 0.5 else
                    "neutral"
                )
                month_str = MONTH_ABBR[c.month] if 1 <= c.month <= 12 else str(c.month)
                text = c.text[:MAX_CHUNK_TEXT_LEN].replace("\n", " ").strip()
                chunk_lines.append(
                    f"[{i}] {c.subreddit} · {month_str} {c.year} · "
                    f"↑{c.upvotes} · {c.topic} · {sentiment_tag}\n"
                    f'    "{text}"'
                )
            parts.append(("CHUNKS", "## RETRIEVED QUOTES\n" + "\n\n".join(chunk_lines)))

        # 6. User query (always last, never dropped)
        query_part = f"## USER QUESTION\n{query}"

        # Assemble with budget enforcement
        # Fixed parts (never dropped): METHODOLOGY + STATS + EVENTS + query
        # Priority optional: TIMELINE > DIGEST > NARRATIVE > CHUNKS
        fixed_labels = {"METHODOLOGY", "STATS", "EVENT"}
        priority_optional = ["TIMELINE", "DIGEST", "NARRATIVE", "CHUNKS"]

        fixed_text = "\n\n---\n\n".join(t for l, t in parts if l in fixed_labels)
        fixed_text += "\n\n---\n\n" + query_part
        budget_remaining = MAX_CONTEXT_CHARS - len(fixed_text)

        # Sort optional parts by priority order
        priority_idx = {lbl: i for i, lbl in enumerate(priority_optional)}
        optional_parts = sorted(
            [(l, t) for l, t in parts if l not in fixed_labels],
            key=lambda lt: priority_idx.get(lt[0], 99),
        )

        included_optional = []
        for label, text in optional_parts:
            if len(text) <= budget_remaining:
                included_optional.append(text)
                budget_remaining -= len(text) + 12  # separator overhead
            else:
                # Truncate chunks; skip everything else
                if label == "CHUNKS":
                    truncated = text[:budget_remaining - 20]
                    included_optional.append(truncated + "\n[... truncated]")
                break  # drop remaining

        all_parts = (
            [t for l, t in parts if l in fixed_labels]
            + included_optional
            + [query_part]
        )
        final_text = "\n\n---\n\n".join(all_parts)

        return AssembledContext(
            text      = final_text,
            chunks    = chunks,
            events    = matched_events,
            has_facts = bool(facts_text),
            token_est = len(final_text) // 4,
        )
