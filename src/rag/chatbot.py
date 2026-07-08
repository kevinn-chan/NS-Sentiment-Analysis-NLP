"""
RAG — NSChatbot orchestrator.

Main entry point. Loads all assets at startup, routes queries,
and returns streaming or complete responses.

Usage:
    bot = NSChatbot()
    result = bot.route("What do NSmen say about IPPT?")
    for chunk in bot.stream(result):
        print(chunk, end="", flush=True)
"""

import logging
from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd

from src.rag.config import FACT_TABLE
from src.rag.context_assembler import AssembledContext, ContextAssembler
from src.rag.query_router import QueryFilters, QueryRouter, RoutedQuery
from src.rag.retriever import ChunkResult, Retriever
from src.rag.synthesizer import Synthesizer

log = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class QuantitativeResult:
    type:    str = "quantitative"
    answer:  str = ""
    query:   str = ""

@dataclass
class QualitativeResult:
    type:    str = "qualitative"
    context: AssembledContext = None
    query:   str = ""

    @property
    def chunks(self) -> list[ChunkResult]:
        return self.context.chunks if self.context else []

    @property
    def events(self) -> list[dict]:
        return self.context.events if self.context else []


QueryResult = QuantitativeResult | QualitativeResult


# ── Quantitative handler ──────────────────────────────────────────────────────

class FactTableHandler:

    def __init__(self, fact_table: pd.DataFrame | None):
        self.ft = fact_table

    def handle(self, query: str, filters: QueryFilters) -> str:
        if self.ft is None:
            return (
                "⚠️ The statistical fact table is not yet available — it will be built after "
                "Stage 8 (temporal aggregation) is complete. Quantitative queries are not yet supported."
            )

        df = self.ft.copy()

        # Apply filters — prefer granularity "year" or "year_topic" or "year_sub" over row-level
        # for cleaner aggregates. Use month=0 rows (= annual aggregates) when no month requested.
        if filters.years:
            df = df[df["year"].isin(filters.years)]
        if filters.subreddits:
            df = df[df["subreddit"].isin(filters.subreddits) | (df["subreddit"] == "all")]
        else:
            df = df[df["subreddit"] == "all"]     # default: all subreddits combined

        if filters.topics:
            df = df[df["topic_macro"].isin(filters.topics)]
        else:
            df = df[df["topic_macro"] == "all"]   # default: all topics combined

        # Use annual aggregates (month == 0) unless a specific month was requested
        if not filters.months:
            df = df[df["month"] == 0]

        if df.empty:
            return (
                f"No data found for the specified filters: "
                f"years={filters.years}, topics={filters.topics}, subreddits={filters.subreddits}. "
                "The corpus covers 2018–2025 across r/singapore, r/askSingapore, and r/NationalServiceSG."
            )

        # Choose primary metric column
        metric_col = {
            "negative":   "mean_sent_neg",
            "positive":   "mean_sent_pos",
            "neutral":    "mean_sent_neu",
            "sentiment":  "mean_sent_neg",
            "commitment": "mean_commit_net",
            "committed":  "pct_committed",
            "critical":   "pct_critical",
            "discourse":  "doc_count",
        }.get(filters.metric or "sentiment", "mean_sent_neg")

        if metric_col not in df.columns:
            metric_col = "mean_sent_neg"

        # Sort and build year-over-year table
        yearly = df.sort_values("year").reset_index(drop=True)

        topic_label  = filters.topics[0] if filters.topics else "all topics"
        sr_label     = filters.subreddits[0] if filters.subreddits else "all subreddits"
        metric_label = metric_col.replace("mean_sent_", "").replace("mean_commit_", "commitment ").title()

        lines = [f"**{metric_label} — {topic_label} — {sr_label}**\n"]

        for pos, row in enumerate(yearly.itertuples(index=False)):
            yr  = int(row.year)
            val = float(getattr(row, metric_col, float("nan")))
            n   = int(getattr(row, "doc_count", 0))
            if pos > 0:
                prev_val = float(getattr(yearly.iloc[pos - 1], metric_col, float("nan")))
                delta = val - prev_val
                pct   = (delta / prev_val * 100) if prev_val != 0 else 0.0
                arrow = "▲" if delta > 0 else "▼"
                lines.append(
                    f"  {yr}: {val:.3f}  {arrow}{abs(delta):.3f} ({abs(pct):.1f}% vs {yr-1})  "
                    f"[{n:,} docs]"
                )
            else:
                lines.append(f"  {yr}: {val:.3f}  [{n:,} docs]")

        lines.append("\n*Source: pre-aggregated fact table. Based on SingBERT v7 (κ=0.602).*")
        return "\n".join(lines)


# ── Main chatbot ──────────────────────────────────────────────────────────────

class NSChatbot:
    """
    Main entry point for the NS Sentiment RAG chatbot.

    Loads all assets at startup (~15s), then handles queries in <5s.
    Designed to be instantiated once at Streamlit dashboard start.
    """

    def __init__(self):
        log.info("Initialising NSChatbot …")

        self.router    = QueryRouter()
        self.retriever = Retriever()
        self.assembler = ContextAssembler()
        self.synthesizer = Synthesizer()

        # Fact table for quantitative handler
        ft = self.assembler.fact_table
        self.fact_handler = FactTableHandler(ft)

        log.info("NSChatbot ready ✓")

    def route(self, query: str) -> QueryResult:
        """
        Route query → return a QuantitativeResult or QualitativeResult.
        For streaming, call .stream(result) after this.
        """
        routed: RoutedQuery = self.router.route(query)
        log.info(f"Query intent: {routed.intent}  filters: {routed.filters}")

        f = routed.filters
        # Rich-context intents always go through the full assembler + LLM path,
        # even if the router classified them as "quantitative".
        rich_intent = (
            f.compare_subreddits
            or f.compare_topics
            or bool(f.ranking_dim)
            or f.ask_methodology
            or f.commitment_breakdown
        )

        if routed.intent == "quantitative" and not rich_intent:
            answer = self.fact_handler.handle(query, routed.filters)
            return QuantitativeResult(answer=answer, query=query)

        else:
            chunks = self.retriever.retrieve(query, routed.filters)
            log.info(f"Retrieved {len(chunks)} chunks")
            context = self.assembler.assemble(query, routed.filters, chunks)
            log.info(f"Context assembled: {context.token_est} tokens est.")
            return QualitativeResult(context=context, query=query)

    def stream(self, result: QueryResult) -> Iterator[str]:
        """
        Stream LLM synthesis for a QualitativeResult.
        For QuantitativeResult, yields the pre-computed answer string directly.
        """
        if isinstance(result, QuantitativeResult):
            yield result.answer
            return

        yield from self.synthesizer.stream(result.context.text)

    def answer(self, query: str) -> tuple[str, list[ChunkResult]]:
        """
        Non-streaming convenience method for testing.
        Returns (full_text, chunks).
        """
        result = self.route(query)
        text = "".join(self.stream(result))
        chunks = result.chunks if isinstance(result, QualitativeResult) else []
        return text, chunks
