"""
RAG — Spike Detector.

Identifies months with significant deviations from a rolling baseline for:
  • mean_sent_neg   (negative sentiment)
  • mean_commit_net (net commitment: support − critical)
  • doc_count       (discourse volume)

Z-scores are computed over a ±6-month rolling window (min 4 obs).
Thresholds: |z| >= 1.5 = notable, |z| >= 2.0 = major.

For each spike/dip month, the detector matches it against ns_events.json
events that overlap that month, returning the best-matched cause.

Results are cached on first call (the fact table never changes at runtime).
"""

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.rag.config import FACT_TABLE, NS_EVENTS

log = logging.getLogger(__name__)

SPIKE_THRESHOLD_NOTABLE = 1.5   # |z| >= this → report
SPIKE_THRESHOLD_MAJOR   = 2.0   # |z| >= this → emphasise
WINDOW                  = 6     # months each side for rolling z-score
MIN_PERIODS             = 4     # min obs to compute z-score

METRICS = {
    "mean_sent_neg":   ("Negative sentiment",     "neg"),
    "mean_commit_net": ("Net commitment score",   "net"),
    "doc_count":       ("Discourse volume",       "vol"),
}


@dataclass
class SpikeRecord:
    year_month:  str          # e.g. "2020-04"
    year:        int
    month:       int
    metric:      str          # column name
    metric_label:str          # human label
    z_score:     float        # signed z-score
    direction:   str          # "spike" | "dip"
    value:       float        # raw metric value
    baseline:    float        # rolling mean at that point
    severity:    str          # "notable" | "major"
    event_ids:   list[str]    = field(default_factory=list)
    event_titles:list[str]    = field(default_factory=list)
    event_descriptions: list[str] = field(default_factory=list)


class SpikeDetector:
    """
    Computes rolling z-score anomalies across the full monthly time-series
    and annotates each anomaly with matched NS events.
    """

    def __init__(self):
        self._monthly: pd.DataFrame | None = None
        self._events:  list[dict]          = []
        self._spikes:  list[SpikeRecord]   = []
        self._built = False

    def _load(self):
        if self._built:
            return

        # Load fact table — monthly aggregate rows for "all" subreddit + "all" topic
        if not FACT_TABLE.exists():
            log.warning("Spike detector: fact table not found — longitudinal analysis disabled")
            self._built = True
            return

        df = pd.read_parquet(FACT_TABLE)
        monthly = (
            df[
                (df["granularity"] == "month") &
                (df["subreddit"]   == "all") &
                (df["topic_macro"] == "all")
            ]
            .copy()
            .sort_values(["year", "month"])
            .reset_index(drop=True)
        )
        if monthly.empty:
            log.warning("Spike detector: no monthly 'all/all' rows found")
            self._built = True
            return

        monthly["year_month"] = (
            monthly["year"].astype(str) + "-" +
            monthly["month"].astype(str).str.zfill(2)
        )
        self._monthly = monthly

        # Load events
        if NS_EVENTS.exists():
            raw = json.load(open(NS_EVENTS, encoding="utf-8"))
            self._events = raw if isinstance(raw, list) else raw.get("events", [])

        # Compute z-scores and build spike list
        self._compute_spikes()
        self._built = True

    def _rolling_zscore(self, series: pd.Series) -> pd.Series:
        """
        For each point i, compute z = (x_i − mean) / std over a symmetric
        window of ±WINDOW months, excluding the point itself from the baseline.
        Falls back to a 12-month trailing window when series is short.
        """
        n = len(series)
        zscores = np.full(n, np.nan)
        for i in range(n):
            lo = max(0, i - WINDOW)
            hi = min(n, i + WINDOW + 1)
            window_vals = np.concatenate([series.iloc[lo:i].values, series.iloc[i+1:hi].values])
            window_vals = window_vals[~np.isnan(window_vals)]
            if len(window_vals) >= MIN_PERIODS:
                mu  = window_vals.mean()
                std = window_vals.std(ddof=1)
                zscores[i] = (series.iloc[i] - mu) / std if std > 1e-9 else 0.0
        return pd.Series(zscores, index=series.index)

    def _events_for_month(self, year: int, month: int) -> list[dict]:
        """Return events whose date range overlaps this year-month."""
        matched = []
        p_start = pd.Timestamp(year=year, month=month, day=1)
        p_end   = pd.Timestamp(year=year, month=month, day=28)  # conservative
        for ev in self._events:
            try:
                e_start = pd.Timestamp(ev["date_start"])
                e_end   = pd.Timestamp(ev["date_end"])
            except Exception:
                continue
            if e_start <= p_end and e_end >= p_start:
                matched.append(ev)
        return matched

    def _compute_spikes(self):
        df = self._monthly
        spikes: list[SpikeRecord] = []

        for col, (label, _) in METRICS.items():
            if col not in df.columns:
                continue
            series = df[col].astype(float)
            zs     = self._rolling_zscore(series)

            for i, z in enumerate(zs):
                if np.isnan(z) or abs(z) < SPIKE_THRESHOLD_NOTABLE:
                    continue
                row     = df.iloc[i]
                year    = int(row["year"])
                month   = int(row["month"])
                ym      = row["year_month"]
                val     = float(series.iloc[i])

                # Baseline = mean of the window
                lo = max(0, i - WINDOW)
                hi = min(len(series), i + WINDOW + 1)
                wv = np.concatenate([series.iloc[lo:i].values, series.iloc[i+1:hi].values])
                wv = wv[~np.isnan(wv)]
                baseline = float(wv.mean()) if len(wv) > 0 else val

                # Direction: for sent_neg, positive z = bad (more negative sentiment)
                # For commit_net, positive z = good (more committed)
                # For doc_count, direction is neutral — just "spike"
                if col == "mean_commit_net":
                    direction = "spike" if z > 0 else "dip"
                elif col == "doc_count":
                    direction = "spike" if z > 0 else "dip"
                else:  # sent_neg: higher = more negative = a "spike" in negativity
                    direction = "spike" if z > 0 else "dip"

                severity = "major" if abs(z) >= SPIKE_THRESHOLD_MAJOR else "notable"

                evs = self._events_for_month(year, month)
                rec = SpikeRecord(
                    year_month   = ym,
                    year         = year,
                    month        = month,
                    metric       = col,
                    metric_label = label,
                    z_score      = round(float(z), 2),
                    direction    = direction,
                    value        = round(val, 4),
                    baseline     = round(baseline, 4),
                    severity     = severity,
                    event_ids    = [ev["id"] for ev in evs],
                    event_titles = [ev["title"] for ev in evs],
                    event_descriptions = [
                        ev.get("description", "")[:250] for ev in evs
                    ],
                )
                spikes.append(rec)

        # Sort by year/month then by |z| descending
        self._spikes = sorted(spikes, key=lambda r: (r.year, r.month, -abs(r.z_score)))
        log.info(f"Spike detector: found {len(self._spikes)} anomalies across "
                 f"{len(set(r.year_month for r in self._spikes))} months")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_all(self) -> list[SpikeRecord]:
        self._load()
        return self._spikes

    def get_for_years(self, years: list[int]) -> list[SpikeRecord]:
        """Return spikes/dips for the specified years."""
        self._load()
        return [r for r in self._spikes if r.year in years]

    def get_for_range(self, year_from: int, year_to: int) -> list[SpikeRecord]:
        """Return spikes/dips for a year range (inclusive)."""
        self._load()
        return [r for r in self._spikes if year_from <= r.year <= year_to]

    def format_timeline(
        self,
        years:      list[int] | None = None,
        year_from:  int | None       = None,
        year_to:    int | None       = None,
        max_chars:  int              = 2500,
    ) -> str:
        """
        Render a compact longitudinal timeline of spikes/dips for injection
        into the LLM context window.

        Groups by year, shows top anomalies per month, and includes
        attributed event cause where available.

        Args:
            years:     specific list of years to include
            year_from: start of range (if years not given)
            year_to:   end of range (if years not given)
            max_chars: hard cap on output length

        Returns:
            Formatted multi-line string, empty string if no anomalies.
        """
        self._load()

        if years:
            records = [r for r in self._spikes if r.year in years]
        elif year_from is not None and year_to is not None:
            records = self.get_for_range(year_from, year_to)
        else:
            records = self._spikes

        if not records:
            return ""

        # Deduplicate: keep the highest-|z| record per (year_month, metric)
        seen: dict[tuple, SpikeRecord] = {}
        for r in records:
            key = (r.year_month, r.metric)
            if key not in seen or abs(r.z_score) > abs(seen[key].z_score):
                seen[key] = r

        # Group by year then month
        from collections import defaultdict
        MONTH_NAME = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        by_ym: dict[str, list[SpikeRecord]] = defaultdict(list)
        for r in seen.values():
            by_ym[r.year_month].append(r)

        lines = ["## LONGITUDINAL ANOMALY TIMELINE"]
        lines.append(
            "Months where key metrics deviated significantly from rolling baseline "
            "(|z| ≥ 1.5; window = ±6 months):"
        )

        total_chars = sum(len(l) for l in lines)

        for ym in sorted(by_ym.keys()):
            recs = sorted(by_ym[ym], key=lambda r: -abs(r.z_score))
            year  = recs[0].year
            month = recs[0].month
            month_label = MONTH_NAME[month] if 1 <= month <= 12 else str(month)

            # Build month header
            header = f"\n  {month_label} {year}:"
            metric_lines = []
            for r in recs[:3]:   # max 3 metrics per month to stay compact
                arrow  = "▲" if r.direction == "spike" else "▼"
                sev    = "⚡" if r.severity == "major" else "•"
                detail = f"{r.value:.3f} (z={r.z_score:+.1f}, baseline={r.baseline:.3f})"
                cause  = ""
                if r.event_titles:
                    # Use first event title; truncate long titles
                    title = r.event_titles[0][:60]
                    cause = f" → {title}"
                metric_lines.append(
                    f"    {sev} {arrow} {r.metric_label}: {detail}{cause}"
                )

            block = header + "\n" + "\n".join(metric_lines)
            if total_chars + len(block) > max_chars:
                lines.append("\n  [... additional anomalies truncated for brevity]")
                break
            lines.append(block)
            total_chars += len(block)

        # Append event legend if events referenced — reserve last 20% of budget for it
        all_event_ids  = list({eid for r in seen.values() for eid in r.event_ids})
        legend_budget  = max_chars // 4  # at most 25% of total budget for legend
        if all_event_ids and total_chars < max_chars - 50:
            self._load()
            ev_map = {ev["id"]: ev for ev in self._events}
            legend_lines = ["\n  Key events referenced:"]
            legend_chars = len(legend_lines[0])
            for eid in all_event_ids[:6]:
                ev = ev_map.get(eid)
                if ev:
                    blurb = ev.get("description", "")[:160].replace("\n", " ")
                    entry = f"    [{eid}] {ev['title']} ({ev['date_start']}): {blurb}"
                    if legend_chars + len(entry) > legend_budget:
                        break
                    legend_lines.append(entry)
                    legend_chars += len(entry)
            if len(legend_lines) > 1:
                lines.extend(legend_lines)

        return "\n".join(lines)

    def monthly_series(self) -> pd.DataFrame | None:
        """Return the raw monthly aggregate DataFrame for external use."""
        self._load()
        return self._monthly
