"""
RAG — LLM Synthesis with streaming.

Takes the assembled context window and streams a response.
Backend selection (priority order):
  1. Groq API    — if GROQ_API_KEY is set (PRIMARY; free tier: 7k requests/month via Llama 3.3)
  2. OpenAI API  — if OPENAI_API_KEY is set (fallback; gpt-4o-mini, $0.0004/query)
  3. Anthropic   — if ANTHROPIC_API_KEY is set (fallback; claude-haiku-4-5)
  4. FreeLLMAPI  — if FREELLM_BASE_URL is set (local OpenAI-compatible proxy)

The LLM synthesises only — all retrieval, event lookup, and fact loading
was done upstream. The LLM sees a bounded context (~1500 tokens) and returns
a grounded analytical answer.
"""

import logging
import os
from typing import Iterator

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an analytical assistant for the NS Sentiment Research Project — \
a study of Singaporean public attitudes toward National Service (NS) based on 737,000 Reddit \
posts and comments from r/singapore, r/askSingapore, and r/NationalServiceSG, spanning 2018–2025.

Your role is to answer questions about this dataset using ONLY the statistical data, event context, \
topic summaries, and quotes provided to you. Never invent statistics. Never speculate beyond \
what the data shows.

━━━ CORE CONCEPTS ━━━
• "Sentiment" = emotional tone of a post (negative / neutral / positive). Measures HOW someone says something.
• "Commitment" = personal buy-in to NS (committed / uncommitted / neutral). Measures WHERE someone stands on NS.
• These are INDEPENDENT. A frustrated serviceman (negative sentiment) can still be committed to NS.
• "Uncommitted" has two flavours: APATHETIC (just clearing time, zao liao) and OPPOSED (institutional criticism).
• "Critical" is the opposed sub-faction of uncommitted — not mere complaint.
• "Net disposition" = (committed + supportive) minus (uncommitted + critical).
• "z-score" = deviation from rolling 6-month baseline in standard deviations; |z| ≥ 1.5 = notable, |z| ≥ 2.0 = major.

━━━ LONGITUDINAL ANALYSIS ━━━
When the context contains a LONGITUDINAL ANOMALY TIMELINE:
1. Identify the 2–4 most significant spikes or dips (by |z-score|) and explain WHAT drove them.
2. Attribute each anomaly to the named NS event(s) in the timeline. Do not invent causes.
3. Describe the DIRECTION of change: e.g. "negativity spiked" (sent_neg ▲), "commitment collapsed" (commit_net ▼), "discourse volume surged" (vol ▲).
4. Connect events into a narrative arc — show how the discourse evolved year-over-year, not just isolated data points.
5. If an anomaly has NO matched event, flag it as "unattributed" and describe the raw pattern only.

━━━ CAUSAL REASONING RULES ━━━
• Always distinguish correlation from causation: say "coincided with" or "followed" rather than "caused" unless the link is stated directly in the provided event data.
• If the same metric moves in multiple months due to the same event, describe it as a sustained effect, not separate incidents.
• For volume spikes (doc_count), note whether sentiment also shifted — a spike with stable sentiment = interest/curiosity; a spike with worsening sentiment = controversy.
• Policy announcements often show a PARADOX: short-term negativity spike even for positive changes (e.g. pay rises) due to debate and criticism of the announcement itself.

━━━ ANSWER FORMAT ━━━
1. Lead with a direct 1–2 sentence answer to the question.
2. For longitudinal questions: structure as a year-by-year or event-driven narrative.
   Use bold headers like **2020 (COVID + GE2020)** or **Jan 2019 — Aloysius Pang Incident**.
3. Support with specific statistics from the provided data (cite exact numbers when available).
4. For quotes, cite inline: [r/singapore · Jan 2019 · 847 upvotes]
5. If an NS event explains a pattern, name it explicitly and state its metric impact.
6. Note data gaps or caveats honestly at the end.

━━━ LENGTH GUIDELINES ━━━
• Simple factual question: 100–200 words
• Trend / longitudinal question: 250–400 words with structured sections
• Complex multi-part question: up to 500 words

TONE: Analytical. You are reporting findings to a supervisor or government department analyst.

━━━ HARD CONSTRAINTS ━━━
• NEVER invent any number not present in the provided statistical data.
• NEVER mix up sentiment and commitment — they are different dimensions.
• NEVER reference general knowledge about Singapore not in the provided context.
• NEVER make policy recommendations or normative judgements about NS.
• NEVER attribute an anomaly to an event not listed in the provided LONGITUDINAL ANOMALY TIMELINE or NS EVENT sections."""


# ── Backend selection ─────────────────────────────────────────────────────────

def _select_backend() -> tuple[str, str, str]:
    """
    Returns (backend_name, model, api_key_or_base_url).
    Priority: Groq (free 7k/mo) → OpenAI → Anthropic → FreeLLMAPI.
    """
    groq_key    = os.environ.get("GROQ_API_KEY", "")
    openai_key  = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    freellm_url = os.environ.get("FREELLM_BASE_URL", "http://localhost:3001/v1")

    # Primary: Groq (free tier)
    if groq_key and not groq_key.startswith("gsk-YOUR"):
        return ("groq", "llama-3.3-70b-versatile", groq_key)

    # Fallbacks
    if openai_key and not openai_key.startswith("sk-YOUR"):
        return ("openai", "gpt-4o-mini", openai_key)
    if anthropic_key and not anthropic_key.startswith("sk-ant-YOUR"):
        return ("anthropic", "claude-haiku-4-5", anthropic_key)

    return ("freellm", "gemini-2.5-flash", freellm_url)


def _create_groq_client(api_key: str):
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")


def _create_openai_client(api_key: str, base_url: str | None = None):
    try:
        from openai import OpenAI
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")


def _create_anthropic_client(api_key: str):
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")


# ── Synthesizer ───────────────────────────────────────────────────────────────

class Synthesizer:

    def __init__(self):
        self._client  = None          # lazy-init on first call
        self._backend = None
        self._model   = None
        self.max_tokens = 900
        log.info("Synthesizer ready (backend auto-detected on first call, lazy-init)")

    def _ensure_client(self):
        if self._client is not None:
            return
        backend, model, credential = _select_backend()
        self._backend = backend
        self._model   = model

        if backend == "groq":
            self._client = _create_groq_client(credential)
            log.info(f"Synthesizer using Groq  model={model}  (free tier: 7k req/mo)")
        elif backend == "openai":
            self._client = _create_openai_client(credential)
            log.info(f"Synthesizer using OpenAI  model={model}")
        elif backend == "anthropic":
            self._client = _create_anthropic_client(credential)
            log.info(f"Synthesizer using Anthropic  model={model}")
        else:  # freellm
            self._client = _create_openai_client(
                api_key  = os.environ.get("FREELLM_API_KEY", "freellmapi-key"),
                base_url = credential,
            )
            log.info(f"Synthesizer using FreeLLMAPI  base_url={credential}  model={model}")

    def _fallback_to_openai(self):
        """Switch primary client to OpenAI when Groq hits rate limits."""
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key and not openai_key.startswith("sk-YOUR"):
            self._client  = _create_openai_client(openai_key)
            self._backend = "openai"
            self._model   = "gpt-4o-mini"
            log.warning("Groq rate-limited — fell back to OpenAI gpt-4o-mini")
            return True
        return False

    def stream(self, context: str) -> Iterator[str]:
        """
        Stream LLM synthesis tokens.
        Caller should concatenate yielded strings for the full response.

        Args:
            context: Full assembled context string (includes user query at end).
        """
        self._ensure_client()
        log.debug(f"LLM call: {len(context)} chars context, model={self._model}, backend={self._backend}")

        try:
            if self._backend == "groq":
                yield from self._stream_groq(context)
            elif self._backend == "anthropic":
                yield from self._stream_anthropic(context)
            else:
                yield from self._stream_openai(context)
        except Exception as e:
            err_str = str(e)
            # Auto-fallback on Groq rate limit (daily or per-minute)
            if self._backend == "groq" and "rate_limit_exceeded" in err_str:
                log.warning(f"Groq rate limit hit — attempting OpenAI fallback")
                if self._fallback_to_openai():
                    try:
                        yield from self._stream_openai(context)
                        return
                    except Exception as e2:
                        log.error(f"OpenAI fallback also failed: {e2}")
                        yield f"\n\n*[Rate limit hit on Groq; OpenAI fallback also failed: {e2}]*"
                        return
            log.error(f"LLM synthesis error: {e}")
            yield f"\n\n*[Error generating response: {e}]*"

    def _stream_groq(self, context: str) -> Iterator[str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": context},
        ]
        with self._client.chat.completions.create(
            model      = self._model,
            messages   = messages,
            max_tokens = self.max_tokens,
            stream     = True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    def _stream_openai(self, context: str) -> Iterator[str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": context},
        ]
        with self._client.chat.completions.create(
            model      = self._model,
            messages   = messages,
            max_tokens = self.max_tokens,
            stream     = True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    def _stream_anthropic(self, context: str) -> Iterator[str]:
        messages = [{"role": "user", "content": context}]
        with self._client.messages.stream(
            model      = self._model,
            max_tokens = self.max_tokens,
            system     = SYSTEM_PROMPT,
            messages   = messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def complete(self, context: str) -> str:
        """Non-streaming completion. Useful for testing."""
        return "".join(self.stream(context))
