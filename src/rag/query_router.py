"""
RAG — Query Router.

Classifies user query as QUANTITATIVE or QUALITATIVE.
Extracts filters: years, months, subreddits, topic_macros, metric_type.

Entirely rule-based — no LLM, no embeddings. Runs in <5ms.
"""

import re
from dataclasses import dataclass, field


# ── Intent patterns ───────────────────────────────────────────────────────────

QUANTITATIVE_PATTERNS = [
    r'\bhow much\b',
    r'\bwhat (is|was|were) the (percentage|proportion|rate|share|%|pct)\b',
    r'\b(increase|decrease|change|shift|rise|fall|drop|grew|growth|declined|decline)\b.{0,30}\b(year|month|20\d\d)\b',
    r'\bwhich (topic|subreddit|year|month|community|group)\b.{0,30}\b(most|highest|lowest|least|biggest|largest|smallest)\b',
    r'\bcompare\b.{0,50}\b(and|vs|versus)\b',
    r'\byear.over.year\b',
    r'\bover time\b',
    r'\btrend\b.{0,30}\b(over|in|from|between)\b',
    r'\b(20\d\d)\b.{0,20}\b(vs|versus|compared to|relative to)\b.{0,20}\b(20\d\d)\b',
    r'\bwhat (percentage|proportion|%) of\b',
    r'\bhow (many|often|frequently)\b',
    r'\baverage (sentiment|commitment|score|rate)\b',
    r'\bstatistics?\b',
    r'\bnumbers?\b.{0,20}\b(for|on|about)\b',
]

QUALITATIVE_PATTERNS = [
    r'\bexplain\b',
    r'\bwhy\b',
    r'\bwhat caused?\b',
    r'\bwhat (do|did|does|are) (people|nsm[ae]n|singaporeans?|redditors?|servicemen|they|men|guys)\b.{0,30}\b(say|think|feel|believe|talk|discuss|complain|worried)\b',
    r'\bdescribe\b',
    r'\btell me about\b',
    r'\bwhat happened\b',
    r'\bgive me (examples?|instances?|quotes?|sample)\b',
    r'\bwhat.s (the mood|the feeling|people saying|going on)\b',
    r'\bsummar(ise|ize|y)\b',
    r'\bexamples? of\b',
    r'\btypical(ly)?\b',
    r'\bcommon(ly)?\b.{0,20}\b(say|think|feel|complain|express)\b',
    r'\bsentiment (about|toward|towards|on)\b',
    r'\bview(s|point)? (on|about|toward)\b',
    r'\bopinion(s)? (on|about)\b',
    r'\bwhat kind of\b',
    r'\bhow do (people|nsm[ae]n|singaporeans?)\b',
]

# ── Topic keyword mapping ─────────────────────────────────────────────────────

TOPIC_KEYWORD_MAP = {
    "Vocations & Units": [
        "vocation", "unit", "ocs", "scs", "officer", "specialist", "combat",
        "infantry", "armour", "armor", "artillery", "navy", "air force", "rsaf",
        "guard", "signals", "engineer", "logistics", "commando", "ndu", "scdf", "spf",
        "posting", "posted", "unit life",
    ],
    "BMT & Training": [
        "bmt", "basic military training", "tekong", "recruit", "training",
        "field camp", "outfield", "live firing", "exercise", "route march",
        "obstacle course", "safety", "book in", "book out", "in camp",
        "confined to camp", "ctc", "pass out parade",
    ],
    "Physical Fitness & IPPT": [
        "ippt", "fitness", "rt", "remedial training", "run", "physical",
        "napfa", "gold", "silver", "standup", "2.4km", "pull up", "sit up",
        "push up", "physical fitness", "pass ippt", "fail ippt", "ippt window",
    ],
    "Medical & Health": [
        "pes", "pes status", "pes f", "pes e", "pes b", "downpes", "medical board",
        "mc", "medical certificate", "injury", "medical", "health", "specialist",
        "downgrade", "upgrade", "pes exemption", "medical condition",
    ],
    "Mental Health": [
        "mental health", "depression", "suicide", "stress", "anxiety",
        "wellbeing", "psychological", "burnout", "counselling", "counseling",
        "mental wellbeing", "psych", "moodiness", "breakdown", "emotional",
    ],
    "NS Life & Culture": [
        "ns culture", "ns life", "army life", "camp culture", "rank culture",
        "sai kang", "keng", "chao keng", "wayang", "encik", "sergeant",
        "guard duty", "cookhouse", "bunk", "buddy", "ord countdown",
        "book in monday", "ns experience", "ns memories", "ns stories",
    ],
    "Enlistment & Pre-NS": [
        "enlist", "enlistment", "camo", "pre-ns", "disruption", "deferment",
        "defer", "ns start", "when enlist", "enlistment date", "letter",
        "pre enlist", "preparation", "before ns",
    ],
    "Reservist & ICT": [
        "reservist", "ict", "in-camp training", "orns", "call up",
        "reservist duties", "mindef notice", "saf100", "incamp", "high key",
        "low key", "ippt window", "reservist life", "mr", "operationally ready",
    ],
    "ORD & Post-NS": [
        "ord", "rod", "operationally ready date", "ord leave", "post ns",
        "after ns", "ord liao", "ORD countdown", "finishing ns", "last day",
        "cleared ns", "mrd",
    ],
    "Pay & Benefits": [
        "pay", "allowance", "salary", "cpf", "benefits", "income", "wage",
        "compensation", "token", "stipend", "ns allowance", "pay increase",
        "pay raise", "ns pay", "underpaid", "poverty pay", "low pay",
        "mindef pay", "economic cost",
    ],
    "Discipline & Misconduct": [
        "db", "detention barracks", "charge", "extra", "sign extra", "soc",
        "misconduct", "discipline", "abuse", "bully", "bullying", "discrimination",
        "power trip", "toxic", "corrupt", "punish", "punishment", "confined",
    ],
    "Relationships & Social": [
        "relationship", "girlfriend", "gf", "breakup", "friends", "social",
        "ord leave", "dating", "romance", "long distance", "family", "parents",
        "missing family", "lonely", "bonding", "brotherhood",
    ],
    "Admin & Logistics": [
        "admin", "paperwork", "leave", "token", "logistics", "store",
        "emart", "uniform", "equipment issue", "saf kit", "sar", "pass",
        "admin matters", "battalion admin", "s1",
    ],
    "Gear & Equipment": [
        "gear", "equipment", "rifle", "sar21", "weapon", "helmet", "vest",
        "no.4", "camouflage", "boots", "kit", "sbo", "lbv", "uniform",
        "saf gear", "issued gear",
    ],
    "NS Policy & Society": [
        "conscription", "policy", "mandatory", "abolish", "prs", "foreigners",
        "exemption", "defend", "national service policy", "institution", "obligation",
        "social contract", "burden", "worth", "ns equity", "foreign talent",
        "inequality", "unfair", "discrimination", "citizenship", "mindef policy",
        "ns reform", "ns debate", "compulsory", "slavery", "forced", "pr exemption",
    ],
    "Career & Sign-on": [
        "sign on", "signed on", "regular", "career", "regulars", "full time",
        "military career", "officer career", "saf career", "promotion",
        "career progression", "psc scholarship", "mindef career", "ling",
    ],
    "Gender & Diversity": [
        "women", "female", "girl", "gender", "ns for women", "equality",
        "sons", "daughters", "gender equality", "conscription women", "ns equality",
        "diversity", "inclusive", "females ns",
    ],
}

# ── Month mapping ─────────────────────────────────────────────────────────────

MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

SUBREDDIT_MAP = {
    "r/singapore":         ["r/singapore", "singapore subreddit", "/singapore"],
    "askSingapore":        ["r/asksingapore", "asksingapore", "/asksingapore"],
    "NationalServiceSG":   ["r/nationalservicesg", "nationalservicesg", "ns subreddit", "/nationalservicesg"],
}

METRIC_KEYWORDS = {
    "commitment": ["commitment", "committed", "uncommitted", "buy-in", "invested", "dedication", "devoted", "engaged"],
    "sentiment":  ["sentiment", "feeling", "opinion", "attitude", "tone", "mood"],
    "negative":   ["negative", "negativity", "criticism", "critical", "unhappy", "dissatisfied"],
    "positive":   ["positive", "positivity", "support", "supportive", "happy", "satisfied"],
    "discourse":  ["discourse", "discussion", "debate", "controversy", "disagreement", "contention"],
}


# ── Comparison / ranking patterns ────────────────────────────────────────────

SUBREDDIT_COMPARE_PATTERNS = [
    r'\bcompare\b.{0,60}\b(subreddit|r/)',
    r'\br/singapore\b.{0,30}\b(vs|versus|compared to|vs\.)\b.{0,30}\br/(national|ask)',
    r'\br/(national|ask).{0,30}\b(vs|versus|compared to)\b.{0,30}\br/singapore',
    r'\b(difference|differ|different)\b.{0,40}\b(subreddit|r/)',
    r'\bwhich (subreddit|community)\b.{0,40}\b(more|most|less|least|higher|lower|highest|lowest)\b',
    r'\bsubreddit (comparison|breakdown|difference|split)\b',
    r'\bhow (does|do) r/',
    r'\bacross (the )?(three |3 )?(subreddits?|communities|platforms)\b',
]

TOPIC_COMPARE_PATTERNS = [
    r'\bcompare\b.{0,80}\b(bmt|reservist|ippt|mental health|pay|policy|vocation|training|medical)\b',
    r'\b(bmt|reservist|ippt|policy|medical|mental health|pay)\b.{0,30}\b(vs|versus|compared to)\b.{0,30}\b(bmt|reservist|ippt|policy|medical|mental health|pay)\b',
    r'\bdifference between\b.{0,80}\b(topic|bmt|reservist|training|pay|policy)\b',
    r'\bwhich topic\b.{0,40}\b(more|most|less|least|higher|lower|highest|lowest)\b',
]

RANKING_PATTERNS = [
    r'\bwhich (year|topic|subreddit)\b.{0,50}\b(highest|lowest|most|least|biggest|worst|best)\b',
    r'\b(rank|ranking)\b.{0,40}\b(subreddit|topic|year)\b',
    r'\btop (year|topic|subreddit|month)\b.{0,30}\b(for|by|in|with)\b',
    r'\bmost (negative|positive|committed|critical|uncommitted|discussed|active)\b',
    r'\bleast (negative|positive|committed|critical|uncommitted|discussed|active)\b',
    r'\b(highest|lowest) (sentiment|commitment|negativity|positivity|volume|activity)\b',
]

COMMITMENT_BREAKDOWN_PATTERNS = [
    r'\b(committed|critical|neutral|apathetic)\b.{0,20}\b(vs|versus|compared|and)\b.{0,20}\b(committed|critical|neutral|apathetic)\b',
    r'\bpct.{0,20}(committed|critical|neutral)\b',
    r'\bpercent(age)?\b.{0,30}\b(committed|critical|neutral|uncommitted|apathetic)\b',
    r'\b(what|how many|breakdown).{0,30}\b(committed|critical|neutral|uncommitted)\b.{0,20}\b(posts?|chunks?|people|users?)\b',
    r'\bcommitment (breakdown|distribution|split|proportion)\b',
    r'\bclass(ification)?s?.{0,20}(committed|critical|neutral)\b',
]

METHODOLOGY_PATTERNS = [
    r'\bhow (was|were|is|are).{0,30}\b(calculated|computed|measured|scored|classified|labeled|defined)\b',
    r'\bwhat (is|does).{0,30}\b(commitment score|sentiment score|net score|pct_|mean_commit|mean_sent)\b',
    r'\bhow (do you|does the (model|system|dataset|classifier))\b',
    r'\b(methodology|method|approach|pipeline|model|classifier)\b.{0,30}\b(used|works|work|defined)\b',
    r'\bwhat does.{0,20}(committed|critical|uncommitted|neutral).{0,20}mean\b',
    r'\bwhat is.{0,20}(commitment|sentiment).{0,20}(measuring|measure|mean)\b',
    r'\bhow (was|is) (the )?(dataset|data|corpus)\b',
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class QueryFilters:
    years:             list[int]  = field(default_factory=list)
    months:            list[int]  = field(default_factory=list)
    subreddits:        list[str]  = field(default_factory=list)
    topics:            list[str]  = field(default_factory=list)
    metric:            str        = "sentiment"   # sentiment | commitment | negative | positive
    # Intent flags
    compare_subreddits:      bool  = False   # wants side-by-side subreddit breakdown
    compare_topics:          bool  = False   # wants side-by-side topic breakdown
    ranking_dim:             str   = ""      # "subreddit" | "topic" | "year" — what to rank
    ranking_dir:             str   = ""      # "highest" | "lowest"
    ask_methodology:         bool  = False   # asking how scores are calculated
    commitment_breakdown:    bool  = False   # wants pct_committed / pct_critical / pct_neutral breakdown


@dataclass
class RoutedQuery:
    original:   str
    intent:     str           # "quantitative" | "qualitative"
    filters:    QueryFilters


# ── Router ────────────────────────────────────────────────────────────────────

class QueryRouter:

    def classify(self, query: str) -> str:
        """Return 'quantitative' or 'qualitative'."""
        q = query.lower()

        quant_score = sum(1 for p in QUANTITATIVE_PATTERNS if re.search(p, q))
        qual_score  = sum(1 for p in QUALITATIVE_PATTERNS  if re.search(p, q))

        # Bias toward qualitative as the safer default
        if quant_score > qual_score:
            return "quantitative"
        return "qualitative"

    def extract_filters(self, query: str) -> QueryFilters:
        q = query.lower()
        filters = QueryFilters()

        # Years
        years = [int(y) for y in re.findall(r'\b(201[89]|202[0-5])\b', q)]
        filters.years = sorted(set(years))

        # Months
        months = []
        for name, num in MONTH_MAP.items():
            if re.search(r'\b' + re.escape(name) + r'\b', q):
                months.append(num)
        filters.months = sorted(set(months))

        # Subreddits
        for canonical, aliases in SUBREDDIT_MAP.items():
            if any(a in q for a in aliases):
                filters.subreddits.append(canonical)

        # Topics — score each topic by keyword hits
        topic_scores: dict[str, int] = {}
        for topic, keywords in TOPIC_KEYWORD_MAP.items():
            score = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', q))
            if score > 0:
                topic_scores[topic] = score

        if topic_scores:
            top_topics = sorted(topic_scores, key=topic_scores.get, reverse=True)[:2]
            filters.topics = top_topics

        # Metric type
        for metric_name, keywords in METRIC_KEYWORDS.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', q) for kw in keywords):
                filters.metric = metric_name
                break

        # ── Intent flags ──────────────────────────────────────────────────────

        # Subreddit comparison
        if any(re.search(p, q) for p in SUBREDDIT_COMPARE_PATTERNS):
            filters.compare_subreddits = True

        # Topic comparison
        if any(re.search(p, q) for p in TOPIC_COMPARE_PATTERNS):
            filters.compare_topics = True

        # Ranking
        for p in RANKING_PATTERNS:
            m = re.search(p, q)
            if m:
                # Determine what dimension to rank
                if re.search(r'\b(subreddit|community|r/)\b', q):
                    filters.ranking_dim = "subreddit"
                elif re.search(r'\btopic\b', q):
                    filters.ranking_dim = "topic"
                elif re.search(r'\b(year|month)\b', q):
                    filters.ranking_dim = "year"
                else:
                    filters.ranking_dim = "year"  # default
                # Direction
                filters.ranking_dir = "lowest" if re.search(r'\b(lowest|least|best|most positive|most committed)\b', q) else "highest"
                break

        # Methodology
        if any(re.search(p, q) for p in METHODOLOGY_PATTERNS):
            filters.ask_methodology = True

        # Commitment class breakdown
        if any(re.search(p, q) for p in COMMITMENT_BREAKDOWN_PATTERNS):
            filters.commitment_breakdown = True

        return filters

    def route(self, query: str) -> RoutedQuery:
        return RoutedQuery(
            original=query,
            intent=self.classify(query),
            filters=self.extract_filters(query),
        )
