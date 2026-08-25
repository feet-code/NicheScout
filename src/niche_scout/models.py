from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def normalize_url(value: str) -> str:
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return value.strip()
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return value.strip()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def source_domain(url: str) -> str:
    try:
        return urlsplit(url).netloc.lower().removeprefix("www.")
    except ValueError:
        return ""


def idea_fingerprint(target_customer: str, canonical_problem: str) -> str:
    key = f"{normalize_text(target_customer)}|{normalize_text(canonical_problem)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


@dataclass(slots=True)
class Source:
    url: str
    title: str = ""
    domain: str = ""

    def __post_init__(self) -> None:
        self.url = normalize_url(self.url)
        self.domain = self.domain or source_domain(self.url)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Evidence:
    claim: str
    url: str
    title: str
    source_type: str
    observed_at: str | None = None
    verified: bool = False

    def __post_init__(self) -> None:
        self.url = normalize_url(self.url)

    @property
    def domain(self) -> str:
        return source_domain(self.url)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KeywordSignal:
    phrase: str
    intent: str
    search_volume: float | None = None
    cpc_usd: float | None = None
    difficulty: float | None = None
    serp_weakness: float | None = None
    source_url: str | None = None
    measured: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateIdea:
    name: str
    target_customer: str
    audience_cluster: str
    canonical_problem: str
    problem: str
    solution: str
    pricing_model: str
    price_monthly_usd: float
    pain_frequency: str
    pain_severity: int
    minutes_saved_per_occurrence: float
    buyer_budget: str
    build_hours: float
    maintenance_hours_month: float
    infra_cost_month: float
    support_complexity: str
    integration_count: int
    needs_scraping: bool
    regulated_data: bool
    recurrence: str
    retention_mechanism: str
    acquisition_wedge: str
    lead_magnet: str
    money_page: str
    topic: str
    blog_angles: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    keywords: list[KeywordSignal] = field(default_factory=list)
    fingerprint_override: str | None = None

    @property
    def fingerprint(self) -> str:
        return self.fingerprint_override or idea_fingerprint(
            self.target_customer, self.canonical_problem
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResearchReview:
    fingerprint: str
    axis: str
    verdict: str
    rationale: str
    confidence: float
    new_risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResearchMission:
    mode: str
    strategy: str
    objective: str
    questions: list[str]
    sequence: int
    focus_context: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResearchBatch:
    candidates: list[CandidateIdea]
    reviews: list[ResearchReview]
    research_summary: str
    consulted_sources: list[Source] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreCard:
    seo: float
    pain: float
    commercial: float
    feasibility: float
    retention: float
    evidence_confidence: float
    worthiness: float
    uncertainty: float
    risk_penalty: float
    eligible: bool
    recommended_status: str
    kill_reasons: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
