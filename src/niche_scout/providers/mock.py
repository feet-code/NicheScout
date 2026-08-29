from __future__ import annotations

import json

from ..config import Settings
from ..models import (
    CandidateIdea,
    Evidence,
    KeywordSignal,
    ResearchBatch,
    ResearchMission,
    ResearchReview,
    Source,
)


AUDIENCES = [
    "independent commercial cleaning companies",
    "small machine and fabrication shops",
    "boutique property management firms",
    "specialty food manufacturers",
    "local fire protection contractors",
    "commercial landscaping companies",
    "independent equipment rental companies",
    "small freight brokerage teams",
    "regional pest control operators",
    "boutique bookkeeping practices",
    "managed IT service providers",
    "small architecture studios",
]

WORKFLOWS = [
    ("Renewal Radar", "tracks recurring vendor and certificate renewal dates", "renewal deadline tracker"),
    ("Proof Pack", "assembles customer proof documents after each completed job", "service proof report builder"),
    ("Quote Check", "checks quote inputs before a proposal is sent", "quote quality checklist"),
    ("Handoff Log", "captures incomplete job handoffs between office and field staff", "job handoff log"),
    ("Exception Inbox", "triages exceptions currently buried in shared email", "workflow exception inbox"),
    ("Visit Reporter", "turns recurring visit notes into a client-ready report", "client visit report"),
    ("Expiry Board", "assigns owners to expiring equipment and staff records", "expiry assignment board"),
    ("Variance Note", "explains recurring estimate-to-actual variances", "job variance report"),
    ("Intake Guard", "validates customer intake files before work begins", "customer intake validator"),
    ("Follow-up Queue", "tracks promised follow-ups after a service event", "service follow-up queue"),
]


class MockResearchProvider:
    """Deterministic synthetic evidence for smoke tests; never presented as real research."""

    name = "mock"

    def retry_at(self) -> float | None:
        return None

    def research(self, mission: ResearchMission, settings: Settings) -> ResearchBatch:
        if mission.mode == "discover":
            start = (mission.sequence - 1) * settings.discovery_batch_size
            candidates = [
                _candidate(start + offset, settings)
                for offset in range(settings.discovery_batch_size)
            ]
            sources = [
                Source(evidence.url, evidence.title)
                for candidate in candidates
                for evidence in candidate.evidence
            ]
            return ResearchBatch(
                candidates=candidates,
                reviews=[],
                research_summary="Synthetic discovery fixture; no market claims are real.",
                consulted_sources=sources,
                usage={"mock": True, "mission": mission.mode},
                artifacts={"mock-mission.json": json.dumps(mission.to_dict(), indent=2)},
            )

        axis = "red_team" if mission.mode == "red_team" else mission.mode.removeprefix("validate_")
        reviews = [
            ResearchReview(
                fingerprint=str(context["fingerprint"]),
                axis=axis,
                verdict="survives",
                rationale=(
                    "Synthetic review confirms the fixture for restart and tournament testing; "
                    "it is not real underwriting evidence."
                ),
                confidence=0.78,
                new_risks=["Synthetic fixture must never be interpreted as market validation"],
            )
            for context in mission.focus_context
        ]
        return ResearchBatch(
            candidates=[],
            reviews=reviews,
            research_summary=f"Synthetic {axis} review fixture.",
            consulted_sources=[],
            usage={"mock": True, "mission": mission.mode},
            artifacts={"mock-mission.json": json.dumps(mission.to_dict(), indent=2)},
        )


def _candidate(index: int, settings: Settings) -> CandidateIdea:
    audience_number = index // settings.products_per_site
    audience = AUDIENCES[audience_number % len(AUDIENCES)]
    if audience_number >= len(AUDIENCES):
        audience = f"{audience} segment {audience_number + 1}"
    workflow_name, workflow_problem, keyword_root = WORKFLOWS[index % len(WORKFLOWS)]
    name = f"{workflow_name} {index + 1}"
    evidence = [
        Evidence(
            claim=f"Synthetic evidence {part + 1} for candidate {index + 1}",
            url=f"https://fixture-{part + 1}.example/candidate-{index + 1}",
            title=f"Synthetic source {part + 1}",
            source_type=("customer_complaint", "competitor_pricing", "job_posting")[part],
            verified=True,
        )
        for part in range(3)
    ]
    keywords = [
        KeywordSignal(
            phrase=f"{keyword_root} {suffix}",
            intent=("transactional", "commercial", "comparison", "problem_aware")[offset % 4],
            serp_weakness=58 + (offset % 8),
            measured=False,
        )
        for offset, suffix in enumerate(
            ("software", "tool", "template", "app", "alternative", "checklist", "automation", audience)
        )
    ]
    return CandidateIdea(
        name=name,
        target_customer=audience,
        audience_cluster=audience,
        canonical_problem=f"operations coordinator {workflow_problem} for {audience}",
        problem=f"An operations coordinator at {audience} repeatedly {workflow_problem} by hand.",
        solution=f"A narrow self-serve {keyword_root} with reminders, ownership, and export.",
        pricing_model="$39 per business per month",
        price_monthly_usd=39,
        pain_frequency="weekly",
        pain_severity=4,
        minutes_saved_per_occurrence=45,
        buyer_budget="medium",
        build_hours=min(settings.max_build_hours, 16 + index % 12),
        maintenance_hours_month=min(settings.max_maintenance_hours_month, 1.2),
        infra_cost_month=min(settings.max_infra_cost_month, 8),
        support_complexity="low",
        integration_count=1,
        needs_scraping=False,
        regulated_data=False,
        recurrence="weekly",
        retention_mechanism="Recurring history, assignments, reminders, and saved exports",
        acquisition_wedge=f"Free {keyword_root} template",
        lead_magnet=f"{keyword_root.title()} template",
        money_page=f"{keyword_root.title()} software for {audience}",
        topic=f"{keyword_root} for {audience}",
        blog_angles=[keyword.phrase for keyword in keywords],
        risks=["Small synthetic niche fixture"],
        evidence=evidence,
        keywords=keywords,
    )
