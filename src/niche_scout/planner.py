from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .config import Settings
from .db import Repository
from .models import ResearchMission


STRATEGIES: dict[str, str] = {
    "spreadsheet_replacement": (
        "Find recurring workflows still run in fragile spreadsheets where errors, handoffs, "
        "reminders, or audit trails create an expensive failure."
    ),
    "review_pain_mining": (
        "Find repeated low-star review complaints and cancellation reasons that reveal one narrow "
        "feature an incumbent suite serves badly."
    ),
    "document_workflow": (
        "Find repetitive intake, validation, extraction, transformation, approval, or delivery of "
        "ordinary business documents without regulated advice."
    ),
    "vertical_reporting": (
        "Find niche operators who repeatedly assemble client, owner, vendor, or internal reports "
        "from a small stable set of inputs."
    ),
    "stable_integration_gap": (
        "Find manual re-entry between tools that expose stable supported APIs or file exports, not "
        "private endpoints or prohibited scraping."
    ),
    "deadline_and_evidence": (
        "Find recurring expirations, inspections, renewals, maintenance, certificates, or evidence "
        "collection where missing one has a clear business cost."
    ),
    "calculator_funnel": (
        "Find niche quoting, estimating, reconciliation, comparison, or calculator workflows with "
        "solution-seeking searches and a natural free-tool-to-paid-product funnel."
    ),
    "suite_unbundling": (
        "Find a single painful workflow hidden inside an expensive horizontal suite that a focused "
        "low-support product can deliver."
    ),
    "job_posting_workflow": (
        "Use recurring responsibilities in job postings to identify tedious software-addressable work "
        "performed by a clearly named role."
    ),
    "template_to_product": (
        "Find templates, checklists, logs, and forms that imply a repeated workflow users may upgrade "
        "from a static artifact into a narrow tool."
    ),
    "field_service_records": (
        "Find service trades that create estimates, inspection records, certificates, photos, or "
        "customer reports after every job."
    ),
    "data_conversion": (
        "Find recurring import, export, cleanup, reconciliation, mapping, or format-conversion jobs "
        "for a specific business audience."
    ),
}

AXES: dict[str, tuple[str, list[str]]] = {
    "pain": (
        "Test whether the named buyer repeatedly experiences the problem and whether failure has a real cost.",
        [
            "Find independent first-person complaints, workflow descriptions, or job responsibilities.",
            "Estimate frequency and consequence conservatively; reject merely annoying one-off tasks.",
            "Look for evidence that the workaround is still manual or fragmented today.",
        ],
    ),
    "seo": (
        "Test whether a coherent organic-search acquisition surface exists and appears realistically rankable.",
        [
            "Inspect multiple commercial, transactional, template, calculator, and problem-aware queries.",
            "Separate a real query cluster from invented phrases and one vanity keyword.",
            "Look for weak, mismatched, stale, forum-only, or horizontal results without fabricating volume.",
        ],
    ),
    "commercial": (
        "Test willingness to pay, buyer authority, incumbent pricing, and whether information traffic can convert.",
        [
            "Find current paid alternatives, pricing, procurement evidence, or explicit requests for a tool.",
            "Identify free substitutes and determine whether the narrow product is materially better.",
            "Reject audiences without budget or problems that are solved adequately for free.",
        ],
    ),
    "feasibility": (
        "Test whether a solo engineer can build and maintain the narrow outcome within configured limits.",
        [
            "Verify required integrations, data sources, APIs, and recurring operational work.",
            "Look for compliance, support, scraping, platform, and data-cost traps.",
            "Prefer file-based, deterministic, self-serve workflows with low marginal cost.",
        ],
    ),
}


class ResearchPlanner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def next_mission(self, repo: Repository) -> ResearchMission:
        # Deferred and interrupted calls reuse the same deterministic mission on resume.
        sequence = repo.action_count(succeeded_only=True) + 1
        if repo.idea_count() < self.settings.discovery_pool_target:
            strategy = self._ucb_strategy(repo.strategy_stats())
            return ResearchMission(
                mode="discover",
                strategy=strategy,
                objective=STRATEGIES[strategy],
                questions=[
                    "Begin from current public evidence, not an ungrounded product brainstorm.",
                    "Return complementary products for one or two specific audience clusters so five can share a site.",
                    "Require repeated pain, a buyer with budget, a low-cost product wedge, and several useful SEO page types.",
                    "Try to disprove each thesis with free substitutes, entrenched incumbents, or hidden operating costs.",
                ],
                sequence=sequence,
            )

        deep_pool = repo.list_ideas(
            self.settings.deep_research_pool, eligible_only=True
        )
        validation = self._next_validation(repo, deep_pool)
        if validation:
            axis, candidates = validation
            objective, questions = AXES[axis]
            return ResearchMission(
                mode=f"validate_{axis}",
                strategy=f"validate_{axis}",
                objective=objective,
                questions=questions,
                sequence=sequence,
                focus_context=[repo.candidate_context(int(item["id"])) for item in candidates],
            )

        red_pool = repo.list_ideas(self.settings.red_team_pool, eligible_only=True)
        untested = [
            item
            for item in red_pool
            if int(item["red_team_passes"]) < self.settings.minimum_red_team_passes
        ]
        if untested:
            selected = untested[: self.settings.review_batch_size]
            return ResearchMission(
                mode="red_team",
                strategy="red_team",
                objective=(
                    "Act as an adversarial investment committee. Seek decisive evidence that each "
                    "candidate will not rank, convert, remain cheap, or retain customers."
                ),
                questions=[
                    "What free substitute or incumbent makes this unnecessary?",
                    "Is apparent search demand informational rather than product-seeking?",
                    "Which build, API, support, compliance, or maintenance assumption is understated?",
                    "Falsify only with evidence; otherwise record the strongest residual risks.",
                ],
                sequence=sequence,
                focus_context=[repo.candidate_context(int(item["id"])) for item in selected],
            )

        # Required tournament coverage is complete. Spend the remaining finite time budget
        # on the highest expected-value-of-information gap near the final cutoff.
        finalist_pool = repo.list_ideas(self.settings.finalist_target, eligible_only=True)
        if len(finalist_pool) < self.settings.finalist_target:
            strategy = self._ucb_strategy(repo.strategy_stats())
            return ResearchMission(
                mode="discover",
                strategy=strategy,
                objective=STRATEGIES[strategy],
                questions=[
                    "Find replacement candidates for ideas eliminated by hard feasibility or adversarial gates.",
                    "Do not return variants of already-known audience/problem fingerprints.",
                ],
                sequence=sequence,
            )
        ranked = sorted(
            finalist_pool,
            key=lambda item: (
                float(item["worthiness_score"]) * (0.55 + float(item["uncertainty"])),
                -int(item["validation_passes"]),
            ),
            reverse=True,
        )
        lead = ranked[0]
        axis = self._least_researched_axis(repo, int(lead["id"]))
        peers = [lead]
        for item in ranked[1:]:
            if len(peers) >= self.settings.review_batch_size:
                break
            if self._least_researched_axis(repo, int(item["id"])) == axis:
                peers.append(item)
        objective, questions = AXES[axis]
        return ResearchMission(
            mode=f"validate_{axis}",
            strategy=f"evi_{axis}",
            objective="Resolve the highest-value remaining uncertainty. " + objective,
            questions=questions,
            sequence=sequence,
            focus_context=[repo.candidate_context(int(item["id"])) for item in peers],
        )

    def _next_validation(
        self, repo: Repository, pool: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]] | None:
        needs = [
            item
            for item in pool
            if int(item["validation_passes"]) < self.settings.minimum_validation_passes
        ]
        if not needs:
            return None
        needs.sort(
            key=lambda item: float(item["worthiness_score"]) * (0.7 + float(item["uncertainty"])),
            reverse=True,
        )
        lead = needs[0]
        axis = self._least_researched_axis(repo, int(lead["id"]))
        selected = [lead]
        for item in needs[1:]:
            if len(selected) >= self.settings.review_batch_size:
                break
            if self._least_researched_axis(repo, int(item["id"])) == axis:
                selected.append(item)
        return axis, selected

    @staticmethod
    def _least_researched_axis(repo: Repository, idea_id: int) -> str:
        counts = Counter(repo.review_axis_counts(idea_id))
        return min(AXES, key=lambda axis: (counts[axis], list(AXES).index(axis)))

    @staticmethod
    def _ucb_strategy(stats: dict[str, dict[str, Any]]) -> str:
        for strategy in STRATEGIES:
            if strategy not in stats or int(stats[strategy]["pulls"]) == 0:
                return strategy
        total = sum(int(item["pulls"]) for item in stats.values())
        return max(
            STRATEGIES,
            key=lambda strategy: (
                float(stats[strategy]["reward_sum"]) / int(stats[strategy]["pulls"])
                + 1.25 * math.sqrt(math.log(max(2, total)) / int(stats[strategy]["pulls"]))
            ),
        )
