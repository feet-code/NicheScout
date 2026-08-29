from __future__ import annotations

import json
import math
from statistics import fmean
from typing import Any

from .config import Settings
from .models import ScoreCard


FREQUENCY = {
    "annual": 0.15,
    "quarterly": 0.3,
    "monthly": 0.5,
    "weekly": 0.75,
    "daily": 0.95,
    "continuous": 1.0,
    "one_off": 0.1,
}
BUYER_BUDGET = {"low": 0.25, "medium": 0.65, "high": 1.0}
SUPPORT_COST = {"low": 0.0, "medium": 12.0, "high": 28.0}
COMMERCIAL_INTENTS = {"transactional", "commercial", "comparison", "problem_aware"}


class ScoreEngine:
    def __init__(self, settings: Settings):
        self.settings = settings

    def score(self, bundle: dict[str, Any]) -> ScoreCard:
        idea = bundle["idea"]
        evidence = bundle["evidence"]
        keywords = bundle["keywords"]
        reviews = bundle["reviews"]

        verified = [item for item in evidence if int(item["verified"])]
        domains = {item["domain"] for item in verified if item["domain"]}
        source_types = {item["source_type"] for item in verified}

        pain = _clamp(
            float(idea["pain_severity"]) * 11
            + FREQUENCY.get(str(idea["pain_frequency"]), 0.25) * 20
            + min(15, math.log1p(max(0.0, float(idea["minutes_saved_per_occurrence"]))) * 3)
            + min(10, 4 * sum(item["source_type"] == "customer_complaint" for item in verified))
        )

        paid_intent_share = (
            sum(item["intent"] in COMMERCIAL_INTENTS for item in keywords) / len(keywords)
            if keywords
            else 0.0
        )
        measured = [item for item in keywords if int(item["measured"])]
        weakness_values = [
            float(item["serp_weakness"])
            for item in keywords
            if item["serp_weakness"] is not None
        ]
        seo = _clamp(
            min(28, len(keywords) * 2.0)
            + paid_intent_share * 28
            + min(14, len({item["intent"] for item in keywords}) * 3.5)
            + (fmean(weakness_values) * 0.18 if weakness_values else 4)
            + min(12, len(measured) * 2)
            + (8 if idea["lead_magnet"] and idea["money_page"] else 0)
        )

        competitor_pricing = sum(
            item["source_type"] == "competitor_pricing" for item in verified
        )
        commercial = _clamp(
            BUYER_BUDGET.get(str(idea["buyer_budget"]), 0.3) * 28
            + min(22, max(0.0, float(idea["price_monthly_usd"])) / 5)
            + min(18, competitor_pricing * 9)
            + FREQUENCY.get(str(idea["recurrence"]), 0.2) * 17
            + paid_intent_share * 15
        )

        feasibility = 100.0
        feasibility -= min(50, float(idea["build_hours"]) / self.settings.max_build_hours * 34)
        maintenance_cap = max(0.1, self.settings.max_maintenance_hours_month)
        feasibility -= min(28, float(idea["maintenance_hours_month"]) / maintenance_cap * 20)
        infra_cap = max(1.0, self.settings.max_infra_cost_month)
        feasibility -= min(16, float(idea["infra_cost_month"]) / infra_cap * 12)
        feasibility -= SUPPORT_COST.get(str(idea["support_complexity"]), 15.0)
        feasibility -= min(18, int(idea["integration_count"]) * 3)
        feasibility -= 45 if int(idea["needs_scraping"]) else 0
        feasibility -= 25 if int(idea["regulated_data"]) else 0
        feasibility = _clamp(feasibility)

        recurrence = FREQUENCY.get(str(idea["recurrence"]), 0.15)
        retention = _clamp(
            recurrence * 58
            + min(22, len(str(idea["retention_mechanism"])) / 4)
            + (12 if float(idea["price_monthly_usd"]) > 0 else 0)
            + (8 if str(idea["recurrence"]) in {"daily", "weekly", "continuous"} else 0)
        )

        validation_axes = {item["axis"] for item in reviews if item["axis"] != "red_team"}
        review_confidences = [float(item["confidence"]) for item in reviews]
        source_confidence = min(0.48, len(domains) * 0.075 + len(source_types) * 0.045)
        review_confidence = min(0.36, len(validation_axes) * 0.07 + len(reviews) * 0.035)
        measured_confidence = min(0.12, len(measured) * 0.02)
        average_review_quality = fmean(review_confidences) if review_confidences else 0.35
        evidence_confidence = min(
            1.0,
            0.04 + source_confidence + review_confidence * average_review_quality + measured_confidence,
        )

        risks = json.loads(idea["risks_json"] or "[]")
        risk_penalty = min(25.0, len(risks) * 1.6)
        lowered = " ".join(str(item).lower() for item in risks)
        if any(token in lowered for token in ("unstable api", "platform dependency", "terms of service")):
            risk_penalty += 6
        if int(idea["regulated_data"]):
            risk_penalty += 6

        kill_reasons: list[str] = []
        if float(idea["build_hours"]) > self.settings.max_build_hours:
            kill_reasons.append("Estimated build exceeds configured solo-MVP limit")
        if float(idea["maintenance_hours_month"]) > self.settings.max_maintenance_hours_month:
            kill_reasons.append("Estimated monthly maintenance exceeds configured limit")
        if float(idea["infra_cost_month"]) > self.settings.max_infra_cost_month:
            kill_reasons.append("Estimated infrastructure exceeds configured limit")
        if int(idea["needs_scraping"]):
            kill_reasons.append("Core product depends on scraping")
        if int(idea["regulated_data"]):
            kill_reasons.append("Core product handles regulated data")
        if str(idea["support_complexity"]) == "high":
            kill_reasons.append("High-touch support is incompatible with the portfolio")
        if any(item["verdict"] == "falsified" for item in reviews):
            kill_reasons.append("Adversarial research falsified a core assumption")

        geometric = _geomean([seo, pain, commercial, feasibility, retention])
        worthiness = _clamp(
            geometric * (0.54 + 0.46 * evidence_confidence) - risk_penalty
        )
        eligible = not kill_reasons
        if not eligible:
            status = "killed"
        elif int(idea["red_team_passes"]) > 0:
            status = "red_teamed"
        elif int(idea["validation_passes"]) > 0:
            status = "validated"
        else:
            status = "discovered"

        notes = [
            f"{len(domains)} independent verified domains",
            f"{len(keywords)} query hypotheses; {len(measured)} measured",
            f"{int(idea['validation_passes'])} validation and {int(idea['red_team_passes'])} red-team passes",
        ]
        return ScoreCard(
            seo=round(seo, 3),
            pain=round(pain, 3),
            commercial=round(commercial, 3),
            feasibility=round(feasibility, 3),
            retention=round(retention, 3),
            evidence_confidence=round(evidence_confidence, 5),
            worthiness=round(worthiness, 3),
            uncertainty=round(1.0 - evidence_confidence, 5),
            risk_penalty=round(risk_penalty, 3),
            eligible=eligible,
            recommended_status=status,
            kill_reasons=kill_reasons,
            notes=notes,
        )


def _geomean(values: list[float]) -> float:
    safe = [max(1.0, value) for value in values]
    return math.exp(sum(math.log(value) for value in safe) / len(safe))


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
