from __future__ import annotations

import json
import os
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any

from .config import Settings
from .db import Repository
from .models import normalize_text, utc_now


class PortfolioNotReady(RuntimeError):
    pass


def qualified_rows(repo: Repository, settings: Settings) -> list[dict[str, Any]]:
    """Return hard-gate survivors that completed the configured underwriting passes."""

    qualified: list[dict[str, Any]] = []
    for row in repo.list_ideas(eligible_only=True):
        if int(row["validation_passes"]) < settings.minimum_validation_passes:
            continue
        if int(row["red_team_passes"]) < settings.minimum_red_team_passes:
            continue
        bundle = repo.get_idea_bundle(int(row["id"]))
        verified_sources = len(
            {item["url"] for item in bundle["evidence"] if int(item["verified"])}
        )
        if verified_sources < settings.minimum_verified_sources:
            continue
        qualified.append(row)
    return qualified


def qualified_idea_count(repo: Repository, settings: Settings) -> int:
    return len(qualified_rows(repo, settings))


def build_portfolio(repo: Repository, settings: Settings) -> dict[str, Any]:
    candidates = qualified_rows(repo, settings)
    if len(candidates) < settings.finalist_target:
        raise PortfolioNotReady(
            f"Only {len(candidates)} ideas meet the final evidence/review gates; "
            f"{settings.finalist_target} are required. Resume research or loosen explicit config gates."
        )

    groups = _select_and_group(candidates, settings)
    selected = [row for group in groups for row in group]
    ranked = sorted(
        selected,
        key=lambda row: (
            float(row["worthiness_score"]),
            float(row["evidence_confidence"]),
            -int(row["id"]),
        ),
        reverse=True,
    )
    rank_by_id = {int(row["id"]): rank for rank, row in enumerate(ranked, 1)}

    idea_id_by_db_id = {
        int(row["id"]): _idea_id(row)
        for row in selected
    }
    sites: list[dict[str, Any]] = []
    site_by_db_id: dict[int, str] = {}
    for index, group in enumerate(groups, 1):
        site = _site_record(index, group, idea_id_by_db_id)
        sites.append(site)
        for row in group:
            site_by_db_id[int(row["id"])] = site["id"]

    ideas = [
        _idea_record(
            repo,
            row,
            idea_id=idea_id_by_db_id[int(row["id"])],
            site_id=site_by_db_id[int(row["id"])],
            rank=rank_by_id[int(row["id"])],
        )
        for row in ranked
    ]
    sites.sort(key=lambda item: (-float(item["score"]), item["id"]))
    for rank, site in enumerate(sites, 1):
        site["rank"] = rank

    counts = repo.progress_counts()
    return {
        "version": 2,
        "generatedAt": utc_now(),
        "generator": "NicheScout",
        "research": {
            "configFingerprint": settings.fingerprint(),
            "activeHours": round(repo.active_seconds() / 3600, 4),
            "targetActiveHours": settings.target_active_hours,
            "actions": repo.action_status_counts(),
            "discoveredIdeas": counts["ideas"],
            "eligibleIdeas": counts["eligible"],
            "qualifiedIdeas": len(candidates),
            "finalists": settings.finalist_target,
            "sites": settings.site_target,
            "productsPerSite": settings.products_per_site,
            "selection": (
                "Hard feasibility gates, minimum source/review coverage, evidence-weighted score, "
                "then deterministic audience-cohesion grouping with duplicate-problem penalties."
            ),
            "metricPolicy": (
                "Search volume/CPC/difficulty are null unless a cited source explicitly measured them; "
                "unmeasured SERP observations are discounted."
            ),
        },
        "sites": sites,
        "ideas": ideas,
    }


def export_portfolio(
    repo: Repository,
    settings: Settings,
    path: str | Path,
    *,
    report_path: str | Path | None = None,
) -> Path:
    document = build_portfolio(repo, settings)
    output = Path(path)
    _atomic_text(output, json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    report = Path(report_path) if report_path else output.with_name("finalists.md")
    _atomic_text(report, _markdown_report(document))
    repo.mark_finalized(str(output.resolve()), len(document["ideas"]), len(document["sites"]))
    return output.resolve()


def _select_and_group(
    candidates: list[dict[str, Any]], settings: Settings
) -> list[list[dict[str, Any]]]:
    remaining = {int(row["id"]): row for row in candidates}
    groups: list[list[dict[str, Any]]] = []
    audience_site_counts: Counter[str] = Counter()
    for _ in range(settings.site_target):
        if len(remaining) < settings.products_per_site:
            raise PortfolioNotReady("Not enough unused qualified candidates to complete every site")
        seed = max(
            remaining.values(),
            key=lambda row: (
                float(row["worthiness_score"])
                - settings.diversity_penalty
                * audience_site_counts[_cluster_key(row)] ** 0.75,
                float(row["evidence_confidence"]),
                -int(row["id"]),
            ),
        )
        group = [seed]
        del remaining[int(seed["id"])]
        while len(group) < settings.products_per_site:
            peer = max(
                remaining.values(),
                key=lambda row: (
                    _peer_value(row, group),
                    float(row["worthiness_score"]),
                    -int(row["id"]),
                ),
            )
            group.append(peer)
            del remaining[int(peer["id"])]
        groups.append(group)
        audience_site_counts[_cluster_key(seed)] += 1
    return groups


def _peer_value(row: dict[str, Any], group: list[dict[str, Any]]) -> float:
    coherence = fmean(_audience_similarity(row, member) for member in group)
    closest_duplicate = max(_product_similarity(row, member) for member in group)
    quality = float(row["worthiness_score"]) / 100
    confidence = float(row["evidence_confidence"])
    # Coherent users matter most; near-identical products are actively rejected.
    return 0.52 * coherence + 0.30 * quality + 0.10 * confidence - 0.32 * closest_duplicate


def _audience_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_cluster = _tokens(left["audience_cluster"])
    right_cluster = _tokens(right["audience_cluster"])
    exact = 1.0 if normalize_text(left["audience_cluster"]) == normalize_text(right["audience_cluster"]) else 0.0
    cluster = _jaccard(left_cluster, right_cluster)
    customers = _jaccard(_tokens(left["target_customer"]), _tokens(right["target_customer"]))
    topics = _jaccard(_tokens(left["topic"]), _tokens(right["topic"]))
    return min(1.0, 0.40 * exact + 0.34 * cluster + 0.20 * customers + 0.06 * topics)


def _product_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    problems = _jaccard(_tokens(left["canonical_problem"]), _tokens(right["canonical_problem"]))
    solutions = _jaccard(_tokens(left["solution"]), _tokens(right["solution"]))
    return 0.75 * problems + 0.25 * solutions


def _site_record(
    index: int,
    group: list[dict[str, Any]],
    idea_id_by_db_id: dict[int, str],
) -> dict[str, Any]:
    seed = group[0]
    site_id = f"portfolio-{index:03d}-{_slug(seed['audience_cluster'])[:42]}"
    pair_scores = [_audience_similarity(left, right) for left, right in combinations(group, 2)]
    topics = Counter(str(row["topic"]).strip() for row in group if str(row["topic"]).strip())
    audience = str(seed["audience_cluster"]).strip()
    return {
        "id": site_id,
        "name": f"{_title(audience)} Tools",
        "audience": audience,
        "topic": topics.most_common(1)[0][0] if topics else audience,
        "productIds": [idea_id_by_db_id[int(row["id"])] for row in group],
        "score": round(fmean(float(row["worthiness_score"]) for row in group), 3),
        "cohesion": round(fmean(pair_scores) if pair_scores else 1.0, 4),
        "domain": None,
    }


def _idea_record(
    repo: Repository,
    row: dict[str, Any],
    *,
    idea_id: str,
    site_id: str,
    rank: int,
) -> dict[str, Any]:
    bundle = repo.get_idea_bundle(int(row["id"]))
    risks = _json_list(row["risks_json"])
    evidence = [
        {
            "claim": item["claim"],
            "url": item["url"],
            "title": item["title"],
            "sourceType": item["source_type"],
            "verified": bool(item["verified"]),
            "observedAt": item["observed_at"],
        }
        for item in bundle["evidence"]
    ]
    keywords = [
        {
            "phrase": item["phrase"],
            "intent": item["intent"],
            "searchVolume": item["search_volume"],
            "cpcUsd": item["cpc_usd"],
            "difficulty": item["difficulty"],
            "serpWeakness": item["serp_weakness"],
            "measured": bool(item["measured"]),
            "sourceUrl": item["source_url"],
        }
        for item in bundle["keywords"]
    ]
    return {
        "id": idea_id,
        "siteId": site_id,
        "rank": rank,
        "name": row["name"],
        "product": row["solution"],
        "audience": row["target_customer"],
        "audienceCluster": row["audience_cluster"],
        "problem": row["problem"],
        "valueProposition": row["solution"],
        "topic": row["topic"],
        "monetization": row["pricing_model"],
        "startupCost": (
            f"Estimated {float(row['build_hours']):g} build hours, "
            f"${float(row['infra_cost_month']):g}/month infrastructure, and "
            f"{float(row['maintenance_hours_month']):g} maintenance hours/month."
        ),
        "seoAngle": (
            f"{row['acquisition_wedge']}. Primary money page: {row['money_page']}."
        ),
        "score": round(float(row["worthiness_score"]), 3),
        "domain": None,
        "research": {
            "fingerprint": row["fingerprint"],
            "confidence": round(float(row["evidence_confidence"]), 5),
            "uncertainty": round(float(row["uncertainty"]), 5),
            "scores": {
                "seo": row["seo_score"],
                "pain": row["pain_score"],
                "commercial": row["commercial_score"],
                "feasibility": row["feasibility_score"],
                "retention": row["retention_score"],
                "riskPenalty": row["risk_penalty"],
            },
            "assumptions": {
                "priceMonthlyUsd": row["price_monthly_usd"],
                "painFrequency": row["pain_frequency"],
                "painSeverity": row["pain_severity"],
                "minutesSavedPerOccurrence": row["minutes_saved_per_occurrence"],
                "buyerBudget": row["buyer_budget"],
                "buildHours": row["build_hours"],
                "maintenanceHoursMonth": row["maintenance_hours_month"],
                "infraCostMonth": row["infra_cost_month"],
                "supportComplexity": row["support_complexity"],
                "integrationCount": row["integration_count"],
                "recurrence": row["recurrence"],
                "retentionMechanism": row["retention_mechanism"],
            },
            "acquisitionWedge": row["acquisition_wedge"],
            "leadMagnet": row["lead_magnet"],
            "moneyPage": row["money_page"],
            "blogAngles": _json_list(row["blog_angles_json"]),
            "risks": risks,
            "validationPasses": row["validation_passes"],
            "redTeamPasses": row["red_team_passes"],
            "evidence": evidence,
            "keywords": keywords,
            "reviews": [
                {
                    "axis": item["axis"],
                    "verdict": item["verdict"],
                    "rationale": item["rationale"],
                    "confidence": item["confidence"],
                    "newRisks": _json_list(item["new_risks_json"]),
                }
                for item in bundle["reviews"]
            ],
        },
    }


def _markdown_report(document: dict[str, Any]) -> str:
    research = document["research"]
    lines = [
        "# NicheScout finalists",
        "",
        f"Generated: {document['generatedAt']}",
        "",
        f"**{research['finalists']} products grouped into {research['sites']} sites; "
        f"{research['activeHours']:g} active research hours.**",
        "",
        "Measured keyword metrics remain blank unless a cited source supplied them. The downstream SEO system, "
        "GSC, signups, and PostHog are deliberately outside this pre-deployment ranking.",
        "",
        "## Sites",
        "",
        "| Rank | Site | Audience | Products | Score | Cohesion |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for site in document["sites"]:
        lines.append(
            f"| {site['rank']} | {site['name']} | {site['audience']} | {len(site['productIds'])} | "
            f"{site['score']:.1f} | {site['cohesion']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Products",
            "",
            "| Rank | Product | Audience | Site | Score | Confidence |",
            "|---:|---|---|---|---:|---:|",
        ]
    )
    for idea in document["ideas"]:
        lines.append(
            f"| {idea['rank']} | {idea['name']} | {idea['audience']} | {idea['siteId']} | "
            f"{idea['score']:.1f} | {idea['research']['confidence']:.0%} |"
        )
    return "\n".join(lines) + "\n"


def _idea_id(row: dict[str, Any]) -> str:
    return f"{_slug(row['name'])[:42]}-{str(row['fingerprint'])[:8]}"


def _cluster_key(row: dict[str, Any]) -> str:
    return normalize_text(str(row["audience_cluster"]))


def _tokens(value: Any) -> set[str]:
    stop = {"a", "an", "and", "for", "in", "of", "the", "to", "with", "who", "small"}
    return {token for token in normalize_text(str(value)).split() if token not in stop}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _slug(value: Any) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return result or "untitled"


def _title(value: str) -> str:
    return " ".join(part.capitalize() for part in re.findall(r"[A-Za-z0-9]+", value))


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
