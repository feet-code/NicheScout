from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .models import CandidateIdea, ResearchMission, ResearchReview, ScoreCard, utc_now


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    provider TEXT NOT NULL,
    mission_json TEXT NOT NULL,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    source_count INTEGER NOT NULL DEFAULT 0,
    usage_json TEXT NOT NULL DEFAULT '{}',
    artifact_dir TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    target_customer TEXT NOT NULL,
    audience_cluster TEXT NOT NULL,
    canonical_problem TEXT NOT NULL,
    problem TEXT NOT NULL,
    solution TEXT NOT NULL,
    pricing_model TEXT NOT NULL,
    price_monthly_usd REAL NOT NULL,
    pain_frequency TEXT NOT NULL,
    pain_severity INTEGER NOT NULL,
    minutes_saved_per_occurrence REAL NOT NULL,
    buyer_budget TEXT NOT NULL,
    build_hours REAL NOT NULL,
    maintenance_hours_month REAL NOT NULL,
    infra_cost_month REAL NOT NULL,
    support_complexity TEXT NOT NULL,
    integration_count INTEGER NOT NULL,
    needs_scraping INTEGER NOT NULL,
    regulated_data INTEGER NOT NULL,
    recurrence TEXT NOT NULL,
    retention_mechanism TEXT NOT NULL,
    acquisition_wedge TEXT NOT NULL,
    lead_magnet TEXT NOT NULL,
    money_page TEXT NOT NULL,
    topic TEXT NOT NULL,
    blog_angles_json TEXT NOT NULL,
    risks_json TEXT NOT NULL,
    strategy TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'discovered',
    eligible INTEGER NOT NULL DEFAULT 1,
    seo_score REAL NOT NULL DEFAULT 0,
    pain_score REAL NOT NULL DEFAULT 0,
    commercial_score REAL NOT NULL DEFAULT 0,
    feasibility_score REAL NOT NULL DEFAULT 0,
    retention_score REAL NOT NULL DEFAULT 0,
    evidence_confidence REAL NOT NULL DEFAULT 0,
    worthiness_score REAL NOT NULL DEFAULT 0,
    uncertainty REAL NOT NULL DEFAULT 1,
    risk_penalty REAL NOT NULL DEFAULT 0,
    validation_passes INTEGER NOT NULL DEFAULT 0,
    red_team_passes INTEGER NOT NULL DEFAULT 0,
    kill_reasons_json TEXT NOT NULL DEFAULT '[]',
    score_notes_json TEXT NOT NULL DEFAULT '[]',
    first_action_id INTEGER REFERENCES actions(id),
    last_action_id INTEGER REFERENCES actions(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idea_id INTEGER NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
    claim TEXT NOT NULL,
    url TEXT NOT NULL,
    domain TEXT NOT NULL,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    observed_at TEXT,
    verified INTEGER NOT NULL DEFAULT 0,
    action_id INTEGER REFERENCES actions(id),
    created_at TEXT NOT NULL,
    UNIQUE(idea_id, url, claim)
);

CREATE TABLE IF NOT EXISTS keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idea_id INTEGER NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
    phrase TEXT NOT NULL,
    intent TEXT NOT NULL,
    search_volume REAL,
    cpc_usd REAL,
    difficulty REAL,
    serp_weakness REAL,
    measured INTEGER NOT NULL DEFAULT 0,
    source_url TEXT,
    action_id INTEGER REFERENCES actions(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(idea_id, phrase)
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idea_id INTEGER NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
    action_id INTEGER NOT NULL REFERENCES actions(id) ON DELETE CASCADE,
    axis TEXT NOT NULL,
    verdict TEXT NOT NULL,
    rationale TEXT NOT NULL,
    confidence REAL NOT NULL,
    new_risks_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    UNIQUE(idea_id, action_id, axis)
);

CREATE TABLE IF NOT EXISTS strategy_stats (
    strategy TEXT PRIMARY KEY,
    pulls INTEGER NOT NULL DEFAULT 0,
    reward_sum REAL NOT NULL DEFAULT 0,
    last_reward REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_health (
    task TEXT NOT NULL,
    model TEXT NOT NULL,
    blocked_until REAL NOT NULL DEFAULT 0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_status INTEGER,
    last_error TEXT,
    last_success_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(task, model)
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ideas_rank ON ideas(eligible DESC, worthiness_score DESC);
CREATE INDEX IF NOT EXISTS idx_ideas_reviews ON ideas(validation_passes, red_team_passes);
CREATE INDEX IF NOT EXISTS idx_evidence_idea ON evidence(idea_id);
CREATE INDEX IF NOT EXISTS idx_keywords_idea ON keywords(idea_id);
CREATE INDEX IF NOT EXISTS idx_reviews_idea ON reviews(idea_id);
CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status, id);
"""


class Repository:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 30000")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Repository":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def initialize(self, *, config_fingerprint: str | None = None) -> int:
        with self.transaction() as conn:
            conn.executescript(SCHEMA)
            conn.execute("PRAGMA user_version = 2")
            recovered = conn.execute(
                "UPDATE actions SET status = 'interrupted', completed_at = ?, "
                "error = COALESCE(error, 'Process stopped before this action committed') "
                "WHERE status = 'running'",
                (utc_now(),),
            ).rowcount
        if config_fingerprint:
            self.set_meta("last_config_fingerprint", config_fingerprint)
        return int(recovered)

    def start_action(self, mission: ResearchMission, provider: str) -> int:
        mission_json = json.dumps(mission.to_dict(), sort_keys=True)
        with self.transaction() as conn:
            recoverable = conn.execute(
                "SELECT id FROM actions WHERE status IN ('interrupted', 'deferred') "
                "AND provider = ? AND mission_json = ? ORDER BY id DESC LIMIT 1",
                (provider, mission_json),
            ).fetchone()
            if recoverable is not None:
                action_id = int(recoverable["id"])
                conn.execute(
                    "UPDATE actions SET started_at = ?, completed_at = NULL, status = 'running', "
                    "error = NULL WHERE id = ?",
                    (utc_now(), action_id),
                )
                return action_id
            cursor = conn.execute(
                "INSERT INTO actions(started_at, status, provider, mission_json) "
                "VALUES (?, 'running', ?, ?)",
                (utc_now(), provider, mission_json),
            )
        return int(cursor.lastrowid)

    def finish_action(
        self,
        action_id: int,
        *,
        status: str,
        candidate_count: int = 0,
        source_count: int = 0,
        usage: dict[str, Any] | None = None,
        artifact_dir: str | None = None,
        error: str | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """UPDATE actions
                   SET completed_at = ?, status = ?, candidate_count = ?, source_count = ?,
                       usage_json = ?, artifact_dir = ?, error = ?
                   WHERE id = ?""",
                (
                    utc_now(),
                    status,
                    candidate_count,
                    source_count,
                    json.dumps(usage or {}, ensure_ascii=False, default=str),
                    artifact_dir,
                    error,
                    action_id,
                ),
            )

    def upsert_candidate(
        self, candidate: CandidateIdea, action_id: int, strategy: str
    ) -> tuple[int, bool, float]:
        now = utc_now()
        existing = self.connection.execute(
            "SELECT id, worthiness_score FROM ideas WHERE fingerprint = ?",
            (candidate.fingerprint,),
        ).fetchone()
        is_new = existing is None
        old_score = float(existing["worthiness_score"]) if existing else 0.0
        values = (
            candidate.fingerprint,
            candidate.name,
            candidate.target_customer,
            candidate.audience_cluster,
            candidate.canonical_problem,
            candidate.problem,
            candidate.solution,
            candidate.pricing_model,
            candidate.price_monthly_usd,
            candidate.pain_frequency,
            candidate.pain_severity,
            candidate.minutes_saved_per_occurrence,
            candidate.buyer_budget,
            candidate.build_hours,
            candidate.maintenance_hours_month,
            candidate.infra_cost_month,
            candidate.support_complexity,
            candidate.integration_count,
            int(candidate.needs_scraping),
            int(candidate.regulated_data),
            candidate.recurrence,
            candidate.retention_mechanism,
            candidate.acquisition_wedge,
            candidate.lead_magnet,
            candidate.money_page,
            candidate.topic,
            json.dumps(candidate.blog_angles, ensure_ascii=False),
            json.dumps(candidate.risks, ensure_ascii=False),
            strategy,
            action_id,
            action_id,
            now,
            now,
        )
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO ideas(
                    fingerprint, name, target_customer, audience_cluster, canonical_problem,
                    problem, solution, pricing_model, price_monthly_usd, pain_frequency,
                    pain_severity, minutes_saved_per_occurrence, buyer_budget, build_hours,
                    maintenance_hours_month, infra_cost_month, support_complexity,
                    integration_count, needs_scraping, regulated_data, recurrence,
                    retention_mechanism, acquisition_wedge, lead_magnet, money_page, topic,
                    blog_angles_json, risks_json, strategy, first_action_id, last_action_id,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                    name = excluded.name,
                    target_customer = excluded.target_customer,
                    audience_cluster = excluded.audience_cluster,
                    canonical_problem = excluded.canonical_problem,
                    problem = excluded.problem,
                    solution = excluded.solution,
                    pricing_model = excluded.pricing_model,
                    price_monthly_usd = excluded.price_monthly_usd,
                    pain_frequency = excluded.pain_frequency,
                    pain_severity = excluded.pain_severity,
                    minutes_saved_per_occurrence = excluded.minutes_saved_per_occurrence,
                    buyer_budget = excluded.buyer_budget,
                    build_hours = excluded.build_hours,
                    maintenance_hours_month = excluded.maintenance_hours_month,
                    infra_cost_month = excluded.infra_cost_month,
                    support_complexity = excluded.support_complexity,
                    integration_count = excluded.integration_count,
                    needs_scraping = excluded.needs_scraping,
                    regulated_data = excluded.regulated_data,
                    recurrence = excluded.recurrence,
                    retention_mechanism = excluded.retention_mechanism,
                    acquisition_wedge = excluded.acquisition_wedge,
                    lead_magnet = excluded.lead_magnet,
                    money_page = excluded.money_page,
                    topic = excluded.topic,
                    blog_angles_json = excluded.blog_angles_json,
                    risks_json = excluded.risks_json,
                    last_action_id = excluded.last_action_id,
                    updated_at = excluded.updated_at
                """,
                values,
            )
            row = conn.execute(
                "SELECT id FROM ideas WHERE fingerprint = ?", (candidate.fingerprint,)
            ).fetchone()
            idea_id = int(row["id"])
            for evidence in candidate.evidence:
                conn.execute(
                    """INSERT INTO evidence(
                           idea_id, claim, url, domain, title, source_type, observed_at,
                           verified, action_id, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(idea_id, url, claim) DO UPDATE SET
                           verified = MAX(evidence.verified, excluded.verified),
                           observed_at = COALESCE(excluded.observed_at, evidence.observed_at),
                           action_id = excluded.action_id""",
                    (
                        idea_id,
                        evidence.claim,
                        evidence.url,
                        evidence.domain,
                        evidence.title,
                        evidence.source_type,
                        evidence.observed_at,
                        int(evidence.verified),
                        action_id,
                        now,
                    ),
                )
            for keyword in candidate.keywords:
                conn.execute(
                    """INSERT INTO keywords(
                           idea_id, phrase, intent, search_volume, cpc_usd, difficulty,
                           serp_weakness, measured, source_url, action_id, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(idea_id, phrase) DO UPDATE SET
                           intent = excluded.intent,
                           search_volume = COALESCE(excluded.search_volume, keywords.search_volume),
                           cpc_usd = COALESCE(excluded.cpc_usd, keywords.cpc_usd),
                           difficulty = COALESCE(excluded.difficulty, keywords.difficulty),
                           serp_weakness = COALESCE(excluded.serp_weakness, keywords.serp_weakness),
                           measured = MAX(keywords.measured, excluded.measured),
                           source_url = COALESCE(excluded.source_url, keywords.source_url),
                           action_id = excluded.action_id,
                           updated_at = excluded.updated_at""",
                    (
                        idea_id,
                        keyword.phrase.strip().lower(),
                        keyword.intent,
                        keyword.search_volume,
                        keyword.cpc_usd,
                        keyword.difficulty,
                        keyword.serp_weakness,
                        int(keyword.measured),
                        keyword.source_url,
                        action_id,
                        now,
                        now,
                    ),
                )
        return idea_id, is_new, old_score

    def add_reviews(self, reviews: Sequence[ResearchReview], action_id: int) -> list[int]:
        touched: list[int] = []
        now = utc_now()
        with self.transaction() as conn:
            for review in reviews:
                row = conn.execute(
                    "SELECT id, risks_json FROM ideas WHERE fingerprint = ?",
                    (review.fingerprint,),
                ).fetchone()
                if row is None:
                    continue
                idea_id = int(row["id"])
                cursor = conn.execute(
                    """INSERT OR IGNORE INTO reviews(
                           idea_id, action_id, axis, verdict, rationale, confidence,
                           new_risks_json, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        idea_id,
                        action_id,
                        review.axis,
                        review.verdict,
                        review.rationale,
                        max(0.0, min(1.0, review.confidence)),
                        json.dumps(review.new_risks, ensure_ascii=False),
                        now,
                    ),
                )
                if not cursor.rowcount:
                    continue
                risks = list(dict.fromkeys(json.loads(row["risks_json"] or "[]") + review.new_risks))
                is_red_team = review.axis == "red_team"
                status_clause = ", status = 'killed', eligible = 0" if review.verdict == "falsified" else ""
                conn.execute(
                    f"""UPDATE ideas SET
                           validation_passes = validation_passes + ?,
                           red_team_passes = red_team_passes + ?,
                           risks_json = ?, last_action_id = ?, updated_at = ?
                           {status_clause}
                       WHERE id = ?""",
                    (
                        0 if is_red_team else 1,
                        1 if is_red_team else 0,
                        json.dumps(risks, ensure_ascii=False),
                        action_id,
                        now,
                        idea_id,
                    ),
                )
                touched.append(idea_id)
        return touched

    def save_score(self, idea_id: int, score: ScoreCard) -> None:
        row = self.connection.execute(
            "SELECT status FROM ideas WHERE id = ?", (idea_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown idea id: {idea_id}")
        status = "killed" if row["status"] == "killed" else score.recommended_status
        eligible = False if status == "killed" else score.eligible
        with self.transaction() as conn:
            conn.execute(
                """UPDATE ideas SET
                       seo_score = ?, pain_score = ?, commercial_score = ?,
                       feasibility_score = ?, retention_score = ?, evidence_confidence = ?,
                       worthiness_score = ?, uncertainty = ?, risk_penalty = ?, eligible = ?,
                       status = ?, kill_reasons_json = ?, score_notes_json = ?, updated_at = ?
                   WHERE id = ?""",
                (
                    score.seo,
                    score.pain,
                    score.commercial,
                    score.feasibility,
                    score.retention,
                    score.evidence_confidence,
                    score.worthiness,
                    score.uncertainty,
                    score.risk_penalty,
                    int(eligible),
                    status,
                    json.dumps(score.kill_reasons, ensure_ascii=False),
                    json.dumps(score.notes, ensure_ascii=False),
                    utc_now(),
                    idea_id,
                ),
            )

    def get_idea_bundle(self, idea_id: int) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown idea id: {idea_id}")
        return {
            "idea": dict(row),
            "evidence": [
                dict(item)
                for item in self.connection.execute(
                    "SELECT * FROM evidence WHERE idea_id = ? ORDER BY verified DESC, id",
                    (idea_id,),
                )
            ],
            "keywords": [
                dict(item)
                for item in self.connection.execute(
                    "SELECT * FROM keywords WHERE idea_id = ? ORDER BY measured DESC, id",
                    (idea_id,),
                )
            ],
            "reviews": [
                dict(item)
                for item in self.connection.execute(
                    "SELECT * FROM reviews WHERE idea_id = ? ORDER BY id", (idea_id,)
                )
            ],
        }

    def candidate_context(self, idea_id: int) -> dict[str, Any]:
        bundle = self.get_idea_bundle(idea_id)
        idea = bundle["idea"]
        return {
            "id": idea["id"],
            "fingerprint": idea["fingerprint"],
            "name": idea["name"],
            "target_customer": idea["target_customer"],
            "audience_cluster": idea["audience_cluster"],
            "canonical_problem": idea["canonical_problem"],
            "problem": idea["problem"],
            "solution": idea["solution"],
            "price_monthly_usd": idea["price_monthly_usd"],
            "build_hours": idea["build_hours"],
            "maintenance_hours_month": idea["maintenance_hours_month"],
            "topic": idea["topic"],
            "worthiness_score": idea["worthiness_score"],
            "evidence_confidence": idea["evidence_confidence"],
            "validation_passes": idea["validation_passes"],
            "red_team_passes": idea["red_team_passes"],
            "evidence": [
                {key: item[key] for key in ("claim", "url", "source_type", "verified")}
                for item in bundle["evidence"][:10]
            ],
            "keywords": [item["phrase"] for item in bundle["keywords"][:16]],
            "prior_reviews": [
                {key: item[key] for key in ("axis", "verdict", "rationale", "confidence")}
                for item in bundle["reviews"][-8:]
            ],
        }

    def list_ideas(
        self,
        limit: int | None = None,
        *,
        eligible_only: bool = False,
        pool_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["eligible = 1"] if eligible_only else []
        sql = "SELECT * FROM ideas"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY worthiness_score DESC, evidence_confidence DESC, id"
        actual_limit = limit if limit is not None else pool_limit
        params: tuple[Any, ...] = ()
        if actual_limit is not None:
            sql += " LIMIT ?"
            params = (actual_limit,)
        return [dict(row) for row in self.connection.execute(sql, params)]

    def review_axis_counts(self, idea_id: int) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT axis, COUNT(*) AS n FROM reviews WHERE idea_id = ? GROUP BY axis",
            (idea_id,),
        )
        return {str(row["axis"]): int(row["n"]) for row in rows}

    def idea_count(self, *, eligible_only: bool = False) -> int:
        where = " WHERE eligible = 1" if eligible_only else ""
        row = self.connection.execute(f"SELECT COUNT(*) AS n FROM ideas{where}").fetchone()
        return int(row["n"])

    def action_count(self, *, succeeded_only: bool = False) -> int:
        where = " WHERE status = 'succeeded'" if succeeded_only else ""
        row = self.connection.execute(f"SELECT COUNT(*) AS n FROM actions{where}").fetchone()
        return int(row["n"])

    def action_status_counts(self) -> dict[str, int]:
        return {
            str(row["status"]): int(row["n"])
            for row in self.connection.execute(
                "SELECT status, COUNT(*) AS n FROM actions GROUP BY status ORDER BY status"
            )
        }

    def recent_errors(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM actions WHERE status IN ('failed', 'interrupted') "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        ]

    def progress_counts(self) -> dict[str, int]:
        row = self.connection.execute(
            """SELECT
                   COUNT(*) AS ideas,
                   SUM(CASE WHEN eligible = 1 THEN 1 ELSE 0 END) AS eligible,
                   SUM(CASE WHEN eligible = 1 AND validation_passes > 0 THEN 1 ELSE 0 END) AS reviewed,
                   SUM(CASE WHEN eligible = 1 AND red_team_passes > 0 THEN 1 ELSE 0 END) AS red_teamed
               FROM ideas"""
        ).fetchone()
        return {key: int(row[key] or 0) for key in ("ideas", "eligible", "reviewed", "red_teamed")}

    def strategy_stats(self) -> dict[str, dict[str, Any]]:
        return {
            str(row["strategy"]): dict(row)
            for row in self.connection.execute("SELECT * FROM strategy_stats")
        }

    def record_strategy_reward(self, strategy: str, reward: float) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO strategy_stats(strategy, pulls, reward_sum, last_reward, updated_at)
                   VALUES (?, 1, ?, ?, ?)
                   ON CONFLICT(strategy) DO UPDATE SET
                       pulls = pulls + 1,
                       reward_sum = reward_sum + excluded.last_reward,
                       last_reward = excluded.last_reward,
                       updated_at = excluded.updated_at""",
                (strategy, reward, reward, utc_now()),
            )

    def get_model_health(self, task: str, model: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM model_health WHERE task = ? AND model = ?", (task, model)
        ).fetchone()
        return dict(row) if row else {
            "task": task,
            "model": model,
            "blocked_until": 0.0,
            "consecutive_failures": 0,
            "last_status": None,
            "last_error": None,
        }

    def list_model_health(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM model_health ORDER BY task, model"
            )
        ]

    def record_model_failure(
        self,
        task: str,
        model: str,
        *,
        status: int | None,
        error: str,
        blocked_until: float,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO model_health(
                       task, model, blocked_until, consecutive_failures, last_status,
                       last_error, updated_at
                   ) VALUES (?, ?, ?, 1, ?, ?, ?)
                   ON CONFLICT(task, model) DO UPDATE SET
                       blocked_until = excluded.blocked_until,
                       consecutive_failures = model_health.consecutive_failures + 1,
                       last_status = excluded.last_status,
                       last_error = excluded.last_error,
                       updated_at = excluded.updated_at""",
                (task, model, blocked_until, status, error[:2000], utc_now()),
            )

    def record_model_success(self, task: str, model: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO model_health(
                       task, model, blocked_until, consecutive_failures, last_success_at, updated_at
                   ) VALUES (?, ?, 0, 0, ?, ?)
                   ON CONFLICT(task, model) DO UPDATE SET
                       blocked_until = 0,
                       consecutive_failures = 0,
                       last_status = NULL,
                       last_error = NULL,
                       last_success_at = excluded.last_success_at,
                       updated_at = excluded.updated_at""",
                (task, model, utc_now(), utc_now()),
            )

    def set_meta(self, key: str, value: Any) -> None:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO meta(key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
                (key, encoded, utc_now()),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def add_active_seconds(self, seconds: float) -> float:
        current = float(self.get_meta("active_seconds", 0.0))
        updated = max(0.0, current + max(0.0, seconds))
        self.set_meta("active_seconds", updated)
        self.set_meta("last_heartbeat_epoch", time.time())
        return updated

    def active_seconds(self) -> float:
        return float(self.get_meta("active_seconds", 0.0))

    def mark_finalized(self, output_path: str, finalist_count: int, site_count: int) -> None:
        self.set_meta(
            "finalization",
            {
                "completed_at": utc_now(),
                "output_path": output_path,
                "finalist_count": finalist_count,
                "site_count": site_count,
            },
        )

    def finalization(self) -> dict[str, Any] | None:
        return self.get_meta("finalization")
