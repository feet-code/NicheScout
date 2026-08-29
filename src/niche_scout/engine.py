from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from .config import Settings
from .db import Repository
from .events import emit
from .planner import ResearchPlanner
from .providers.base import ResearchProvider
from .scoring import ScoreEngine


class InsufficientFinalists(RuntimeError):
    """The time box ended before enough hard-gate survivors existed."""


@dataclass(slots=True)
class ActionResult:
    action_id: int
    mode: str
    strategy: str
    candidates: int
    new_ideas: int
    reviewed_ideas: int
    reward: float
    best_score: float


@dataclass(slots=True)
class RunResult:
    reason: str
    actions_this_run: int
    active_seconds: float
    eligible_ideas: int
    output_path: str | None = None


class AgentEngine:
    """Run one durable research action at a time."""

    def __init__(
        self,
        settings: Settings,
        repo: Repository,
        provider: ResearchProvider,
        planner: ResearchPlanner | None = None,
        scorer: ScoreEngine | None = None,
    ):
        self.settings = settings
        self.repo = repo
        self.provider = provider
        self.planner = planner or ResearchPlanner(settings)
        self.scorer = scorer or ScoreEngine(settings)

    def run_action(self) -> ActionResult:
        mission = self.planner.next_mission(self.repo)
        action_id = self.repo.start_action(mission, self.provider.name)
        started = time.monotonic()
        emit(
            "research_started",
            action_id=action_id,
            mode=mission.mode,
            strategy=mission.strategy,
            ideas=self.repo.idea_count(),
        )
        try:
            batch = self.provider.research(mission, self.settings)
            artifact_dir = self._persist_artifacts(action_id, mission.mode, batch)

            touched: set[int] = set()
            old_scores: dict[int, float] = {}
            new_ideas = 0
            for candidate in batch.candidates:
                idea_id, is_new, old_score = self.repo.upsert_candidate(
                    candidate, action_id, mission.strategy
                )
                touched.add(idea_id)
                old_scores[idea_id] = old_score
                new_ideas += int(is_new)

            reviewed = self.repo.add_reviews(batch.reviews, action_id)
            touched.update(reviewed)

            scores: list[float] = []
            confidences: list[float] = []
            improvements: list[float] = []
            for idea_id in sorted(touched):
                score = self.scorer.score(self.repo.get_idea_bundle(idea_id))
                self.repo.save_score(idea_id, score)
                scores.append(score.worthiness)
                confidences.append(score.evidence_confidence)
                improvements.append(
                    max(0.0, score.worthiness - old_scores.get(idea_id, score.worthiness))
                )
                emit(
                    "idea_scored",
                    action_id=action_id,
                    idea_id=idea_id,
                    score=score.worthiness,
                    confidence=score.evidence_confidence,
                    status=score.recommended_status,
                )

            reward = self._reward(
                candidate_count=len(batch.candidates),
                new_count=new_ideas,
                scores=scores,
                confidences=confidences,
                improvements=improvements,
            )
            if mission.mode == "discover":
                self.repo.record_strategy_reward(mission.strategy, reward)

            self.repo.finish_action(
                action_id,
                status="succeeded",
                candidate_count=len(batch.candidates),
                source_count=len(batch.consulted_sources),
                usage=batch.usage,
                artifact_dir=str(artifact_dir),
            )
            mark_persisted = getattr(self.provider, "mark_persisted", None)
            if callable(mark_persisted):
                try:
                    mark_persisted(mission)
                except Exception as exc:
                    emit(
                        "stage_cache_cleanup_failed",
                        level="warning",
                        action_id=action_id,
                        message=f"Durable action succeeded; stale cache cleanup failed: {exc}",
                    )
            result = ActionResult(
                action_id=action_id,
                mode=mission.mode,
                strategy=mission.strategy,
                candidates=len(batch.candidates),
                new_ideas=new_ideas,
                reviewed_ideas=len(set(reviewed)),
                reward=reward,
                best_score=max(scores, default=0.0),
            )
            progress = self.repo.progress_counts()
            emit(
                "research_succeeded",
                action_id=action_id,
                mode=mission.mode,
                strategy=mission.strategy,
                ideas=progress["ideas"],
                progress=(
                    f"eligible={progress['eligible']} reviewed={progress['reviewed']} "
                    f"red_teamed={progress['red_teamed']}"
                ),
            )
            return result
        except KeyboardInterrupt:
            self.repo.finish_action(action_id, status="interrupted", error="Interrupted by user")
            emit("research_interrupted", level="warning", action_id=action_id)
            raise
        except Exception as exc:
            retry_at = getattr(exc, "retry_at", None) or self._provider_retry_at()
            status = "deferred" if retry_at else "failed"
            self.repo.finish_action(
                action_id,
                status=status,
                error=f"{type(exc).__name__}: {exc}",
                usage={"retry_at": retry_at} if retry_at else {},
            )
            emit(
                "research_deferred" if retry_at else "research_failed",
                level="warning" if retry_at else "error",
                action_id=action_id,
                wait_seconds=max(0, round(float(retry_at) - time.time())) if retry_at else None,
                message=str(exc),
                error_type=type(exc).__name__,
            )
            raise
        finally:
            # API and analysis time count toward the time box. Quota waits do not.
            self.repo.add_active_seconds(time.monotonic() - started)

    def run_until_budget(
        self,
        *,
        max_actions: int | None = None,
        no_wait: bool = False,
        auto_finalize: bool = True,
        output_path: str | Path | None = None,
    ) -> RunResult:
        invocation_limit = self.settings.max_actions if max_actions is None else max_actions
        completed = 0
        while True:
            finalization = self.repo.finalization()
            if finalization:
                return RunResult(
                    reason="already_finalized",
                    actions_this_run=completed,
                    active_seconds=self.repo.active_seconds(),
                    eligible_ideas=self.repo.idea_count(eligible_only=True),
                    output_path=str(finalization.get("output_path") or ""),
                )

            if self.repo.active_seconds() >= self.settings.target_active_seconds:
                return self._finish_time_box(completed, auto_finalize, output_path)

            if invocation_limit and completed >= invocation_limit:
                return RunResult(
                    reason="max_actions",
                    actions_this_run=completed,
                    active_seconds=self.repo.active_seconds(),
                    eligible_ideas=self.repo.idea_count(eligible_only=True),
                )

            try:
                self.run_action()
                completed += 1
                if self.settings.action_pause_seconds > 0:
                    time.sleep(self.settings.action_pause_seconds)
            except KeyboardInterrupt:
                return RunResult(
                    reason="interrupted",
                    actions_this_run=completed,
                    active_seconds=self.repo.active_seconds(),
                    eligible_ideas=self.repo.idea_count(eligible_only=True),
                )
            except Exception as exc:
                retry_at = getattr(exc, "retry_at", None) or self._provider_retry_at()
                if not retry_at:
                    raise
                if no_wait:
                    return RunResult(
                        reason="models_cooling_down",
                        actions_this_run=completed,
                        active_seconds=self.repo.active_seconds(),
                        eligible_ideas=self.repo.idea_count(eligible_only=True),
                    )
                try:
                    self._wait_for_models(float(retry_at))
                except KeyboardInterrupt:
                    return RunResult(
                        reason="interrupted",
                        actions_this_run=completed,
                        active_seconds=self.repo.active_seconds(),
                        eligible_ideas=self.repo.idea_count(eligible_only=True),
                    )

    def rescore(self, idea_id: int) -> dict[str, Any]:
        score = self.scorer.score(self.repo.get_idea_bundle(idea_id))
        self.repo.save_score(idea_id, score)
        return self.repo.get_idea_bundle(idea_id)["idea"]

    def rescore_all(self) -> int:
        ideas = self.repo.list_ideas()
        for idea in ideas:
            self.rescore(int(idea["id"]))
        return len(ideas)

    def _finish_time_box(
        self,
        completed: int,
        auto_finalize: bool,
        output_path: str | Path | None,
    ) -> RunResult:
        from .exporter import export_portfolio, qualified_idea_count

        eligible = self.repo.idea_count(eligible_only=True)
        qualified = qualified_idea_count(self.repo, self.settings)
        if qualified < self.settings.finalist_target:
            raise InsufficientFinalists(
                f"The time box ended with {eligible} hard-gate survivors but only {qualified} "
                f"completed every final evidence/review gate; {self.settings.finalist_target} are required. "
                "Extend target_active_hours and resume."
            )
        path: str | None = None
        if auto_finalize:
            exported = export_portfolio(
                self.repo,
                self.settings,
                output_path or self.settings.output_dir / "ideas.json",
            )
            path = str(exported)
        return RunResult(
            reason="time_budget_complete",
            actions_this_run=completed,
            active_seconds=self.repo.active_seconds(),
            eligible_ideas=eligible,
            output_path=path,
        )

    def _wait_for_models(self, retry_at: float) -> None:
        while time.time() < retry_at:
            remaining = max(0.0, retry_at - time.time())
            emit(
                "quota_wait_heartbeat",
                level="warning",
                wait_seconds=round(remaining),
                ideas=self.repo.idea_count(),
                message="All configured models are cooling down; Ctrl+C is safe",
            )
            time.sleep(min(self.settings.heartbeat_seconds, remaining))

    def _provider_retry_at(self) -> float | None:
        getter = getattr(self.provider, "retry_at", None)
        return getter() if callable(getter) else None

    def _persist_artifacts(self, action_id: int, mode: str, batch: Any) -> Path:
        destination = self.settings.artifacts_dir / f"{action_id:06d}-{mode}"
        destination.mkdir(parents=True, exist_ok=True)
        artifacts = dict(batch.artifacts)
        artifacts["batch.json"] = json.dumps(
            {
                "research_summary": batch.research_summary,
                "candidates": [candidate.to_dict() for candidate in batch.candidates],
                "reviews": [review.to_dict() for review in batch.reviews],
                "consulted_sources": [source.to_dict() for source in batch.consulted_sources],
                "usage": batch.usage,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        for filename, content in artifacts.items():
            _atomic_text(destination / Path(filename).name, str(content))
        return destination

    @staticmethod
    def _reward(
        *,
        candidate_count: int,
        new_count: int,
        scores: list[float],
        confidences: list[float],
        improvements: list[float],
    ) -> float:
        if not scores:
            return 0.0
        novelty = new_count / max(1, candidate_count)
        quality = fmean(scores) / 100
        confidence = fmean(confidences)
        improvement = min(1.0, fmean(improvements) / 15) if improvements else 0.0
        return round(
            min(1.0, 0.30 * novelty + 0.35 * quality + 0.25 * confidence + 0.10 * improvement),
            5,
        )


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
