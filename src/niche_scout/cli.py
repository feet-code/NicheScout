from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from .config import Settings
from .db import Repository
from .engine import AgentEngine, InsufficientFinalists
from .events import configure, emit
from .exporter import PortfolioNotReady, export_portfolio, qualified_idea_count
from .planner import ResearchPlanner
from .providers.mock import MockResearchProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="niche-scout",
        description="Finite, restart-safe pre-deployment micro-SaaS research tournament",
    )
    parser.add_argument("--config", default="config.toml", help="TOML config (default: config.toml)")
    parser.add_argument("--verbose", action="store_true", help="Print full JSON events")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create durable state directories and SQLite database")

    doctor = sub.add_parser("doctor", help="Check configuration, key, SDK, disk, and database")
    doctor.add_argument("--live", action="store_true", help="Make one ungrounded Gemini test call")

    for command in ("run", "resume"):
        run = sub.add_parser(command, help="Run/resume until the active-time budget is complete")
        run.add_argument("--mock", action="store_true", help="Use deterministic fake evidence")
        run.add_argument("--max-actions", type=int, help="Stop after N successful actions this invocation")
        run.add_argument("--active-hours", type=float, help="Override target active research hours")
        run.add_argument("--no-wait", action="store_true", help="Exit when every model is cooling down")
        run.add_argument("--no-finalize", action="store_true", help="Do not auto-export at the time limit")
        run.add_argument("--out", help="Final ideas.json path")

    status = sub.add_parser("status", help="Show persisted tournament/model progress")
    status.add_argument("--mock", action="store_true", help="Inspect the isolated mock state")

    leaderboard = sub.add_parser("leaderboard", help="Show current evidence-weighted ranking")
    leaderboard.add_argument("--limit", type=int, default=25)

    show = sub.add_parser("show", help="Print one idea with evidence, queries, and reviews")
    show.add_argument("idea_id", type=int)

    finalize = sub.add_parser("finalize", help="Export the deterministic version-2 portfolio")
    finalize.add_argument("--force", action="store_true", help="Export before time limit (quality gates remain)")
    finalize.add_argument("--out", help="Output path (default: exports/ideas.json)")

    errors = sub.add_parser("errors", help="Show recent failed/interrupted actions")
    errors.add_argument("--limit", type=int, default=20)

    sub.add_parser("rescore", help="Recalculate all deterministic scores")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    settings = Settings.load(config_path if config_path.exists() else None)
    if getattr(args, "active_hours", None) is not None:
        settings = replace(settings, target_active_hours=args.active_hours)
        settings.validate()
    if getattr(args, "mock", False):
        settings = replace(
            settings,
            db_path=settings.db_path.with_name("mock-" + settings.db_path.name),
            output_dir=settings.output_dir / "mock",
            artifacts_dir=settings.artifacts_dir.with_name("mock-artifacts"),
            event_log_path=settings.event_log_path.with_name("mock-events.jsonl"),
        )
    settings.ensure_directories()
    configure(settings.event_log_path, verbose=args.verbose)

    with Repository(settings.db_path) as repo:
        recovered = repo.initialize(config_fingerprint=settings.fingerprint())
        if recovered:
            emit(
                "interrupted_actions_recovered",
                level="warning",
                actions=recovered,
                message="Recovered unfinished action(s); their deterministic mission will be reused",
            )

        if args.command == "init":
            emit(
                "initialized",
                message="NicheScout state initialized",
                db_path=str(settings.db_path.resolve()),
                event_log=str(settings.event_log_path.resolve()),
            )
            return

        if args.command == "doctor":
            _doctor(settings, repo, live=args.live)
            return

        if args.command in {"run", "resume"}:
            if args.max_actions is not None and args.max_actions < 1:
                raise SystemExit("--max-actions must be at least 1")
            provider = MockResearchProvider() if args.mock else _gemini_provider(settings, repo)
            engine = AgentEngine(settings, repo, provider)
            try:
                with _run_lock(settings.db_path.with_suffix(".lock")):
                    result = engine.run_until_budget(
                        max_actions=args.max_actions,
                        no_wait=args.no_wait,
                        auto_finalize=not args.no_finalize,
                        output_path=args.out,
                    )
            except (InsufficientFinalists, PortfolioNotReady) as exc:
                raise SystemExit(f"Portfolio not ready: {exc}") from exc
            print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
            return

        if args.command == "status":
            print(json.dumps(_status(settings, repo), indent=2, ensure_ascii=False, default=str))
            return

        if args.command == "leaderboard":
            _leaderboard(repo.list_ideas(args.limit))
            return

        if args.command == "show":
            print(json.dumps(repo.get_idea_bundle(args.idea_id), indent=2, ensure_ascii=False))
            return

        if args.command == "finalize":
            if not args.force and repo.active_seconds() < settings.target_active_seconds:
                remaining = (settings.target_active_seconds - repo.active_seconds()) / 3600
                raise SystemExit(
                    f"Research time box has {remaining:.2f} active hours remaining. "
                    "Use --force only if you intentionally want an early snapshot."
                )
            try:
                output = export_portfolio(
                    repo,
                    settings,
                    args.out or settings.output_dir / "ideas.json",
                )
            except PortfolioNotReady as exc:
                raise SystemExit(f"Portfolio not ready: {exc}") from exc
            print(output)
            return

        if args.command == "errors":
            errors = repo.recent_errors(args.limit)
            print(json.dumps(errors, indent=2, ensure_ascii=False))
            return

        if args.command == "rescore":
            count = AgentEngine(settings, repo, MockResearchProvider()).rescore_all()
            emit("rescored", ideas=count)


def _gemini_provider(settings: Settings, repo: Repository):
    from .providers.gemini import GeminiResearchProvider

    return GeminiResearchProvider(settings, repo)


def _status(settings: Settings, repo: Repository) -> dict[str, Any]:
    counts = repo.progress_counts()
    active = repo.active_seconds()
    try:
        qualified = qualified_idea_count(repo, settings)
    except Exception:
        qualified = 0
    next_mission = ResearchPlanner(settings).next_mission(repo)
    return {
        "database": str(settings.db_path.resolve()),
        "activeHours": round(active / 3600, 4),
        "targetActiveHours": settings.target_active_hours,
        "remainingActiveHours": round(max(0.0, settings.target_active_seconds - active) / 3600, 4),
        "progress": {
            **counts,
            "qualified": qualified,
            "finalistTarget": settings.finalist_target,
            "siteTarget": settings.site_target,
        },
        "actions": repo.action_status_counts(),
        "nextMission": {"mode": next_mission.mode, "strategy": next_mission.strategy},
        "models": repo.list_model_health(),
        "finalization": repo.finalization(),
        "eventLog": str(settings.event_log_path.resolve()),
    }


def _leaderboard(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No candidates yet. Smoke-test with: niche-scout run --mock --max-actions 1")
        return
    print("ID\tSCORE\tCONF\tVAL\tRED\tSTATUS\tIDEA")
    for row in rows:
        print(
            f"{row['id']}\t{float(row['worthiness_score']):.1f}\t"
            f"{float(row['evidence_confidence']):.0%}\t{row['validation_passes']}\t"
            f"{row['red_team_passes']}\t{row['status']}\t{row['name']}"
        )


def _doctor(settings: Settings, repo: Repository, *, live: bool) -> None:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    check("python", sys.version_info >= (3, 11), platform.python_version())
    sdk_installed = _module_exists("google.genai")
    check(
        "google-genai",
        sdk_installed,
        "installed" if sdk_installed else "run: python -m pip install -e .",
    )
    key = os.getenv(settings.api_key_env, "")
    check(settings.api_key_env, bool(key), "set" if key else "missing from .env/environment")
    integrity = repo.connection.execute("PRAGMA integrity_check").fetchone()[0]
    check("sqlite", integrity == "ok", str(integrity))
    free_bytes = shutil.disk_usage(settings.db_path.parent).free
    check("disk", free_bytes >= 500 * 1024 * 1024, f"{free_bytes / 1024**3:.2f} GiB free")
    check(
        "model-policy",
        True,
        "configuration passed the free-text and Search-grounding capability allowlists",
    )

    if live and all(item["ok"] for item in checks[:3]):
        from .providers.gemini import GeminiGateway

        try:
            response = GeminiGateway(settings, repo).generate_text(
                task="doctor",
                models=settings.synthesis_models,
                system="You are a connectivity check.",
                prompt="Reply with exactly: NICHE_SCOUT_OK",
                grounded=False,
            )
            check("live-gemini", "NICHE_SCOUT_OK" in response.text, f"model={response.model}")
        except Exception as exc:
            check("live-gemini", False, f"{type(exc).__name__}: {exc}")

    print(json.dumps(checks, indent=2))
    failed = [item for item in checks if not item["ok"]]
    if failed:
        raise SystemExit(f"Doctor found {len(failed)} problem(s).")


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


@contextmanager
def _run_lock(path: Path):
    """Cross-platform advisory lock preventing two writers from running the agent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SystemExit(
                f"Another NicheScout process already holds {path}. Stop it before starting a second writer."
            ) from exc
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


if __name__ == "__main__":
    main()
