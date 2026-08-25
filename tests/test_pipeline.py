from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from niche_scout.config import Settings
from niche_scout.db import Repository
from niche_scout.engine import AgentEngine
from niche_scout.exporter import export_portfolio, qualified_idea_count
from niche_scout.planner import ResearchPlanner
from niche_scout.providers.mock import MockResearchProvider


class PipelineTests(unittest.TestCase):
    def test_small_tournament_exports_exact_grouped_portfolio(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = _settings(root)
            settings.ensure_directories()
            with Repository(settings.db_path) as repo:
                repo.initialize(config_fingerprint=settings.fingerprint())
                engine = AgentEngine(settings, repo, MockResearchProvider())
                with redirect_stdout(io.StringIO()):
                    for _ in range(80):
                        engine.run_action()
                        if qualified_idea_count(repo, settings) >= settings.finalist_target:
                            break
                self.assertGreaterEqual(qualified_idea_count(repo, settings), 10)
                output = export_portfolio(repo, settings, root / "ideas.json")
                document = json.loads(output.read_text(encoding="utf-8"))

                self.assertEqual(document["version"], 2)
                self.assertEqual(len(document["ideas"]), 10)
                self.assertEqual(len(document["sites"]), 2)
                self.assertEqual({len(site["productIds"]) for site in document["sites"]}, {5})
                product_ids = {idea["id"] for idea in document["ideas"]}
                self.assertEqual(len(product_ids), 10)
                self.assertEqual(
                    product_ids,
                    {product for site in document["sites"] for product in site["productIds"]},
                )
                site_ids = {site["id"] for site in document["sites"]}
                self.assertTrue(all(idea["siteId"] in site_ids for idea in document["ideas"]))
                self.assertTrue((root / "finalists.md").exists())

    def test_interrupted_action_reuses_id_and_mission(self) -> None:
        with TemporaryDirectory() as directory:
            settings = _settings(Path(directory))
            with Repository(settings.db_path) as repo:
                repo.initialize()
                mission = ResearchPlanner(settings).next_mission(repo)
                action_id = repo.start_action(mission, "mock")

            with Repository(settings.db_path) as repo:
                recovered = repo.initialize()
                self.assertEqual(recovered, 1)
                resumed_mission = ResearchPlanner(settings).next_mission(repo)
                resumed_id = repo.start_action(resumed_mission, "mock")
                self.assertEqual(resumed_mission.to_dict(), mission.to_dict())
                self.assertEqual(resumed_id, action_id)

    def test_configured_hard_gate_kills_expensive_candidate(self) -> None:
        with TemporaryDirectory() as directory:
            settings = _settings(Path(directory))
            provider = MockResearchProvider()
            with Repository(settings.db_path) as repo:
                repo.initialize()
                mission = ResearchPlanner(settings).next_mission(repo)
                candidate = provider.research(mission, settings).candidates[0]
                candidate.build_hours = settings.max_build_hours + 1
                action_id = repo.start_action(mission, provider.name)
                idea_id, _, _ = repo.upsert_candidate(candidate, action_id, mission.strategy)
                engine = AgentEngine(settings, repo, provider)
                idea = engine.rescore(idea_id)
                self.assertEqual(idea["eligible"], 0)
                self.assertEqual(idea["status"], "killed")


def _settings(root: Path) -> Settings:
    return replace(
        Settings(),
        db_path=root / "state.db",
        output_dir=root / "exports",
        artifacts_dir=root / "artifacts",
        event_log_path=root / "events.jsonl",
        target_active_hours=1,
        finalist_target=10,
        products_per_site=5,
        discovery_pool_target=20,
        deep_research_pool=15,
        red_team_pool=10,
        discovery_batch_size=5,
        review_batch_size=3,
        action_pause_seconds=0,
    )


if __name__ == "__main__":
    unittest.main()
