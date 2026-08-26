from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from niche_scout.config import Settings
from niche_scout.db import Repository
from niche_scout.models import ResearchMission, Source
from niche_scout.providers.gemini import (
    GeminiCall,
    GeminiGateway,
    GeminiResearchProvider,
    ResearchBatchSchema,
)


class _RateLimitError(RuntimeError):
    status_code = 429


class _ModelPermissionError(RuntimeError):
    status_code = 403


class _Response:
    text = "ok"
    parsed = None

    def model_dump(self, **_kwargs):
        return {"usage_metadata": {"prompt_token_count": 3}}


class _Models:
    def __init__(self):
        self.calls: list[str] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(model)
        if model == "first-model":
            raise _RateLimitError("429 quota; retry in 60s")
        return _Response()


class _Client:
    def __init__(self, models=None):
        self.models = models or _Models()


class _PermissionModels(_Models):
    def generate_content(self, *, model, contents, config):
        self.calls.append(model)
        if model == "first-model":
            raise _ModelPermissionError("403 model/tool is not enabled for this project tier")
        return _Response()


class _Gateway:
    def __init__(self):
        self.calls = 0

    def generate_text(self, **_kwargs):
        self.calls += 1
        return GeminiCall(
            model="grounded",
            text="dossier",
            parsed=None,
            sources=[Source("https://one.example/source", "One")],
            usage={},
            raw={},
        )

    def generate_structured(self, **_kwargs):
        self.calls += 1
        parsed = ResearchBatchSchema(
            research_summary="cached fixture", candidates=[], reviews=[]
        )
        return GeminiCall(
            model="structured",
            text=parsed.model_dump_json(),
            parsed=parsed,
            sources=[],
            usage={},
            raw={},
        )


class GeminiFallbackTests(unittest.TestCase):
    def test_rate_limited_model_cools_down_and_chain_advances(self) -> None:
        with TemporaryDirectory() as directory:
            settings = replace(
                Settings(),
                db_path=Path(directory) / "state.db",
                zero_cost_mode=False,
                base_cooldown_seconds=60,
            )
            with Repository(settings.db_path) as repo:
                repo.initialize()
                client = _Client()
                gateway = GeminiGateway(settings, repo, client=client, now=lambda: 1000.0)
                result = gateway.generate_text(
                    task="test",
                    models=("first-model", "second-model"),
                    system="system",
                    prompt="prompt",
                    grounded=False,
                )
                self.assertEqual(result.model, "second-model")
                self.assertEqual(client.models.calls, ["first-model", "second-model"])
                health = repo.get_model_health("test", "first-model")
                self.assertEqual(health["consecutive_failures"], 1)
                self.assertGreater(health["blocked_until"], 1000)

    def test_model_permission_failure_advances_to_older_grounded_model(self) -> None:
        with TemporaryDirectory() as directory:
            settings = replace(
                Settings(),
                db_path=Path(directory) / "state.db",
                zero_cost_mode=False,
            )
            with Repository(settings.db_path) as repo:
                repo.initialize()
                models = _PermissionModels()
                gateway = GeminiGateway(
                    settings,
                    repo,
                    client=_Client(models),
                    now=lambda: 1000.0,
                )
                result = gateway.generate_text(
                    task="grounded-test",
                    models=("first-model", "second-model"),
                    system="system",
                    prompt="prompt",
                    grounded=True,
                )
                self.assertEqual(result.model, "second-model")
                self.assertEqual(models.calls, ["first-model", "second-model"])

    def test_completed_api_stages_resume_from_disk_cache(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = replace(
                Settings(),
                db_path=root / "state.db",
                artifacts_dir=root / "artifacts",
            )
            mission = ResearchMission(
                mode="discover",
                strategy="fixture",
                objective="fixture",
                questions=[],
                sequence=1,
            )
            with Repository(settings.db_path) as repo:
                repo.initialize()
                first_gateway = _Gateway()
                first = GeminiResearchProvider(
                    settings, repo, gateway=first_gateway  # type: ignore[arg-type]
                )
                first.research(mission, settings)
                self.assertEqual(first_gateway.calls, 2)

                second_gateway = _Gateway()
                second = GeminiResearchProvider(
                    settings, repo, gateway=second_gateway  # type: ignore[arg-type]
                )
                second.research(mission, settings)
                self.assertEqual(second_gateway.calls, 0)
                second.mark_persisted(mission)
                self.assertFalse(second._cache_dir(mission).exists())


if __name__ == "__main__":
    unittest.main()
