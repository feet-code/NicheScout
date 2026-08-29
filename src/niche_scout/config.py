from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, get_args, get_origin


# These model IDs have a documented free input/output tier as of 2026-08-25.
# Search-grounding capability and free-tier grounding entitlement are separate:
# validate the former here, while the account/project controls the latter.
FREE_TEXT_MODELS = {
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
}
SEARCH_GROUNDED_MODELS = {
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
}


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("state/niche_scout.db")
    output_dir: Path = Path("exports")
    artifacts_dir: Path = Path("state/artifacts")
    event_log_path: Path = Path("state/logs/events.jsonl")

    target_active_hours: float = 72.0
    finalist_target: int = 500
    # Soft grouping target. Cohesion, not this count, decides when a group closes.
    products_per_site: int = 5
    max_products_per_site: int = 8
    audience_similarity_threshold: float = 0.34
    discovery_pool_target: int = 1500
    deep_research_pool: int = 800
    red_team_pool: int = 600
    minimum_validation_passes: int = 2
    minimum_red_team_passes: int = 1
    discovery_batch_size: int = 8
    review_batch_size: int = 3
    minimum_verified_sources: int = 3
    max_actions: int = 0
    heartbeat_seconds: int = 30
    action_pause_seconds: float = 3.0

    max_build_hours: float = 40.0
    max_maintenance_hours_month: float = 4.0
    max_infra_cost_month: float = 50.0
    diversity_penalty: float = 12.0

    zero_cost_mode: bool = True
    api_key_env: str = "GEMINI_API_KEY"
    grounded_models: tuple[str, ...] = (
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
    )
    synthesis_models: tuple[str, ...] = (
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
    )
    deep_models: tuple[str, ...] = (
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
    )
    request_timeout_seconds: int = 180
    base_cooldown_seconds: int = 60
    maximum_cooldown_seconds: int = 21600
    max_output_tokens: int = 14000

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Settings":
        raw: dict[str, Any] = {}
        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file does not exist: {path}")
            with path.open("rb") as handle:
                document = tomllib.load(handle)
            raw = {**document.get("research", {}), **document.get("gemini", {})}

        defaults = cls()
        values: dict[str, Any] = {}
        for item in fields(cls):
            default_value = getattr(defaults, item.name)
            value = raw.get(item.name, default_value)
            env_name = f"NICHE_SCOUT_{item.name.upper()}"
            if env_name in os.environ:
                value = _coerce(os.environ[env_name], item.type, default_value)
            elif isinstance(default_value, tuple) and isinstance(value, list):
                value = tuple(str(part) for part in value)
            if item.name.endswith("_path") or item.name.endswith("_dir"):
                value = Path(value)
            values[item.name] = value
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.target_active_hours <= 0:
            raise ValueError("target_active_hours must be positive")
        if self.finalist_target < 1:
            raise ValueError("finalist_target must be positive")
        if self.products_per_site < 1:
            raise ValueError("products_per_site must be positive")
        if self.max_products_per_site < self.products_per_site:
            raise ValueError(
                "max_products_per_site must be at least the preferred products_per_site"
            )
        if not 0 <= self.audience_similarity_threshold <= 1:
            raise ValueError("audience_similarity_threshold must be between 0 and 1")
        if not (
            self.discovery_pool_target >= self.deep_research_pool
            >= self.red_team_pool
            >= self.finalist_target
        ):
            raise ValueError(
                "Expected discovery_pool_target >= deep_research_pool >= red_team_pool >= finalist_target"
            )
        if self.discovery_batch_size < 1 or self.review_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.minimum_validation_passes < 1 or self.minimum_red_team_passes < 1:
            raise ValueError("minimum review pass counts must be positive")
        if self.heartbeat_seconds < 5:
            raise ValueError("heartbeat_seconds must be at least 5")
        if self.max_build_hours <= 0 or self.max_maintenance_hours_month < 0:
            raise ValueError("build and maintenance limits must be non-negative")
        if not self.grounded_models or not self.synthesis_models or not self.deep_models:
            raise ValueError("Every Gemini model chain must contain at least one model")
        unsupported_grounding = set(self.grounded_models) - SEARCH_GROUNDED_MODELS
        if unsupported_grounding:
            raise ValueError(
                "grounded_models rejects models without documented Google Search grounding: "
                + ", ".join(sorted(unsupported_grounding))
            )
        if self.zero_cost_mode:
            all_models = (
                set(self.grounded_models)
                | set(self.synthesis_models)
                | set(self.deep_models)
            )
            unknown_text = all_models - FREE_TEXT_MODELS
            if unknown_text:
                raise ValueError(
                    "zero_cost_mode rejects models without a documented free text tier: "
                    + ", ".join(sorted(unknown_text))
                )

    @property
    def site_target(self) -> int:
        """Estimated site count at the preferred size; actual grouping is flexible."""
        return (self.finalist_target + self.products_per_site - 1) // self.products_per_site

    @property
    def target_active_seconds(self) -> float:
        return self.target_active_hours * 3600

    def ensure_directories(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)

    def fingerprint(self) -> str:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def _coerce(value: str, annotation: Any, default: Any = None) -> Any:
    origin = get_origin(annotation)
    if origin is tuple or get_args(annotation) or isinstance(default, tuple):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    text = str(annotation)
    if "bool" in text or isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if "int" in text or (isinstance(default, int) and not isinstance(default, bool)):
        return int(value)
    if "float" in text or isinstance(default, float):
        return float(value)
    return value
