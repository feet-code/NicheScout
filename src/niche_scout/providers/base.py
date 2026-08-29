from __future__ import annotations

from typing import Protocol

from ..config import Settings
from ..models import ResearchBatch, ResearchMission


class ResearchProvider(Protocol):
    name: str

    def research(self, mission: ResearchMission, settings: Settings) -> ResearchBatch: ...

    def retry_at(self) -> float | None: ...
