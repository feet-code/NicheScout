from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import Lock
from typing import Any

from .models import utc_now


_log_path: Path | None = None
_verbose = False
_lock = Lock()


def configure(path: str | Path | None, *, verbose: bool = False) -> None:
    global _log_path, _verbose
    _log_path = Path(path) if path else None
    _verbose = verbose
    if _log_path:
        _log_path.parent.mkdir(parents=True, exist_ok=True)


def emit(event: str, *, level: str = "info", message: str | None = None, **fields: Any) -> None:
    payload = {"ts": utc_now(), "level": level, "event": event, **fields}
    if message:
        payload["message"] = message
    encoded = json.dumps(payload, ensure_ascii=False, default=str)
    with _lock:
        if _log_path:
            with _log_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
        if _verbose or level in {"warning", "error"}:
            stream = sys.stderr if level == "error" else sys.stdout
            print(encoded, file=stream, flush=True)
        else:
            summary = message or event.replace("_", " ")
            details = " ".join(
                f"{key}={value}"
                for key, value in fields.items()
                if key in {"action_id", "mode", "strategy", "model", "ideas", "progress", "wait_seconds"}
            )
            print(f"[{payload['ts']}] {summary}{' | ' + details if details else ''}", flush=True)
