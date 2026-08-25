#!/usr/bin/env sh
set -eu
exec .venv/bin/python -m niche_scout --config config.toml run "$@"
