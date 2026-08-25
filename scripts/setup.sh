#!/usr/bin/env sh
set -eu

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
[ -f config.toml ] || cp config.example.toml config.toml
[ -f .env ] || cp .env.example .env
.venv/bin/python -m niche_scout --config config.toml init
echo "Setup complete. Put your Gemini key in .env, then run ./scripts/run.sh."
