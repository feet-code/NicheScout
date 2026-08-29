$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e .

if (-not (Test-Path "config.toml")) {
    Copy-Item "config.example.toml" "config.toml"
}
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

& .\.venv\Scripts\python.exe -m niche_scout --config config.toml init
Write-Host "Setup complete. Put your Gemini key in .env, then run scripts\run.ps1."
