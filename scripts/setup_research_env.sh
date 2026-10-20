#!/usr/bin/env bash
# Reproduce the CI research environment (Python 3.12 + requirements-research.txt) and run the V2 test command.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v python3.12 >/dev/null || { echo "python3.12 required (CI uses 3.12)"; exit 1; }
if [ "$(git rev-parse --is-shallow-repository)" = true ]; then git fetch --unshallow origin; fi
python3.12 -m venv .venv-research
.venv-research/bin/pip install -q -r requirements-research.txt
.venv-research/bin/python --version
.venv-research/bin/python -m pytest -q tests/test_nfl_v2_*.py
