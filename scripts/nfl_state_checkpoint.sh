#!/usr/bin/env bash
# No pull/rebase: compare-and-swap refuses any concurrent advancement of immutable state.
set -euo pipefail
python "${GITHUB_WORKSPACE}/nfl_shadow_ownership.py" checkpoint --root "$PWD"
