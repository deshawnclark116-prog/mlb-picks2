#!/usr/bin/env bash
# Durable checkpoint of the nfl-shadow-state working tree (run with cwd = the state checkout). Append-only files only (.gitignore excludes blobs/tmp/locks). Retries; never rewrites history.
set -u
git config user.name "nfl-shadow-bot"; git config user.email "nfl-shadow-bot@users.noreply.github.com"
git add -A
if git diff --cached --quiet; then echo "checkpoint: no state change"; exit 0; fi
git commit -q -m "shadow state checkpoint $(date -u +%Y-%m-%dT%H:%M:%SZ) run ${GITHUB_RUN_ID:-local}"
for i in 1 2 3 4; do
  if git push -q origin "HEAD:${STATE_BRANCH:-nfl-shadow-state}"; then
    echo "checkpoint: pushed $(git rev-parse --short HEAD)"
    # presentation only: when this checkpoint added forecast batches, ask the read-only publisher to run now (best effort; never blocks the checkpoint)
    if [ -n "${GH_TOKEN:-}" ] && git diff --name-only HEAD~1 HEAD 2>/dev/null | grep -q '^forecasts/batches/'; then
      gh api -X POST "repos/${GITHUB_REPOSITORY}/actions/workflows/nfl_new_engine_publish.yml/dispatches" -f ref=main >/dev/null 2>&1 && echo "checkpoint: publisher dispatched" || true
    fi
    exit 0
  fi
  git pull -q --rebase origin "${STATE_BRANCH:-nfl-shadow-state}" || true
  sleep $((i*3))
done
echo "checkpoint: PUSH FAILED"; exit 1
