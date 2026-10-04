#!/usr/bin/env bash
# Durable checkpoint of the nfl-shadow-state working tree (run with cwd = the state checkout). Append-only files only (.gitignore excludes blobs/tmp/locks). Retries; never rewrites history.
set -u
git config user.name "nfl-shadow-bot"; git config user.email "nfl-shadow-bot@users.noreply.github.com"
git add -A
if git diff --cached --quiet; then echo "checkpoint: no state change"; exit 0; fi
git commit -q -m "shadow state checkpoint $(date -u +%Y-%m-%dT%H:%M:%SZ) run ${GITHUB_RUN_ID:-local}"
for i in 1 2 3 4; do
  if git push -q origin "HEAD:${STATE_BRANCH:-nfl-shadow-state}"; then echo "checkpoint: pushed $(git rev-parse --short HEAD)"; exit 0; fi
  git pull -q --rebase origin "${STATE_BRANCH:-nfl-shadow-state}" || true
  sleep $((i*3))
done
echo "checkpoint: PUSH FAILED"; exit 1
