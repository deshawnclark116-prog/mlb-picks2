# Independent NFL deadline supervisor (optional, NOT deployed)

**State:** code and offline tests exist in PR #70. This is an independent
orchestration path, NOT a replacement for the frozen model and NOT evidence
that any live forecast was delivered. Do not enable it until PR #70 is
reviewed, the repaired collector/publisher run from `main`, and the operator
authorizes worker hosting and GitHub token permissions.

## Why it exists

The system previously relied on GitHub scheduled Actions arriving within a
four-minute pre-cutoff capture window. Observed schedule delays of 11–90
minutes make this unacceptable for unattended pregame forecasts. GitHub's
second scheduled watchdog runs on the same schedule platform, so it is not
an independent trigger.

`nfl_shadow_external_supervisor.py` runs as a single long-lived process on a
separate hosting provider. It fetches the live game schedule independently,
derives T24 and T90 cutoff times, checks the GitHub state ledger and current
public NFL artifact, and dispatches the **same** serialized GitHub collector
up to 45 minutes before a cutoff. The collector still waits until its real
four-minute window; no forecast is captured early or backfilled.

The worker can dispatch the publisher when a valid completed forecast has
not appeared publicly. Seven minutes after an actual cutoff, missing DONE or
missing published rows lead to durable incident comments on issue #71.
It cannot silently turn `MISSED_REAL_CUTOFF` into an accepted forecast.

## Deployment contract (requires authorization)

Use a **single-replica persistent Linux background worker**, for example an
approved Render worker. Required runtime: Python 3.12 with `numpy` and a
checkout of the **reviewed and deployed** repository commit.

Start command:

```bash
python -u nfl_shadow_external_supervisor.py --interval-sec 60 --state-file /var/data/nfl-supervisor-state.json
```

Required environment variable:

- `NFL_SUPERVISOR_GITHUB_TOKEN`: fine-grained, repository-scoped token with
  **Actions: read/write**, **Contents: read**, and **Issues: read/write** for
  `deshawnclark116-prog/mlb-picks2`. This token must be stored in the hosting
  provider's secrets manager and never committed to Git.

The `/var/data` directory must be persistent and writable. Do not run two
replicas; the worker has a process-local flock and a durable JSON deduplication
log, but flock alone is not a cross-host distributed lock. Any duplicate
workflow dispatch is additionally controlled by GitHub Actions concurrency
and the collector's Git expected-SHA lease.

This worker **never** writes to `nfl-shadow-state` or `docs/`, and never
imports or executes the predictor. It can dispatch only workflows on `main`.
It uses the ordinary GitHub Actions API; no secret endpoint and no arbitrary
Git ref input is accepted.

## Verification before declaring autonomy

1. Confirm the repaired code actually runs from `main` and the old publisher
   can no longer overwrite it.
2. Confirm supervisor logs show a 45-minute prewarm decision for the first
   due game. Verify the resulting Actions workflow is queued/running with
   the repaired code well before the final 4-minute capture period.
3. Verify genuine T24 **and** T90 pre-cutoff receipts in `nfl-shadow-state`,
   durable source hashes and expected-SHA checkpoints.
4. Verify that the committed public JSON contains *exactly* the forecast IDs
   in immutable batch receipts. The publisher runs the separate
   `nfl_shadow_delivery_audit.py --check-schedule` and retains its artifact
   on both success and failure.
5. Disconnect the network/simulate dispatch errors in an isolated environment
   and verify the worker emits explicit `SUPERVISOR_POLL_FAILED` diagnostics,
   not a fabricated success.
6. Demonstrate missed-cutoff and unpublished-DONE incidents actually reach
   GitHub issue #71. Record the alert latency.
7. Measure coverage across multiple games and runner changes. A dispatched
   GitHub workflow can itself be queued/delayed; independent prompting
   reduces schedule latency dependence, but **does not guarantee** execution
   if GitHub Actions itself is unavailable. Full independence from GitHub
   worker execution requires a separate model-execution host and a new
   rigorously audited state-writer protocol.

## Current status

The supervisor is a deployment-ready candidate **only** once it passes CI,
is reviewed, and receives an authorized hosting connection. It is not live
and has not demonstrated real autonomous coverage. Keep the incident open.
