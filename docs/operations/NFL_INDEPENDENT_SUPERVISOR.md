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

## Approval-gated Render deployment candidate

A concrete Render Blueprint has now been staged at
`docs/operations/nfl_supervisor.render.yaml`. It is intentionally **not**
a root-level `render.yaml`; creating the resource requires an explicit
Blueprint import and approval, and does not affect any existing Render service.

The connected Render workspace currently has an existing Quiet Money
background worker, but that belongs to a different project and is suspended.
**Do not repurpose or restart it for NFL.** The proposed NFL service is a
new, single-instance worker on a paid compute plan with a 1 GB persistent
disk. Review Render's current price and disk charges before approving.

Before provisioning:

1. Have the operator review and deploy PR #70's scheduler/publisher to
   `main` and prove the workflow uses the corresponding deployed SHA.
2. Confirm the supervisor's `main` branch is at the reviewed commit, and
   that the GitHub Actions workflow-dispatch event works for both collector
   and publisher.
3. Approve the new worker's recurring charges and one-replica configuration.
4. Create a **new fine-grained token** with Actions read/write, Contents
   read, and Issues read/write for this one repository; provision it as the
   Render `NFL_SUPERVISOR_GITHUB_TOKEN` secret, never as plaintext in files.
5. Import the blueprint as a separate service, confirm its persistent
   `/var/data` mount, and inspect the first live independent schedule and
   API/read-only provenance requests.
6. Monitor the PHI/JAX T24 and T90 receipts across the real deadlines.
   Any manually dispatched recovery stays tagged manual and never counts as
   unattended delivery.

The supervisor now treats incomplete published row counts and duplicate
public IDs as failures, issues pre-cutoff warnings when no collector is
running, and escalates failed source/API polls to issue #71 where GitHub
is reachable. The publisher's independent hash-and-ID parity check remains
the final delivery authority; row-count parity alone is not a full
cryptographic receipt check.

**Known residual risks:** This process dispatches GitHub Actions; it does
not run the forecast engine outside GitHub. A widespread GitHub Actions
execution outage can still miss the five-minute capture window. The
worker also cannot post to issue #71 during a GitHub-wide API outage;
its stderr/error logs and Render service-failure monitoring must remain
independently observable. Neither risk may be called solved on deployment.

## Current status

The supervisor is a deployment-ready candidate **only** once it passes CI,
is reviewed, and receives an authorized hosting connection. It is not live
and has not demonstrated real autonomous coverage. Keep the incident open.
