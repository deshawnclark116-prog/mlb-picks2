# NFL independent deadline supervisor — deployment and acceptance

This is a safety-critical **trigger/observer**, NOT the forecast engine, an odds source,
or a method for filling missed historical cutoffs. Read
[the incident report](NFL_SHADOW_REPAIR.md) before changing anything. The user
authorized a paid Render 512 MB background worker and 1 GB disk on 2026-10-10.

## What was live-tested without side effects

The `nfl_supervisor_live_preflight.yml` workflow authenticates with the temporary
GitHub Actions job token, retrieves **the real schedule**, the immutable
`nfl-shadow-state/dispatch_ledger.jsonl`, the main published
`docs/nfl_phase1_shadow.json`, and the actual collector/publisher workflow
statuses, then runs the **same decision logic** the worker runs.
`--probe` **does not dispatch a workflow, post an issue, or write local state**.
That token is short-lived and **is not usable by the persistent Render worker**.

An ordinary preflight success is NOT a claim that a live T24/T90 capture,
GitHub API write access, or Render worker is ready.

## Paid Render background worker configuration

Blueprint source: `docs/operations/nfl_supervisor.render.yaml`

- Worker type: `worker`, NOT web service or hourly cron.
- GitHub source: `deshawnclark116-prog/mlb-picks2`, branch `main`.
- Oregon, plan `0.5c-512mb`, exactly **one** instance.
- Persistent disk: `/var/data`, **1 GB**; sole purpose is dedupe and alert state.
- Poll: every **60 seconds**; only prewarm/dispatch the existing serialized
  `nfl_phase1e_shadow.yml` worker, never run a second model process in Render.
- Install `numpy scipy` (same set as the live preflight).
- Start: `python -u nfl_shadow_external_supervisor.py --interval-sec 60
  --state-file /var/data/nfl-supervisor-state.json`.
- Env variable: `NFL_SUPERVISOR_GITHUB_TOKEN`, entered privately in Render.
  The Blueprint deliberately says `sync: false`.
- Render published compute price checked 2026-10-10: $7/month for the 512 MB
  worker plus $0.25/month for 1 GB disk, barring other usage or plan changes.

This Render connection's available creation actions do not include a persistent
background worker or Blueprint deployment. Use the **Render dashboard**:
https://dashboard.render.com/select-repo?type=blueprint or
Dashboard > New > Blueprint > choose the `mlb-picks2` repository,
select `main` and set **Blueprint Path** to exactly
`docs/operations/nfl_supervisor.render.yaml`.
Review the worker/disk/cost, enter the GitHub credential securely when prompted,
and choose **Deploy Blueprint**. Do not create a second copy if the named worker
already exists. Render supports Blueprint paths outside the repo root.

### Least-privilege GitHub credential

GitHub > Settings > Developer settings > Personal access tokens >
Fine-grained tokens > Generate. Resource owner: the repository owner,
repository access: **only `mlb-picks2`**. Minimum repository permissions:

- **Contents: Read** — inspect ledger/public output in the two branches.
- **Actions: Read and write** — list real runs, dispatch the two known
  workflows. A read-only token is not enough.
- **Issues: Read and write** — post durable alerts on existing
  [incident #71](https://github.com/deshawnclark116-prog/mlb-picks2/issues/71).
- Metadata read is included automatically.

Use an expiration you can rotate before it lapses (for example 90 days).
Never add the token to a file, a GitHub issue, a PR comment, a chat message
or an Actions log. **Paste it only into Render's secret input.** If the token
expires or is revoked, worker logs/incident reports must mark the system
unhealthy; it may not fall back to false-delivery claims.

## Live acceptance — required before calling the incident closed

1. Confirm one running Render background worker, persistent disk mounted,
   with `inspect_sources` decisions logged at least every 60 seconds.
   Verify no `SUPERVISOR_POLL_FAILED` or `SUPERVISOR_ALERT_FAILED`.
2. Confirm Github source permissions and GitHub **write** permission by observing
   a legitimately due pre-cutoff dispatch (no arbitrary test dispatch claimed
   as a successful forecast).
3. For upcoming *real* game/horizon keys, independently verify the
   original pre-cutoff source retrieval, existing prefit qualification,
   DONE/partial durable ledger with immutable batch, matching published ID set,
   and `docs/nfl_phase1_shadow.json` **on main**.
4. Confirm T24 and T90 for at least the same genuinely upcoming game; a
   successful Actions run/green CI by itself is insufficient.
5. Record separately misses, source errors, stuck queue, owner binding and
   publisher mismatch. Check the existing incident #71 for alerts.
6. Never rewrite historical misses, weaken the frozen five-minute snapshot
   vintage guard, backdate source timestamps, or manually promote the model.

If the worker cannot boot because the secret is missing, do NOT leave a paid
crash-looping service and claim the installation is finished. Complete the
credential in Render and re-verify the above live events.

Until all acceptance steps pass, the correct status is **deployed / observing**
at most, never **model validated / full forecast delivery guaranteed**.
