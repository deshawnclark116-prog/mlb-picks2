# NFL Phase 1E shadow — operational runbook (durable dispatch)

**Status:** operational shadow infrastructure only. **R11 (simulation N) remains a BLOCKER**; no final freeze file exists and none is created here. Phase 1 is not production-ready; nothing here touches production serving or `docs/nfl_predictions.json`. The forecasts written are operational shadow evidence at an operational N (default 100,000). A snapshot retrieved inside the registered capture window of a genuinely future cutoff, with a prefit that pre-dates that cutoff, **is clean-forward temporal evidence**; R11 means only that N is operational and the result is **not freeze evidence and not production-promotion evidence**.

## Evidence caveat (read first)
The Thursday 2026-10-01 PIT@CLE forecast in `adhoc_pit_cle_20261001/` is **AD_HOC_PREGAME**: retrieved 2026-10-01T23:03:57Z (~71 min before kickoff), horizon label `AD_HOC_PREGAME`. It is **not clean T24/T90 evidence** and **not freeze evidence**, and must never be relabelled. The PIT@CLE T24 and T90 keys are `MISSED_REAL_CUTOFF` and stay that way.

## Execution path (no interactive process)
`.github/workflows/nfl_phase1e_shadow.yml` — cron every 10 min + `workflow_dispatch`; one run at a time (`concurrency: nfl-phase1e-shadow`, `cancel-in-progress: false`); runs only `nfl_phase1e_scheduler.py`; does not build any sport and does not touch `all_sports_predictions.yml`.
State: orphan branch `nfl-shadow-state` (ledgers, forecast / v2 stores, prefit artifacts, `status.json`, `readiness.json`); large snapshot blobs in the Actions cache (hash-verified on read — a lost or altered blob invalidates that stored snapshot and the key FAILS loudly; it is never silently replaced). GitHub may delay/drop scheduled runs: a cutoff without a valid pre-cutoff snapshot is `MISSED_REAL_CUTOFF`, never backfilled.

Each invocation (`scheduler run`): (1) probe the real schedule and store the snapshot; (2) plan keys `(game, horizon, cutoff)` from the **real kickoff** (cutoff = kickoff − 24 h / − 90 min; nothing is hard-coded per team or week); (3) for keys whose cutoff is within the 20-min lead, retrieve every source from the real provider, store it in the CAS and run Phase 1 **and** the frozen v2 comparator from that one snapshot; (4) close expired keys; (5) run the **prefit** if no cutoff is due within 50 min; (6) write the heartbeat and readiness.

## Prefit vs cutoff forecasting
The weekly walk-forward artifact (≈20 min to fit) is created **before** the cutoff by the prefit step once all earlier-week games are complete in the provider data (`prefit_ledger.jsonl`: version, training cutoff, source identity, bundle sha256, created_at; the artifact directory is `artifacts/<sha256>`). A live forecast only **loads** it and verifies every blob and the bundle hash; it never fits. Before the cutoff a missing / invalid prefit (`prefit_missing`, `prefit_hash_mismatch`, `prefit_code_identity_mismatch`) stays retryable; at the cutoff without a valid pre-cutoff artifact the key is terminal FAILED `PREFIT_NOT_READY_AT_CUTOFF`, and a later prefit can never rescue it. Prefit is independent of N and of the cutoff; the content of the model is unchanged.

## Prefit must pre-date the cutoff (audit patch)
A forecast for a given game/horizon may use a prefit only if **both** `created_at <= cutoff` and `source_identity.fit_retrieval_ts <= cutoff` (and the record is self-consistent). The prefit is also checked against the **current** `code_identity()` (`prefit_code_identity_mismatch`) and by hash. The check runs before any live capture. If no valid pre-cutoff prefit exists when the cutoff is reached the key is closed **FAILED `PREFIT_NOT_READY_AT_CUTOFF`**: no post-cutoff prefit is ever used for it, it can never become DONE, timestamps are never backdated, and a snapshot already captured stays as evidence only.

## Exact cutoff semantics
Per live event (`live_events.jsonl`, append-only): intended horizon, official kickoff used, computed cutoff, **actual retrieval timestamp** (after the last byte), `retrieval_minus_cutoff_seconds`, schedule snapshot sha256 (Phase 1 stored form) and raw sha256 (v2 side), snapshot set/content ids, prefit sha256 + creation time, model version, Phase 1 and v2 status, joint readiness. Rules: retrieval ≤ cutoff or the key is `MISSED_REAL_CUTOFF`; the workflow may **wake** up to 20 min early but **waits inside the job** and starts the capture only 4 min before the cutoff (pre-registered); a completed retrieval more than 5 min before the cutoff is **rejected** (`early_snapshot_rejected`), so a valid T90 event is ~90-95 min before kickoff (recorded: `seconds_before_cutoff`, `effective_minutes_before_kickoff`); the retrieval timestamp (after the last byte) is authoritative and a retrieval that finishes after the cutoff is MISSED_REAL_CUTOFF (cutoffs <~8 min apart can cost the later group its window — recorded honestly); no retrieval is attempted after the cutoff; no forecast is generated at/after kickoff (`GuardedStore`).

## Restart / process death
Same snapshot → same identity (`group` + content id). Death after the snapshot is stored → the next invocation reuses those exact bytes (verified), even after the cutoff. After the cutoff without a stored valid snapshot → `MISSED_REAL_CUTOFF`. Duplicate invocation → verified duplicates, 0 new records. Same id + different bytes → `HardError`. A kickoff revision creates a new key; old keys are never edited.

## Joint Phase 1 + frozen v2
Both run in one `run_group` from the same snapshot, kickoff, cutoff and schedule provenance. If v2 cannot be produced the key is `PARTIAL_V2_MISSING` (joint evidence incomplete); no v2 output is substituted.

## Inspect readiness / status
`status.json` (last invocation, next cutoffs, planned keys, DONE/FAILED/MISSED_REAL_CUTOFF counts, last real-provider retrieval, readiness summary) and `readiness.json` (checks: workflow configured, source registry, store writable, schedule resolvable, prefit artifacts verified, Phase 1 runner, frozen v2, optional deep source reachability) on branch `nfl-shadow-state`. Locally: `python nfl_phase1e_scheduler.py status --root <state>` and `python nfl_phase1e_scheduler.py readiness --root <state>` (exit 2 if not ready).

## Manual recovery (`workflow_dispatch`)
Actions → "NFL Phase 1E shadow" → Run workflow: `mode=readiness` (deep check; persists `readiness.json` + `status.json`; the run is red, exit 2, when READY=false), `mode=prefit` (prefit-only: stores schedule provenance, runs the guarded prefit from the real schedule, writes status; dispatches nothing, captures no live snapshot), `mode=run` with `duration_min=25` near a cutoff (keeps ticking each minute), `mode=status`. A manual run never backdates: it can only act on a cutoff still in the future or resume a stored pre-cutoff snapshot.

## Known limits
Scheduled runs can be late (a missed cutoff is recorded, not repaired). The Actions cache is best effort. First deployment needs one successful prefit (`mode=prefit`) before the first cutoff; until then readiness is NOT READY (visible in `status.json`).
