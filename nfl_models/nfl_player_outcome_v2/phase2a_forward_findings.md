# Phase2A-FWD: clean forward V2 shadow forecasting (research only; NOT production)

## Status

The first real forward forecasts exist: **11 T24 forecasts for `2026_05_TB_DAL`** (11 players, 5 DAL and 6 TB; Thursday kickoff 2026-10-09T00:15Z), generated 2026-10-07T00:19:38Z, i.e. about 24 hours before the 2026-10-08T00:15Z T24 cutoff, in an append-only hash-chained ledger. No T90 forecast exists yet (its window opens 2026-10-08T21:15Z). No cutoff has been missed. Every team-game logged `ABSTAIN_QB_STATE_UNCERTIFIED` (2 abstentions); no QB forecast exists. V2 is **not promoted**; production is untouched.

## Order of events (visible in git)

1. `24d8e92` protocol (windows, eligibility, censoring, metrics, comparators, abstention, lock rule) committed first.
2. `282c15b` engine, grader, QB-state ingestion interface and harness, tests and workflow.
3. `e5ea942` engine lock (sha256 of the engine code, the frozen upstream code and artifacts, the protocol).
4. `21fdcb7` test-guard widening; tests green (317+ V2 tests, Python 3.12 / numpy 2.3.5); everything pushed.
5. `b3d9142` the first real forecasts, generated only after steps 1-4 were pushed.

## Surviving chain used (no new model, no fitting)

Team opportunity: frozen Phase1B. Target share and carry share: frozen Phase1D allocators. Receiving and rushing efficiency: frozen Phase1F/Phase1B incumbent. The engine calls the frozen `nfl_v2_phase1e_integrated.receipt` unchanged. Phase1G (catch rate, completed air per catch, YAC per catch) is attached as a diagnostic only and never changes a final projection. QB opportunity and efficiency are blocked: QB-position players and every QB projection abstain. Monte Carlo is not included.

## Cutoff and immutability rules (enforced in code and tests)

- T24 window = kickoff-48h .. kickoff-24h; T90 window = kickoff-3h .. kickoff-90min. Before the window: `NOT_YET_DUE` (nothing written). After the cutoff: `MISSED_<cutoff>_CUTOFF`, logged once, never created, never backfilled. The CLI has no clock override; the guard `generated_at < cutoff <= kickoff` is re-evaluated before every append, and input retrieval must also precede the cutoff.
- Ledgers (`phase2a_ledger/`): hash-chained JSONL; duplicates are skipped, existing lines are never rewritten; tests also verify the git history only appends. Inputs seen by the engine are digest-recorded; the live 2026 stats and roster bytes are archived content-addressed in `phase2a_inputs/`.
- No target-game outcome is readable: history is filtered to strictly earlier weeks before any index is built, any target-week stat row refuses the game (`REFUSED_OUTCOME_DATA_PRESENT`), the schedule loader allowlists eight columns (no result, score or betting column), and comparators are built separately after the V2 forecast is frozen.

## Next eligible games and cutoffs (schedule timestamps only; week not assumed)

| Game | Kickoff (UTC) | T24 window (opens - cutoff) | T90 window (opens - cutoff) | Status |
|---|---|---|---|---|
| 2026_05_TB_DAL | Fri 10-09 00:15Z | Wed 10-07 00:15Z - Thu 10-08 00:15Z | Thu 10-08 21:15Z - Thu 10-08 22:45Z | T24 CREATED; T90 not yet due |
| 2026_05_PHI_JAX | Sun 10-11 13:30Z | Fri 10-09 13:30Z - Sat 10-10 13:30Z | Sun 10-11 10:30Z - Sun 10-11 12:00Z | T24 not yet due |
| 2026_05_CHI_GB | Sun 10-11 17:00Z | Fri 10-09 17:00Z - Sat 10-10 17:00Z | Sun 10-11 14:00Z - Sun 10-11 15:30Z | T24 not yet due |
| 2026_05_CIN_MIA | Sun 10-11 17:00Z | Fri 10-09 17:00Z - Sat 10-10 17:00Z | Sun 10-11 14:00Z - Sun 10-11 15:30Z | T24 not yet due |
| 2026_05_CLE_NYJ | Sun 10-11 17:00Z | Fri 10-09 17:00Z - Sat 10-10 17:00Z | Sun 10-11 14:00Z - Sun 10-11 15:30Z | T24 not yet due |
| 2026_05_HOU_TEN | Sun 10-11 17:00Z | Fri 10-09 17:00Z - Sat 10-10 17:00Z | Sun 10-11 14:00Z - Sun 10-11 15:30Z | T24 not yet due |
| 2026_05_IND_PIT | Sun 10-11 17:00Z | Fri 10-09 17:00Z - Sat 10-10 17:00Z | Sun 10-11 14:00Z - Sun 10-11 15:30Z | T24 not yet due |
| 2026_05_LV_NE | Sun 10-11 17:00Z | Fri 10-09 17:00Z - Sat 10-10 17:00Z | Sun 10-11 14:00Z - Sun 10-11 15:30Z | T24 not yet due |
| 2026_05_MIN_NO | Sun 10-11 17:00Z | Fri 10-09 17:00Z - Sat 10-10 17:00Z | Sun 10-11 14:00Z - Sun 10-11 15:30Z | T24 not yet due |
| 2026_05_NYG_WAS | Sun 10-11 17:00Z | Fri 10-09 17:00Z - Sat 10-10 17:00Z | Sun 10-11 14:00Z - Sun 10-11 15:30Z | T24 not yet due |
| 2026_05_DEN_LAC | Sun 10-11 20:05Z | Fri 10-09 20:05Z - Sat 10-10 20:05Z | Sun 10-11 17:05Z - Sun 10-11 18:35Z | T24 not yet due |
| 2026_05_DET_ARI | Sun 10-11 20:25Z | Fri 10-09 20:25Z - Sat 10-10 20:25Z | Sun 10-11 17:25Z - Sun 10-11 18:55Z | T24 not yet due |
| 2026_05_SF_SEA | Sun 10-11 20:25Z | Fri 10-09 20:25Z - Sat 10-10 20:25Z | Sun 10-11 17:25Z - Sun 10-11 18:55Z | T24 not yet due |
| 2026_05_BAL_ATL | Mon 10-12 00:20Z | Sat 10-10 00:20Z - Sun 10-11 00:20Z | Sun 10-11 21:20Z - Sun 10-11 22:50Z | T24 not yet due |
| 2026_05_BUF_LA | Tue 10-13 00:15Z | Sun 10-11 00:15Z - Mon 10-12 00:15Z | Mon 10-12 21:15Z - Mon 10-12 22:45Z | T24 not yet due |

The first scheduled game still inside its T90 window is TB @ DAL (T90 window 2026-10-08 21:15Z - 22:45Z). Week 6 and later games enter the same rule automatically when their windows open.

## Eligibility, receipts and comparators

A player is forecast only if he is a frozen Phase1D meaningful participant (prior-3 average >= 3 targets or >= 5 carries with at least two prior team games) on the active target-week roster; no fringe zeroes are added. Each receipt records eligibility reason, role confidence, availability state and confidence (roster status only: injury designations are not an input), data completeness, uncertainty flags, blocked information, source hashes and the full causal chain (team opportunity, share, expected targets/carries, catch rate, air and YAC diagnostics, efficiency, final projection). For every forecast a separate comparator row, also created before the cutoff, freezes the competent-human projection, a simple prior-3-game baseline and V1 when V1 published a projection before the cutoff (V1 had not published Week 5 at generation, recorded as `V1_NOT_PUBLISHED_BEFORE_CUTOFF`).

## Grader and performance report

`nfl_v2_phase2a_forward_grade.py` is built and tested on synthetic ledgers. It verifies the chain and the pre-cutoff proof of every forecast (and, optionally, the first git commit time of each line), grades only games the schedule marks final with official stats present for both teams, keeps RAW and CLEAN ledgers (only documented in-game events in `phase2a_censors.json` are censored), grades T24 and T90 separately, pairs them, and compares V2 against the human, simple and V1 projections on identical player-games. `phase2a_forward_report.json` is committed with status `NO_GRADED_GAMES_YET`; no game has finished.

## QB-state interface and provider harness

`nfl_v2_qb_state_ingestion.py` provides `ingest_qb_state`, `state_at`, `certify_for_forecast` and `evaluate_provider` on top of the Phase1M canonical schema and archive, with append-only raw and normalized layers. Sportradar and SportsDataIO adapters are conceptual field maps (`CONCEPTUAL_FIELD_MAP_NOT_VALIDATED_AGAINST_ANY_REAL_PAYLOAD`); nothing connects to a vendor, reads credentials or runs unauthorized. Fixtures prove the harness logic only (`FIXTURE_ONLY_NOT_PROVIDER_EVIDENCE`): they produce all three classifications. Both providers are `NOT_EVALUATED_NO_PROVIDER_PAYLOAD_TESTED`; Phase1L stays `BLOCKED_STARTER_STATE_DATA`.

## How to run it (no scheduler or production change was made)

The workflow `nfl_v2_phase2a_forward.yml` has `workflow_dispatch` (modes forecast / grade) but a dispatch workflow only runs from the default branch, and PR #64 must not be merged, so the supported procedure today is manual:

```bash
pip install -r requirements-research.txt
python nfl_v2_phase2a_forward_forecast.py --mode due --data-dir /tmp/p2a-data   # run inside each T24 and T90 window
git add nfl_models/nfl_player_outcome_v2/phase2a_ledger nfl_models/nfl_player_outcome_v2/phase2a_inputs nfl_models/nfl_player_outcome_v2/phase2a_forecast_manifest.json && git commit && git push
python nfl_v2_phase2a_forward_grade.py --stats <stats_player_week_2026.csv> --schedule <games.csv> --git-provenance   # after games finish
```

Without a scheduled trigger every window that no one runs is recorded as a missed cutoff at the next run; adding a `schedule:` trigger is a deliberate operator decision that this phase did not make.

## Honest limitations

- Availability is roster status only; injury designations are not an input, so injured or inactive players can appear (flagged `ROSTER_STATUS_ONLY_AVAILABILITY`). Pre-game absence is the model's responsibility and is never censored.
- The T24 forecast for Thursday was created near the start of its window, so it carries almost no information beyond the T90 forecast except newer roster data; the paired T24/T90 comparison measures only that.
- Engine v1.0 fetches the 2026 season files by name; a new season needs a new engine version.
- The schedule release asset was returning 404 at generation; the engine fell back to the public nfldata `games.csv` (digest and URL are recorded per forecast).
- Sample sizes are tiny until many games are graded; no conclusion about accuracy exists yet.
