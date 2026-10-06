# Phase1N-T — team opportunity / game-state engine V2

**Verdict: `REJECTED_TEAM_OPPORTUNITY_REPLACEMENT`.** No candidate family beat the strongest comparator on team plays, dropbacks or rush attempts on the 2024 selection weeks, so the development gate failed and the phase froze immediately. **2025 was not opened. The 2026 W1–4 diagnostic could not be run. Phase1D integration and the final-player-stat test were not reached.** No rescue tuning, grid extension or threshold change was made.

Locked states are unchanged: receiving `FROZEN_AT_PHASE1F_PENDING_NEW_INFORMATION`, receiver depth `SURVIVED_SIGNAL_NOT_PROMOTED`, rushing efficiency `REJECTED_EFFICIENCY_REPLACEMENT`, QB opportunity `BLOCKED_STARTER_STATE_DATA`, QB source acquisition `PHASE1L_BLOCKED_PENDING_VENDOR_ACCESS`. No sportsbook field, Monte Carlo, vendor data, blocked starter state, Week 5+ row, production/frontend/scheduler/grading/calibration/forecast change or other-sport file was touched.

Order of work (git history): protocol (`5369d89`) → source layer, audit, ridge chain, tests, workflow (`20c8848`) → development results, human baseline, lock, receipts (`0392ee3`) → this findings file, snapshot and registry pointer.

## Question and design

Can a coherent, pregame, interpretable team-state chain predict team **plays**, **dropbacks** and **designed rush attempts** materially better than the strongest transparent comparator, the Phase1B team-volume layer and frozen Phase1C-T? Every candidate outputs `plays = dropbacks + rush attempts` exactly (a rate chain: expected plays × expected dropback rate = dropbacks; the remainder is rushes). Models are closed-form ridge regressions estimated on **2024 weeks 1–8** and selected on **2024 weeks 9–18** (λ ∈ {1, 10, 100}, EWMA decay ∈ {0.65, 0.85}, window ∈ {5, 8}). 2023 supplies history only. History is strictly pregame with the preregistered lag (a prior game must be at least 3 days before the target date). 2025 would have opened only if a family qualified on all three quantities.

## PBP team-state source audit (2023–2024 pinned bytes)

Semantics are fixed in the protocol and tested: an **opportunity play** is a run, pass or QB spike that is not a two-point try or kneel; **dropbacks** include sacks, scrambles and spikes (nflfastR records spikes with `qb_dropback = 0`, so they are added); **designed rushes** are non-dropback rushes; **plays = dropbacks + designed rushes** with zero violations in both seasons. A scramble counts once, as a dropback. Penalty/no-play rows, kneels and two-point tries are excluded and counted separately (2024: 405 kneels, 135 two-point tries, 2,704 no-play rows; 71 spikes; 1,062 scrambles; 16 overtime games).

Reconciliation to official stats is essentially exact: PBP pass attempts equal official team attempts in **544 of 544** team-games in each season; PBP rushes + scrambles + kneels equal official carries in 543 of 544; official targets average 0.955 of attempts (the fixed ratio used for the secondary targets estimate). `attempts ≤ dropbacks` holds everywhere.

**Drive reconstruction was rejected by the preregistered rule.** Drive ids exist on 100% of opportunity plays and 99.93% of drives have a single offense, but the two offenses' drive counts differ by at most 1 in only **89.0% (2023) and 90.8% (2024)** of games against the required 95%. Mean 10.3–10.7 drives and 5.8–5.9 plays per drive per team-game. Family C was therefore **not fitted** (`REJECTED_DRIVE_RECONSTRUCTION`), and the drive-based oracle decompositions were not run. The 95% threshold was registered before the audit and was not relaxed afterward.

The 2026 diagnostic is unavailable: the current `pbp_2026` and `stats_player_week_2026` release assets no longer match the Phase1H sha256 pins and now post-date Week 4, so they were not opened. 2023–2025 pinned bytes verified.

## Competent-human team baseline (frozen, no optimization)

`plays = 0.8·(0.5·own recent-5 + 0.5·opponent recent-5 allowed) + 0.2·league`; dropback rate the same blend of ratios; `dropbacks = plays × rate`; `rushes = plays − dropbacks`. 2024 weeks 9–18 MAE: **plays 6.96, dropbacks 6.28, rush 5.80**; dropback-rate error 7.84 pp. It is nearly as good as the best candidate.

## Comparators on the 2024 selection weeks (298 team-games, MAE)

| Comparator | Plays | Dropbacks | Rush attempts |
|---|---:|---:|---:|
| **Frozen Phase1C-T, definition-offset-corrected (strongest on all three)** | **6.806** | **6.214** | **5.744** |
| Competent human | 6.962 | 6.282 | 5.804 |
| Recent-5 mean | 7.242 | 6.805 | 5.888 |
| Recent-3 mean | 7.522 | 6.989 | 6.131 |

Phase1C-T counts kneels and excludes spikes, so its outputs carry a definition offset. A fixed offset (plays +0.94, dropbacks +0.04, rush +2.90, estimated on 2024 weeks 1–8 only) was subtracted. Phase1C-T was selected on all of 2024, an in-sample advantage for the comparator. The frozen Phase1B layer has no plays/dropback head, so it is compared on attempts, targets and official carries (below).

## Candidate families (2024 weeks 9–18, MAE; gain = comparator − candidate, negative means worse; up95 = paired game-block 95% upper bound of candidate − comparator, which must be below 0)

| Family | Plays | Dropbacks | Rush | Rate pp | Verdict |
|---|---:|---:|---:|---:|---|
| A own recent volume | 6.936 (−0.13, up95 +0.19) | 6.423 (−0.21, +0.39) | 5.989 (−0.25, +0.39) | 8.41 | REJECTED_NO_PRACTICAL_GAIN |
| B own + opponent pace (+ home/rest) | 6.966 (−0.16, +0.26) | 6.402 (−0.19, +0.37) | 5.976 (−0.23, +0.37) | 8.37 | REJECTED_NO_PRACTICAL_GAIN |
| C drive chain | not fitted | | | | REJECTED_DRIVE_RECONSTRUCTION |
| D neutral tendency | 6.966 (−0.16, +0.26) | 6.438 (−0.23, +0.44) | 5.980 (−0.24, +0.36) | 8.39 | REJECTED_NO_PRACTICAL_GAIN |
| E opponent invitation | 6.966 (−0.16, +0.26) | 6.408 (−0.19, +0.37) | 5.842 (−0.10, +0.23) | 8.11 | REJECTED_NO_PRACTICAL_GAIN |
| F pregame scenarios | 6.905 (−0.10, +0.20) | 6.404 (−0.19, +0.33) | 5.775 (−0.03, +0.17) | 8.08 | REJECTED_NO_PRACTICAL_GAIN |
| G full coherent chain | | | | | NOT_EVALUATED_PARENTS_DID_NOT_SURVIVE |

Required to qualify: gain ≥ max(0.35 plays or dropbacks / 0.30 rush, 5% of the comparator's MAE), upper-95 below 0, no more >15 misses than the comparator, and exact coherence. Every cell is worse than the strongest comparator, not just short of the bar. Best candidate by summed normalized MAE: F. Every selected configuration landed on the λ = 100 grid edge. The protocol fixed the grid and forbade extension, so that edge was not explored.

Chain candidates beat the *simple* recent-3/5 means on plays and sit within 0.06 plays of the competent-human baseline (A and F are 0.03–0.06 better, B/D/E 0.00 worse); on dropbacks they are 0.12–0.16 worse than the human baseline and on rush attempts F is 0.03 better while the others are up to 0.19 worse. The team-state structure carries real but small signal; it does not replace what already exists.

## Coherence

All candidates: `plays − dropbacks − rushes` = 0 (max |error| 0.0), no negative components, predicted attempts ≤ dropbacks and targets ≤ attempts, dropback rates inside [0.2, 0.9] with 0% clipped. An independent-heads ablation (family E features, non-promotable) fits plays, dropbacks and rushes separately: MAEs 6.95 / 6.52 / 5.91 versus the chain's 6.97 / 6.41 / 5.84. Because ridge with identical features and penalty is linear in the target, the independent heads also satisfy plays = dropbacks + rushes exactly (measured residual 0.0), so this ablation does **not** demonstrate an incoherence cost; it only shows the rate chain is slightly better on dropbacks and rushes. Accounting was not the reason for failure.

## Coverage by slice (best family F vs strongest comparator, selection weeks)

Regulation (274 games): plays 6.72 vs 6.67. Overtime (24): plays 9.04 vs 8.32; dropbacks 7.82 vs 6.59. Home/away: no material difference (plays 6.81 vs 6.76 home; 7.00 vs 6.85 away). Later weeks 13–18 plays 6.74 vs 6.72; weeks 9–12 plays 7.18 vs 6.95. Actual high-possession tercile (109): plays **10.25 vs 8.94** (worse); actual low-possession tercile (104): 7.58 vs 8.32 (better). Catastrophic-miss rate (>15): plays 8.4%, dropbacks 6.7%, rush 1.7% (comparator plays 6.0%). The possession and overtime slices stratify by realized counts and are descriptive only.

## Oracle forensics (postgame-descriptive, best family F, never used for fitting)

- Actual plays × predicted rate: dropback MAE **4.95**. Predicted plays × actual rate: **4.23**. Chain: **6.40**. Both components matter and they interact: removing only the plays error leaves 4.23 / 6.40 = 66% of the chain's dropback MAE, and removing only the rate error leaves 4.95 / 6.40 = 77%.
- The dropback rate is the larger shared problem: the pregame mixture misses it by **8.08 pp**, a constant by 8.65 pp, the human baseline by 7.84 pp. Even an oracle that knows the realized final-margin class and applies that class's scenario rate only reaches **7.11 pp**. Realized scripts move the rate hard (LEADS 53.5% vs TRAILS 67.4% dropbacks; the pregame mixture is biased +4.9 pp when the team led and −6.9 pp when it trailed), but nothing pregame predicts which script happens. |play error| correlates 0.003 with |realized margin|, so margin size does not explain play error; overtime does (0.12; OT games are underpredicted by 8.0 plays).
- Drive decompositions (actual drives × predicted plays/drive, predicted drives × actual plays/drive) were not run because drive reconstruction was rejected.

## Secondary: attempts, targets, official carries vs frozen Phase1B (same 298 rows)

Attempts: candidate 5.91 vs Phase1B 5.78 (bias −1.55 vs −0.30). Targets: 5.68 vs 5.50. Carries: 5.93 vs 5.93. Phase1B is as good or better, and the chain's dropbacks are biased low (−1.7), consistent with an intercept estimated on weeks 1–8 and a rising pass rate in weeks 9–18. Secondary only; not a promotion criterion.

## Receipts and the NYJ–CHI question

`phase1n_team_receipts.jsonl.gz` holds all 544 2024 team-games for the best family: expected plays / dropback rate / dropbacks / rush rate / rush attempts, scenario weights and effects, pregame expected margin, actual drives/plays/dropbacks/rushes/rate, component and total errors, selection-residual uncertainty, every comparator's forecast, source hashes and the features. Expected drives/plays-per-drive are null by design (rejected reconstruction). The Week 4 2026 NYJ–CHI imbalance cannot be inspected because no 2026 bytes are pinned. The largest 2024 selection-weeks imbalance, **KC @ DEN in Week 18 (33 vs 71 plays)**, shows the same failure type: both teams were forecast at about 60.5 plays and about 38 dropbacks, with a near-even pregame margin (−0.4 / +0.4) and flat scenario weights (competitive 0.40). A Week 18 game like this is plausibly a resting-starters situation (not verified from the source files) that no pregame team-history feature can see; that is the kind of information the QB/starter-state gate (Phase1M) is meant to address.

## Largest remaining team-opportunity failure source

**Game-script-driven dropback rate**, compounded by tails (high-possession and overtime games). The play count is only slightly better than the human baseline, and the rate is barely better than a constant. These errors come from realized script, not from accounting, drive structure or opponent invitation, none of which added signal over the strongest existing comparator.

## Not reached, and why

- **2025 untouched validation:** `NOT_RUN_DEVELOPMENT_GATE_FAILED` (no 2025 file opened). 2025 team-volume aggregates for Phase1B and Phase1C-T were visible earlier; no Phase1N candidate or threshold used them.
- **2026 W1–4 burned diagnostic:** `NOT_RUN_PINNED_BYTES_UNAVAILABLE`. A promotion would also have been impossible without it, by the preregistered verdict rule.
- **Phase1D integration** and **final player stats:** `NOT_RUN` (team layer not promoted; no cancellation credit can be claimed).

## Honest caveats

- The first development run completed without error. After seeing results I (a) added the explicit class-oracle line to the postgame oracle diagnostics (a protocol-listed descriptive item) and regenerated the artifacts; nothing in any gate, selection or verdict changed.
- Estimation used only 246 team-games, and every selected λ sat at the grid edge; a different shrinkage regime was not tested because the protocol forbids rescue tuning.
- Phase1C-T enjoys an in-sample selection advantage on 2024 and carries a definition offset corrected with a weeks 1–8 constant; both favor the comparator, and the candidates still failed to beat even the human baseline on dropbacks.
- Scenario weights use a fixed 13.5-point scale and a simple margin-difference model; a better internal strength model was not pursued.

## Reproducibility repair (after the first CI run)

The first CI run failed at "Reproduce the pinned PBP team-state source audit": the committed audit and the CI-generated audit differed at line 89, `targets_total_vs_attempts_mean_ratio` (0.9547998395579832 committed vs 0.9547998395579826 in CI; 2024: ...733 vs ...7324). **Cause: Python 3.12 made `sum()` of floats compensated, so the same mean computed with `sum(x)/len(x)` differs in the last digits between my Python 3.11 and CI's 3.12. It was not source drift:** the allowlisted 2023-2024 schedule digest was unchanged on a fresh download. The mean now uses an explicit left-to-right accumulation (`ordered_mean`), which reproduces the committed values exactly on 3.11 and 3.12, so the audit bytes did not change. Separately, the schedule input is no longer read from the mutable upstream `games.csv`: the exact 544 allowlisted 2023-2024 regular-season rows (digest `3d53248e...`) are committed as `phase1n_frozen_schedule_2023_2024.json` with provenance, and every stage consumes only that artifact. No result, selection, metric, threshold or verdict changed; only the lock's code hash and the snapshot's artifact hashes were updated.

## Reproduction

```bash
python nfl_v2_phase1n_team_sources.py --fetch --data-dir /tmp/phase1n-data --out /tmp/phase1n-audit.json   # downloads pinned pbp/stats only; the schedule comes from the committed frozen artifact
python nfl_v2_phase1n_team_opportunity.py --data-dir /tmp/phase1n-data --stage develop --out-dir /tmp/phase1n-out
python nfl_v2_phase1n_team_opportunity.py --data-dir /tmp/phase1n-data --stage validate --out-dir /tmp/phase1n-out   # refuses: gate failed
python nfl_v2_phase1n_team_opportunity.py --data-dir /tmp/phase1n-data --stage burned2026 --out-dir /tmp/phase1n-out # refuses: pinned bytes unavailable
python -m pytest -q tests/test_nfl_v2_*.py
```

CI compares every regenerated artifact byte for byte, uploads them and updates one compact PR comment. It opens only pinned 2023/2024 bytes.
