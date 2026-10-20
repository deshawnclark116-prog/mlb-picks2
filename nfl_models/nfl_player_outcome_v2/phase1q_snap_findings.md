# Phase1Q-SNAP: snap-share change / deployment inflection engine

**Verdict: REJECTED_SNAP_ROLE_REPLACEMENT** (receiving: REJECTED_SNAP_ROLE_REPLACEMENT; rushing: REJECTED_SNAP_ROLE_REPLACEMENT). This is the Phase1O family B hypothesis, now actually tested after Phase1P resolved the identity blocker (99.78% exact PFR->GSIS mapping). Frozen Phase1D stays the comparator. Workload only. 2025 was never opened (the development gate failed); 2026 W1-4 is `NOT_RUN_PINNED_BYTES_UNAVAILABLE`.

## Design (preregistered before any result)

- Candidate: `max(0, Phase1D share + ridge delta)` for non-QB players with a prior snap series, renormalised so the non-QB mass equals Phase1D's (shares sum to 1 for receiving; QB shares are held fixed for rushing so scrambles never mix into the RB allocation). Ridge lambda in {10, 100, 1000}, fit on 2024 W1-8, selected on W9-18.
- Information: only PRIOR completed-game snaps from games at least 4 days before the target game (one day more conservative than the V2 history rule; the snap source has no publication timestamp). Target-game snaps are postgame-only and appear only in oracle forensics and receipts.
- Families A snap level, B acceleration, C regime, D snap/opportunity divergence, E teammate redistribution; F (position routing) only if a basic family survived. The six non-snap Phase1O families were not rerun.
- Required comparators: Phase1D (C1) and a frozen competent-human snap baseline (C3S: Phase1D share x (1 + clip(persistent snap gain/loss))). Thresholds unchanged from Phase1O: gain >= max(0.10 targets / 0.20 carries, 5%), bootstrap upper-95 < 0, catastrophic-miss tolerance 0.005.

## Receiving head: 2024 W9-18 selection (1303 Phase1D fixed meaningful rows)

Oracle player-opportunity MAE (predicted share x actual team opportunity): Phase1D **1.942**, prior-3 share 2.088, competent-human snap baseline 1.949. Share MAE: Phase1D 0.0616, human snap 0.0618. Rows with a snap series: 1293 of 1303.

| Family | lambda | Oracle MAE | Gain vs Phase1D | Required | Bootstrap upper-95 (cand - Phase1D) | Gain vs human | Survives |
|---|---:|---:|---:|---:|---:|---:|---|
| A_snap_level | 100 | 1.927 | 0.015 | 0.100 | -0.008 | 0.022 | False |
| B_snap_acceleration | 1000 | 1.937 | 0.005 | 0.100 | 0.000 | 0.012 | False |
| C_snap_regime | 100 | 1.933 | 0.009 | 0.100 | -0.002 | 0.016 | False |
| D_snap_opportunity_divergence | 1000 | 1.948 | -0.006 | 0.100 | 0.015 | 0.002 | False |
| E_teammate_redistribution | 1000 | 1.940 | 0.002 | 0.100 | 0.001 | 0.010 | False |
| F_position_routing | - | - | - | - | - | - | NOT_TESTED_NO_BASIC_SNAP_FAMILY_SURVIVED |

Snap-change flags (pregame; last snap minus previous trailing-3 >= +/-0.15; actual change = share vs prior-3 reference by tau 0.05):
- SNAP_ROLE_UP: flagged 249, actual changes in population 299, precision 0.285, recall 0.237, false-promotion rate 0.715; workload MAE on the flagged rows: Phase1D 1.937, human snap 1.878.
- SNAP_ROLE_DOWN: flagged 157, actual changes in population 408, precision 0.306, recall 0.118, false-promotion rate 0.694; workload MAE on the flagged rows: Phase1D 2.060, human snap 2.135.

Best family (`A_snap_level`) gain vs Phase1D by slice: HAS_SNAP_SERIES n=1293 0.014; SNAP_CHANGED n=406 0.029; SNAP_ROLE_DOWN n=157 -0.013; SNAP_ROLE_STABLE n=897 0.008; SNAP_ROLE_UP n=249 0.056. The pregame snap flags have low precision for share changes (up flags 0.26-0.29, down flags 0.31-0.52) and the candidate's gains are within noise on every slice.

Postgame-only forensics: role error MAE 1.927; team-volume error (Phase1B projection x actual share) MAE 0.923. By actual target-game snap change: TARGET_GAME_SNAP_DOWN n=169 MAE 2.143; TARGET_GAME_SNAP_STABLE n=884 MAE 1.828; TARGET_GAME_SNAP_UP n=239 MAE 2.126. Of 239 rows where the target-game snap share jumped by 0.15 or more, 153 did not see a matching share rise: a same-game snap jump often does not become opportunity.

## Rushing head: 2024 W9-18 selection (450 Phase1D fixed meaningful rows restricted to RB/FB/HB)

Oracle player-opportunity MAE (predicted share x actual team opportunity): Phase1D **3.083**, prior-3 share 3.380, competent-human snap baseline 3.090. Share MAE: Phase1D 0.1188, human snap 0.1189. Rows with a snap series: 444 of 450.

| Family | lambda | Oracle MAE | Gain vs Phase1D | Required | Bootstrap upper-95 (cand - Phase1D) | Gain vs human | Survives |
|---|---:|---:|---:|---:|---:|---:|---|
| A_snap_level | 1000 | 3.113 | -0.030 | 0.200 | 0.062 | -0.024 | False |
| B_snap_acceleration | 1000 | 3.111 | -0.028 | 0.200 | 0.050 | -0.022 | False |
| C_snap_regime | 1000 | 3.102 | -0.019 | 0.200 | 0.052 | -0.012 | False |
| D_snap_opportunity_divergence | 1000 | 3.111 | -0.028 | 0.200 | 0.057 | -0.021 | False |
| E_teammate_redistribution | 10 | 3.098 | -0.015 | 0.200 | 0.035 | -0.008 | False |
| F_position_routing | - | - | - | - | - | - | NOT_TESTED_NO_BASIC_SNAP_FAMILY_SURVIVED |

Snap-change flags (pregame; last snap minus previous trailing-3 >= +/-0.15; actual change = share vs prior-3 reference by tau 0.1):
- SNAP_ROLE_UP: flagged 88, actual changes in population 99, precision 0.261, recall 0.232, false-promotion rate 0.739; workload MAE on the flagged rows: Phase1D 2.907, human snap 2.970.
- SNAP_ROLE_DOWN: flagged 67, actual changes in population 136, precision 0.522, recall 0.257, false-promotion rate 0.478; workload MAE on the flagged rows: Phase1D 3.459, human snap 3.386.

Best family (`E_teammate_redistribution`) gain vs Phase1D by slice: HAS_SNAP_SERIES n=444 -0.015; SNAP_CHANGED n=155 0.038; SNAP_ROLE_DOWN n=67 0.101; SNAP_ROLE_STABLE n=295 -0.042; SNAP_ROLE_UP n=88 -0.011. The pregame snap flags have low precision for share changes (up flags 0.26-0.29, down flags 0.31-0.52) and the candidate's gains are within noise on every slice.

Postgame-only forensics: role error MAE 3.098; team-volume error (Phase1B projection x actual share) MAE 2.517. By actual target-game snap change: TARGET_GAME_SNAP_DOWN n=73 MAE 4.292; TARGET_GAME_SNAP_STABLE n=286 MAE 2.560; TARGET_GAME_SNAP_UP n=83 MAE 3.952. Of 83 rows where the target-game snap share jumped by 0.15 or more, 38 did not see a matching share rise: a same-game snap jump often does not become opportunity.

## Decision

No snap family met the practical threshold against Phase1D on either head (best gain 0.015 targets against the required 0.10; every carries candidate is worse than Phase1D against the required 0.20 gain), so no survivor combination or position-routing test was formed, the **development gate failed** and 2025 stayed unopened. The competent-human snap baseline is almost indistinguishable from Phase1D (1.949 vs 1.942 targets; 3.090 vs 3.083 carries), which is itself informative: a transparent persistent-snap adjustment adds nothing. Coherence held exactly (max deviation 0.0e+00). No hyperparameter, family, threshold or feature was changed after the results were seen.

## Answers to the receipt questions

- *Did the player get more snaps but not more opportunities?* Frequently (see the forensic counts above); snap rises are noisy and mostly reflect game script and availability.
- *Did snaps actually lead workload?* Not measurably beyond what Phase1D already captures from prior shares (snap level correlates about 0.78 with Phase1D share, so the information is largely redundant).
- *Was Phase1D already capturing the change?* Largely yes; the snap candidates move Phase1D by hundredths of a target.
- *Was the team environment the main error?* For rushing the coupled team-volume error (about 2.5 carries) is comparable to the role error (about 3.1); for receiving the role error (about 1.9) dominates the volume error (about 0.9). Phase1N is not reopened.

## Largest remaining role failure

Realized availability and script variation that no pregame deployment signal removes: the oracle role error stays near 1.9 targets and 3.1 carries per player-game.

## Honest caveats

- Selection and lambda choice used the same W9-18 rows, which favours the candidates; they still failed.
- The 4-day lag almost never excludes the most recent game (only unusually short turnarounds); the snap results are therefore close to what an unlagged signal would show, and the conclusion does not depend on the lag.
- F (position routing) was correctly not tested because no basic family survived; a lack of a survivor means no position interaction is justified.
- Snap data only; participation, depth charts and injuries remain unusable under the timestamp rules.

## Reproduction

```bash
pip install -r requirements-research.txt   # Python 3.12, numpy 2.3.5
python nfl_v2_phase1o_role_sources.py --fetch --data-dir /tmp/p1q-data
python nfl_v2_phase1q_snap_role.py --stage develop --data-dir /tmp/p1q-data --out-dir /tmp/p1q-out
python nfl_v2_phase1q_snap_role.py --stage validate --data-dir /tmp/p1q-data --out-dir /tmp/p1q-out   # refuses: gate failed
python nfl_v2_phase1q_snap_role.py --stage burned2026 --out-dir /tmp/p1q-out
python -m pytest -q tests/test_nfl_v2_*.py
```
