# Phase1K-R — PBP-only rushing efficiency mechanics

**REJECTED_EFFICIENCY_REPLACEMENT. No development family passed; no selected specification.**

The protocol and source audit were committed before real-season fitting. Implementation/tests/workflow were committed next, with no computed performance. The 2024-only development lock was then committed. The preregistered immediate stop applies: **no new 2025 validation scores, no new 2026 W1–4 diagnostic scores, no refit, no combined A/B rescue, and no full predicted-carry projection**. These results do not establish a replacement. Old frozen Phase1F/H/I findings were read as requested; 2025 is not globally fresh and has not been used to choose this phase's architecture.

Receiving remains **FROZEN_AT_PHASE1F_PENDING_NEW_INFORMATION**. Receiver depth remains **SURVIVED_SIGNAL_NOT_PROMOTED**. Neither is reopened.

## Source audit and historical limitations

Only hashed nflverse PBP and official weekly rushing/identity statistics were used. No FTN, box counts, OL proxy, personnel absence, paid vendor data, sportsbook line/odds or Monte Carlo input. Source-only coverage checks cover 2023–2026; the frozen 2026 corpus contains only W1–4. Retrieval first checks the release digest and rejects revised bytes before downloading; there is no latest-data fallback.

| Season | RB/FB/HB audited carries | Missing run location | Raw missing gap |
|---|---:|---:|---:|
| 2023 | 11,738 | 0.094% | 26.640% |
| 2024 | 11,831 | 0.059% | 25.644% |
| 2025 | 11,882 | 0.050% | 26.772% |
| 2026 | 2,788 | 0.036% | 27.224% |

EPA/success, distance, down, yardline, formation/no-huddle, score context, core IDs and carry labels are populated. `success` means EPA >0, not independent blocking quality. Negative carries are yards <0; zero is separate; 10+/20+ derive from realized historical carry yards. QB scrambles and kneels never enter candidate player/defense/league aggregates. PBP cannot prove all non-scramble QB carries are designed; **verified designed-QB intent remains BLOCKED_DATA**, so QB is outside this promotion target.

**Gap-audit limitation, corrected without rescue:** the initial <=5% missingness gate was applied to the entire carry population and was too conservative for gap semantics. Source-only follow-up confirms **zero missing gap labels on left/right RB carries** in every frozen season (8,611 / 8,797 / 8,701 / 2,029 carries). Missing middle/unknown gaps are structural. Thus gap labels are conditionally available, not missing on a quarter of otherwise eligible left/right carries. The original audit/protocol/lock stay frozen: F tests left/middle/right only; gap is **AVAILABLE_CONDITIONALLY_NOT_TESTED_THIS_PHASE**, not a proven failed football signal. No extra model was fit after this correction. The frozen snapshot carries this machine-readable clarification.

Historical PBP is a corrected retrospective release, not an original first-publication archive. Features require strict earlier season/week **and source game date +48h <= target game date −24h at 00:00 UTC**. This conservative completion/cutoff bound excludes same-week results and recent incompletely elapsed games; it does not prove historical T24/T90 provider vintages. EPA itself is nflfastR-derived. OL/starter/personnel/box restrictions from J remain blocked and unmodified.

## Fixed population and interpretable mechanics

Pregame membership uses prior-known RB/FB/HB identities, at least two completed current-team stat games and last-three mean carries >=5. Target participation, actual carry count and target-game direction/context never determine membership. Missing player stat rows become zero labels only after whole-game/team official-stat coverage validation. This is an efficiency population, not a new availability model: prior-known departed/inactive players can remain zero-carry candidates. Skill follows stable identity across teams; current-team mix/role does not.

2024 W1–8 supplies **358 positive-carry fit rows**. W9–18 contains **633 fixed candidates: 438 positive-carry and 195 zero-carry rows**. All models use identical candidate keys. Zero-carry rows stay in receipts and are excluded only from undefined YPC/rate metrics, avoiding dilution by easy zeros. All 438 positive rows reconcile PBP carry counts and yards exactly to official labels. No actual-carry minimum or candidate-specific deletion is used. Cold starts/new-team players without the locked prior history are outside scope; these results cannot certify them or FB/HB players (none had positive selection rows).

The six independent models are train-only standardized ridge YPC residual corrections to the exact frozen Phase1B efficiency used by F/E. Alphas 0.1/1/10 are selected only on later 2024. Player/defense profiles use 8 prior games and 50/160 carry shrinkage; conditional yields use 20. Output YPC is bounded [0,12]. Twelve-decimal artifact/model precision prevents irrelevant numerical-platform drift; it is not a new searched parameter. The old incumbent configuration was already selected using all 2024 in an earlier phase; it is held fixed here, not retuned. This is conservative benchmarking against a locked incumbent, not a fresh development claim for it.

- A: player YPC, median, negative/zero/success/10+/20+, short-yard and goal-line profile.
- B: RB-specific defensive YPC/EPA/success/negative/explosive allowance, per carry rather than yards/game.
- C: A+B only if both independent parents pass; **not run**.
- D: <10, 10–19, 20+ probabilities and conditional yields; routine + explosive contribution.
- E: <=0, 1–9, >=10 state probabilities/yields and prior success rate.
- F: prior current-team left/middle/right/unknown mixture, shrunk player contextual yield and defensive allowance; no realized target direction.
- G: prior current-team goal-line/short-yard/early-down/other × shotgun/under-center mixture and no-huddle tendency; no realized target situation.

## Development efficiency comparison

Paired delta CIs use 2,000 seeded **14-calendar-day moving blocks of complete games**, never row bootstrap. Negative delta favors the candidate. Selection is oracle-carry MAE first; every candidate must also pass independent YPC/component, practical gain and catastrophic-miss guards.

| Fixed comparator | Oracle-carry yards MAE | YPC MAE |
|---|---:|---:|
| competent_human | 15.209 | 1.498 |
| phase1e_efficiency_same_as_f | 14.608 | 1.451 |
| phase1f_incumbent | 14.608 | 1.451 |
| position_league | 14.704 | 1.449 |
| recent_player_ypc | 16.300 | 1.577 |

| Family | Oracle-carry yards MAE | YPC MAE | 95% paired delta CI | Decision |
|---|---:|---:|---|---|
| A_player_profile | 14.445 | 1.425 | [-0.289, -0.001] | FROZEN_REJECTED |
| B_defense_allowance | 14.437 | 1.426 | [-0.300, 0.003] | FROZEN_REJECTED |
| C_player_defense | — | — | — | Not run: failed A/B parents |
| D_explosive | 14.409 | 1.424 | [-0.359, 0.065] | FROZEN_REJECTED |
| E_states | 14.437 | 1.423 | [-0.303, 0.032] | FROZEN_REJECTED |
| F_location | 14.422 | 1.424 | [-0.319, 0.013] | FROZEN_REJECTED |
| G_context | 14.429 | 1.425 | [-0.318, 0.013] | FROZEN_REJECTED |

Required gain was max(0.50 yard, 2%) oracle-carry MAE and max(0.03 YPC, 1%), plus bootstrap upper bound <0 and no >1 percentage-point worsening in >40/>50 misses. **Every tested family failed both absolute practical gates.** A's selected delta CI narrowly excludes zero, but its 0.163-yard gain is only 1.11% and its 0.026 YPC gain is below 0.03. This is exactly why statistical detectability alone cannot promote a candidate. Other CIs include zero. No failed families are combined.

The best numerical D result improves oracle MAE only **0.199 yard (1.37%)** and YPC MAE 0.027. Shrunk negative/success/10+ rates generally beat raw recent-five player rates, but 20+ rate MAE worsens (A/D/E/F/G 0.02735 vs 0.02557); a weak rate comparator win does not overcome the full independent efficiency failure. Incumbent has no state probabilities, so the component-rate comparator is explicitly raw recent-five PBP history, never a probability invented from its mean. Counts, predicted/observed rates and component errors are in the lock.

Incumbent YPC signed bias **0.264**, oracle signed bias **1.368 yards**, median absolute error **10.943**. Within ±5/10/15/20: **25.8% / 46.8% / 61.2% / 74.9%**. Misses >30/40/50: **12.1% / 5.3% / 1.8%**. Candidate intervals/bias/miss metrics and slices are in the lock. Candidate >40 rates rise to about 5.9–6.2%, with >50 about 2.3–2.5%; these do not exceed the preregistered 1pp guard, but provide no reason to promote.

High prior-role slice has 148 positive rows; lower-role has 290. D oracle MAE is 18.588 vs 12.276 respectively. Selection is W9–18, so early W1–4 slice has no rows; FB/HB likewise has no positive evidence. Do not invent confirmation slice evidence for unrun 2025/2026. Registry/failure decisions remain frozen.

## Oracle diagnostics: where the remaining error lives

All are **POSTGAME_ORACLE_DIAGNOSTIC_ONLY**, after frozen pregame evaluation, on the same 438 positive rows. These replace mixtures/counts in unchanged prior conditional-yield tables. They are descriptive table diagnostics, **not the ridge candidate with an oracle covariate**, not promotion evidence and not newly achievable pregame accuracy.

| Prior conditional-yield table | Pregame mixture MAE | Actual mixture/count oracle MAE |
|---|---:|---:|
| Run location | 15.744 | 15.813 |
| Formation/context | 15.964 | 16.192 |
| Routine/explosive tail | 15.008 | 8.851 |

Knowing actual context/location does not help these tested tables. Knowing tail occurrence is the largest tested efficiency lever, reducing table MAE by about 41%; substantial within-state yield error still remains. The most severe incumbent errors include Barkley's 255 yards on 26 carries and Taylor's 218 on 29, despite correct carry counts in the primary test. This does **not** prove all residual variance is irreducible or that prohibited box/OL/personnel inputs would fix it.

**Full workload tests are NOT_RUN_INDEPENDENT_EFFICIENCY_GATE.** No fresh predicted-carry/full-yard or workload-oracle comparison is fabricated. Old F found workload and efficiency both problematic on its different all-rusher cohort; its headline MAEs must not be compared directly to this RB-only, stricter-cutoff cohort. If future efficiency passes, exact frozen E workload receipts/assumptions must be handled explicitly; no target-roster or OL proxy may be invented.

## Reproduction and scope

```bash
python nfl_v2_phase1k_sources.py --fetch --data-dir /tmp/phase1k-data --out /tmp/phase1k-audit.json
python nfl_v2_phase1k_rushing_efficiency.py --data-dir /tmp/phase1k-data --stage develop --lock /tmp/phase1k-lock.json
python nfl_v2_phase1k_rushing_efficiency.py --data-dir /tmp/phase1k-data --stage confirm --out /tmp/phase1k-results.json --receipts /tmp/phase1k-receipts.jsonl.gz
python -m pytest -q tests/test_nfl_v2_*.py
```

`confirm` checks the committed protocol/lock and, on this empty survivor set, opens only 2023/2024 to reproduce the development ledger/diagnostics. It does not open 2025/2026 for predictions. CI runs audit → lock reproduction → conditional confirmation/stop → exact artifacts/ledger comparison → upload and compact existing-PR comment. The frozen raw source corpus was re-downloaded after digest verification and the audit reproduced byte-for-byte.

Receipt ledger: **6,963 deterministic gzip rows**, all eleven models/comparators on the same 633 candidate keys, with carry/efficiency/rate/mix/history/cutoff/error receipts. Predicted carries/full direct projections are explicitly null because their gate was not reached. No scientific store or existing forecast record changes.

The only existing-file changes are additive registry metadata and a test-only Phase1J scope endpoint fixed to its own published head; a new independent K test proves all other **1,163 prior files byte-identical** to starting HEAD. Source inventory and all E/F/H/I/J scientific artifacts/code stay unchanged, as do production/frontend/scheduler/grading/calibration and every other sport. No new PR and no merge.

**Final: REJECTED_EFFICIENCY_REPLACEMENT. Retain Phase1F rushing efficiency. Freeze all six tested failures; C remains unrun after parent failure. No rescue, no receiving reopen, no Week5+ access.**
