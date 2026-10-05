# Phase1I-A — target-depth and QB/receiver catchability

**REJECTED_EFFICIENCY_REPLACEMENT. No replacement selected in 2024; no rescue.**

Receiver depth history survives as a **component-only signal against the position-only reference**, not as an efficiency replacement. It does not satisfy the QB+receiver architecture requirement or the overall incumbent gates. All other tested additions are rejected. Routes-run, alignment, per-route separation and defender assignment remain BLOCKED_DATA. Pressure/blitz were audited but not tested in this isolating experiment.

## Locked chronology and source feasibility

Protocol/source audit committed before real-data performance; implementation/tests/CI committed before development; the 2024 selection lock committed before confirmation. 2024 W1–8 supplies fitted league/position priors; W9–18 selects concentration (two fixed grids) and architecture. Those specifications are frozen, with priors refit on all regular-season 2024. All six fixed ablations are then evaluated once on 2025 and on burned 2026 W1–4. No 2025/2026 tuning, failed-family combination, Monte Carlo, sportsbook input, or Week 5+ access. CI only reproduces committed specifications/results and fails on source revisions.

2025 was untouched **by Phase1I selection**; prior phases already used/reported this season. It is not a newly discovered season-wide holdout. No literal historical T24/T90 source-publication equivalence is claimed for retrospective provider releases.

| Season | REG PBP targets | Passer ID | Receiver ID | Air yards | Completion | Coarse location | Pressure | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 17483 | 17483 | 17483 | 17483 | 17483 | 17483 | 17483 | 17442 |
| 2024 | 17013 | 17013 | 17013 | 17013 | 17013 | 17013 | 17013 | 16992 |
| 2025 | 16609 | 16609 | 16609 | 16609 | 16609 | 16609 | 16609 | 16597 |
| 2026 | 3936 | 3936 | 3936 | 3936 | 3936 | 3936 | 0 | 0 |

PBP supplies the requested depth chain. `pass_location` is coarse left/middle/right, **not alignment**. Historical participation `route` labels describe the targeted receiver only; they do not supply all routes run. Official public nflreadr dictionaries and the nflverse NGS release were investigated. NGS separation/cushion is a weekly/season aggregate, not route-level QB/receiver/defender evidence. It is not substituted as a tracking feature. 2026 participation/pressure/coverage is not published in the frozen corpus; no stale proxy or Phase1H correction is used.

All six source types for 2023–2026 are SHA256-verified against the frozen Phase1H corpus. No new provider files or silent revised labels are accepted. See `phase1i_source_audit.json` for counts, official URLs and blocked reasons.

## Football chain and hierarchy

Fixed buckets: behind LOS (<0), 0–9 ([0,10)), 10–19 ([10,20)), 20+ ([20,infinity)). No edge changes.

`AirYPT = sum(depth_probability[b] * E[air | completed,b] * P(completion | b))`

`YPT = AirYPT + exact frozen Phase1G catch_probability * YAC_per_catch`

Conditional air uses actual completed PBP air yards, including negative air; using air conditional on completion avoids assuming catch-independent depth inside wide buckets. This is an observed completion model, not tracking-derived throw catchability or an ability to separate drops from uncatchable passes.

League priors use only fit-period targets. WR/TE/RB priors shrink toward league with 50 pseudo-targets. Entity windows are the last eight *targeted games* strictly before both target week and target date. Depth uses a Dirichlet posterior; bucket completion uses Beta-style pseudo-targets; conditional completed air uses pseudo-catches. Two preregistered strengths are tested: QB/receiver/pair/opponent = 50/25/80/100 or 100/50/160/200. Hierarchy = position → QB by receiver position → receiver → pair. Pair corrections are disabled below 20 prior pair targets. Overlapping parent/entity samples are pooled empirical Bayes, not independent evidence.

QB assignment uses latest **strictly prior-week** team roster QB identities, ignores status/participation, and ranks prior current-team target passes over three games with 1/.5/.25 decay. Career-history fallback and unknown-QB fallback are explicit. Target-game passer, target-week roster identities and lineup oracle never enter a feature. This is not a dedicated starter model: stale transactions, injuries and replacement decisions remain limitations. Receipt weights are relative prior-pass evidence, not calibrated starter probabilities. Stable GSIS histories follow QB/receiver/pairs across teams; team QB assignment uses current-team passing history.

No new model changes YAC, rescales it using new completion rates, fits final yards, or uses Phase1H corrections. Phase1F has only a native scalar YPT here: its depth/air/catch metrics are NA. Phase1G has native air/catch but no depth PMF; its overall catch rate is applied uniformly only for the explicit by-depth comparator, not misrepresented as a native depth model.

## Required separate ablations

| Architecture | Routed change |
|---|---|
| receiver_depth_only | Receiver bucket PMF and conditional air; position completion |
| qb_depth_only | QB-position bucket PMF and conditional air; position completion |
| qb_receiver_hierarchy | Receiver shrunk toward QB-position profile; position completion |
| qb_receiver_pair | Pair profile shrunk toward hierarchy, >=20 prior targets; position completion |
| opponent_depth_allowance | Opponent positional PMF blended .15/.30; no catch/air adjustment |
| depth_catchability | Position PMF/air; hierarchical QB/receiver/pair bucket completion plus .15 opponent logit adjustment |

Every family must earn >=0.5% completed-air MAE gain and positive blocked-bootstrap evidence against position reference; oracle-yard MAE cannot worsen. Depth families must also improve depth TV >=0.5% with deep-frequency error no worse. Catchability must improve overall and macro depth catch-rate error >=0.5%. Hierarchy/pair must earn >=0.5% additional air gain over their parents. A full replacement additionally needs both QB and receiver, >=1% air gain vs G, >=0.5% oracle gain vs F/history, no worse overall catch vs G, and negative upper 95% blocked-bootstrap bounds. No earned catch/opponent family existed, so **no combinations were built**.

## 2024 development (W9–18)

Fit plays 7541; refit plays 16988; selection rows 1303, positive-target efficiency rows 1265. Selected replacement: **none**.

| Architecture | Air YPT MAE | Oracle yards MAE | Depth TV | Catch MAE | Component gate | Failed component checks |
|---|---:|---:|---:|---:|---|---|
| depth_catchability | 2.8400 | 17.4321 | 0.3341 | 0.2049 | REJECTED | air_block_support, catch_depth_gain, oracle_not_worse |
| opponent_depth_allowance | 2.8576 | 17.2246 | 0.3340 | 0.2061 | REJECTED | air_block_support, air_component_gain, depth_distribution_gain, oracle_not_worse |
| qb_depth_only | 2.8311 | 17.1624 | 0.3304 | 0.2059 | SURVIVES | none |
| qb_receiver_hierarchy | 2.7653 | 17.0839 | 0.3170 | 0.2049 | REJECTED | parent_0_air_gain |
| qb_receiver_pair | 2.7619 | 17.1517 | 0.3164 | 0.2049 | REJECTED | parent_0_air_gain |
| receiver_depth_only | 2.7667 | 17.0570 | 0.3172 | 0.2048 | SURVIVES | none |

The hierarchy improved over the position reference but not materially over receiver-only. Pair history did not earn a material incremental improvement over hierarchy. Catchability slightly improved mean catch error but worsened oracle yards and did not pass depth-catch/bootstrap gates. No overall architecture passed; validation cannot change that decision. All fixed rejected architectures are frozen, never substitutes chosen using confirmation performance.

## diagnostic_2026_wk1_4

Oracle-yard/YPT/catch metrics exclude zero-target rows. Air/depth metrics additionally require PBP target count, catches and receiving yards to reconcile exactly with official stats and completed air+YAC to reconcile per play. Missing/unreconciled labels never become zeros. Conditional bucket catch error only scores rows with observed targets in that bucket; metric denominators are in JSON.

| Architecture | N | Air N | Air YPT MAE | Oracle yards MAE | YPT MAE | Depth TV | Overall catch error | Deep-frequency error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| depth_catchability | 464 | 458 | 3.0319 | 16.8747 | 3.6889 | 0.3341 | 0.2079 | 0.1253 |
| history | 464 | 458 | 3.0354 | 17.4856 | 3.8040 | 0.3273 | 0.2101 | 0.1232 |
| opponent_depth_allowance | 464 | 458 | 3.0710 | 16.9828 | 3.7054 | 0.3341 | 0.2061 | 0.1252 |
| phase1f | 464 | NA | NA | 16.6658 | 3.6691 | NA | NA | NA |
| phase1g | 464 | 458 | 2.9651 | 16.5885 | 3.6493 | NA | 0.2053 | NA |
| position_depth | 464 | 458 | 3.0715 | 16.9949 | 3.7062 | 0.3341 | 0.2060 | 0.1253 |
| qb_depth_only | 464 | 458 | 3.0371 | 16.8980 | 3.6781 | 0.3326 | 0.2060 | 0.1260 |
| qb_receiver_hierarchy | 464 | 458 | 2.9714 | 16.8149 | 3.6764 | 0.3253 | 0.2042 | 0.1229 |
| qb_receiver_pair | 464 | 458 | 2.9613 | 16.8550 | 3.6807 | 0.3251 | 0.2040 | 0.1224 |
| receiver_depth_only | 464 | 458 | 2.9802 | 16.8123 | 3.6771 | 0.3247 | 0.2041 | 0.1225 |

| Architecture | Catch <0 | Catch 0–9 | Catch 10–19 | Catch 20+ | Macro depth catch | Explosive air YPT MAE | Completed 20+ frequency error | Signed yard bias | >20 miss | >30 miss | >40 miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| depth_catchability | 0.2723 | 0.2502 | 0.3690 | 0.3748 | 0.3166 | 1.8778 | 0.0689 | -1.7572 | 0.3147 | 0.1530 | 0.0905 |
| history | 0.2931 | 0.2544 | 0.3787 | 0.3747 | 0.3252 | 1.9456 | 0.0809 | -0.0474 | 0.3211 | 0.1853 | 0.1034 |
| opponent_depth_allowance | 0.2627 | 0.2488 | 0.3578 | 0.3734 | 0.3107 | 1.8234 | 0.0670 | -2.4621 | 0.3147 | 0.1573 | 0.0819 |
| phase1f | NA | NA | NA | NA | NA | NA | NA | -0.8355 | 0.3103 | 0.1444 | 0.0797 |
| phase1g | 0.3396 | 0.2646 | 0.3612 | 0.4253 | 0.3477 | NA | NA | -0.9893 | 0.3082 | 0.1422 | 0.0841 |
| position_depth | 0.2627 | 0.2488 | 0.3578 | 0.3734 | 0.3107 | 1.8232 | 0.0670 | -2.5161 | 0.3147 | 0.1573 | 0.0797 |
| qb_depth_only | 0.2627 | 0.2488 | 0.3578 | 0.3734 | 0.3107 | 1.8406 | 0.0675 | -1.8990 | 0.3082 | 0.1509 | 0.0948 |
| qb_receiver_hierarchy | 0.2627 | 0.2488 | 0.3578 | 0.3734 | 0.3107 | 1.8481 | 0.0677 | -1.2754 | 0.3082 | 0.1487 | 0.0884 |
| qb_receiver_pair | 0.2627 | 0.2488 | 0.3578 | 0.3734 | 0.3107 | 1.8432 | 0.0675 | -1.2266 | 0.3060 | 0.1466 | 0.0884 |
| receiver_depth_only | 0.2627 | 0.2488 | 0.3578 | 0.3734 | 0.3107 | 1.8401 | 0.0675 | -1.5329 | 0.3060 | 0.1530 | 0.0841 |

### Paired two-week moving-block bootstrap

2,000 draws, seed 165, whole games kept together, circular two-NFL-calendar-week blocks within each season; no row-wise bootstrap. CIs are descriptive, not multiple-comparison-adjusted claims. Negative delta favors candidate. Four burned 2026 weeks give very limited evidence.

| Architecture | Air vs G delta | 95% CI | Oracle vs F delta | 95% CI |
|---|---:|---|---:|---|
| depth_catchability | 0.0668 | [0.0436, 0.0865] | 0.2089 | [-0.0828, 0.4565] |
| opponent_depth_allowance | 0.1059 | [0.0773, 0.1303] | 0.3170 | [-0.0327, 0.6138] |
| qb_depth_only | 0.0720 | [0.0461, 0.0940] | 0.2322 | [-0.1215, 0.5323] |
| qb_receiver_hierarchy | 0.0062 | [-0.0245, 0.0368] | 0.1491 | [0.0634, 0.2355] |
| qb_receiver_pair | -0.0038 | [-0.0362, 0.0283] | 0.1892 | [0.0806, 0.2987] |
| receiver_depth_only | 0.0150 | [-0.0223, 0.0521] | 0.1465 | [0.0179, 0.2763] |

### Slices

Slice checks are descriptive, never a tuning/promotion escape. All comparators and secondary metrics for every slice are retained in JSON.

| Architecture | Slice | N | Air YPT MAE | Oracle yards MAE |
|---|---|---:|---:|---:|
| depth_catchability | RB | 70 | 2.0571 | 10.1608 |
| depth_catchability | TE | 122 | 2.6458 | 13.5327 |
| depth_catchability | WR | 272 | 3.4572 | 20.1016 |
| depth_catchability | early | 464 | 3.0319 | 16.8747 |
| depth_catchability | established | 0 | NA | NA |
| opponent_depth_allowance | RB | 70 | 2.0916 | 10.2432 |
| opponent_depth_allowance | TE | 122 | 2.7225 | 13.5443 |
| opponent_depth_allowance | WR | 272 | 3.4805 | 20.2595 |
| opponent_depth_allowance | early | 464 | 3.0710 | 16.9828 |
| opponent_depth_allowance | established | 0 | NA | NA |
| qb_depth_only | RB | 70 | 2.0209 | 10.1273 |
| qb_depth_only | TE | 122 | 2.6838 | 13.2377 |
| qb_depth_only | WR | 272 | 3.4583 | 20.2822 |
| qb_depth_only | early | 464 | 3.0371 | 16.8980 |
| qb_depth_only | established | 0 | NA | NA |
| qb_receiver_hierarchy | RB | 70 | 1.9706 | 10.1193 |
| qb_receiver_hierarchy | TE | 122 | 2.6843 | 13.2260 |
| qb_receiver_hierarchy | WR | 272 | 3.3587 | 20.1477 |
| qb_receiver_hierarchy | early | 464 | 2.9714 | 16.8149 |
| qb_receiver_hierarchy | established | 0 | NA | NA |
| qb_receiver_pair | RB | 70 | 1.9824 | 10.1616 |
| qb_receiver_pair | TE | 122 | 2.6795 | 13.2786 |
| qb_receiver_pair | WR | 272 | 3.3406 | 20.1817 |
| qb_receiver_pair | early | 464 | 2.9613 | 16.8550 |
| qb_receiver_pair | established | 0 | NA | NA |
| receiver_depth_only | RB | 70 | 1.9858 | 10.0753 |
| receiver_depth_only | TE | 122 | 2.6924 | 13.2666 |
| receiver_depth_only | WR | 272 | 3.3661 | 20.1364 |
| receiver_depth_only | early | 464 | 2.9802 | 16.8123 |
| receiver_depth_only | established | 0 | NA | NA |

## validation_2025

Oracle-yard/YPT/catch metrics exclude zero-target rows. Air/depth metrics additionally require PBP target count, catches and receiving yards to reconcile exactly with official stats and completed air+YAC to reconcile per play. Missing/unreconciled labels never become zeros. Conditional bucket catch error only scores rows with observed targets in that bucket; metric denominators are in JSON.

| Architecture | N | Air N | Air YPT MAE | Oracle yards MAE | YPT MAE | Depth TV | Overall catch error | Deep-frequency error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| depth_catchability | 2175 | 2146 | 2.8669 | 17.0395 | 3.7820 | 0.3353 | 0.1997 | 0.1230 |
| history | 2175 | 2146 | 2.8490 | 17.6614 | 3.8879 | 0.3209 | 0.2033 | 0.1207 |
| opponent_depth_allowance | 2175 | 2146 | 2.8715 | 17.0414 | 3.7892 | 0.3352 | 0.1996 | 0.1230 |
| phase1f | 2175 | NA | NA | 16.8751 | 3.7622 | NA | NA | NA |
| phase1g | 2175 | 2146 | 2.8070 | 16.8475 | 3.7572 | NA | 0.1993 | NA |
| position_depth | 2175 | 2146 | 2.8705 | 17.0437 | 3.7890 | 0.3353 | 0.1996 | 0.1230 |
| qb_depth_only | 2175 | 2146 | 2.8597 | 16.9880 | 3.7798 | 0.3307 | 0.1994 | 0.1235 |
| qb_receiver_hierarchy | 2175 | 2146 | 2.7868 | 16.8219 | 3.7510 | 0.3198 | 0.1979 | 0.1207 |
| qb_receiver_pair | 2175 | 2146 | 2.7809 | 16.8473 | 3.7539 | 0.3197 | 0.1977 | 0.1207 |
| receiver_depth_only | 2175 | 2146 | 2.7818 | 16.8073 | 3.7486 | 0.3199 | 0.1979 | 0.1204 |

| Architecture | Catch <0 | Catch 0–9 | Catch 10–19 | Catch 20+ | Macro depth catch | Explosive air YPT MAE | Completed 20+ frequency error | Signed yard bias | >20 miss | >30 miss | >40 miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| depth_catchability | 0.2610 | 0.2606 | 0.3496 | 0.3952 | 0.3166 | 1.8154 | 0.0660 | -0.6893 | 0.3108 | 0.1687 | 0.0864 |
| history | 0.2685 | 0.2664 | 0.3576 | 0.3961 | 0.3222 | 1.7791 | 0.0729 | 0.1999 | 0.3306 | 0.1766 | 0.0984 |
| opponent_depth_allowance | 0.2584 | 0.2578 | 0.3523 | 0.3926 | 0.3153 | 1.8048 | 0.0659 | -1.1573 | 0.3113 | 0.1632 | 0.0855 |
| phase1f | NA | NA | NA | NA | NA | NA | NA | -0.4970 | 0.3126 | 0.1637 | 0.0869 |
| phase1g | 0.3244 | 0.2741 | 0.3540 | 0.4571 | 0.3524 | NA | NA | -0.5027 | 0.3122 | 0.1660 | 0.0864 |
| position_depth | 0.2584 | 0.2578 | 0.3523 | 0.3926 | 0.3153 | 1.8040 | 0.0659 | -1.1869 | 0.3108 | 0.1646 | 0.0855 |
| qb_depth_only | 0.2584 | 0.2578 | 0.3523 | 0.3926 | 0.3153 | 1.8135 | 0.0663 | -0.9652 | 0.3090 | 0.1646 | 0.0846 |
| qb_receiver_hierarchy | 0.2584 | 0.2578 | 0.3523 | 0.3926 | 0.3153 | 1.7853 | 0.0652 | -0.9074 | 0.3099 | 0.1609 | 0.0887 |
| qb_receiver_pair | 0.2584 | 0.2578 | 0.3523 | 0.3926 | 0.3153 | 1.7837 | 0.0652 | -0.8754 | 0.3044 | 0.1637 | 0.0906 |
| receiver_depth_only | 0.2584 | 0.2578 | 0.3523 | 0.3926 | 0.3153 | 1.7804 | 0.0651 | -0.9807 | 0.3094 | 0.1637 | 0.0887 |

### Paired two-week moving-block bootstrap

2,000 draws, seed 165, whole games kept together, circular two-NFL-calendar-week blocks within each season; no row-wise bootstrap. CIs are descriptive, not multiple-comparison-adjusted claims. Negative delta favors candidate. Four burned 2026 weeks give very limited evidence.

| Architecture | Air vs G delta | 95% CI | Oracle vs F delta | 95% CI |
|---|---:|---|---:|---|
| depth_catchability | 0.0599 | [0.0371, 0.0806] | 0.1643 | [0.0222, 0.2984] |
| opponent_depth_allowance | 0.0644 | [0.0396, 0.0866] | 0.1663 | [-0.0393, 0.3924] |
| qb_depth_only | 0.0526 | [0.0291, 0.0727] | 0.1129 | [-0.0654, 0.3073] |
| qb_receiver_hierarchy | -0.0203 | [-0.0481, 0.0080] | -0.0532 | [-0.1627, 0.0544] |
| qb_receiver_pair | -0.0261 | [-0.0573, 0.0067] | -0.0278 | [-0.1510, 0.0889] |
| receiver_depth_only | -0.0252 | [-0.0537, 0.0040] | -0.0679 | [-0.1633, 0.0210] |

### Slices

Slice checks are descriptive, never a tuning/promotion escape. All comparators and secondary metrics for every slice are retained in JSON.

| Architecture | Slice | N | Air YPT MAE | Oracle yards MAE |
|---|---|---:|---:|---:|
| depth_catchability | RB | 350 | 1.9873 | 12.0136 |
| depth_catchability | TE | 540 | 2.5059 | 14.4600 |
| depth_catchability | WR | 1285 | 3.2605 | 19.4923 |
| depth_catchability | early | 478 | 2.5465 | 16.8466 |
| depth_catchability | established | 1697 | 2.9570 | 17.0938 |
| opponent_depth_allowance | RB | 350 | 1.9794 | 12.0466 |
| opponent_depth_allowance | TE | 540 | 2.4898 | 14.1929 |
| opponent_depth_allowance | WR | 1285 | 3.2773 | 19.5990 |
| opponent_depth_allowance | early | 478 | 2.5551 | 16.7465 |
| opponent_depth_allowance | established | 1697 | 2.9604 | 17.1245 |
| qb_depth_only | RB | 350 | 1.9055 | 11.7707 |
| qb_depth_only | TE | 540 | 2.4800 | 14.1561 |
| qb_depth_only | WR | 1285 | 3.2815 | 19.5992 |
| qb_depth_only | early | 478 | 2.5585 | 16.8286 |
| qb_depth_only | established | 1697 | 2.9443 | 17.0330 |
| qb_receiver_hierarchy | RB | 350 | 1.8852 | 11.5252 |
| qb_receiver_hierarchy | TE | 540 | 2.4387 | 14.1022 |
| qb_receiver_hierarchy | WR | 1285 | 3.1809 | 19.4075 |
| qb_receiver_hierarchy | early | 478 | 2.5064 | 16.7878 |
| qb_receiver_hierarchy | established | 1697 | 2.8656 | 16.8315 |
| qb_receiver_pair | RB | 350 | 1.8781 | 11.5029 |
| qb_receiver_pair | TE | 540 | 2.4351 | 14.0958 |
| qb_receiver_pair | WR | 1285 | 3.1744 | 19.4592 |
| qb_receiver_pair | early | 478 | 2.5045 | 16.8456 |
| qb_receiver_pair | established | 1697 | 2.8587 | 16.8478 |
| receiver_depth_only | RB | 350 | 1.8932 | 11.5642 |
| receiver_depth_only | TE | 540 | 2.4299 | 14.0630 |
| receiver_depth_only | WR | 1285 | 3.1740 | 19.3886 |
| receiver_depth_only | early | 478 | 2.5019 | 16.7445 |
| receiver_depth_only | established | 1697 | 2.8605 | 16.8249 |

## Frozen decision and remaining error

| Family / component | Final status | Interpretation |
|---|---|---|
| depth_catchability | REJECTED | No survival/promotion from burned diagnostics or rescue tuning |
| opponent_depth_allowance | REJECTED | No survival/promotion from burned diagnostics or rescue tuning |
| pressure_blitz | NOT_TESTED_ISOLATE_DEPTH_CHAIN | Source audited; isolated depth chain; rejected H modifiers not reused |
| qb_depth_only | REJECTED | No survival/promotion from burned diagnostics or rescue tuning |
| qb_receiver_hierarchy | REJECTED | No survival/promotion from burned diagnostics or rescue tuning |
| qb_receiver_pair | REJECTED | No survival/promotion from burned diagnostics or rescue tuning |
| receiver_depth_only | SURVIVES | Signal against position-only reference; NOT an efficiency replacement and does not meet both-entity architecture |
| route_level_separation | BLOCKED_DATA | No legitimate complete route-level and cutoff-safe source established |
| routes_run_alignment_assignment | BLOCKED_DATA | No legitimate complete route-level and cutoff-safe source established |

Overall: **REJECTED_EFFICIENCY_REPLACEMENT**. Full predicted-workload projections: **NOT_RUN_EFFICIENCY_GATE_FAILED**. Receipt direct projections are explanatory arithmetic, not a scored final-yard stage. No alternate candidate was promoted after seeing 2025.

Completed-air production remains the larger per-target component error. Frozen G air/YAC MAE is 2.8070/2.2443 in 2025 and 2.9651/2.1211 in burned 2026. The depth chain improves representation and conditional catch stratification, but receiver/pair gains over the strong G air incumbent are below 1% and unsupported at the upper95 bound; oracle-yard gains over F are below 0.5%. Conditional completion alone has no independent survival. These are not grounds to select a new parameter using validation.

A genuinely new protocol/data source would need verified pregame QB starter/replacement state and/or complete receiver routes, alignment, depth intent and throw catchability/accuracy or defender assignment (including timing-safe route separation, not weekly aggregate proxies). This experiment cannot resolve those from PBP alone. New route/depth specifications require new preregistration and genuinely clean evidence, not reuse of these 2025/2026 results. Week 5+ is reserved, unaccessed and unscored.

## Artifacts, receipts and scope

`phase1i_receipts.jsonl.gz` retains every fixed ablation row: QB identity/selection weights/prior-roster week; receiver; predicted targets; four depth, conditional-air and completion factors; component contributions; unchanged incumbent YAC; total YPT/direct projection; legal historical sample counts/last dates; evaluation labels and reconciliation flag. All frozen I candidates share identical YAC on the same row. Existing F/G/H forecast artifacts are not overwritten.

The Phase1H scope-test endpoint was fixed to its already-published historical head so it continues auditing the frozen H change instead of rejecting a later independent protocol. H code, protocol, fitted/frozen artifacts and results are byte-identical. The new I scope test SHA256-checks every protected base file. No production, frontend, scheduler, calibration, distributions, other sports, or legacy science changed.

Data-quality exclusions/assignment counts (not performance selection):

```json
{
  "2024_component_reconciled": 2276,
  "2024_component_unreconciled": 35,
  "2024_qb_PRIOR_CURRENT_TEAM_PASSING": 2190,
  "2024_qb_UNKNOWN_PRIOR_QB": 121,
  "2025_component_reconciled": 2200,
  "2025_component_unreconciled": 29,
  "2025_qb_PRIOR_CURRENT_TEAM_PASSING": 2229,
  "2026_component_reconciled": 473,
  "2026_component_unreconciled": 7,
  "2026_qb_PRIOR_CURRENT_TEAM_PASSING": 480,
  "missing_common_comparator_rows": 8
}
```
