# NHL Scoring S0 - goals / assists / points feasibility (no model fit)

Protocol `phase_scoring_s0_protocol.json`; results `phase_scoring_s0_feasibility.json`.

| question | result |
|---|---|
| S1 label coverage | COVERAGE_EXACT (all 9 seasons, 0 gaps, 0 nulls) |
| S2 identities | POINTS == GOALS + ASSISTS **exact on all 397,778 skater-games**; EV+PP+SH splits exact; goals <= shots violated in 30 rows (all +1; 18/18 through 2023 sit inside the 113 official-SOG differences: the stat line omitted the goal's shot) |
| S3 vs play-by-play | goals / assists mismatches 0 / 0 on 285,146 player-games (2017-2023); 3 PBP scorers outside the skater table are goalies (empty-net goals) |
| S4 team goals | 21,282 team-games equal, 816 = shootout winner +1; 6 others = goalie goals (3 verified in PBP, 3 in 2024-25 unverifiable) |
| S5 conversion inputs | goals per shot C/L/R ~0.12, D ~0.05; 24% zero-shot rows; ~97% of target rows have >=10 prior appearances |
| S6 season consistency | STABLE (assists per goal 1.67-1.69 every season) |
| S7 joint structure | JOINT_GENERATION_SUPPORTED: every goal has <= 2 assists (36,293 / 8,651 / 3,104 goals with 2 / 1 / 0); player assists vs team goals r = 0.29 |

Verdict: `S0_LABELS_RECONCILED_JOINT_GENERATION_FEASIBLE`. Points can be derived from the joint goals / assists process. No model was tuned.
