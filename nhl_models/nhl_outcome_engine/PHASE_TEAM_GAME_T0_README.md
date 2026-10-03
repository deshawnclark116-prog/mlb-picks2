# NHL Team-game T0 - team goals / moneyline architecture audit (no model fit)

Protocol `phase_team_game_t0_protocol.json`; results `phase_team_game_t0_feasibility.json`. No sportsbook data.

| question | result |
|---|---|
| T1 labels | COMPLETE (11,052 games 2017-2025; scores, period type, final state all present; no ties) |
| T2 legacy label | LEGACY_LABEL_EQUALS_FINAL_WINNER (9,781/9,781); OT/SO decides 20.7-24.9% of games per season |
| T3 score vs skater goals | shared with S0 S4 (reconciled with explained differences) |
| T4 goal distributions | regulation goals Poisson-like per team (dispersion 1.013 / 1.003) but observed regulation-tie rate 22.4% vs 16.7% under independent Poisson; home-away corr -0.079 -> verdict **LIMITED**: independence understates ties, the joint model must carry dependence / tie mass |
| T5 derivation | DERIVATION_IDENTIFIABLE: P(home win) = P(home>away in regulation) + P(regulation tie) * P(home wins OT/SO | tie); P(home wins | OT/SO) = 0.512; OT decides 67% of regulation ties, SO 33% |
| T6 PIT inputs | CONSTRUCTIBLE_WITH_LISTED_BLOCKERS: blockers = goalie start confirmation, special-teams / penalty acquisition not frozen yet, joint-dependence modelling |

Verdict: `T0_ARCHITECTURE_FEASIBLE_WITH_LISTED_BLOCKERS`. The old binary moneyline model remains LEGACY_STABLE_COMPARATOR.
