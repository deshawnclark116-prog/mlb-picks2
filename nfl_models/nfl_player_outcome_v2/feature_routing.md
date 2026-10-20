# NFL V2 feature routing

This file defines what each current information source is *for*. A feature cannot wander into unrelated parts of the model.

| Source / signal | Football purpose | Allowed destination |
|---|---|---|
| Schedule, opponent, home/away, rest | Establish game context | team environment |
| Historical play-by-play | Learn pace, play calling, opportunity and event behavior | team environment, role history, efficiency history |
| Team play/rush/dropback history | Estimate available team opportunity | team environment |
| Player carries/targets/routes/snaps | Measure current role | role and usage |
| Injury reports | Availability and redistribution caused by absences | availability, role |
| Weekly rosters | Define who can plausibly participate | availability / candidate universe |
| Timestamped depth charts | Starter/backup hierarchy and replacement prior | role prior |
| Teammate absence | Reallocate opportunity | role/share |
| Organizational-intent prior | Resolve ambiguous role before enough current usage exists | role prior only |
| Red-zone / goal-line history | Valuable/scoring opportunity | role and event hazards |
| Opponent allowed rates | Measure opponent effect relative to league/team norm | team environment and validated efficiency components |
| Box / stacked-front tendencies | Run-defense mechanism | rushing matchup |
| Blitz / pressure | QB/pass mechanism | sack, completion, passing efficiency |
| Man/zone/coverage when current and timestamp-safe | Passing-game matchup mechanism | route/target/catch/air-yard components if validated |
| DL/LB/DB/OL reported absences | Personnel degradation | relevant matchup components |
| Player rush efficiency | Production per carry | rushing efficiency |
| Air yards | Downfield usage/production | receiving and passing efficiency |
| Catch rate | Target-to-catch conversion | receptions |
| YAC | Catch-to-yard conversion | receiving/passing yards |
| QB completion/TD/INT/sack history | Per-dropback outcomes | passing efficiency/events |
| Role-change signals | Detect that old usage is stale | weighting / role state |
| Uncertainty signals | Decide how much trust to place in projection | uncertainty / no-action decision |
| Monte Carlo | Propagate already-modeled uncertainty | distribution only |
| Sportsbook lines/odds | Compare frozen model to executable market | downstream only; never model |

## Currently blocked / limited

- True timestamp-safe offensive-line quality is not available in the current repository.
- Historical pregame OL starter identity is not available for 2023-24 from the current depth-chart source.
- Some 2026 coverage/pressure information is stale or incomplete and must be labeled as such.

Blocked data stays blocked. We do not invent substitutes and call them the real thing.
