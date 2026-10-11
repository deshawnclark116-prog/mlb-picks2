# Tennis V2 feature routing (Phase0, design only — nothing here is built)

Routing answers one question per candidate input: **which layer of the match distribution may it feed, and is it safe pre-match?** Evidence labels refer to `phase0_*` artifacts.
Layers follow the protocol standard: PLAYER STATE → SURFACE/TOURNAMENT → SERVE → RETURN → FORMAT → INTERACTION → point/service-game mechanics → set distribution → match distribution → derived outcomes.
No betting-market quantity may enter any science layer (see `protocol.json` firewall). Execution data lives in a separate downstream record.

| Input (available in the pinned TML data) | Layer | Pre-match safe? | Evidence / caveat |
|---|---|---|---|
| Result history (winner/loser, date, round, tournament) | PLAYER STATE (rating) | yes, in CANONICAL order only | as-implemented text-`match_id` order processes later rounds before earlier rounds in ~36% (ATP) of cross-round pairs (`phase0_population_definition.json` order_leakage) |
| Rank, rank points, age | PLAYER STATE (second, independent rating signal) | yes (value as printed for the tournament week) | rank-logistic alone is calibrated (ECE 0.02) and the 50/50 Elo+rank human blend beats the incumbent Brier with interval above 0 in 3 of 4 tour-periods |
| Surface (exact, from tournament history) | SURFACE | yes when KNOWN_EXACT / HISTORICAL_TOURNEY_INFERENCE | live pick log: 98.6% HARD_FALLBACK because ESPN sponsor names never match TML names (`phase0_surface_audit.json`); no indoor/outdoor or speed field |
| Best-of (3/5), round | FORMAT | yes | best-of alone explains most of total-games skill (`bestof_fit` beats the incumbent) |
| Aces, double faults, serve points, 1st-in, 1st/2nd points won, break points | SERVE / RETURN (own and opponent-derived return) | history only; never the same match | DATA_SUFFICIENT on both tours (`phase0_serve_return_feasibility.json`): coverage ≥ 0.95, identity violations < 0.03%, out-of-time R² improvement for serve-point win 9.8% ATP / 4.4% WTA |
| Serve games, break points saved/faced | SERVICE-GAME MECHANICS (hold probability check) | history only | closed-form hold from serve-point win tracks observed hold on average (mean 0.803 vs 0.795 ATP) |
| Minutes played | FATIGUE / TRAVEL proxy | history only | not evaluated in Phase0; **tempting leakage if used for the current match** |
| Retirement marker (RET/DEF) | POPULATION rule | label only | excluded from performance labels; winner kept for ratings; settlement is a downstream book rule |
| Player identity (TML id) | IDENTITY gate | ids only | live ESPN→TML resolution has a surname guess (2.9% of logged players) and ambiguous rows; V2 forbids forecasts for non-EXACT/NORMALIZED_EXACT identities |
| Tournament level, qualifying/Challenger | POPULATION | n/a | Challenger and qualifying rows are ABSENT from both tours; WTA contains ITF-level rows (level `I`) |
| Ranking of opponents faced / strength of schedule | INTERACTION | history only | not evaluated in Phase0 |

Never routed (banned from science): any line, price, implied probability, bookmaker field, consensus, or the incumbent's `model_prob` / `predicted_mean` from the live log.
Not available in this source and therefore *not routable without a new source*: point-by-point data, court speed, indoor/outdoor, altitude, injury/withdrawal timing, scheduled start times for history, handedness/height in the schema used.
