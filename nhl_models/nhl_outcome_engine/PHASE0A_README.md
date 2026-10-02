# NHL outcome engine — Phase 0A (data / chronology feasibility, shots on goal)

No model was trained, nothing tuned, no frontend, existing NHL models and predictions untouched. Evidence: `phase0a_data_feasibility.json` (probe: `nhl_outcome_phase0a_probe.py`, builder: `nhl_outcome_phase0a_build.py`).

## Verdict
A leak-safe SOG outcome engine is **partly feasible**.

**Feasible (PASS):** everything from TOI downward. On 8 hash-selected games (2018, 2021, 2023, 2025 seasons; 288 skater-games) the NHL web API boxscore, play-by-play, stats-REST summary, stats-REST time-on-ice and shift charts agree with each other: 0 SOG mismatches (play-by-play shootout shots excluded), 0 TOI mismatches, EV+PP+SH = total in every row, shift sums and counts match after de-duplication. One numeric `playerId`/`teamId`/`gameId` is shared by all sources. Boxscore / play-by-play / strength TOI go back to 2009, shift charts to 2010. Event-level shot attempts (shot, miss, block, goal with shooter ids, strength code, coordinates) exist.

**Not feasible as-is (BLOCKER):** the front of the chain — availability, goalie, lineup / power-play deployment *as known before the cutoff*. Every endpoint returns only the current value; scratches, the goalie `starter` flag and dressed lists for past games are post-hoc truth, and no pre-game version is archived. Live, scratches appear only ~1 minute before puck drop; no confirmed-starter field exists. Schedule time changes are overwritten too.

## Defects found in what exists
- `nhl_models/nhl_model.sqlite` `skater_games` omits players (zero player-seasons with two teams; 39 of 288 sampled dressed skaters have no row, including players with SOG). It cannot be the outcome source of truth. It also has no puck-drop time, strength, shift or event data.
- Shift charts contain duplicate rows (up to 19 in one game); shootout shots look like SOG in play-by-play.
- `games.week` and the team-state tracker are 7-day synthetic buckets, not exact chronology.

## Must solve next (needs approval; Phase 0B)
1. Availability policy: (a) model availability/role from prior-game participation, TOI and shifts only, and (b) start forward, timed, content-addressed capture of pre-game scratches / rosters / goalies now (season began 2026-09-29).
2. Rebuild a complete chronological skater-game + event + shift table from the API (de-duplicated, shootout excluded). Estimated ~1.3 s/game ≈ 30 min per 1,312-game season; stop and confirm before running.
3. Test power-play-unit derivation from shifts + situation codes on a small sample.
4. Fix the cutoff rule (puck drop − 90 min / − 24 h) and the allowed sources at each.
