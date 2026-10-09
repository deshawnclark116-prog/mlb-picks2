# Phase2B-DATA: forward availability capture (data capture only; Phase2A v1.0 untouched)

## Part 1 summary: the red legacy audit
`NFL V2 research audit` failed because its fetch step downloaded `https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv`, a release asset that now returns HTTP 404 upstream (confirmed locally). The audit used that file for one thing: the set of final (season, week, team) game-teams (`load_final_teams`: result or both scores present). I froze exactly that as `phase0_frozen_final_games_2026_w1_4.json` (64 regular-season 2026 W1-4 games, allowlisted fields game_id, season, game_type, week, home_team, away_team, final; no scores, results or betting columns; sha256-verified rows). Provenance: the exact release-asset bytes retrieved 2026-10-06 (sha256 `1896f36a...`) and a cross-check against the public nfldata `games.csv` retrieved 2026-10-07, which agree on every allowlisted field and finality (7,340 rows compared, 0 differences). The audit now reads the artifact (digest-checked; a changed row, flag, team or count fails) and no workflow downloads a schedule. Proof the science is unchanged: running the real Week 4 audit with the old CSV semantics and with the frozen artifact on identical inputs produced byte-identical JSON (sha256 `f301d22b...`), and no Phase0/Week4/Phase1 scientific artifact changed. All 18 workflows were green afterwards.

## Question
Can a timestamp-valid forward archive of offensive-player availability at T24 and T90 be built without historical backfill or leakage? This phase only collects. It does not ask whether availability improves projections, and Phase2A engine v1.0, its T24 ledger rows, comparator rows and lock are byte-identical to the first-forecast commit (tested).

## Sources audited (2026-10-07; no vendor payload was available or tested)
| Source | Classification | Why |
|---|---|---|
| nflverse weekly rosters (status) | **ACCEPTED_FORWARD_CAPTURE** (roster_status only, LIMITED_INFORMATION) | CC-BY-4.0 release already an accepted V2 input; current-week rows exist before kickoff; no per-record timestamp, so knowledge time = our retrieval time |
| nflverse injuries release | PROMISING_NEEDS_TERMS | derived from official NFL injury reports with no stated reuse terms; observed `injuries_2026.csv` has 16 columns, weeks 1-4 only, and no `date_modified` although the dictionary documents one |
| NFL.com official injury report | BLOCKED_LICENSE | terms 1.3: "Systematic retrieval ... is prohibited absent our express prior written consent"; 11(f) bars spiders/robots; no scraper built |
| Team / aggregator injury pages | BLOCKED_LICENSE | no authorization; a viewable page is not permission |
| Sportradar, SportsDataIO | PROMISING_NEEDS_ACCESS | documented shape fits, no credentials or sample payload |
| Manually supplied authorized records | ACCEPTED_FORWARD_CAPTURE (channel, empty) | requires an authorization reference and original publication time; none exist |
| nflverse snaps / participation | BLOCKED_TIMING | postgame |
| nflverse depth charts | PROMISING_NEEDS_TERMS | roles, not availability; reuse rights unresolved |

**No accepted source provides injury designations or practice participation**, so the injury-aware questions (roster ACT but officially inactive, QUESTIONABLE -> OUT) cannot be populated yet; the interface and archive are built and wait for an accepted source.

## Schema and as-of semantics
`phase2b_availability_schema.json` carries every requested field (state in ACTIVE, EXPECTED_ACTIVE, QUESTIONABLE, DOUBTFUL, OUT, INACTIVE, IR, PUP, LIMITED, FULL, DNP, RETURNING, UNKNOWN; practice/game status; injury body part and detail; roster status; provider, product, record id; published/retrieved/effective times; cutoff eligibility; valid_from/valid_to; revision_sequence; original_vintage, backfilled, stale; raw and normalized sha256; confidence). `availability_at(player, game, cutoff_time)` uses only records with published_at <= cutoff AND retrieved_at <= cutoff (retrieval time is the knowledge time when the source gives none), excludes postgame and backfilled-without-original-vintage records, and takes the highest visible revision. A later update never leaks backward: QUESTIONABLE at T24 then OUT at T90 returns QUESTIONABLE at T24 and OUT at T90 (tested, including a record retrieved after the T90 cutoff). No record means UNKNOWN, never ACTIVE. Windows are exactly the Phase2A windows; a late attempt is logged `MISSED_CAPTURE` and never backfilled.

## Real captures
One real capture exists: **`2026_05_TB_DAL` at T24, 48 offensive roster-status rows** retrieved 2026-10-07T01:17:57Z (inside the T24 window, before the 2026-10-08T00:15Z cutoff): EXPECTED_ACTIVE 29, IR 5, UNKNOWN 14 (practice squad / retired). They link read-only to the 11 Phase2A T24 forecasts, all of which were EXPECTED_ACTIVE at their cutoff, consistent with their recorded roster status. No capture was missed. Nothing was captured for T90 (its window has not opened) and nothing for other games (their windows have not opened).

## Observational link and value report
`link_phase2a` maps every Phase2A forecast to the state visible at its own cutoff by (game_id, player_id, cutoff_type) and writes nothing to Phase2A. `availability_report` (committed empty, `NO_GRADED_GAMES_YET`) will count forecasts by state group, give MAE by state, T24 -> T90 status changes and the number of catastrophic misses (receiving yards > 40, rushing yards > 30, targets > 4, carries > 8, receptions > 3) associated with a downgrade, a late inactive or an incomplete state, descriptively and without causal claims.

## QB
QB starter / health payloads are not stored here: `route_qb_payload` delegates to `ingest_qb_state`, and `qb_state_link` is `certify_for_forecast` (currently BLOCKED, NO_ACCEPTED_QB_STATE_SOURCE). The only QB rows in this archive are roster-status observations flagged `ROSTER_STATUS_ONLY_NOT_A_QB_STATE`. No QB source is certified.

## Operator aid
`python nfl_v2_phase2b_availability.py --mode next-due` is read-only (no clock override, creates nothing) and prints NEXT_DUE_FORECAST_WINDOW, opens_at, cutoff_at and games_due. At commit time: **T90, opens 2026-10-08T21:15:00Z, cutoff 2026-10-08T22:45:00Z, games_due `2026_05_TB_DAL`**. The T90 forecast was not generated (the clock was far outside its window). Run the Phase2A forecast command and `--mode capture-roster` inside that window.

## Honest limitations
Roster status is not a game-day designation (ACT does not mean healthy), so the only real capture says little about injuries. The value of availability for projection is unmeasured and no model change is made or implied; any use needs a separate preregistered phase after enough forward records and an accepted injury source.
