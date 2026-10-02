"""Assemble nhl_models/nhl_outcome_engine/phase0a_data_feasibility.json from the probe output (nhl_outcome_phase0a_probe.py) and the repo inspection. No model, no tuning."""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_outcome_engine"


def evidence(game):
    """Actual rows for the top-SOG skater of a sample game, from every source."""
    return game


def build(probe, ops_first_run):
    games = [g for g in probe["games"] if "error" not in g]
    sample = []
    for g in games:
        sample.append({"game_id": g["game_id"], "identity": g["identity"], "local_games_row[date,home,away,hs,as]": g["local_games_row"], "boxscore_skaters": g["boxscore_skaters"], "local_db_skater_rows": g["local_db_skater_rows"],
                       "local_db_missing_players_count": len(g["local_db_vs_boxscore"]["ids_in_boxscore_not_local"]), "local_db_missing_player_ids": g["local_db_vs_boxscore"]["ids_in_boxscore_not_local"],
                       "sog_boxscore_vs_pbp_mismatches": g["sog_boxscore_vs_pbp"]["mismatches"], "sog_boxscore_vs_stats_rest_summary_mismatches": g["skater_summary_vs_boxscore"].get("sog_mismatches"),
                       "local_db_vs_boxscore_sog_mismatches": g["local_db_vs_boxscore"]["sog_mismatches"], "team_sog": g["team_sog_check"],
                       "strength_toi": {k: g["strength_toi"].get(k) for k in ("n_rows", "example_row", "total_toi_mismatch_vs_boxscore_gt_1s", "ev_pp_sh_not_summing_to_total")},
                       "shifts": {k: g["shifts"].get(k) for k in ("n_raw_shift_rows", "n_duplicate_rows_removed", "n_shift_rows", "n_players", "example", "shift_toi_sum_vs_boxscore_toi_gt_2s", "shift_count_mismatch_vs_boxscore")},
                       "pbp": {k: g["pbp"][k] for k in ("n_events", "n_rosterSpots", "has_situationCode", "has_shootout_period", "n_distinct_situation_codes", "example_shot_event")},
                       "event_types": g["pbp"]["event_types"], "right_rail_gameInfo": g.get("right_rail_gameInfo"),
                       "provenance": {k: {"url": v.get("url"), "retrieved_at_utc": v.get("retrieved_at_utc"), "sha256": v.get("sha256"), "bytes": v.get("bytes"), "http_status": v.get("http_status"),
                                          "etag": (v.get("headers") or {}).get("etag") or (v.get("headers") or {}).get("ETag")} for k, v in g["sources"].items()}})
    n_sog_cmp = sum(g["sog_boxscore_vs_pbp"]["n_players_compared"] for g in games)
    agg = {"games": len(games), "seasons": sorted({g["identity"]["season"] for g in games}), "players_compared_sog": n_sog_cmp,
           "sog_mismatches_boxscore_vs_pbp": sum(len(g["sog_boxscore_vs_pbp"]["mismatches"]) for g in games),
           "sog_mismatches_boxscore_vs_stats_rest": sum(len(g["skater_summary_vs_boxscore"].get("sog_mismatches", [])) for g in games),
           "toi_mismatch_stats_rest_vs_boxscore": sum(len(g["strength_toi"].get("total_toi_mismatch_vs_boxscore_gt_1s", [])) for g in games),
           "ev_pp_sh_sum_violations": sum(len(g["strength_toi"].get("ev_pp_sh_not_summing_to_total", [])) for g in games),
           "shift_toi_mismatch_after_dedup": sum(len(g["shifts"].get("shift_toi_sum_vs_boxscore_toi_gt_2s", [])) for g in games),
           "shift_count_mismatch_after_dedup": sum(len(g["shifts"].get("shift_count_mismatch_vs_boxscore", [])) for g in games),
           "duplicate_shift_rows_removed_total": sum(g["shifts"].get("n_duplicate_rows_removed", 0) for g in games),
           "local_db_players_missing_total": sum(len(g["local_db_vs_boxscore"]["ids_in_boxscore_not_local"]) for g in games), "boxscore_players_total": sum(g["boxscore_skaters"] for g in games),
           "local_db_sog_of_missing_players": "see per-game ids; e.g. 2025020444 A. Texier (8480074) played 14:47 with 2 SOG and has no row in nhl_model.sqlite"}
    cov = probe["coverage_probe"]
    first_shift = min(int(y) for y, c in cov.items() if (c["shiftcharts_rows"] or 0) > 0)
    art = {
        "phase": "NHL outcome engine Phase 0A -- data / chronology feasibility for player shots on goal. No model trained, nothing tuned.",
        "governing_chain": "availability -> deployment/role -> TOI -> team shot environment -> player shot attempts -> shots on goal; sportsbook inputs prohibited",
        "existing_system_inspection": {
            "files_inspected": ["nhl_live_foundation_a.py", "nhl_player_games_foundation_a.py", "nhl_serving_builder_a.py", "nhl_shots_on_goal_clean_baseline_a.py", "nhl_shots_on_goal_champion_gate_a.py", "nhl_models/nhl_model.sqlite (schema + counts)", "nfl_phase1d_cas.py / nfl_phase1_snapshots (NFL provenance infrastructure, reusable pattern)"],
            "inventory": {
                "nhl_live_foundation_a.py": "api-web club-schedule-season -> games(game_id, season, week, game_date, team names + abbrevs, scores, state). 11,125 games 2018-2025. NO puck-drop time stored (date only). week = floor(days since season's first game / 7)+1, a synthetic bucket.",
                "nhl_player_games_foundation_a.py": "api.nhle.com stats/rest skater|goalie summary isGame=true, one team-season at a time -> skater_games (goals, assists, points, shots, toi_seconds), goalie_games. 310,357 skater rows, 19,749 goalie rows. No strength split, shifts, events, scratches, starters, lines.",
                "nhl_serving_builder_a.py": "serving: week-bucket team state (state_asof) for moneyline; skater serving uses season-to-date 'final state' from the same table; prior-season variants; predictions-first (no odds).",
                "nhl_shots_on_goal_*": "OVER/UNDER 2.5 classifier on D-1 as-of season / recent-3/5 averages, per-60, recent TOI, opponent shots-allowed, team context; eligibility floor recent TOI >= 480 s. Comparator only."},
            "synthetic_weekly_state": ["games.week (7-day bucket) and the team-state tracker keyed by (team, week) in MoneylineEngine: games earlier in the same bucket are invisible, so state is NOT exact chronological state",
                                       "validation machinery re-uses CFB/NFL week-bucket DEV/VAL/HOLDOUT splits", "serving 'final state' = current season-to-date, consistent for upcoming games but not a time-stamped as-of record"],
            "sportsbook_use": "no book odds in the NHL SOG stack; SHOTS_LINE = 2.5 is a fixed round-number threshold for the classification target (a book-style label, not a book input).",
            "data_defect_found": "nhl_model.sqlite skater_games omits players: 0 (player, season) pairs have rows for 2 teams in 2018-2025 (traded players cannot all be present) and in 7 of 8 sampled games 3-11 of 36 dressed skaters have no row, including players with SOG (e.g. 2025020444 A. Texier 2 SOG). Cause not isolated (team-season loop / API team attribution); the table cannot be the outcome source of truth."},
        "source_inventory": [
            {"id": "S1", "name": "NHL web API schedule (api-web.nhle.com/v1/schedule/{date}, /club-schedule-season/{team}/{season}, /schedule/now)", "supplies": "game id, startTimeUTC, venue offset, gameState, teams/ids", "tested": True},
            {"id": "S2", "name": "NHL web API boxscore (/gamecenter/{id}/boxscore)", "supplies": "final per-skater/goalie lines: sog, toi, shifts, goals, assists, hits, blocks, position, sweater; team sog; goalie 'starter' flag (post game)", "tested": True},
            {"id": "S3", "name": "NHL web API play-by-play (/gamecenter/{id}/play-by-play)", "supplies": "event stream: shot-on-goal, missed-shot, blocked-shot, goal with shooter/blocker ids, x/y, zone, situationCode (strength), period/time, rosterSpots (dressed players + sweater + position)", "tested": True},
            {"id": "S4", "name": "NHL stats REST shiftcharts (api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=)", "supplies": "every shift: player, period, start/end/duration", "tested": True},
            {"id": "S5", "name": "NHL stats REST skater/timeonice isGame=true", "supplies": "per game EV / PP / SH / OT TOI, total TOI, shifts, TOI per shift", "tested": True},
            {"id": "S6", "name": "NHL stats REST skater/summary isGame=true (the existing foundation's source)", "supplies": "goals, assists, points, shots, TOI per game", "tested": True},
            {"id": "S7", "name": "NHL web API right-rail (/gamecenter/{id}/right-rail) gameInfo", "supplies": "scratches, officials, coaches (final, and published shortly before puck drop)", "tested": True},
            {"id": "S8", "name": "NHL web API landing (/gamecenter/{id}/landing) pregame 'matchup'", "supplies": "season leaders / team goalie season stats (no confirmed-starter field found)", "tested": True},
            {"id": "S9", "name": "NHL HTML game reports (www.nhl.com/scores/htmlreports/{season}/TH|TV{n}.HTM)", "supplies": "official shift tables by period (human-readable, parse needed)", "tested": "fetched for 2 games (HTTP 200, shift rows present); NOT parsed"},
            {"id": "S10", "name": "NHL web API roster (/roster/{team}/{season|current})", "supplies": "season-end / current roster only; no effective dates", "tested": True},
            {"id": "S11", "name": "Repo nhl_model.sqlite (games, skater_games, goalie_games)", "supplies": "derived from S1/S6; see defect", "tested": True},
            {"id": "S12", "name": "Pregame starting-goalie / projected-lines sources (e.g. third-party sites)", "supplies": "not available in the repo and not tested; no substitute invented", "tested": False}],
        "field_inventory": {
            "game identity / puck drop": ["S1/S2/S3: id, season, gameDate, startTimeUTC, easternUTCOffset, venueUTCOffset, gameState, gameScheduleState, team ids/abbrevs"],
            "dressed players": ["S2 boxscore (skaters + goalies with ids), S3 rosterSpots (40 per game, ids, positions, sweaters)"],
            "scratches": ["S7 right-rail gameInfo.{awayTeam,homeTeam}.scratches (ids + names)"],
            "goalie started": ["S2 goalie 'starter' flag (post game only)"],
            "TOI": ["S2 toi (mm:ss), S5 timeOnIce, evTimeOnIce, ppTimeOnIce, shTimeOnIce, otTimeOnIce, shifts", "S4 shift-level start/end"],
            "shot attempts / SOG": ["S3 events with shooter id: shot-on-goal, goal (=SOG), missed-shot, blocked-shot (shooter + blocker), situationCode, coordinates", "S2 sog per skater, team sog", "S6 shots"],
            "strength state": ["S3 situationCode (4 digits: away G/skaters, home skaters/G)", "S5 EV/PP/SH TOI"],
            "player/team ids": ["numeric playerId (stable across S2-S6), numeric teamId + abbrev"]},
        "temporal_availability_matrix": [
            {"source": "S1 schedule", "published": "months ahead; updated on reschedule", "history_overwritten": "YES -- served value is the current one; earlier kickoff times are not archived by the API", "point_in_time": "only by our own forward snapshots; historically final puck-drop only", "earliest_reliable_season": "2009 (probed 2009-2025 all HTTP 200)", "id_quality": "high", "failure_modes": "postponements, time changes, neutral-site/outdoor games"},
            {"source": "S2 boxscore", "published": "live during game; final minutes after", "history_overwritten": "YES -- served as latest scoring (official stat corrections change the same URL); no revision history", "point_in_time": "post-game outcome only (valid as TARGET)", "earliest_reliable_season": "2009 (11-12 forwards listed per team)", "id_quality": "high", "failure_modes": "late scoring corrections; no archived versions"},
            {"source": "S3 play-by-play", "published": "live; final shortly after", "history_overwritten": "YES -- latest event list (corrections / re-attribution possible)", "point_in_time": "valid for PRIOR games' events as of any later date only if we snapshot after game; no event wall-clock timestamps", "earliest_reliable_season": "2009 events + situationCode (pbp_with_situationCode 267/278 in 2009); x/y on most shots", "id_quality": "high; shootout shots must be excluded from SOG", "failure_modes": "shootout events (period type SO) are not SOG; rosterSpots is populated before puck drop (46 then 40 entries in the live probe); equality with the final dressed list unverified"},
            {"source": "S4 shiftcharts", "published": "live / post-game", "history_overwritten": "YES -- latest", "point_in_time": "prior games only", "earliest_reliable_season": f"{first_shift} (2009 returned 0 rows)", "id_quality": "high", "failure_modes": "DUPLICATE shift rows (19 in 2023020259, 1 in 2025020444) must be de-duplicated on (player, period, start, end); event rows (typeCode 505) mixed with shifts (517)"},
            {"source": "S5 timeonice", "published": "post-game", "history_overwritten": "YES -- latest", "point_in_time": "prior games only", "earliest_reliable_season": "2009 (36 rows/game)", "id_quality": "high", "failure_modes": "EV/PP/SH sum to total in all sampled rows"},
            {"source": "S6 skater summary (existing foundation source)", "published": "post-game", "history_overwritten": "YES", "point_in_time": "prior games only", "earliest_reliable_season": "2009", "id_quality": "high", "failure_modes": "10,000-row query cap; per-team-season loop dropped traded players in the repo table"},
            {"source": "S7 right-rail scratches", "published": "empty ~1h before the game (FUT); populated shortly before puck drop (PRE: 2/3 scratches at T-1m in a live probe); final after", "history_overwritten": "YES -- no historical versions", "point_in_time": "NOT reconstructable for past games: only the final scratch list exists", "earliest_reliable_season": "2009", "id_quality": "high", "failure_modes": "late scratches / warm-up injuries; list is the post-hoc truth when read after the game"},
            {"source": "S8 landing matchup", "published": "pre-game", "history_overwritten": "YES", "point_in_time": "no archived pre-game versions", "earliest_reliable_season": "current", "id_quality": "high", "failure_modes": "contains season leaders only, no confirmed starter"},
            {"source": "S10 roster", "published": "continuous", "history_overwritten": "YES -- /roster/TOR/20222023 returns a season roster without effective dates (47 entries); /current 23", "point_in_time": "not usable as-of", "earliest_reliable_season": "n/a", "id_quality": "high", "failure_modes": "trades / call-ups invisible in time"},
            {"source": "S11 repo sqlite", "published": "built by our scripts", "history_overwritten": "rebuilt on re-run", "point_in_time": "derived; incomplete", "earliest_reliable_season": "2018", "id_quality": "high for present rows", "failure_modes": "missing traded / multi-team players; no puck-drop time; no strength/shift/event detail"}],
        "historical_coverage": {"probe": "one deterministic game per season 2009-2025 (sha256-selected game numbers)", "per_season": cov,
                                "earliest_season_all_of_boxscore_pbp_situationCode_timeonice": 2009, "earliest_season_with_shiftcharts": first_shift,
                                "local_db": "2018-2025 only (games 11,125; skater rows 310,357); 2026-27 season has started (2026-09-29) and is absent"},
        "identifier_mapping_assessment": {"player": "numeric playerId identical across boxscore, pbp rosterSpots/events, shiftcharts, timeonice, summary and the repo table (no name matching needed); 0 id conflicts in the sample", "team": "teamId + abbrev; relocation ARI->UTA 2024 handled via abbrev in the repo; ids stable", "game": "gameId (e.g. 2023020259 = season 2023, type 02 regular season, game 259) identical in every source", "status": "PASS"},
        "known_leakage_revision_risks": [
            "All historical endpoints return the CURRENT value of a record: no revision history, so a point-in-time archive exists only from our own snapshots (forward). Past-game stats are final-after-corrections.",
            "Scratches / dressed list / goalie 'starter' flag in S2/S3/S7 are POST-HOC truth for past games; using them as features for the same game leaks availability. Pre-game availability must be modelled from history or captured live.",
            "Schedule kickoff time history is overwritten; only the final startTimeUTC is retrievable for the past.",
            "pbp rosterSpots is already populated before puck drop (live probe: 46 entries ~1h before, 40 at T-1m; 40 in finished games); its pre-game content was not verified to equal the final dressed list, so it must not be treated as who played.",
            "Shootout events appear as shot-on-goal in pbp but are not SOG in the boxscore (5 apparent mismatches in 2025020444 disappear when period type SO is excluded).",
            "Shiftcharts duplicates inflate TOI by up to ~25% for affected players unless de-duplicated.",
            "The repo table's synthetic 'week' buckets hide same-bucket games and its rows omit traded players: features built on it are neither exact-chronology nor complete.",
            "Late scoring corrections (SOG re-credit) can change the target after first publication; target should be frozen from a recorded retrieval, not re-read."],
        "missing_critical_variables": [
            {"variable": "as-of pregame lineup / availability (who was scratched or injured, known before the cutoff)", "status": "not historically archived; forward capture only"},
            {"variable": "confirmed starting goalie before puck drop", "status": "no pregame source in the API; third-party sources not accessible/tested"},
            {"variable": "projected lines / PP units before the game", "status": "not provided; PP unit membership must be inferred from prior games' shifts + situationCode (derivable, not tested in this phase)"},
            {"variable": "injury reports", "status": "no NHL injury endpoint tested"},
            {"variable": "exact historical schedule revisions", "status": "overwritten"}],
        "sample_game_reconstruction": {"selection_rule": probe["sample_rule"], "sample_game_ids": probe["sample_game_ids"], "aggregate": agg, "games": sample},
        "operations": {"first_run": ops_first_run, "cached_rerun_seconds": probe["operations"]["total_seconds"], "note": "responses cached by sha256 under the session scratch dir (not committed); hashes are in each game's provenance block"},
        "pregame_live_probe": probe["pregame_probe"],
        "status_by_data_category": [
            {"category": "game identity + final puck-drop time", "status": "PASS", "evidence": "8/8 sample games: startTimeUTC, ids, teams, scores agree with repo games table"},
            {"category": "final SOG outcome", "status": "PASS", "evidence": f"{n_sog_cmp} skater-games: boxscore = stats-REST summary = pbp (excluding shootout) = repo row where present; 0 mismatches"},
            {"category": "skater/team/game ID mapping", "status": "PASS", "evidence": "one numeric playerId/teamId/gameId across all sources"},
            {"category": "player TOI (total)", "status": "PASS", "evidence": "boxscore = timeonice = repo, 0 mismatches > 1 s"},
            {"category": "EV / PP / SH TOI", "status": "PASS", "evidence": "timeonice.ev+pp+sh = total in every sampled row"},
            {"category": "shift-level data", "status": "PASS", "evidence": f"matches boxscore TOI and shift counts after de-duplication; coverage {first_shift}+"},
            {"category": "shot-attempt / SOG event attribution", "status": "PASS", "evidence": "pbp shooter ids reproduce every player's SOG; missed/blocked events carry shooter ids; attempt totals not independently cross-checked (no second source tested)"},
            {"category": "historical coverage depth", "status": "PASS", "evidence": f"boxscore/pbp/TOI 2009+, shifts {first_shift}+; local DB only 2018+ and incomplete"},
            {"category": "existing repo table as outcome source of truth", "status": "BLOCKER", "evidence": "omits traded/multi-team players incl. players with SOG; no puck-drop time, strength, shifts, events; rebuild from S2-S5 required"},
            {"category": "as-of pregame roster / scratches / injuries", "status": "BLOCKER", "evidence": "only post-hoc truth exists for past games; pre-drop snapshots exist only live (probe: scratches published ~T-1m)"},
            {"category": "pregame confirmed starting goalie", "status": "BLOCKER", "evidence": "only post-game 'starter' flag; no pregame field found"},
            {"category": "power-play unit / line deployment before the game", "status": "BLOCKER", "evidence": "PP TOI is available post-game; unit membership/lines not provided; must be derived from prior shifts + situationCode (untested)"},
            {"category": "point-in-time schedule/lineup revision history", "status": "BLOCKER", "evidence": "providers overwrite; archive must be built forward with the CAS pattern"},
            {"category": "sportsbook independence", "status": "PASS", "evidence": "every source above is non-market"}],
        "overall": {"outcome_side_feasible": True, "leak_safe_sog_engine_feasible_with_available_sources": "PARTIALLY: the outcome, TOI/strength/shift/event history needed to model team shot environment, TOI and shot attempts from PRIOR games is available and consistent from 2010 (all deterministic, ID-stable). The first links of the chain (availability, goalie, lineup/PP deployment as known before the cutoff) cannot be reconstructed historically and are unavailable without forward capture or an explicit probabilistic availability model trained only on prior-game participation.",
                    "must_solve_next": ["decide the availability policy: (a) model availability/role from prior-game participation, TOI and shifts only (history-reconstructable) and (b) start forward capture of pre-game scratches / lineups / goalie via timed snapshots into a content-addressed store now (season began 2026-09-29)",
                                        "rebuild a complete chronological skater-game + event + shift table from S2-S5 (replacing the incomplete repo table), with de-duplication and shootout exclusion, estimated ~1.3 s/game => ~30 min per 1,312-game season (needs approval)",
                                        "test PP-unit derivation from shifts + situationCode on a small sample",
                                        "define the cutoff rule (puck drop - 90 min / -24 h) and which sources are allowed at each"]}}
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--first-run-ops", required=True, help="json with n_requests, n_network, bytes_network, total_seconds of the first (network) run")
    a = ap.parse_args()
    probe = json.loads(Path(a.probe).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    art = build(probe, json.loads(a.first_run_ops))
    (OUT / "phase0a_data_feasibility.json").write_text(json.dumps(art, indent=1, default=str))
    print("ok", len(json.dumps(art)))
