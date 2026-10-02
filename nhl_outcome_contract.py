"""
NHL_OUTCOME_CONTRACT -- the historical information contract (Phase 0B) as executable, deterministic helpers. No model, no I/O.

A player-game prediction at cutoff T may use only information from games COMPLETED strictly before T (start + GAME_MAX_MINUTES <= T), the schedule identity of the target game, and
lagged aggregates of those completed games. Nothing from the target game itself (dressed list, scratches, starter flag, shifts, TOI, events, SOG) may enter.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
CONTRACT_VERSION = "nhl-historical-contract-1"
GAME_MAX_MINUTES = 210            # conservative regulation + OT + shootout + stoppages: a game counts as completed only if start + 210 min <= T
HORIZON_MINUTES = {"T24H": 1440, "T90": 90, "T30": 30, "T10": 10}      # candidate horizons; the recommended subset is in phase0b_forward_snapshot_contract.json
LOOKBACK_TEAM_GAMES = 10          # a player is 'known to the team' if he appeared for it in the last 10 completed team games before T


def parse_utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def cutoff_time(start, horizon):
    """T = scheduled puck drop - horizon. `start` must be timezone-aware UTC."""
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")
    return start.astimezone(UTC) - timedelta(minutes=HORIZON_MINUTES[horizon])


def completed_before(game_start, cutoff):
    """True iff the game is certainly over before the cutoff (strict). Deterministic; independent of the actual end time."""
    return game_start.astimezone(UTC) + timedelta(minutes=GAME_MAX_MINUTES) <= cutoff


def prior_rows(rows, target_game_id, cutoff):
    """rows: iterable of dicts with gameId and startTimeUTC (ISO 'Z'). Returns the rows allowed at `cutoff` for the target game: completed before T, never the target game itself."""
    return [r for r in rows if r["gameId"] != target_game_id and completed_before(parse_utc(r["startTimeUTC"]), cutoff)]


def candidate_universe(rows, target_game_id, team, cutoff):
    """Historical candidate universe for `team` in the target game, at `cutoff` T (deterministic under any input-row order).

    1. only rows of games != target game that are certainly over before T (start + GAME_MAX_MINUTES <= T);
    2. the distinct prior games played by `team`, ordered by (startTimeUTC, gameId);
    3. the last LOOKBACK_TEAM_GAMES of them;
    4. candidates = playerIds who APPEARED FOR THAT SAME TEAM in those games. Nothing else adds a candidate: prior games for another team do not establish membership; a player with his
       first appearance for the team in the target game is `unobservable_at_T` (no historical point-in-time roster / transaction source exists in Phase 0B).
    Returns {"candidates": sorted ids, "team_games": ordered game ids used, "n_prior_team_games_available": int}."""
    allowed = prior_rows(rows, target_game_id, cutoff)
    games = {}
    for r in allowed:
        if r["team"] == team:
            games[r["gameId"]] = r["startTimeUTC"]
    ordered = sorted(games, key=lambda g: (games[g], g))
    used = ordered[-LOOKBACK_TEAM_GAMES:]
    keep = set(used)
    cands = sorted({r["playerId"] for r in allowed if r["team"] == team and r["gameId"] in keep}, key=lambda x: (str(type(x)), x))
    return {"candidates": cands, "team_games": used, "n_prior_team_games_available": len(ordered)}


def current_team_appearances(rows, target_game_id, team, player_id, cutoff, n=5):
    """CURRENT-TEAM role history: the player's allowed prior appearances for `team` only (old-team rows are never role history), last n, ordered (startTimeUTC, gameId).
    Old-team rows may be used for player SKILL history elsewhere; they never establish membership or current-team role."""
    allowed = [r for r in prior_rows(rows, target_game_id, cutoff) if r["team"] == team and r["playerId"] == player_id]
    allowed.sort(key=lambda r: (r["startTimeUTC"], r["gameId"]))
    return allowed[-n:]


def diagnose_unobservable(player_id, team, target_game_id, rows, cutoff):
    """POSTGAME / GRADING-ONLY diagnostic for a target participant who was not a candidate. Never feeds candidacy."""
    allowed = [r for r in prior_rows(rows, target_game_id, cutoff) if r["playerId"] == player_id]
    if not allowed:
        return "NO_LOADED_HISTORY"
    if any(r["team"] == team for r in allowed):
        return "STALE_SAME_TEAM_HISTORY_OUTSIDE_MEMBERSHIP_WINDOW"
    return "PRIOR_NHL_HISTORY_ELSEWHERE"


def shot_counts_from_events(events, shooter_of=None):
    """events: play-by-play event dicts. Returns {playerId: Counter(shot-on-goal, goal, missed-shot, blocked-shot)} counting REGULATION and OVERTIME events only:
    shootout (periodType SO) events are never SOG or attempts."""
    out = {}
    for e in events:
        t = e.get("typeDescKey")
        if t not in ("shot-on-goal", "goal", "missed-shot", "blocked-shot"):
            continue
        if (e.get("periodDescriptor") or {}).get("periodType") == "SO":
            continue
        d = e.get("details") or {}
        pid = d.get("shootingPlayerId") or d.get("scoringPlayerId")
        if pid:
            out.setdefault(pid, Counter())[t] += 1
    return out


def sog(c):
    return c["shot-on-goal"] + c["goal"]


def attempts(c):
    return sog(c) + c["missed-shot"] + c["blocked-shot"]


def strength_consistent(row, tol=1):
    return row["ev"] is not None and abs((row["ev"] or 0) + (row["pp"] or 0) + (row["sh"] or 0) - (row["toi"] or 0)) <= tol or abs((row["ev"] or 0) + (row["pp"] or 0) + (row["sh"] or 0) + (row.get("ot") or 0) - (row["toi"] or 0)) <= tol


CONTRACT = {
    "version": CONTRACT_VERSION,
    "prediction_unit": "player-game (skater), outcome = shots on goal (regular season; shootout excluded)",
    "cutoff": {"definition": "T = scheduled puck drop - horizon, using the puck-drop time retrievable at the time of the forecast (forward) or the final startTimeUTC (history, see risks)", "horizons_candidate_minutes": HORIZON_MINUTES,
               "completed_before_rule": f"a source game g' is usable iff start(g') + {GAME_MAX_MINUTES} min <= T and g' != target game (nhl_outcome_contract.completed_before)", "same_day_effect": "a back-to-back / same-day earlier game is excluded unless it is certainly over"},
    "allowed": ["target game identity: gameId, scheduled startTimeUTC, home/away team ids, season", "all completed games strictly before T (rule above)", "lagged participation (who appeared for which team)", "lagged total TOI", "lagged EV / PP / SH TOI and shares", "lagged shift counts and TOI per shift",
                "lagged shot attempts / SOG / missed / blocked (shootout excluded)", "lagged team and opponent shot environment, strength-state mix, penalties", "lagged goalie usage of the opponent (who started recent games)", "schedule-derived context (rest days, back-to-back, travel) from the schedule"],
    "forbidden": ["the target game's final dressed list / rosterSpots as ground truth for who played", "target-game scratches (right-rail gameInfo)", "target-game goalie 'starter' flag", "target-game shifts / TOI / EV-PP-SH split", "any target-game event (shots, penalties, goals)", "any boxscore / stats value of the target game",
                  "any postgame correction or re-credit unavailable before T", "sportsbook lines or odds", "season-aggregate rows that include the target game (e.g. /skater/summary isGame=false read after the game)", "the repo's synthetic `week` bucket as a time index"],
    "candidate_universe_policy": {
        "rule": f"for target game G, team TEAM, cutoff T: allowed rows = games != G with start + {GAME_MAX_MINUTES} min <= T; TEAM's distinct prior games ordered by (startTimeUTC, gameId); the last {LOOKBACK_TEAM_GAMES}; candidates = playerIds who appeared FOR THAT TEAM in them (nhl_outcome_contract.candidate_universe)",
        "nothing_else_adds_a_candidate": ["prior games for another team do not establish current-team membership", "target-game boxscore / rosterSpots / scratches / position / TOI / PP / shifts / SOG / events are grading-only"],
        "unobservable_at_T": "a target participant absent from the candidate set; counted and reported (player and SOG coverage); never inserted from target truth. A point-in-time roster / transaction source would be needed to add him; Phase 0B has established none for history (the forward snapshots may add such a source later).",
        "postgame_diagnostics_grading_only": ["NO_LOADED_HISTORY (true debut or history outside the loaded window)", "PRIOR_NHL_HISTORY_ELSEWHERE (other-team history only)", "STALE_SAME_TEAM_HISTORY_OUTSIDE_MEMBERSHIP_WINDOW"],
        "skill_vs_role": "once membership is established, prior NHL history from ALL teams may feed PLAYER SKILL features; current-team ROLE / DEPLOYMENT features use current-team appearances only (current_team_appearances)"},
    "deployment_features_probe": "PP / EV ALLOCATION shares (player PP or EV TOI / sum of team skater PP or EV TOI in a prior game; not true PP opportunity shares), last <=5 current-team appearances, PP rank among candidates, recent-vs-long change (mean last 2 - mean last 5, needs >=3 appearances)",
    "target_definition": "SOG from the boxscore of the target game after the game, frozen from a recorded retrieval (late corrections recorded, not re-read); shootout shots excluded",
    "known_historical_limitation": "the historical schedule start time is the final one (earlier revisions overwritten) and past availability is unarchived: history can evaluate the prior-game-only model; the forward snapshots (phase0b_forward_snapshot_contract.json) are the only as-of availability evidence",
}
