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


def classify_player(player_id, team, prior_player_rows):
    """Explicit cold-start / unknown policy. prior_player_rows: the player's rows allowed at T (most recent last), each with `team`.
    KNOWN    appeared for `team` in the lookback -> player-level lagged features allowed
    ACQUIRED has prior games, none for `team` in the lookback -> own lagged features allowed, team-specific role unknown (flagged)
    COLD     no prior NHL game at all in the loaded history -> no player-level features; league position prior only (flagged); never backfilled from the target game"""
    if not prior_player_rows:
        return "COLD"
    recent = prior_player_rows[-LOOKBACK_TEAM_GAMES:]
    return "KNOWN" if any(r["team"] == team for r in recent) else "ACQUIRED"


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
    "cold_start_policy": {"states": {"KNOWN": f"appeared for the team in the last {LOOKBACK_TEAM_GAMES} completed team games before T", "ACQUIRED": "has prior games, none for this team in the lookback (traded / signed): own lagged features allowed, team role unknown -> flagged",
                                     "COLD": "no prior NHL game in the loaded history: no player-level features; league position prior only; flagged"},
                          "universe_rule": "the candidate universe at T is the set of KNOWN and ACQUIRED players recoverable from prior games (plus, FORWARD ONLY, the pregame roster snapshot). A player who plays the target game but is not in the universe is 'unobservable_at_T': he is counted and reported (coverage), excluded from the prediction set, and NEVER inserted from target-game truth.",
                          "no_backfill": "existence of a player may not be inferred from the target game's boxscore, pbp, shifts or scratches list"},
    "target_definition": "SOG from the boxscore of the target game after the game, frozen from a recorded retrieval (late corrections recorded, not re-read); shootout shots excluded",
    "known_historical_limitation": "the historical schedule start time is the final one (earlier revisions overwritten) and past availability is unarchived: history can evaluate the prior-game-only model; the forward snapshots (phase0b_forward_snapshot_contract.json) are the only as-of availability evidence",
}
