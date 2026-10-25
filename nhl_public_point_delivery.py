"""Source-safe NHL public slate date and event dedupe.

Pure standard library: original archived model predictions and immutable
shot forecast/grade ledgers MUST NOT be rewritten to solve a UI collision.
"""
from datetime import datetime,timezone
from zoneinfo import ZoneInfo

def eastern_game_calendar_date(at_utc=None):
    """Real NHL local calendar, independently testable around UTC rollover."""
    moment = at_utc or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise ValueError("NHL_GAME_DATE_REQUIRES_AWARE_TIMESTAMP")
    return moment.astimezone(ZoneInfo("America/New_York")).date().isoformat()


def infer_target_date(con, today_str):
    """Earliest US/Eastern NHL date containing an UNFINISHED scheduled game.

    Never switch to tomorrow solely because UTC passed midnight while games
    are still underway in North America. Never publish finished games as
    fresh predictions. An unverified source result is not 'all games final'.
    """
    r = con.execute(
        """SELECT MIN(game_date) FROM games
           WHERE game_date >= ?
             AND (home_score IS NULL OR away_score IS NULL)""",
        (today_str,),
    ).fetchone()
    return r[0] if r and r[0] else None


def select_one_public_points_outcome(picks):
    """One public 0.5-point outcome per authentic player/game.

    Before viewing outcomes we select the CURRENT season model when both
    valid generations exist; otherwise fallback to the prior-season model.
    Do not choose the larger probability (optimism bias), average
    non-equivalent calibrators, or alter either original research/log record.
    Different games/dates/players remain distinct.
    """
    from collections import Counter
    kept, seen = [], {}
    priority = {"points": 0, "points_early_season": 1}
    hidden = Counter()
    for i, original in enumerate(picks):
        p = dict(original)
        market = p.get("market")
        if market not in priority:
            kept.append(p)
            continue
        player_id = p.get("player_id")
        team = str(p.get("team") or "").strip().upper()
        opponent = str(p.get("opponent") or "").strip().upper()
        if not player_id or not team or not opponent or team == opponent:
            hidden["missing_exact_player_or_team_identity"] += 1
            continue
        key = (p.get("season"), p.get("game_date"),
               frozenset((team, opponent)), str(player_id),
               p.get("kickoff_utc"))
        if key not in seen or priority[market] < priority[seen[key][1]["source_market"]]:
            if key in seen:
                hidden["superseded_by_current_season"] += 1
            seen[key] = (i, {**p, "source_market": market,
                             "market": "points"})
        else:
            hidden["superseded_by_current_season"] += 1
    # Keep original feed game/clock sort stable and give no winner-seeking
    # bonus to model generations; source logs remain fully immutable.
    kept.extend(v for _, v in sorted(seen.values(), key=lambda x: x[0]))
    audits = {
        "schema": "NHL_ANYTIME_POINT_UNIQUE_PLAYER_GAME_PUBLICATION_V1",
        "selection_policy": "PREFER_CURRENT_SEASON_MODEL_IF_AVAILABLE_ELSE_PRIOR",
        "raw_anytime_point_rows": sum(p.get("market") in priority for p in picks),
        "public_anytime_point_rows": len(seen),
        "superseded_model_generation_rows": hidden["superseded_by_current_season"],
        "quarantined_unidentified_rows": hidden["missing_exact_player_or_team_identity"],
        "historical_original_model_rows_mutated": False,
        "no_probability_maximization_or_calibrator_blending": True,
    }
    return kept, audits


