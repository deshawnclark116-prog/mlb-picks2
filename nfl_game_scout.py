#!/usr/bin/env python3
"""Pregame, player-specific NFL scouting evidence. Research only; no pick edits.

Source-backed mechanism checks, not an invented man-coverage or blocking matchup.
Every source must predate the forecast week. Unknown assignments stay unknown.
This file intentionally refuses to turn descriptive matchup statistics into an
unvalidated yardage multiplier or a high-confidence probability.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "NFL_PLAYER_GAME_SCOUT_EVIDENCE_V1"
MARKETS = {"rushing_yards", "receiving_yards"}
OL = {"T", "G", "C", "OL", "OT", "OG"}
DL = {"DE", "DT", "NT", "DL", "EDGE"}
DB = {"CB", "S", "SS", "FS", "DB"}
LB = {"LB", "ILB", "OLB", "MLB"}
STATUS = {"out", "ir", "pup", "sus", "questionable", "doubtful", "dnr", "na"}


def instant(v):
    out = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    if out.tzinfo is None:
        raise ValueError("TIMESTAMP_MUST_HAVE_OFFSET")
    return out.astimezone(timezone.utc)


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def namekey(s):
    return "".join(c for c in str(s or "").lower() if c.isalnum())


def prior(record, season, week):
    try:
        return (int(record["season"]), int(record["week"])) < (season, week)
    except (KeyError, TypeError, ValueError):
        return False


def depth(row):
    a = number(row.get("air_yards"))
    if a is None:
        return "unknown"
    return "deep_20plus" if a >= 20 else "intermediate_10_19" if a >= 10 else "short_under_10"


def run_gap(row):
    gap = str(row.get("run_gap") or "").lower()
    return "inside_guard_center" if gap in {"guard", "center"} else "tackle_or_edge" if gap in {"tackle", "end"} else "unknown"


def read_csv(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def profile(entries, category):
    """Only previously completed games, with sample counts and no invented zeros."""
    group = defaultdict(lambda: [0, 0.0])
    total = 0
    for row in entries:
        if category == "receiving_yards":
            if row.get("play_type") != "pass" or not row.get("receiver_player_id"):
                continue
            bucket = depth(row)
        else:
            if row.get("play_type") != "run" or not row.get("rusher_player_id"):
                continue
            bucket = run_gap(row)
        y = number(row.get("yards_gained"))
        if y is None:
            continue
        group[bucket][0] += 1
        group[bucket][1] += y
        total += 1
    return {
        "observed_opportunities": total,
        "by_play_type": {
            k: {"n": v[0], "yards_per_observed_play": round(v[1] / v[0], 3),
                "observed_share": round(v[0] / total, 4) if total else None}
            for k, v in sorted(group.items())
        },
        "no_unobserved_route_or_block_assignment_inferred": True,
    }


def recent_plays(rows, key, id_field, season, week, max_games=8, current_team=None):
    """Select latest distinct games, not latest N plays; current and future excluded."""
    valid = [r for r in rows if prior(r, season, week) and r.get(id_field) == key
             and (current_team is None or r.get("posteam") == current_team)]
    games = sorted({(int(r["season"]), int(r["week"]), r.get("game_id")) for r in valid})
    selected = set(games[-max_games:])
    return [r for r in valid if (int(r["season"]), int(r["week"]), r.get("game_id")) in selected]


def matchup_observations(rows, defense, market, season, week, max_games=6):
    valid = [r for r in rows if prior(r, season, week) and r.get("defteam") == defense]
    games = sorted({(int(r["season"]), int(r["week"]), r.get("game_id")) for r in valid})
    selected = set(games[-max_games:])
    sample = [r for r in valid if (int(r["season"]), int(r["week"]), r.get("game_id")) in selected]
    return {"games": len(selected), **profile(sample, market)}



def defender_coverage_history(pfr_rows, season, week, max_games=12):
    """PFR's defender-attributed targets allowed; NOT route-by-route matchups."""
    grouped = defaultdict(list)
    for r in pfr_rows:
        if not prior(r, season, week) or r.get("game_type") != "REG":
            continue
        pid = r.get("pfr_player_id")
        if pid and number(r.get("def_targets")) is not None:
            grouped[pid].append(r)
    result = {}
    for pid, rows in grouped.items():
        recent = sorted(rows, key=lambda r: (int(r["season"]), int(r["week"])))[-max_games:]
        nt = sum(number(r.get("def_targets")) or 0 for r in recent)
        if nt < 5:
            continue
        ya = sum(number(r.get("def_yards_allowed")) or 0 for r in recent)
        co = sum(number(r.get("def_completions_allowed")) or 0 for r in recent)
        result[pid] = {
            "charting_games": len(recent), "defender_attributed_targets": round(nt),
            "charted_yards_per_target_allowed": round(ya / nt, 3),
            "charted_completion_rate_allowed": round(co / nt, 3),
            "not_individual_wr_assignment": True,
            "historical_pfr_charting_not_film_verified": True,
        }
    return result


def past_personnel(snaps, team, role, season, week, injuries, coverage=None, max_games=3):
    key = "defense_pct" if role == "defense" else "offense_pct"
    positions = (DL | LB | DB) if role == "defense" else OL
    arr = [r for r in snaps if prior(r, season, week) and r.get("team") == team
           and r.get("position") in positions and number(r.get(key)) is not None]
    games = sorted({(int(r["season"]), int(r["week"])) for r in arr})
    selected = set(games[-max_games:])
    grouped = defaultdict(list)
    for r in arr:
        if (int(r["season"]), int(r["week"])) not in selected:
            continue
        val = number(r[key])
        if val is not None and val > 0:
            grouped[(namekey(r.get("player")), r.get("position"))].append((str(r.get("player")), val, r.get("pfr_player_id")))
    out = []
    for (n, pos), observations in grouped.items():
        avg = sum(z[1] for z in observations) / len(observations)
        if avg < 0.3:
            continue
        out.append({"player": observations[-1][0], "position": pos,
                    "previous_games_observed": len(observations),
                    "mean_past_snap_share": round(avg, 3),
                    "injury_source_status": injuries.get((team, n)),
                    "game_day_active_verified": False,
                    "assignment_verified": False,
                    "prior_defender_charted_coverage": (coverage or {}).get(observations[-1][2])
                        if role == "defense" else None})
    return sorted(out, key=lambda d: -d["mean_past_snap_share"])[:12]


def direction(role):
    observations = [x for x in role.get("recent_team_game_observations", [])
                    if x.get("stat_row_observed") and number(x.get("share_when_observed")) is not None]
    if len(observations) < 2:
        return {"status": "INSUFFICIENT_HISTORY", "change": None}
    first, last = observations[0], observations[-1]
    change = float(last["share_when_observed"]) - float(first["share_when_observed"])
    return {"status": "OBSERVED_NOT_FORECAST", "earliest_week": first.get("week"),
            "latest_week": last.get("week"), "first_share": round(float(first["share_when_observed"]), 3),
            "last_share": round(float(last["share_when_observed"]), 3),
            "change_percentage_points": round(100 * change, 1)}


def evidence_match(player, defense, market):
    """Exact evidence, not a claim about who guards or blocks whom."""
    p, d = player["by_play_type"], defense["by_play_type"]
    labels = ("short_under_10", "intermediate_10_19", "deep_20plus") if market == "receiving_yards" else ("inside_guard_center", "tackle_or_edge")
    aligned = []
    for b in labels:
        pp, dd = p.get(b), d.get(b)
        if not pp or not dd or pp["n"] < 3 or dd["n"] < 8:
            continue
        aligned.append({"mechanism": b, "player_prior_opportunity_share": pp["observed_share"],
                        "player_prior_yards_per_play": pp["yards_per_observed_play"],
                        "defense_prior_yards_allowed_per_play": dd["yards_per_observed_play"],
                        "player_sample_n": pp["n"], "defense_sample_n": dd["n"],
                        "interpretation": "DESCRIPTIVE_MATCHUP_NOT_VALIDATED_CAUSAL_EFFECT"})
    return aligned


def build(board, role_doc, plays, snaps, injury_doc, *, now=None, pfr_def=None):
    now = instant(now or datetime.now(timezone.utc))
    if not isinstance(board.get("scheduled_games"), list) or not isinstance(board.get("picks"), list):
        raise ValueError("INVALID_BOARD")
    if role_doc.get("schema") != "nfl-role-evidence-coverage-v1":
        raise ValueError("INVALID_ROLE_SOURCE")
    if (board.get("season"), board.get("week")) != (role_doc.get("season"), role_doc.get("week")):
        raise ValueError("ROLE_WEEK_MISMATCH")
    generation = instant(board["generated_at_utc"])
    if instant(role_doc["forecast_generated_at_utc"]) != generation:
        raise ValueError("ROLE_FORECAST_MISMATCH")
    if generation > now or instant(role_doc["captured_at_utc"]) > now:
        raise ValueError("FUTURE_SOURCE_TIMESTAMP")
    injury_time = instant(injury_doc["captured_at_utc"])
    if injury_time > now:
        raise ValueError("FUTURE_INJURY_TIMESTAMP")
    season, week = int(board["season"]), int(board["week"])
    # Historical source rows must never silently incorporate the game being predicted.
    plays = [r for r in plays if prior(r, season, week) and r.get("season_type") == "REG"]
    snaps = [r for r in snaps if prior(r, season, week) and r.get("game_type") == "REG"]
    coverage = defender_coverage_history(pfr_def or [], season, week)
    injury = {}
    for r in injury_doc.get("players", []):
        status = str(r.get("injury_status") or "").lower()
        team = r.get("team")
        if team and status in STATUS:
            injury[(team, namekey(r.get("player")))] = status
    games = {}
    for g in board["scheduled_games"]:
        home, away = g["home_team"], g["away_team"]
        kickoff = instant(g["kickoff_utc"])
        if home == away or kickoff <= generation:
            raise ValueError("INVALID_UPCOMING_GAME")
        games[frozenset((home, away))] = (g, kickoff)
    roles = {}
    by_team_market = defaultdict(list)
    for r in role_doc["player_market_evidence"]:
        if r.get("market") not in MARKETS:
            continue
        k = (r.get("player_id"), r.get("team"), r.get("market"))
        if k in roles:
            raise ValueError("DUPLICATE_ROLE_EVIDENCE")
        roles[k] = r
        by_team_market[(r["team"], r["market"])].append(r)
    unique = set()
    cards = []
    for pick in board["picks"]:
        market = pick.get("market")
        if market not in MARKETS:
            continue
        key = (pick.get("player_id"), pick.get("team"), market)
        if key in unique:
            raise ValueError("DUPLICATE_PUBLISHED_PLAYER_MARKET")
        unique.add(key)
        gid = frozenset((pick.get("team"), pick.get("opponent")))
        if gid not in games:
            raise ValueError("UNKNOWN_GAME_FOR_PLAYER")
        game, kickoff = games[gid]
        if instant(pick["kickoff_utc"]) != kickoff:
            raise ValueError("PICK_KICKOFF_MISMATCH")
        if kickoff <= now:
            continue
        role = roles.get(key)
        if role is not None and instant(role["kickoff_utc"]) != kickoff:
            raise ValueError("ROLE_KICKOFF_MISMATCH")
        own = recent_plays(plays, key[0], "receiver_player_id" if market == "receiving_yards" else "rusher_player_id",
                               season, week, current_team=pick["team"])
        mine = profile(own, market)
        defense = matchup_observations(plays, pick["opponent"], market, season, week)
        competing = []
        for other in by_team_market[(pick["team"], market)]:
            if other.get("player_id") == key[0]:
                continue
            competing.append({"player": other.get("player"), "player_id": other.get("player_id"),
                "observed_recent_opportunities_per_game": other.get("recent_opportunities_per_observed_game"),
                "observed_recent_team_share": other.get("role_share_volume_weighted"),
                "recent_share_direction": direction(other),
                "confirmed_active": False})
        competing.sort(key=lambda x: -(number(x["observed_recent_team_share"]) or 0))
        matchup = evidence_match(mine, defense, market)
        risks = []
        if role is None: risks.append("MISSING_CURRENT_ROLE_EVIDENCE")
        elif role.get("warnings"): risks += role["warnings"]
        if role is not None and role.get("observed_stat_rows", 0) < 3:
            risks.append("INCOMPLETE_RECENT_ROLE_HISTORY")
        if defense["games"] < 3: risks.append("DEFENSE_HISTORICAL_COVERAGE_LOW")
        if mine["observed_opportunities"] < 10: risks.append("PLAYER_PLAY_SAMPLE_LOW")
        if not matchup: risks.append("NO_SAMPLE_SUFFICIENT_PLAY_TYPE_MATCHUP")
        if pick.get("confidence") == "HIGH" and pick.get("projection_probability_source", "").startswith("UNAVAILABLE"):
            risks.append("PUBLISHED_HIGH_CONFIDENCE_WITHOUT_PLAYER_DISTRIBUTION")
        cards.append({
            "player": pick["player"], "player_id": key[0], "team": pick["team"],
            "opponent": pick["opponent"], "market": market,
            "kickoff_utc": kickoff.isoformat(), "published_projection_yards": pick.get("projected_median"),
            "published_model_version": pick.get("yardage_model_version", pick.get("model_source")),
            "published_confidence_label": pick.get("confidence"),
            "published_probability_source": pick.get("projection_probability_source"),
            "model_reach_probability_reported": pick.get("projection_reach_probability_measured"),
            "confidence_is_not_a_verified_individual_probability": True,
            "observed_role": None if role is None else {
                "recent_opportunities_per_observed_game": role.get("recent_opportunities_per_observed_game"),
                "recent_team_share": role.get("role_share_volume_weighted"),
                "recent_share_range": role.get("role_share_game_to_game_range"),
                "recent_share_direction": direction(role),
                "observed_prior_team_games": role.get("observed_stat_rows"),
                "roster_status": role.get("roster_status"),
                "active_starter_verified": role.get("has_verified_starting_role") is True,
            },
            "teammate_opportunity_competition": competing[:5],
            "player_prior_play_profile": mine,
            "opponent_prior_defensive_play_profile": defense,
            "matched_play_mechanisms": matchup,
            "likely_defensive_personnel_from_previous_snaps": past_personnel(
                snaps, pick["opponent"], "defense", season, week, injury, coverage=coverage),
            "likely_own_offensive_line_from_previous_snaps": past_personnel(
                snaps, pick["team"], "offense", season, week, injury),
            "unverified_player_vs_corner_assignment": True,
            "unverified_ol_vs_dl_blocking_assignment": True,
            "coaching_game_plan_not_verified": True,
            "data_warnings": sorted(set(risks)),
            "forecast_change_yards": 0,
            "forecast_change_reason": "NO_VALIDATED_MATCHUP_EFFECT_OR_NEW_MODEL_FIT",
        })
    cards.sort(key=lambda x: (x["kickoff_utc"], x["team"], x["market"], x["player"]))
    return {"schema": SCHEMA, "season": season, "week": week,
            "forecast_generated_at_utc": generation.isoformat(), "scout_generated_at_utc": now.isoformat(),
            "source_injury_captured_at_utc": injury_time.isoformat(),
            "source_policy": "ONLY_PRIOR_COMPLETED_WEEKS;HISTORICAL_RELEASE_RETRIEVAL_TIMING_NOT_ARCHIVED",
            "player_versus_player_matchups_verified": False,
            "new_forecast_model_implemented": False, "projection_or_pick_modified": False,
            "player_cards": cards,
            "summary": {"scouted_player_markets": len(cards),
                        "player_markets_with_play_type_comparisons": sum(bool(c["matched_play_mechanisms"]) for c in cards),
                        "high_confidence_without_player_distribution": sum(
                            "PUBLISHED_HIGH_CONFIDENCE_WITHOUT_PLAYER_DISTRIBUTION" in c["data_warnings"] for c in cards),
                        "missing_role_evidence": sum(c["observed_role"] is None for c in cards),
                        "named_personnel_evidence_cards": sum(
                            bool(c["likely_defensive_personnel_from_previous_snaps"]) for c in cards),
                        "named_defender_charting_rows": sum(
                            bool(p["prior_defender_charted_coverage"])
                            for c in cards for p in c["likely_defensive_personnel_from_previous_snaps"]),
                        "player_markets_with_named_defender_charting": sum(
                            any(bool(p["prior_defender_charted_coverage"])
                                for p in c["likely_defensive_personnel_from_previous_snaps"]) for c in cards)}}



def render_markdown(report):
    """Readable scout notes with provenance and caveats, not fabricated advantages."""
    lines = [
        "# NFL game-by-game scouting evidence",
        "",
        f"Season {report['season']} week {report['week']} | Frozen board: {report['forecast_generated_at_utc']}",
        "",
        "**Research evidence only.** These notes do not alter the existing yardage "
        "projections, identify verified coverage/blocking assignments, or imply "
        "calibrated player-specific probabilities.",
        "",
    ]
    by_game = defaultdict(list)
    for c in report["player_cards"]:
        gid = (c["kickoff_utc"], tuple(sorted((c["team"], c["opponent"]))))
        by_game[gid].append(c)
    for (kickoff, teams), cards in sorted(by_game.items()):
        lines += [f"## {teams[0]} vs {teams[1]} — {kickoff}", ""]
        for c in cards:
            market = "receiving" if c["market"] == "receiving_yards" else "rushing"
            lines += [f"### {c['player']} ({c['team']}) — {market} yards", ""]
            lines.append(f"Published projection: **{c['published_projection_yards']} yards** "
                         f"({c['published_model_version']}); **not adjusted** by these notes.")
            role = c["observed_role"]
            if role:
                n = role.get("recent_opportunities_per_observed_game")
                share = role.get("recent_team_share")
                desc = f"{n} observed {('targets' if market == 'receiving' else 'carries')} per game"
                if share is not None:
                    desc += f"; {100 * share:.1f}% volume-weighted team share"
                ch = role["recent_share_direction"]
                if ch.get("change_percentage_points") is not None:
                    desc += f"; share changed {ch['change_percentage_points']:+.1f} percentage points across observed games"
                lines.append(f"- **Role evidence:** {desc}. "
                             f"Roster label: {role['roster_status']}; starter confirmed: {role['active_starter_verified']}.")
            else:
                lines.append("- **Role evidence:** unavailable; no starting role has been assumed.")
            friends = c["teammate_opportunity_competition"][:3]
            if friends:
                tx = ", ".join(f"{p['player']} ({p.get('observed_recent_team_share')})" for p in friends)
                lines.append(f"- **Competing teammates** (recent volume-weighted shares): {tx}. "
                             "These shares are observational, not next-game targets.")
            p = c["player_prior_play_profile"]
            d = c["opponent_prior_defensive_play_profile"]
            lines.append(f"- **Play-level sample:** {p['observed_opportunities']} player events from up to eight prior games; "
                         f"{d['observed_opportunities']} opponent defensive events in {d['games']} prior games.")
            comparisons = c["matched_play_mechanisms"]
            for m in comparisons:
                lines.append(f"- **Observed {m['mechanism']}:** player took "
                             f"{100 * m['player_prior_opportunity_share']:.1f}% of their charted events in this category "
                             f"(n={m['player_sample_n']}); previous opponent defense allowed "
                             f"{m['defense_prior_yards_allowed_per_play']} yards per such play "
                             f"(n={m['defense_sample_n']}). *Not a validated matchup adjustment.*")
            if not comparisons:
                lines.append("- **Play-type matchup:** insufficient overlap/sample to compare reliably.")
            want = {"CB", "S", "SS", "FS", "DB"} if market == "receiving" else DL | LB
            defenders = [p for p in c["likely_defensive_personnel_from_previous_snaps"]
                         if p["position"] in want][:4]
            if defenders:
                dx = ", ".join(
                    f"{p['player']} {p['position']} ({100 * p['mean_past_snap_share']:.0f}% prior snap share"
                    + (f"; injury source {p['injury_source_status']}" if p["injury_source_status"] else "")
                    + ")" for p in defenders)
                lines.append(f"- **Opponent personnel seen in prior games:** {dx}. "
                             "Game-day availability and individual assignments unverified.")
                for p in defenders:
                    cov = p.get("prior_defender_charted_coverage")
                    if cov:
                        lines.append(f"  - {p['player']} prior defender charting: "
                                     f"{cov['defender_attributed_targets']} attributed targets over "
                                     f"{cov['charting_games']} games; "
                                     f"{cov['charted_yards_per_target_allowed']} yards/target allowed. "
                                     "Not proven to be matched against this receiver.")
            if market == "rushing":
                ol = c["likely_own_offensive_line_from_previous_snaps"][:5]
                if ol:
                    lines.append("- **Previous OL participants:** "
                                 + ", ".join(p['player'] for p in ol)
                                 + ". Blocks/individual DL matchups unverified.")
            if c["data_warnings"]:
                lines.append("- **Evidence warnings:** " + "; ".join(c["data_warnings"]))
            lines += ["- **Missing game-specific certainty:** exact WR/CB assignments, OL/DL blocks, "
                      "playcalling and active status are not established by this evidence.", ""]
    lines += ["---", "",
              "All statements above depend on observed historical events and current provided receipts. "
              "This is not an independently validated model promotion or a recommended bet.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--injuries", required=True)
    ap.add_argument("--pbp", action="append", default=[], required=True)
    ap.add_argument("--snaps", action="append", default=[], required=True)
    ap.add_argument("--pfr-def", action="append", default=[], help="PFR weekly defender charting source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md-out", help="Optional human-readable game-by-game scout report")
    ap.add_argument("--now", help="ISO 8601 UTC; defaults to runtime UTC")
    a = ap.parse_args()
    board, roles, inj = [json.loads(Path(x).read_text()) for x in (a.board, a.roles, a.injuries)]
    pbp = [r for p in a.pbp for r in read_csv(p)]
    snaps = [r for p in a.snaps for r in read_csv(p)]
    pfr = [r for p in a.pfr_def for r in read_csv(p)]
    result = build(board, roles, pbp, snaps, inj, now=a.now, pfr_def=pfr)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if a.md_out:
        Path(a.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md_out).write_text(render_markdown(result))
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
