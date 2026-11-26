#!/usr/bin/env python3
"""College basketball score-distribution model: ML, spreads, and totals.

Research-only. Real boxscores and schedules, game-date-forward snapshots, no
sportsbook data as a model input. Never describe a retrospectively reconstructed
forecast as a first-seen pregame prediction. Each market is the same joint
(home-score, away-score) predictive distribution, not three independent picks.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCHEMA = "CBB_GAME_INTELLIGENCE_V0_RESEARCH"
STAT_KEYS = ("pace", "ortg", "drtg", "efg", "tov", "orb", "ftr",
             "efg_allowed", "tov_forced", "orb_allowed", "ftr_allowed")
LEAGUE_PRIOR = {"pace": 68.5, "ortg": 105., "drtg": 105., "efg": .492,
                "tov": .18, "orb": .29, "ftr": .34,
                "efg_allowed": .492, "tov_forced": .18,
                "orb_allowed": .29, "ftr_allowed": .34}
FEATURE_NAMES = (
    "mechanical_margin", "mechanical_total", "neutral", "shared_pace",
    "home_ortg", "away_ortg", "home_drtg", "away_drtg",
    "efg_match_home", "efg_match_away", "tov_match_home", "tov_match_away",
    "orb_match_home", "orb_match_away", "ftr_match_home", "ftr_match_away",
    "home_current_games", "away_current_games", "rotation_concentration_diff",
    "rotation_change_diff", "home_recent_form", "away_recent_form",
)


def num(value, default=None):
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def boolean(v):
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v) if v is not None and not pd.isna(v) else False


def phi(z):
    return 0.5 * (1 + math.erf(float(z) / math.sqrt(2)))


def score_probabilities(margin, total, margin_sd, total_sd, home_spread=None, total_line=None):
    """Coherent marginal events from a shared normal margin/total score model.

    home_spread is an American-style point handicap: e.g. -5.5 favorite.
    P(home covers)=P((home-away)+home_spread>0). No lines => no cover claims.
    """
    if margin_sd <= 0 or total_sd <= 0:
        raise ValueError("invalid learned standard deviations")
    out = {
        "projected_home_score": round((total + margin) / 2, 1),
        "projected_away_score": round((total - margin) / 2, 1),
        "projected_home_margin": round(margin, 1),
        "projected_total": round(total, 1),
        "p_home_win": round(phi(margin / margin_sd), 4),
        "p_away_win": round(1 - phi(margin / margin_sd), 4),
        "margin_80pct_interval": [round(margin - 1.28155 * margin_sd, 1),
                                  round(margin + 1.28155 * margin_sd, 1)],
        "total_80pct_interval": [round(total - 1.28155 * total_sd, 1),
                                 round(total + 1.28155 * total_sd, 1)],
    }
    out["home_spread_line"] = home_spread
    out["p_home_cover"] = round(phi((margin + float(home_spread)) / margin_sd), 4) if home_spread is not None else None
    out["total_line"] = total_line
    out["p_over"] = round(phi((total - float(total_line)) / total_sd), 4) if total_line is not None else None
    return out


def read_data(paths):
    frames = [pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
              for p in paths]
    if not frames:
        raise ValueError("no input files")
    return pd.concat(frames, ignore_index=True)


def box_data(boxes, players=None):
    """Parse completed team rows, refusing impossible boxscores."""
    rows = {}
    for b in boxes.to_dict("records"):
        gid = str(b.get("game_id"))
        tid = str(b.get("team_id"))
        if not gid or not tid:
            continue
        score = num(b.get("team_score"))
        fga, fgm = num(b.get("field_goals_attempted")), num(b.get("field_goals_made"))
        three, fta = num(b.get("three_point_field_goals_made")), num(b.get("free_throws_attempted"))
        oreb, to = num(b.get("offensive_rebounds")), num(b.get("total_turnovers"))
        dreb = num(b.get("defensive_rebounds"))
        if None in (score, fga, fgm, three, fta, oreb, to, dreb) or fga < 20 or fgm > fga:
            continue
        pos = fga - oreb + to + .475 * fta
        if not 35 <= pos <= 140 or not 0 <= score <= 220:
            continue
        rows[(gid, tid)] = {
            "score": score, "poss": pos, "fga": fga, "fgm": fgm, "three": three,
            "fta": fta, "oreb": oreb, "dreb": dreb, "tov": to,
        }
    player_stats = defaultdict(dict)
    if players is not None:
        need = ("game_id", "team_id", "athlete_id", "minutes", "points")
        if all(col in players.columns for col in need):
            has_names = "athlete_display_name" in players.columns
            cols = list(need) + (["athlete_display_name"] if has_names else [])
            for row in players[cols].itertuples(index=False, name=None):
                gid, tid, pid, minutes, pts = row[:5]
                name = str(row[5]) if has_names and pd.notna(row[5]) else str(pid)
                minval = num(minutes)
                if minval is None or minval <= 0:
                    continue
                player_stats[(str(gid), str(tid))][str(pid)] = (
                    min(50., minval), num(pts, 0.), name)
    return rows, player_stats


def game_observation(own, opp):
    # Shared possessions avoid treating two different ball counts as independent.
    poss = (own["poss"] + opp["poss"]) / 2
    rebound_opportunities = own["oreb"] + opp["dreb"]
    return {
        "pace": poss,
        "ortg": 100 * own["score"] / poss,
        "drtg": 100 * opp["score"] / poss,
        "efg": (own["fgm"] + .5 * own["three"]) / own["fga"],
        "tov": own["tov"] / poss,
        "orb": own["oreb"] / rebound_opportunities if rebound_opportunities else LEAGUE_PRIOR["orb"],
        "ftr": own["fta"] / own["fga"],
        "efg_allowed": (opp["fgm"] + .5 * opp["three"]) / opp["fga"],
        "tov_forced": opp["tov"] / poss,
        "orb_allowed": opp["oreb"] / (opp["oreb"] + own["dreb"]) if (opp["oreb"] + own["dreb"]) else LEAGUE_PRIOR["orb_allowed"],
        "ftr_allowed": opp["fta"] / opp["fga"],
    }


def profile(history, team, season):
    prior = [x for x in history[team] if x["season"] == season - 1][-20:]
    current = [x for x in history[team] if x["season"] == season][-12:]
    # Prior season is discounted to 3 pseudo-games; no later-season records exist.
    previous_w = [3. / len(prior)] * len(prior) if prior else []
    recent_w = [0.88 ** (len(current) - i - 1) for i in range(len(current))]
    denom = 5 + sum(previous_w) + sum(recent_w)
    results = {k: (LEAGUE_PRIOR[k] * 5 +
                    sum(w * x["stats"][k] for x, w in zip(prior, previous_w)) +
                    sum(w * x["stats"][k] for x, w in zip(current, recent_w))) / denom
               for k in STAT_KEYS}
    results["n_current"] = len(current)
    results["recent_form"] = (
        sum(x["stats"]["ortg"] - x["stats"]["drtg"] for x in current[-3:]) / min(3, len(current))
        if current else 0.
    )
    recent_player_minutes = defaultdict(float)
    older_player_minutes = defaultdict(float)
    recent_player_pts = defaultdict(float)
    recent_player_games = defaultdict(int)
    player_names = {}
    for x in current[-3:]:
        for pid, data in x["players"].items():
            mins, pts = data[:2]
            player_names[pid] = data[2] if len(data) > 2 else pid
            recent_player_minutes[pid] += mins
            recent_player_pts[pid] += pts
            recent_player_games[pid] += 1
    for x in current[:-3]:
        for pid, data in x["players"].items():
            older_player_minutes[pid] += data[0]
    sorted_minutes = sorted(recent_player_minutes.values(), reverse=True)
    total_minutes = sum(sorted_minutes)
    results["rotation_concentration"] = sum(sorted_minutes[:5]) / total_minutes if total_minutes else .65
    # Changes in rotation exposure, rather than presumed verified starters.
    results["rotation_change"] = (
        sum(mins for pid, mins in recent_player_minutes.items() if pid not in older_player_minutes) / total_minutes
        if total_minutes and older_player_minutes else 0.
    )
    results["player_minutes_observed"] = bool(total_minutes)
    results["observed_recent_rotation"] = [
        {"athlete_id": pid, "name": player_names.get(pid, pid),
         "minutes_in_last_3_observed_team_games": round(mins, 1),
         "points_per_observed_appearance": round(
             recent_player_pts[pid] / recent_player_games[pid], 1),
         "minutes_share_of_observed_rotation": round(mins / total_minutes, 4)
         if total_minutes else None, "starter_verified_for_next_game": False}
        for pid, mins in sorted(recent_player_minutes.items(),
                                key=lambda z: -z[1])[:6]
    ]
    return results


def game_scout(home, away, neutral):
    """Pregame *observations*, not a trained change in score or win probability."""
    def side(attack, defense):
        return {
            "offensive_points_per_100_possessions_prior": round(attack["ortg"], 2),
            "opponent_defensive_points_allowed_per_100_possessions_prior": round(defense["drtg"], 2),
            "shooting_efficiency_vs_defensive_allowance_proxy_pct_points": round(
                100 * (attack["efg"] - defense["efg_allowed"]), 2),
            "turnover_rate_vs_opponent_forced_rate_pct_points": round(
                100 * (attack["tov"] - defense["tov_forced"]), 2),
            "offensive_rebound_rate_vs_opponent_allowed_pct_points": round(
                100 * (attack["orb"] - defense["orb_allowed"]), 2),
            "foul_drawing_rate_vs_opponent_allowed": round(
                attack["ftr"] - defense["ftr_allowed"], 3),
            "prior_team_games_this_season": attack["n_current"],
            "recent_3_game_raw_net_rating": round(attack["recent_form"], 2),
            "top_previous_rotation_players": attack["observed_recent_rotation"],
        }
    return {
        "evidence_source": "ESPN_HOOPR_PREVIOUS_COMPLETED_TEAM_AND_PLAYER_BOXSCORES",
        "evidence_not_validated_individual_matchup_effects": True,
        "neutral_site": bool(neutral),
        "home_pace_prior": round(home["pace"], 2),
        "away_pace_prior": round(away["pace"], 2),
        "shared_pace_proxy": round((home["pace"] + away["pace"]) / 2, 2),
        "tempo_mismatch_possessions": round(home["pace"] - away["pace"], 2),
        "home_offense_vs_away_defense": side(home, away),
        "away_offense_vs_home_defense": side(away, home),
        "confirmed_starting_fives": False,
        "injuries_source_verified": False,
        "coaching_play_call_or_referee_film_verified": False,
        "missing": ["verified starters and injuries",
                    "coach and scheme playcalling",
                    "travel and rest adjustments",
                    "individual defender matchups",
                    "actual sportsbook quotes where absent"],
    }


def vector(home, away, neutral):
    pace = (home["pace"] + away["pace"]) / 2
    heff = home["ortg"] + away["drtg"] - LEAGUE_PRIOR["drtg"]
    aeff = away["ortg"] + home["drtg"] - LEAGUE_PRIOR["drtg"]
    # Approximate points from possession times adjusted efficiency. Learned
    # ridge model estimates correction, including home edge and game context.
    mechanical_margin = pace * (heff - aeff) / 100 + (0 if neutral else 3.5)
    mechanical_total = pace * (heff + aeff) / 100
    vals = [
        mechanical_margin, mechanical_total, float(neutral), pace,
        home["ortg"], away["ortg"], home["drtg"], away["drtg"],
        home["efg"] - away["efg_allowed"], away["efg"] - home["efg_allowed"],
        away["tov_forced"] - home["tov"], home["tov_forced"] - away["tov"],
        home["orb"] - away["orb_allowed"], away["orb"] - home["orb_allowed"],
        home["ftr"] - away["ftr_allowed"], away["ftr"] - home["ftr_allowed"],
        min(home["n_current"], 12), min(away["n_current"], 12),
        home["rotation_concentration"] - away["rotation_concentration"],
        home["rotation_change"] - away["rotation_change"],
        home["recent_form"], away["recent_form"]
    ]
    return np.array(vals, dtype=float), np.array([mechanical_margin, mechanical_total])


def build_rows(schedule, boxes, players=None, asof=None):
    asof = asof or datetime.now(timezone.utc)
    if asof.tzinfo is None:
        raise ValueError("asof must be timezone-aware")
    boxmap, playerstats = box_data(boxes, players)
    schedule = schedule.copy()
    schedule["tip"] = pd.to_datetime(schedule["game_date_time"], utc=True, errors="coerce")
    schedule = schedule.dropna(subset=["tip", "home_id", "away_id", "game_id"])
    schedule["date_only"] = schedule["tip"].dt.date
    schedule = schedule.sort_values(["date_only", "tip", "game_id"]).drop_duplicates("game_id", keep="last")
    history = defaultdict(list)
    completed, upcoming = [], []
    # Snapshot *every* game on a calendar day before applying any boxscore
    # from that day. Same-day games cannot leak into an earlier start time.
    for _, day in schedule.groupby("date_only", sort=True):
        to_update = []
        for g in day.to_dict("records"):
            tip = pd.Timestamp(g["tip"]).to_pydatetime()
            season = int(g.get("season"))
            home_id, away_id = str(g["home_id"]), str(g["away_id"])
            gid = str(g["game_id"])
            if home_id == away_id:
                continue
            neutral = boolean(g.get("neutral_site"))
            h, a = profile(history, home_id, season), profile(history, away_id, season)
            x, naive = vector(h, a, neutral)
            event = {"game_id": gid, "tip_utc": tip.isoformat(), "season": season,
                     "home_team_id": home_id, "away_team_id": away_id,
                     "home": str(g.get("home_display_name") or g.get("home_name") or home_id),
                     "away": str(g.get("away_display_name") or g.get("away_name") or away_id),
                     "neutral_site": neutral, "home_prior_games": h["n_current"],
                     "away_prior_games": a["n_current"], "player_box_coverage": (
                         h["player_minutes_observed"] and a["player_minutes_observed"]),
                     "x": x, "naive": naive}
            is_done = boolean(g.get("status_type_completed"))
            hb, ab = boxmap.get((gid, home_id)), boxmap.get((gid, away_id))
            if is_done and hb and ab and tip < asof:
                hs, ascore = num(g.get("home_score")), num(g.get("away_score"))
                if hs is None or ascore is None:
                    continue
                if abs(hs - hb["score"]) > .01 or abs(ascore - ab["score"]) > .01:
                    # Reject inconsistent sources, not cherry-pick score.
                    continue
                event["y"] = np.array([hs - ascore, hs + ascore], dtype=float)
                completed.append(event)
                to_update.append((season, home_id, away_id, gid, hb, ab))
            elif tip > asof and not is_done:
                event["game_intelligence"] = game_scout(h, a, neutral)
                upcoming.append(event)
        for season, h_id, a_id, gid, hb, ab in to_update:
            history[h_id].append({"season": season, "stats": game_observation(hb, ab),
                                  "players": playerstats.get((gid, h_id), {})})
            history[a_id].append({"season": season, "stats": game_observation(ab, hb),
                                  "players": playerstats.get((gid, a_id), {})})
    return completed, upcoming


def as_matrix(rows):
    return np.stack([r["x"] for r in rows]), np.stack([r["y"] for r in rows])


def fitted(rows, alpha):
    x, y = as_matrix(rows)
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    model.fit(x, y)
    return model


def metrics(rows, predictions, sd):
    y = np.stack([r["y"] for r in rows])
    e = predictions - y
    p = np.array([phi(float(m / sd)) for m in predictions[:, 0]])
    truth = (y[:, 0] > 0).astype(float)
    baselines = np.stack([r["naive"] for r in rows])
    naive_p = np.array([phi(float(m / sd)) for m in baselines[:, 0]])
    def err(a):
        return round(float(np.mean(np.abs(a))), 4)
    return {
        "games": len(rows),
        "margin_mae": err(e[:, 0]),
        "total_mae": err(e[:, 1]),
        "moneyline_accuracy": round(float(np.mean((p >= .5) == truth)), 4),
        "moneyline_brier": round(float(np.mean((p - truth) ** 2)), 5),
        "mechanical_baseline_margin_mae": err(baselines[:, 0] - y[:, 0]),
        "mechanical_baseline_total_mae": err(baselines[:, 1] - y[:, 1]),
        "mechanical_baseline_moneyline_brier": round(float(np.mean((naive_p - truth) ** 2)), 5),
        "home_team_only_accuracy": round(float(np.mean(truth == 1)), 4),
        "predicted_home_win_rate": round(float(np.mean(p)), 4),
        "observed_home_win_rate": round(float(np.mean(truth)), 4),
    }


def evaluate_and_forecast(completed, upcoming, horizon_days=7, audit_output=None,
                          audit_markdown=None, uncertainty_output=None,
                          uncertainty_markdown=None, neutral_output=None,
                          neutral_markdown=None):
    train = [r for r in completed if r["season"] in (2023, 2024)]
    validation = [r for r in completed if r["season"] == 2025]
    holdout = [r for r in completed if r["season"] == 2026]
    if min(len(train), len(validation), len(holdout)) < 200:
        raise ValueError("INSUFFICIENT_SEASON_COHORTS: need 2023-2026 historical data")
    _, vy = as_matrix(validation)
    candidates = []
    for alpha in (10, 100, 1000):
        m = fitted(train, alpha)
        vp = m.predict(as_matrix(validation)[0])
        errors = np.mean(np.abs(vp - vy), axis=0)
        candidates.append((float(errors[0] + errors[1]), alpha))
    candidates.sort()
    alpha = candidates[0][1]
    # Validation season is not part of the initial training or holdout.
    vm = fitted(train, alpha)
    vp = vm.predict(as_matrix(validation)[0])
    ve = vp - vy
    sd = np.maximum(np.std(ve, axis=0, ddof=1), [7., 9.])
    corr = float(np.corrcoef(ve.T)[0, 1])
    corr = max(-.95, min(.95, corr)) if math.isfinite(corr) else 0.
    # After choosing only on 2025, test on untouched 2026 season.
    hm = fitted(train + validation, alpha)
    hp = hm.predict(as_matrix(holdout)[0])
    report = {
        "split": "develop_2023_2024;select_and_estimate_uncertainty_2025;test_once_2026",
        "alpha_selected_using_2025": alpha,
        "validation": metrics(validation, vp, sd[0]),
        "untouched_2026_holdout": metrics(holdout, hp, sd[0]),
        "residual_margin_sd_from_2025": round(float(sd[0]), 3),
        "residual_total_sd_from_2025": round(float(sd[1]), 3),
        "residual_margin_total_correlation_from_2025": round(corr, 4),
        "probabilities_are_normal_residual_estimates_not_prospectively_calibrated": True,
        "spread_and_total_cover_accuracy_not_tested_without_archived_historical_lines": True,
    }
    if audit_output is not None or audit_markdown is not None:
        if not audit_output or not audit_markdown:
            raise ValueError("both audit paths must be specified")
        from cbb_probability_stress import make_audit, save
        audit = make_audit(completed, validation, vp, holdout, hp, sd)
        save(audit, audit_output, audit_markdown)
    if uncertainty_output is not None or uncertainty_markdown is not None:
        if not uncertainty_output or not uncertainty_markdown:
            raise ValueError("both uncertainty audit paths must be specified")
        from cbb_uncertainty_diagnostic import experiment, save as save_uncertainty
        result = experiment(validation, vp, holdout, hp, sd)
        save_uncertainty(result, uncertainty_output, uncertainty_markdown)
    # 2027 has no role in coefficient fit: its played games only update *as-of*
    # team and player histories for today's report.
    final = fitted(train + validation + holdout, alpha)
    asof = datetime.now(timezone.utc)
    games = []
    for g in sorted(upcoming, key=lambda x: x["tip_utc"]):
        if g["season"] != 2027 or g["home_prior_games"] + g["away_prior_games"] < 0:
            continue
        tip = datetime.fromisoformat(g["tip_utc"])
        if tip < asof or tip > asof + timedelta(days=horizon_days):
            continue
        margin, total = final.predict(g["x"][None, :])[0]
        probs = score_probabilities(float(margin), float(total), float(sd[0]), float(sd[1]))
        games.append({
            **{k: v for k, v in g.items() if k not in ("x", "naive", "y")},
            **probs,
            "source": "ESPN_HOOPR_HISTORICAL_TEAM_PLAYER_BOXSCORES_AND_SCHEDULE",
            "not_confirmed_starting_fives": True,
            "not_a_bookmaker_fade_or_value_pick": True,
        })
    if neutral_output is not None or neutral_markdown is not None:
        if not neutral_output or not neutral_markdown:
            raise ValueError("both neutral challenge paths must be specified")
        from cbb_neutral_challenger import (
            challenger_audit, neutral_specialist, shadow_upcoming,
            elo_future_probabilities, save as save_neutral,
        )
        neutral_result = challenger_audit(
            completed, train, validation, holdout, vm, vp, hm, hp, float(sd[0]))
        future_neutral = [g for g in upcoming if g["neutral_site"] and g["season"] == 2027
                          and asof < datetime.fromisoformat(g["tip_utc"])
                          <= asof + timedelta(days=horizon_days)]
        elo_today = elo_future_probabilities(completed, future_neutral)
        special_live = neutral_specialist(train + validation + holdout)
        neutral_result["upcoming_neutral_research_shadow"] = shadow_upcoming(
            future_neutral, final, special_live, asof, float(sd[0]), elo=elo_today)
        neutral_result["upcoming_shadow_is_not_official_predictions"] = True
        save_neutral(neutral_result, neutral_output, neutral_markdown)
    return report, games


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team-box", action="append", required=True)
    ap.add_argument("--player-box", action="append", default=[])
    ap.add_argument("--schedule", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--audit-json", help="Optional diagnostics (not promotion evidence)")
    ap.add_argument("--audit-md", help="Optional readable diagnostic report")
    ap.add_argument("--uncertainty-json", help="2025-trained uncertainty challenger; research only")
    ap.add_argument("--uncertainty-md", help="Readable uncertainty diagnostic")
    ap.add_argument("--neutral-json", help="Neutral expert comparison and upcoming shadow; research only")
    ap.add_argument("--neutral-md", help="Neutral expert readable report")
    args = ap.parse_args()
    team, schedule = read_data(args.team_box), read_data(args.schedule)
    player = read_data(args.player_box) if args.player_box else None
    asof = datetime.now(timezone.utc)
    completed, future = build_rows(schedule, team, player, asof=asof)
    report, games = evaluate_and_forecast(
        completed, future, horizon_days=args.days,
        audit_output=args.audit_json, audit_markdown=args.audit_md,
        uncertainty_output=args.uncertainty_json,
        uncertainty_markdown=args.uncertainty_md,
        neutral_output=args.neutral_json,
        neutral_markdown=args.neutral_md)
    out = {
        "schema": SCHEMA, "generated_utc": asof.isoformat(),
        "status": "RESEARCH_ONLY_NO_PRETIP_FIRST_SEEN_ARCHIVE",
        "no_live_frontend_or_production_changes": True,
        "historical_complete_game_rows": len(completed),
        "upcoming_scheduled_game_rows": len(future),
        "features": FEATURE_NAMES,
        "report": report, "game_predictions": games,
        "limits": [
            "No verified 2027 pregame injuries, confirmed starters, lineup matchups or bookmaker lines",
            "No historic opening/closing spread and total line coverage grades",
            "Historical backtest source snapshots were reconstructed retrospectively",
            "Do not claim prospective accuracy from retroactively reconstructed games"
        ],
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"historical": len(completed), "upcoming": len(future),
                      "forecast_rows": len(games), "validation": report["validation"],
                      "holdout": report["untouched_2026_holdout"]}, sort_keys=True))


if __name__ == "__main__":
    main()
