#!/usr/bin/env python3
"""NFL opportunity-first hypothesis 0.1. Research ONLY; no serving integration.

Prerecorded structural choices (no post-holdout tuning):
  Expected opportunities = 65% recent-three player attempts + 35% full-season
  attempts, averaged with recent player share * recent team volume.
  Efficiency = last-eight yards/attempt shrunk toward as-of league efficiency.
  Opponent adjustment = bounded +/-10% as-of efficiency, sample-shrunk.
All features from weeks strictly BEFORE evaluated week, for the whole slate.
Never equate workload stability with a probability of beating sportsbook lines.
A player not recorded in the historical stats release is not scored as zero;
coverage bias is reported and promotion is BLOCKED until solved and this model
beats the *actual* locked model on the same first-seen predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean

MARKETS = {
    "rushing_yards": ("carries", "rushing_yards", {"RB", "FB"}, 5.0, 30.0),
    "receiving_yards": ("targets", "receiving_yards", {"WR", "TE", "RB"}, 3.0, 20.0),
}
MODEL_ID = "nfl_opportunity_first_shadow_v0.1_frozen"
PARAMS = {"recent_weight": 0.65, "direct_vs_share_weight": 0.5,
          "max_defense_adjustment": 0.10, "defense_pseudo_opportunities": 50,
          "min_prior_player_games": 3, "opportunity_efficiency_lookback": 8,
          "eligible_players": "observed_target_game_only_RESEARCH_SELECTION_BIAS"}


def avg(values):
    return mean(values) if values else None


def finite_positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def forecast_player(history, team_history, opponent_history, league_history, *,
                    season, market):
    """Returns independent expected attempts, per-attempt efficiency and yards.

    history items: (season, week, team, attempts, yards, observed_team_attempts).
    team_history items: (season, week, volume). Opponent: (attempts, yards).
    All are passed strictly as-of from walkforward(), not read from target game.
    """
    if market not in MARKETS:
        raise ValueError("unknown market")
    _, _, _, min_vol, pseudo = MARKETS[market]
    recent = history[-3:]
    current = [g for g in history if g[0] == season]
    if len(recent) < 3 or not current or avg([g[3] for g in recent]) < min_vol:
        return None
    if recent[-1][2] != recent[0][2]:
        return None  # new team: role context is too ambiguous for this version
    same_team = recent[-1][2]
    rec3 = [g for g in recent if g[2] == same_team]
    if len(rec3) < 3:
        return None
    opportunities_recent = avg([g[3] for g in rec3])
    opportunities_season = avg([g[3] for g in current if g[2] == same_team])
    if opportunities_season is None:
        return None

    t3 = [g[2] for g in team_history if g[0] == season and g[2] is not None][-3:]
    shares = [g[3] / g[5] for g in rec3 if g[5] > 0]
    if not t3 or len(shares) != 3:
        return None
    observed_share = avg(shares)
    projected_team_volume = avg(t3)
    share_opportunities = observed_share * projected_team_volume
    direct = (PARAMS["recent_weight"] * opportunities_recent
              + (1-PARAMS["recent_weight"]) * opportunities_season)
    attempts = PARAMS["direct_vs_share_weight"] * direct + (
        1-PARAMS["direct_vs_share_weight"]) * share_opportunities

    prior_league_vol = sum(g[0] for g in league_history)
    prior_league_yards = sum(g[1] for g in league_history)
    if prior_league_vol <= 0:
        return None
    league_eff = prior_league_yards / prior_league_vol
    last8 = history[-PARAMS["opportunity_efficiency_lookback"]:]
    vol = sum(g[3] for g in last8)
    yards = sum(g[4] for g in last8)
    efficiency = (yards + pseudo * league_eff) / (vol + pseudo)

    opp_vol = sum(g[0] for g in opponent_history)
    opp_yds = sum(g[1] for g in opponent_history)
    if opp_vol >= 10 and league_eff > 0:
        raw = (opp_yds / opp_vol / league_eff) - 1
        shrunk = raw * (opp_vol / (opp_vol + PARAMS["defense_pseudo_opportunities"]))
        matchup_adj = clamp(shrunk, -PARAMS["max_defense_adjustment"],
                            PARAMS["max_defense_adjustment"])
    else:
        matchup_adj = 0.0

    previous = [g[3] for g in history[-8:-3] if g[2] == same_team]
    role_change = opportunities_recent / avg(previous) if len(previous) >= 3 and avg(previous) > 0 else None
    return {
        "projected_yards": max(0, attempts * efficiency * (1 + matchup_adj)),
        "expected_opportunities": attempts,
        "yards_per_opportunity": efficiency,
        "league_eff_asof": league_eff,
        "last3_share": observed_share,
        "last3_opportunities": opportunities_recent,
        "season_opportunities": opportunities_season,
        "projected_team_opportunities": projected_team_volume,
        "role_trend_ratio": role_change,
        "role_direction": ("EXPANDING" if role_change and role_change >= 1.35 else
                           "DECLINING" if role_change and role_change <= 0.75 else
                           "NO_CONFIRMED_CHANGE"),
        "opponent_efficiency_adjustment": matchup_adj,
        "history_games": len(history),
        "no_verified_future_starter_status": True,
    }


def walkforward(con, seasons=(2025, 2026)):
    all_rows = con.execute("""
        SELECT player_id,player_name,position,team,opponent,season,week,game_id,
               COALESCE(carries,0),COALESCE(rushing_yards,0),
               COALESCE(targets,0),COALESCE(receiving_yards,0)
        FROM player_games WHERE season_type='REG'
        ORDER BY season,week,game_id,player_id
    """).fetchall()
    by_week = defaultdict(list)
    for r in all_rows:
        if not r[7] or not r[0]:
            continue
        by_week[(r[5], r[6])].append(r)
    history = defaultdict(list)
    team_history = defaultdict(list)
    opp_history = defaultdict(list)
    league_history = defaultdict(list)
    outcomes = []
    eligible_and_observed = defaultdict(int)

    for (season, week), week_rows in sorted(by_week.items()):
        totals = defaultdict(float)
        opp_aggs = defaultdict(lambda: [0.0, 0.0])
        league_aggs = defaultdict(lambda: [0.0, 0.0])
        teams_games = set()
        for r in week_rows:
            pid, name, position, team, opponent, s, w, game_id, carries, rush_y, targets, recv_y = r
            teams_games.add((game_id, team))
            for market, (_, _, positions, _, _) in MARKETS.items():
                if position not in positions:
                    continue
                attempt, yds = (carries,rush_y) if market=="rushing_yards" else (targets,recv_y)
                if not finite_positive(attempt) or not finite_positive(yds):
                    continue
                totals[(game_id,team,market)] += attempt
                # Opponent receives performance allowed BY the opposing team's offense.
                if opponent:
                    opp_aggs[(opponent,market)][0] += attempt
                    opp_aggs[(opponent,market)][1] += yds
                league_aggs[market][0] += attempt
                league_aggs[market][1] += yds

        # Evaluate the entire week before learning anything in this week.
        if season in seasons:
            for r in week_rows:
                pid, name, pos, team, opp, s, w, gid, carries, rush_y, targets, recv_y = r
                for market, (_, _, positions, _, _) in MARKETS.items():
                    if pos not in positions:
                        continue
                    vol, yds = (carries,rush_y) if market=="rushing_yards" else (targets,recv_y)
                    if not finite_positive(vol) or not finite_positive(yds):
                        continue
                    eligible_and_observed[(season,market)] += 1
                    ph = history[(pid,market)]
                    f = forecast_player(
                        ph, team_history[(team,market)], opp_history[(opp,market)],
                        league_history[market], season=season, market=market)
                    if not f:
                        continue
                    last3_y = avg([x[4] for x in ph[-3:]])
                    outcomes.append({
                        "game_id": gid, "season": season, "week": week,
                        "player_id": pid, "player": name, "team": team, "opponent": opp,
                        "market": market, "actual": float(yds),
                        "shadow": f["projected_yards"], "baseline_last3": last3_y,
                        "role_direction": f["role_direction"],
                        "expected_opportunities": f["expected_opportunities"],
                        "recent_share": f["last3_share"],
                        "role_trend_ratio": f["role_trend_ratio"],
                        "opponent_adjustment": f["opponent_efficiency_adjustment"],
                    })

        # Now learn current completed week's facts only for future weeks.
        for r in week_rows:
            pid, name, pos, team, opp, s, w, gid, carries, rush_y, targets, recv_y = r
            for market, (_, _, positions, _, _) in MARKETS.items():
                if pos not in positions:
                    continue
                vol, yds = (carries,rush_y) if market=="rushing_yards" else (targets,recv_y)
                if not finite_positive(vol) or not finite_positive(yds):
                    continue
                history[(pid,market)].append((s,w,team,float(vol),float(yds),
                                               totals[(gid,team,market)]))
        for game_id, team in teams_games:
            for market in MARKETS:
                total = totals[(game_id,team,market)]
                if total > 0:
                    team_history[(team,market)].append((season,week,total))
        for key,(vol,yds) in opp_aggs.items():
            if vol > 0:
                opp_history[key].append((vol,yds))
        for market,(vol,yds) in league_aggs.items():
            if vol > 0:
                league_history[market].append((vol,yds))
    return outcomes, eligible_and_observed


def metrics(rows):
    if not rows:
        return {"n":0}
    error = [abs(r["shadow"]-r["actual"]) for r in rows]
    baseline = [abs(r["baseline_last3"]-r["actual"]) for r in rows]
    return {
        "n":len(rows),
        "mae_opportunity":round(mean(error),3),
        "mae_asof_last3":round(mean(baseline),3),
        "uplift_vs_last3":round(mean(baseline)-mean(error),3),
        "bias":round(mean(r["shadow"]-r["actual"] for r in rows),3),
        "within_10_yards":round(sum(e<=10 for e in error)/len(error),4),
    }


def ci_game_clusters(rows, n=800, seed=20261011):
    """Same NFL game is one resample cluster; no falsely independent players."""
    if len({r["game_id"] for r in rows}) < 3:
        return None
    by_game=defaultdict(list)
    for row in rows:
        by_game[row["game_id"]].append(row)
    games=list(by_game)
    rng=random.Random(seed)
    lifts=[]
    for _ in range(n):
        sample=[row for _ in games for row in by_game[rng.choice(games)]]
        lifts.append(mean(abs(r["baseline_last3"]-r["actual"])
                          - abs(r["shadow"]-r["actual"]) for r in sample))
    lifts.sort()
    return [round(lifts[int((n-1)*.025)],3),round(lifts[int((n-1)*.975)],3)]



def compare_first_seen(outcomes, db_path, ledger_path):
    """Paired, genuinely frozen forecast comparison on EXACT SAME players/games.

    Uses the independent canonical first-seen grader, not its own relaxed
    snapshot or a retrospective final-fit V2 model. No promotion on this
    already-consulted 2026 sample regardless of outcome.
    """
    import nfl_numeric_accuracy_audit as canonical
    doc=canonical.evaluate(canonical.load_ledger(ledger_path), db_path)
    from datetime import timedelta
    game_kickoffs=defaultdict(list)
    with sqlite3.connect(f"file:{db_path}?mode=ro",uri=True) as db:
        for ss,ww,kickoff in db.execute(
            "SELECT season,week,kickoff_utc FROM games WHERE kickoff_utc IS NOT NULL"
        ):
            game_kickoffs[(ss,ww)].append(canonical.as_utc(kickoff))
    last_completed_week={}
    max_prior=None
    for key in sorted(game_kickoffs):
        last_completed_week[key]=max_prior
        latest=max(game_kickoffs[key])+timedelta(hours=4)
        max_prior=max(latest,max_prior) if max_prior is not None else latest
    import re
    import unicodedata

    def name(v):
        z=unicodedata.normalize("NFKD",str(v or ""))
        z="".join(ch for ch in z if not unicodedata.combining(ch)).lower()
        z=re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b","",z)
        return " ".join(re.sub(r"[^a-z ]","",z).split())

    shadow={}
    duplicate=0
    for row in outcomes:
        key=(row["season"],row["week"],row["market"],row["team"],name(row["player"]))
        if key in shadow:
            duplicate+=1
            shadow[key]=None
        else:
            shadow[key]=row
    matched=[]
    skips=defaultdict(int)
    for incumbent in doc["forecasts"]:
        key=(incumbent["season"],incumbent["week"],incumbent["market"],
             incumbent["team"],name(incumbent["player"]))
        row=shadow.get(key)
        if row is None:
            skips["no_unique_shadow_role_projection"]+=1
            continue
        if abs(row["actual"]-incumbent["actual"])>0.0001:
            skips["actual_outcome_mismatch"]+=1
            continue
        cutoff=last_completed_week.get((incumbent["season"],incumbent["week"]))
        if cutoff is not None and canonical.as_utc(incumbent["logged_at"])<cutoff:
            skips["prior_week_data_unavailable_at_forecast_time"]+=1
            continue
        # As-of strictness: an entire prior week must have been completed
        # before the FIRST-SEEN public forecast timestamp. Otherwise reject.
        # The shadow includes all prior-week team/defense games, not merely
        # the player's own previous game.
        matched.append({
            "season":row["season"],"week":row["week"],"market":row["market"],
            "game_id":row["game_id"],"player":row["player"],"team":row["team"],
            "actual":row["actual"],"shadow":row["shadow"],
            "champion":incumbent["projected_median"],
            "baseline_last3":incumbent["baseline_last3"],
            "role_direction":row["role_direction"],
            "incumbent_source":str(incumbent["model_source"] or "unknown"),
            "logged_at":incumbent["logged_at"]
        })
    def paired(rows):
        if not rows:
            return {"n":0}
        shadow_mae=mean(abs(x["actual"]-x["shadow"]) for x in rows)
        champion_mae=mean(abs(x["actual"]-x["champion"]) for x in rows)
        games=defaultdict(list)
        for row in rows:
            games[row["game_id"]].append(
                abs(row["actual"]-row["champion"])-abs(row["actual"]-row["shadow"]))
        ci=None
        if len(games)>=3:
            rng=random.Random(20261012)
            blocks=list(games.values())
            samples=[]
            for _ in range(1200):
                choose=[blocks[rng.randrange(len(blocks))] for __ in blocks]
                samples.append(mean(value for sample in choose for value in sample))
            samples.sort()
            ci=[round(samples[29],3),round(samples[1169],3)]
        return {"n":len(rows),
                "shadow_mae":round(shadow_mae,3),
                "first_seen_champion_mae":round(champion_mae,3),
                "shadow_lift_against_incumbent":round(champion_mae-shadow_mae,3),
                "paired_game_clusters":len(games),"ci95_game_cluster_lift":ci}

    # Full source of truth for the incumbent is the canonical grader.
    return {
        "status":"RESEARCH_ONLY_NO_PROMOTION",
        "incumbent_first_seen_eligible":doc["eligible_sample"],
        "matched":len(matched),
        "shadow_overlap_coverage":round(len(matched)/doc["eligible_sample"],3)
             if doc["eligible_sample"] else None,
        "duplicates_shadow":duplicate,
        "exclusions":dict(skips),
        "overall":paired(matched),
        "by_market":{k:paired([r for r in matched if r["market"]==k]) for k in MARKETS},
        "by_role":{k:paired([r for r in matched if r["role_direction"]==k])
                  for k in ("EXPANDING","DECLINING","NO_CONFIRMED_CHANGE")},
        "by_source":{k:paired([r for r in matched if r["incumbent_source"]==k])
                    for k in sorted({r["incumbent_source"] for r in matched})},
        "examples":matched[-10:],
        "critical_limitations":["2026 outcomes previously consulted, not an untouched holdout",
          "Historical opportunity model selects observed game rows only",
          "No timestamped next-game lineup/active roster; no market probability"],
    }


def report(outcomes, eligible):
    groups={}
    for season in (2025,2026):
        for mkt in MARKETS:
            rows=[r for r in outcomes if r["season"]==season and r["market"]==mkt]
            res=metrics(rows)
            res["eligible_observed_rows"]=eligible[(season,mkt)]
            res["observed_cohort_coverage"]=round(
                res["n"]/eligible[(season,mkt)],3) if eligible[(season,mkt)] else None
            res["game_cluster_ci_lift"]=ci_game_clusters(rows)
            res["by_role"]={role:metrics([r for r in rows if r["role_direction"]==role])
                            for role in ("EXPANDING","DECLINING","NO_CONFIRMED_CHANGE")}
            groups[f"{season}_{mkt}"]=res
    return {
        "model":MODEL_ID,"parameter_lock":PARAMS,"uses_target_week_data_in_features":False,
        "data_scope":"prior-week as-of NFL player_games stats, teams and opponent; no articles",
        "evidence_limitations":[
            "Only players recorded with box-score rows in evaluated game are scored. "
            "Missing/inactive players cannot be treated as zero; selection bias blocks promotion.",
            "Current-week depth charts, snap commitments, late injuries, offensive-line changes "
            "and verified available player status are NOT provided by this stat-only foundation.",
            "No odds/bet side/probabilities are estimated; league and defense tendencies are as-of.",
            "No locked champion matched-prediction benchmark yet; baseline is last-three average.",
            "2026 prior outcomes have already been consulted; it is NOT an untouched holdout."],
        "promotion_status":"RESEARCH_ONLY_NOT_ELIGIBLE_FOR_PRODUCTION",
        "cohorts":groups,
        "rows_scored":len(outcomes),
        "examples":outcomes[-15:],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--db",required=True)
    p.add_argument("--out",default=None)
    p.add_argument("--first-seen-ledger",default=None)
    args=p.parse_args()
    con=sqlite3.connect(f"file:{args.db}?mode=ro",uri=True)
    rows, eligible=walkforward(con)
    con.close()
    result=report(rows,eligible)
    if args.first_seen_ledger:
        result["paired_first_seen_vs_locked_champion"]=compare_first_seen(
            rows,args.db,args.first_seen_ledger)
    if args.out:
        path=Path(args.out)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"model":result["model"],"status":result["promotion_status"],
                      "rows_scored":result["rows_scored"],
                      "paired_champion":result.get("paired_first_seen_vs_locked_champion"),
                      "cohorts":{k:{x:v[x] for x in ("n","mae_opportunity","mae_asof_last3",
                               "uplift_vs_last3","game_cluster_ci_lift","observed_cohort_coverage")}
                                 for k,v in result["cohorts"].items()}},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
