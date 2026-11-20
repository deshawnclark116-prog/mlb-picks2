#!/usr/bin/env python3
"""NFL V7 roster-conditioned opportunity challenger. NEVER the live champion.

Structural hypothesis: a player absent from the team's last two games can
return; V6B's last-two-game player universe cannot represent that event.
Use only an independently captured, pre-kickoff ESPN roster to admit an
older (last-eight-team-games) player. Preserve V6B's finite joint team
attempt/target/carry simulation and a separately identified OTHER bucket.

A roster listing is NOT an active/inactive-game-day certification and is
NOT a probability of playing. All reported distributions are CONDITIONAL
on the selected candidate universe. No bookmaker odds, retroactive team
membership, or postgame statuses may enter this model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import unicodedata

import numpy as np

import nfl_v6_joint_opportunity_shadow as v6

VERSION = "NFL_V7_PREGAME_ROSTER_RETURNER_JOINT_SHADOW_A"
MAX_PRIOR_TEAM_GAMES = 8
MIN_SIMULATIONS = 10000
ESPN_ROSTER = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{slug}/roster"
OUT_STATUSES = frozenset(("out", "injured reserve", "injuredreserve", "suspended",
    "suspension", "pup", "physically unable to perform", "non-football injury",
    "injuredreserveorout", "commissioner exempt list"))
CAUTION_STATUSES = frozenset(("questionable", "doubtful", "limited", "day-to-day"))


def normalized(name):
    s = unicodedata.normalize("NFKD", str(name or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", s)
    return " ".join(re.sub("[^a-z ]", " ", s).split())


def parse_roster(team, payload, observed_at_utc, kickoff_utc):
    """Validate a captured source, after its network read, not before it."""
    observed, kickoff = v6.utc(observed_at_utc), v6.utc(kickoff_utc)
    if observed >= kickoff:
        raise ValueError("ROSTER_NOT_OBSERVED_BEFORE_KICKOFF")
    if not isinstance(payload, dict) or not isinstance(payload.get("athletes"), list):
        raise ValueError("INVALID_ESPN_ROSTER_PAYLOAD")
    people = {}
    seen_items = 0
    for group in payload["athletes"]:
        if not isinstance(group, dict):
            continue
        tag = str(group.get("position") or "").strip().lower()
        for athlete in group.get("items") or []:
            if not isinstance(athlete, dict):
                continue
            name = athlete.get("displayName") or athlete.get("fullName")
            norm = normalized(name)
            if not norm:
                continue
            statuses = {str(athlete.get("status") or "").strip().lower()}
            for injury in athlete.get("injuries") or []:
                if isinstance(injury, dict):
                    statuses.add(str(injury.get("status") or "").strip().lower())
            if tag in ("injuredreserveorout", "suspended"):
                statuses.add(tag)
            if statuses & OUT_STATUSES:
                flag = "ROSTER_LISTED_UNAVAILABLE"
            elif statuses & CAUTION_STATUSES:
                flag = "ROSTER_LISTED_CAUTION_NOT_CONFIRMED_ACTIVE"
            else:
                flag = "ROSTER_LISTED_NOT_CONFIRMED_ACTIVE"
            # Duplicate normalized identities cannot certify an athlete.
            people.setdefault(norm, []).append({"name": str(name), "status": flag})
            seen_items += 1
    if seen_items == 0:
        raise ValueError("EMPTY_ESPN_ROSTER_CANNOT_CERTIFY")
    return {"team": team, "status": "ROSTER_CAPTURED", "observed_at_utc":
        observed.isoformat(), "source": ESPN_ROSTER.format(
        slug={"LA": "lar", "WAS": "wsh"}.get(team, team.lower())),
        "items": seen_items, "players": people,
        "no_game_day_active_certification": True}


def roster_candidates(team, team_games, histories, league, roster):
    """Return V6 baseline + source-confirmed returners, never hindsight actives.

    The V6 last-two-game baseline remains a reference for a fair paired
    comparison, but the challenger requires the contemporaneous roster for
    every admitted player. Not found, ambiguous or listed OUT => excluded.
    """
    if roster.get("status") != "ROSTER_CAPTURED" or roster.get("team") != team:
        raise ValueError("VERIFIED_TEAM_ROSTER_REQUIRED")
    baseline = v6.participants(team, team_games, histories, league)
    baseline_ids = {p["id"] for p in baseline}
    old = team_games[-MAX_PRIOR_TEAM_GAMES:]
    ids = {r["id"] for g in old for r in g["rows"] if r["team"] == team}
    results, audit = [], {"v6_last2": len(baseline), "returners_admitted": 0,
                          "unmatched_or_ambiguous": 0, "listed_unavailable": 0}
    for pid in sorted(ids):
        hist = histories.get((pid, team), [])[-v6.LAST:]
        if not hist:
            continue
        last = hist[-1]
        identity = roster["players"].get(normalized(last["name"]), [])
        if len(identity) != 1:
            audit["unmatched_or_ambiguous"] += 1
            continue
        if identity[0]["status"] == "ROSTER_LISTED_UNAVAILABLE":
            audit["listed_unavailable"] += 1
            continue
        target = sum(float(h["targets"]) for h in hist)
        carry = sum(float(h["carries"]) for h in hist)
        attempts = sum(float(h["attempts"]) for h in hist)
        if target + carry + attempts <= 0:
            continue
        # Gap in ACTUAL team fixtures, not historical player rows: an
        # unobserved stat row does not prove the player did not dress.
        last_idx = next((i for i in range(len(old)-1, -1, -1)
                         if any(r["id"] == pid for r in old[i]["rows"])), None)
        gap = len(old) - 1 - last_idx if last_idx is not None else None
        results.append({"id": pid, "player": last["name"], "team": team,
            "position": last["pos"], "n_games": len(hist), "last_seen": last["kickoff"],
            "targets": target, "carries": carry, "attempts": attempts,
            "ypt": (sum(h["rec_yards"] for h in hist) + 8*league["rec_ypt"])/(target+8),
            "ypc": (sum(h["rush_yards"] for h in hist) + 12*league["rush_ypc"])/(carry+12),
            "ypa": (sum(h["pass_yards"] for h in hist) + 16*league["pass_ypa"])/(attempts+16),
            "roster_status": identity[0]["status"], "team_fixture_gap": gap,
            "v6_last2_candidate": pid in baseline_ids, "game_day_active_verified": False})
        if pid not in baseline_ids:
            audit["returners_admitted"] += 1
    audit["challenger_candidates"] = len(results)
    audit["roster_player_items"] = roster["items"]
    return results, audit


def simulate_game(game, by_team, histories, league, rosters, role_source=None,
                  sims=20000, seed=v6.SEED):
    """Recompute all teammate shares, rather than adding yards post hoc."""
    if sims < MIN_SIMULATIONS:
        raise ValueError("INSUFFICIENT_SIMULATIONS")
    home, away = game["home_team"], game["away_team"]
    if not home or not away or home == away:
        raise ValueError("INVALID_MATCHUP")
    rng = np.random.default_rng(seed)
    common = rng.lognormal(-.5*.12**2, .12, sims)
    outputs, summaries, seen = [], {}, set()
    for team, opp in ((home, away), (away, home)):
        ctx = v6.context(team, opp, by_team, league)
        people, audit = roster_candidates(team, by_team.get(team, []),
                                          histories, league, rosters[team])
        ids = {p["id"] for p in people}
        if ids & seen:
            raise ValueError("SAME_PLAYER_ON_BOTH_TEAMS")
        seen |= ids
        if ctx["own_games"] < 3 or ctx["opponent_games"] < 3 or not people:
            summaries[team] = {"status": "INSUFFICIENT_PREGAME_SOURCE", **audit}
            continue
        total = np.clip(np.rint(ctx["team_plays"]*common*rng.lognormal(
            -.5*.08**2, .08, sims)).astype(int), 22, 104)
        rate = rng.beta(max(1, ctx["pass_rate"]*65),
                        max(1, (1-ctx["pass_rate"])*65), sims)
        passing = rng.binomial(total, rate)
        rushing = total-passing
        receivers = [p for p in people if p["targets"] > 0]
        rushers = [p for p in people if p["carries"] > 0]
        qb = sorted((p for p in people if p["attempts"] >= 10),
                    key=lambda p: (p["last_seen"], p["attempts"]), reverse=True)
        selected_qb = qb[0] if qb else None
        t_fraction = v6.clip(sum(g["targets"] for g in by_team[team][-v6.LAST:]) /
            max(1, sum(g["passes"] for g in by_team[team][-v6.LAST:])), .5, 1)
        if role_source is None:
            t_alpha = np.asarray([p["targets"]+.5 for p in receivers]+[5.], float)
            r_alpha = np.asarray([p["carries"]+.5 for p in rushers]+[4.], float)
        else:
            from nfl_v6_snap_role_evidence import allocate_role_priors
            prior = allocate_role_priors(people, histories, team, role_source,
                ctx["team_plays"], ctx["team_plays"]*ctx["pass_rate"]*t_fraction,
                ctx["team_plays"]*(1-ctx["pass_rate"]))
            idx = {p["id"]: i for i,p in enumerate(people)}
            t_alpha = np.asarray([prior["target_alpha"][idx[p["id"]]]
                for p in receivers]+[prior["target_alpha"][-1]], float)
            r_alpha = np.asarray([prior["carry_alpha"][idx[p["id"]]]
                for p in rushers]+[prior["carry_alpha"][-1]], float)
        targets = rng.binomial(passing, t_fraction)
        target_alloc = np.zeros((sims,len(t_alpha)), np.int16)
        carry_alloc = np.zeros((sims,len(r_alpha)), np.int16)
        for j in range(sims):
            target_alloc[j] = rng.multinomial(int(targets[j]), rng.dirichlet(t_alpha))
            carry_alloc[j] = rng.multinomial(int(rushing[j]), rng.dirichlet(r_alpha))
        pass_yards = np.zeros(sims, float)
        qb_factor = v6.clip(selected_qb["ypa"]/max(.1,league["pass_ypa"])
                            if selected_qb else 1., .75, 1.25)
        for players, counts, market in ((receivers,target_alloc,"receiving_yards"),
                                        (rushers,carry_alloc,"rushing_yards")):
            for i,p in enumerate(players):
                opps = counts[:,i]
                if market == "receiving_yards":
                    mean_per = v6.clip(p["ypt"]*ctx["opp_pass_factor"]*qb_factor,1,22)
                    yards = rng.gamma(np.maximum(.01,opps*1.25), mean_per/1.25)
                    pass_yards += np.where(opps>0,yards,0)
                    stat = "rec_yards"
                else:
                    mean_per = v6.clip(p["ypc"]*ctx["opp_rush_factor"],1,10)
                    yards = rng.gamma(np.maximum(.01,opps*1.8),mean_per/1.8)
                    stat = "rush_yards"
                yards[opps == 0] = 0
                outputs.append({"player_id":p["id"],"player":p["player"],
                    "team":team,"opponent":opp,"market":market,
                    "roster_status":p["roster_status"],
                    "team_fixture_gap":p["team_fixture_gap"],
                    "returner_added_to_player_universe":not p["v6_last2_candidate"],
                    "probability_event":"CONDITIONAL_ON_SELECTED_ROSTER_SCENARIO",
                    "naive_recent3_mean":round(float(np.mean(
                        [h[stat] for h in histories[(p["id"],team)][-3:]])),1),
                    **v6.distribution(yards,opps)})
        if selected_qb:
            other = target_alloc[:,-1]
            y = rng.gamma(np.maximum(.01,other*1.25), v6.clip(
                league["rec_ypt"]*ctx["opp_pass_factor"]*qb_factor,1,22)/1.25)
            pass_yards += y*(other>0)
            outputs.append({"player_id":selected_qb["id"],
                "player":selected_qb["player"],"team":team,"opponent":opp,
                "market":"passing_yards","starting_qb_verified":False,
                "probability_event":"CONDITIONAL_ON_SELECTED_ROSTER_SCENARIO",
                **v6.distribution(pass_yards,passing)})
        summaries[team] = {"status":"ROSTER_CONDITIONAL_NOT_GAME_DAY_ACTIVE",
            **audit,"simulated_targets":round(float(targets.mean()),2),
            "simulated_carries":round(float(rushing.mean()),2),
            "unknown_others_target_and_carry_bucket":True,
            "source_is_offensive_snaps_not_routes":True}
    return {"game_id":game["game_id"],"home":home,"away":away,
        "kickoff_utc":game["kickoff_utc"],"simulations":sims,
        "teams":summaries,"projections":outputs,
        "no_calibrated_probability_of_playing":True}


def capture(db, snap_files, out_dir, *, now=None, sims=20000, fetch=None):
    """Exclusive pretip source receipt; never regenerate a game after kickoff."""
    import requests
    at = v6.utc(now) if now else datetime.now(timezone.utc)
    if sims < MIN_SIMULATIONS:
        raise ValueError("INSUFFICIENT_SIMULATIONS")
    if not snap_files or any(not Path(p).is_file() for p in snap_files):
        raise ValueError("MISSING_VERIFIED_SNAP_FILES")
    if fetch is None:
        fetch = requests.get
    conn = sqlite3.connect("file:"+str(Path(db).absolute())+"?mode=ro",uri=True)
    conn.row_factory = sqlite3.Row
    dest = Path(out_dir); dest.mkdir(parents=True,exist_ok=True)
    results={"captured":[],"already_frozen":[],"no_roster_source":[],"window_missed":[]}
    try:
        games = [dict(x) for x in conn.execute(
            "SELECT game_id,season,week,home_team,away_team,kickoff_utc "
            "FROM games WHERE kickoff_utc IS NOT NULL ORDER BY kickoff_utc")]
        for game in games:
            kickoff = v6.utc(game["kickoff_utc"])
            if not 0 < (kickoff-at).total_seconds() <= 30*3600:
                continue
            file = dest / (game["game_id"]+".json")
            if file.exists():
                results["already_frozen"].append(game["game_id"])
                continue
            rosters, raw_sha = {}, {}
            for team in (game["home_team"],game["away_team"]):
                slug = {"LA":"lar","WAS":"wsh"}.get(team,team.lower())
                try:
                    response = fetch(ESPN_ROSTER.format(slug=slug),timeout=20)
                    response.raise_for_status()
                    raw = response.json()
                    observed = datetime.now(timezone.utc) if now is None else at
                    if observed >= kickoff:
                        raise ValueError("SOURCE_ARRIVED_AFTER_KICKOFF")
                    rosters[team] = parse_roster(team,raw,observed,kickoff)
                    raw_sha[team]=hashlib.sha256(json.dumps(raw,sort_keys=True,
                        separators=(",",":")).encode()).hexdigest()
                except (ValueError,KeyError,TypeError,requests.RequestException) as e:
                    results["no_roster_source"].append({
                        "game_id":game["game_id"],"team":team,"error":type(e).__name__})
                    break
            if len(rosters)!=2:
                continue
            # Factual player histories are only from games that completed
            # >24h before the ACTUAL source capture time.
            cutoff = min(kickoff-timedelta(hours=2),
                         min(v6.utc(x["observed_at_utc"]) for x in rosters.values()))
            by_team,histories = v6.game_history(v6.asof_rows(conn,cutoff))
            league = v6.league_rates(by_team)
            from nfl_v6_snap_role_evidence import read_snap_roles
            roles=read_snap_roles(snap_files,db,cutoff)
            seed=v6.SEED+int(hashlib.sha256(game["game_id"].encode()).hexdigest()[:8],16)
            simulated=simulate_game(game,by_team,histories,league,rosters,
                                    role_source=roles,sims=sims,seed=seed)
            if (datetime.now(timezone.utc) if now is None else at) >= kickoff:
                results["window_missed"].append(game["game_id"])
                continue
            # Preserve the real numeric production forecast at this pretip
            # observation, not a reconstructed after-the-game baseline.
            from nfl_v7_incumbent_snapshot import capture_incumbent
            incumbent = capture_incumbent(
                Path(__file__).resolve().parent/"docs"/"nfl_predictions.json",
                game, observed_at_utc=max(
                    r["observed_at_utc"] for r in rosters.values()))
            receipt={"schema":VERSION,"status":"RESEARCH_ONLY_NOT_PROMOTED",
                "game_id":game["game_id"],"kickoff_utc":game["kickoff_utc"],
                "capture_observed_at_utc":max(x["observed_at_utc"] for x in rosters.values()),
                "source_cutoff_utc":cutoff.isoformat(),
                "roster_sha256":raw_sha,"roster_receipts":{t:{
                    "source":r["source"],"items":r["items"],
                    "observed_at_utc":r["observed_at_utc"],
                    "no_game_day_active_certification":True} for t,r in rosters.items()},
                "model_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "snap_source_sha256":{Path(p).name:hashlib.sha256(
                    Path(p).read_bytes()).hexdigest() for p in snap_files},
                "model":simulated,"frozen_live_incumbent":incumbent,
                "official_bets":0,
                "conditional_player_predictions_not_calibrated":True,
                "comparison_required":"SAME_GAME_SAME_PLAYER_FROZEN_V6B_AND_SERVING_ORIGINAL"}
            try:
                with file.open("x") as fp:
                    json.dump(receipt,fp,indent=2)
                    fp.write("\n")
            except FileExistsError:
                results["already_frozen"].append(game["game_id"])
            else:
                results["captured"].append(game["game_id"])
    finally:
        conn.close()
    return results


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--db",default="nfl_models/nfl_model.sqlite")
    p.add_argument("--snap-files",nargs="+",required=True)
    p.add_argument("--out-dir",default="docs/nfl_v7_firstseen")
    p.add_argument("--simulations",type=int,default=20000)
    args=p.parse_args()
    print("NFL_V7_PREGAME_CAPTURE",json.dumps(capture(
        args.db,args.snap_files,args.out_dir,sims=args.simulations)))
