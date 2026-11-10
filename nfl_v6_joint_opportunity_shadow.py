#!/usr/bin/env python3
"""NFL V6 shadow: as-of, coupled team/role simulation. NOT the live champion.

This deliberately does not use sportsbook totals, player prop lines, in-game
boxscores, article projections, or late game-day roster snapshots. Candidate
players are last-observed participants, NOT confirmed active players. Results
are conditional research projections, never certified bets.

Distinct from the serving bootstrap: team plays and pass mix are generated
once per game draw; all players compete for the same finite attempts/targets.
Opponent defensive performance and QB effectiveness are estimated exclusively
from games finished before the as-of cutoff. A 24-hour stats-availability lag
prevents using results that could not yet have been published at forecast time.

Every numeric prediction and probability is computed from exactly the same
joint draws. Do not promote until prospectively compared with the frozen
serving incumbent on identical player-game rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

VERSION = "NFL_V6_JOINT_OPPORTUNITY_SHADOW_A"
SNAP_VERSION = "NFL_V6B_SNAP_ROLE_SHADOW_A"
PERSONNEL_VERSION = "NFL_V6C_PERSONNEL_CONDITIONED_SHADOW_A"
N = 20000
LAST = 8
LAG_HOURS = 24
SEED = 827136
MARKETS = ("receiving_yards", "rushing_yards", "passing_yards")


def utc(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timezone is required")
    return dt.astimezone(timezone.utc)


def asof_rows(conn, cutoff):
    available_before = (utc(cutoff) - timedelta(hours=LAG_HOURS)).isoformat()
    sql = """SELECT p.player_id,p.player_name,p.position,p.team,p.opponent,
      p.game_id,p.season,p.week,COALESCE(p.targets,0),COALESCE(p.receptions,0),
      COALESCE(p.receiving_yards,0),COALESCE(p.carries,0),
      COALESCE(p.rushing_yards,0),COALESCE(p.attempts,0),
      COALESCE(p.passing_yards,0),g.kickoff_utc
      FROM player_games p INNER JOIN games g ON p.game_id=g.game_id
      WHERE g.kickoff_utc < ? AND g.home_score IS NOT NULL
        AND g.away_score IS NOT NULL AND p.season_type='REG'
      ORDER BY g.kickoff_utc,p.game_id,p.team,p.player_id"""
    out = []
    for x in conn.execute(sql, (available_before,)):
        out.append(dict(zip(("id","name","pos","team","opp","game_id","season",
          "week","targets","receptions","rec_yards","carries","rush_yards",
          "attempts","pass_yards","kickoff"),x)))
    return out


def game_history(rows):
    games = {}
    player = defaultdict(list)
    for r in rows:
        key = (r["game_id"],r["team"])
        if key not in games:
            games[key] = {"team":r["team"],"opp":r["opp"],"kickoff":r["kickoff"],
              "passes":0,"rushes":0,"pass_yards":0,"rush_yards":0,
              "targets":0,"rows":[]}
        t = games[key]
        t["passes"] += r["attempts"]
        t["rushes"] += r["carries"]
        t["pass_yards"] += r["pass_yards"]
        t["rush_yards"] += r["rush_yards"]
        t["targets"] += r["targets"]
        t["rows"].append(r)
        player[(r["id"],r["team"])].append(r)
    by_team = defaultdict(list)
    for t in games.values():
        if 20 <= t["passes"]+t["rushes"] <= 105:
            by_team[t["team"]].append(t)
    for t in by_team.values():
        t.sort(key=lambda x:x["kickoff"])
    return by_team, player


def weighted(seq, field, default=0):
    if not seq:
        return float(default)
    vals = [(float(z[field]),0.78**i) for i,z in enumerate(reversed(seq[-LAST:]))]
    return sum(x*w for x,w in vals)/sum(w for _,w in vals)


def clip(value, low, high):
    return float(max(low,min(high,value)))


def league_rates(by_team):
    games = [g for hist in by_team.values() for g in hist]
    passing = sum(g["passes"] for g in games)
    rushing = sum(g["rushes"] for g in games)
    if passing < 50 or rushing < 50:
        raise ValueError("insufficient as-of league attempts")
    return {"pass_ypa":sum(g["pass_yards"] for g in games)/passing,
      "rush_ypc":sum(g["rush_yards"] for g in games)/rushing,
      "rec_ypt":sum(sum(p["rec_yards"] for p in g["rows"]) for g in games)/
                max(1,sum(g["targets"] for g in games)),
      "team_plays":sum(g["passes"]+g["rushes"] for g in games)/len(games),
      "pass_rate":passing/(passing+rushing)}


def context(team, opp, by_team, league):
    own = by_team.get(team,[])[-LAST:]
    faced = [g for hist in by_team.values() for g in hist if g["opp"]==opp][-LAST:]
    faced.sort(key=lambda x:x["kickoff"])
    own_p = weighted(own,"passes",league["team_plays"]*league["pass_rate"])
    own_r = weighted(own,"rushes",league["team_plays"]*(1-league["pass_rate"]))
    def_p = weighted(faced,"passes",own_p)
    def_r = weighted(faced,"rushes",own_r)
    # Bayesian shrinkage of the opponent-adjusted pace toward team history.
    base_plays = .7*(own_p+own_r)+.3*(def_p+def_r)
    pass_rate = clip((own_p+.33*def_p+4*league["pass_rate"])/
                     max(1,own_p+own_r+.33*(def_p+def_r)+4),.28,.79)
    faced_pass = sum(g["passes"] for g in faced)
    faced_rush = sum(g["rushes"] for g in faced)
    allowed_pass = (sum(g["pass_yards"] for g in faced)+50*league["pass_ypa"])/max(1,faced_pass+50)
    allowed_rush = (sum(g["rush_yards"] for g in faced)+50*league["rush_ypc"])/max(1,faced_rush+50)
    return {"team_plays":clip(base_plays,35,82),"pass_rate":pass_rate,
      "opp_pass_factor":clip(allowed_pass/max(.1,league["pass_ypa"]),.8,1.2),
      "opp_rush_factor":clip(allowed_rush/max(.1,league["rush_ypc"]),.8,1.2),
      "own_games":len(own),"opponent_games":len(faced)}


def participants(team, team_games, histories, league):
    last = team_games[-2:]
    if not last:
        return []
    # Only observations known before cutoff; no future weekly roster labels.
    present = {(r["id"],r["team"]) for g in last for r in g["rows"] if r["team"]==team}
    out=[]
    for pid,_ in sorted(present):
        hist=histories[(pid,team)][-LAST:]
        if not hist:
            continue
        recent=hist[-1]
        opps=sum(g["targets"]+g["carries"]+g["attempts"] for g in hist)
        if opps==0:
            continue
        target=sum(g["targets"] for g in hist)
        carry=sum(g["carries"] for g in hist)
        attempts=sum(g["attempts"] for g in hist)
        out.append({"id":pid,"player":recent["name"],"position":recent["pos"],
          "team":team,"n_games":len(hist),"last_seen":recent["kickoff"],
          "targets":target,"carries":carry,"attempts":attempts,
          "ypt":(sum(g["rec_yards"] for g in hist)+8*league["rec_ypt"])/(target+8),
          "ypc":(sum(g["rush_yards"] for g in hist)+12*league["rush_ypc"])/(carry+12),
          "ypa":(sum(g["pass_yards"] for g in hist)+16*league["pass_ypa"])/(attempts+16),
          "role_unverified":True})
    return out


def distribution(samples, opportunities):
    mean=float(np.mean(samples))
    # An accuracy event centered on the reported number, not an optimized
    # low target. Uncalibrated and to be compared with future truth.
    target=int(math.floor(mean+0.5))
    p_exact=float(np.mean(samples>=target))
    return {"projected_mean":round(mean,1),
      "projected_median":round(float(np.median(samples)),1),
      "p10":round(float(np.quantile(samples,.10)),1),
      "p90":round(float(np.quantile(samples,.90)),1),
      "rounded_mean":target,
      "prob_reach_rounded_mean":round(p_exact,4),
      "prob_within_15_yards_of_rounded_mean":round(float(np.mean(
          np.abs(samples-target)<=15)),4),
      "modeled_opportunities_mean":round(float(np.mean(opportunities)),2),
      "simulations":len(samples),
      "is_forward_calibrated":False,
      "active_status_verified":False}


def simulate_matchup(game, by_team, histories, league, sims=N, seed=SEED, role_source=None, personnel=None):
    if sims<10000:
        raise ValueError("at least ten thousand joint-game simulations required")
    home,away=game["home_team"],game["away_team"]
    if home==away:
        raise ValueError("duplicate teams")
    rng=np.random.default_rng(seed)
    # Shared pace innovation couples teams across the same simulated game.
    common=rng.lognormal(-.5*.12**2,.12,sims)
    outputs=[]
    team_summaries={}
    for team,opp in ((home,away),(away,home)):
        ctx=context(team,opp,by_team,league)
        people=participants(team,by_team.get(team,[]),histories,league)
        # Source-asserted OUT players no longer consume expected targets/carries.
        # Unasserted does not mean available: everyone remains role-unverified.
        team_personnel=(personnel["teams"].get(team) if personnel else None)
        removed=[]
        if team_personnel:
            out_ids=team_personnel["inactive_ids"]
            removed=[p["player"] for p in people if str(p["id"]) in out_ids]
            people=[p for p in people if str(p["id"]) not in out_ids]
        if ctx["own_games"]<3 or ctx["opponent_games"]<3 or not people:
            team_summaries[team]={"status":"INSUFFICIENT_PRIOR_SOURCE",
              "context":ctx,"players":len(people)}
            continue
        total_plays=np.clip(np.rint(ctx["team_plays"]*common*
          rng.lognormal(-.5*.08**2,.08,sims)).astype(int),22,104)
        rates=rng.beta(max(1,ctx["pass_rate"]*65),
                       max(1,(1-ctx["pass_rate"])*65),sims)
        passing=rng.binomial(total_plays,rates)
        rushing=total_plays-passing
        receivers=[p for p in people if p["targets"]>0]
        rushers=[p for p in people if p["carries"]>0]
        qb=sorted((p for p in people if p["attempts"]>=10),
                  key=lambda p:(p["last_seen"],p["attempts"]),reverse=True)
        if personnel is None:
            selected_qb=qb[0] if qb else None
            qb_role_status="HISTORICAL_HEURISTIC_NOT_VERIFIED"
        else:
            asserted=(team_personnel or {}).get("starter_id")
            matched=[p for p in qb if str(p["id"])==asserted] if asserted else []
            selected_qb=matched[0] if len(matched)==1 else None
            qb_role_status=("SOURCE_ASSERTED_STARTER_MATCHED_IN_HISTORY" if selected_qb
                else "SOURCE_ASSERTED_STARTER_NOT_IN_CANDIDATE_HISTORY" if asserted
                else "NO_SOURCE_ASSERTED_STARTER")
        # Past target share cannot exceed total current QB attempts; when a
        # feed is incomplete, use an explicit non-player remainder.
        t_fraction=clip(sum(g["targets"] for g in by_team[team][-LAST:])/
             max(1,sum(g["passes"] for g in by_team[team][-LAST:])),.5,1.)
        role_priors=None
        if role_source is None:
            # Sealed V6A empirical-opportunity branch. Keep the original
            # historical grade comparable; no source information added.
            t_alpha=np.array([p["targets"]+.5 for p in receivers]+[5.],float)
            r_alpha=np.array([p["carries"]+.5 for p in rushers]+[4.],float)
        else:
            from nfl_v6_snap_role_evidence import allocate_role_priors
            role_priors=allocate_role_priors(people,histories,team,role_source,
                 ctx["team_plays"],
                 ctx["team_plays"]*ctx["pass_rate"]*t_fraction,
                 ctx["team_plays"]*(1-ctx["pass_rate"]))
            ix={p["id"]:i for i,p in enumerate(people)}
            t_alpha=np.array([role_priors["target_alpha"][ix[p["id"]]]
                 for p in receivers]+[role_priors["target_alpha"][-1]],float)
            r_alpha=np.array([role_priors["carry_alpha"][ix[p["id"]]]
                 for p in rushers]+[role_priors["carry_alpha"][-1]],float)
        total_targets=rng.binomial(passing,t_fraction)
        target_alloc=np.zeros((sims,len(t_alpha)),dtype=np.int16)
        carry_alloc=np.zeros((sims,len(r_alpha)),dtype=np.int16)
        for j in range(sims):
            target_alloc[j]=rng.multinomial(int(total_targets[j]),rng.dirichlet(t_alpha))
            carry_alloc[j]=rng.multinomial(int(rushing[j]),rng.dirichlet(r_alpha))
        pass_yard_total=np.zeros(sims,float)
        qb_factor=clip((selected_qb["ypa"]/max(.1,league["pass_ypa"])) if selected_qb else 1.,.75,1.25)
        for i,p in enumerate(receivers):
            opp_count=target_alloc[:,i]
            mean_per=clip(p["ypt"]*ctx["opp_pass_factor"]*qb_factor,1,22)
            # Gamma compounding has genuine target-level zero probability,
            # and the player's target count is shared across the team.
            yards=rng.gamma(np.maximum(.01,opp_count*1.25),
                            mean_per/1.25).astype(float)
            yards[opp_count==0]=0
            pass_yard_total+=yards
            outputs.append({"player_id":p["id"],"player":p["player"],
              "team":team,"opponent":opp,"market":"receiving_yards",
              "probability_event":"P_YARDS_AT_LEAST_ROUNDED_MODEL_MEAN",
              "naive_recent3_mean":round(float(np.mean([g["rec_yards"] for g in
                histories[(p["id"],team)][-3:]])),1),
              **({"snap_role_source_status":
                role_priors["players"][p["id"]]["role"]["status"]}
                if role_priors else {}),
              **({"qb_role_status":qb_role_status}
                 if personnel is not None else {}),
              **distribution(yards,opp_count)})
        for i,p in enumerate(rushers):
            opp_count=carry_alloc[:,i]
            mean_per=clip(p["ypc"]*ctx["opp_rush_factor"],1,10)
            yards=rng.gamma(np.maximum(.01,opp_count*1.8),
                            mean_per/1.8).astype(float)
            yards[opp_count==0]=0
            outputs.append({"player_id":p["id"],"player":p["player"],
              "team":team,"opponent":opp,"market":"rushing_yards",
              "probability_event":"P_YARDS_AT_LEAST_ROUNDED_MODEL_MEAN",
              "naive_recent3_mean":round(float(np.mean([g["rush_yards"] for g in
                histories[(p["id"],team)][-3:]])),1),
              **({"snap_role_source_status":
                role_priors["players"][p["id"]]["role"]["status"]}
                if role_priors else {}),
              **distribution(yards,opp_count)})
        # Keep QB passing yards consistent with team-wide receiver results;
        # incomplete/missing pass-catchers sit in the OTHER bucket.
        if selected_qb:
            others=target_alloc[:,-1]
            pass_yard_total+=rng.gamma(np.maximum(.01,others*1.25),
                 clip(league["rec_ypt"]*ctx["opp_pass_factor"]*qb_factor,1,22)/1.25) * (others>0)
            outputs.append({"player_id":selected_qb["id"],"player":selected_qb["player"],
              "team":team,"opponent":opp,"market":"passing_yards",
              "probability_event":"P_YARDS_AT_LEAST_ROUNDED_MODEL_MEAN",
              "starting_qb_verified":False,
              "naive_recent3_mean":round(float(np.mean([g["pass_yards"] for g in
                histories[(selected_qb["id"],team)][-3:]])),1),
              **distribution(pass_yard_total,passing)})
        team_summaries[team]={"status":"SIMULATED_ROLE_UNVERIFIED",
          "context":ctx,"players":len(people),
          **({"personnel_source_status":"EXACT_GAME_PREGAME_ASSERTIONS_ONLY",
              "qb_role_status":qb_role_status,
              "source_asserted_out_removed":removed,
              "source_asserted_out_count":len((team_personnel or {}).get("inactive_ids",())),
              "unknown_active_status_for_unasserted_players":True}
             if personnel is not None else {}),
          "projected_plays":round(float(total_plays.mean()),1),
          "projected_pass_attempts":round(float(passing.mean()),1),
          "projected_rush_attempts":round(float(rushing.mean()),1),
          "players_missing_from_target_allocation":True,
          **({"snap_source_role": {
              "matched_candidates":role_priors["observed_snap_role_players"],
              "total_candidates":role_priors["candidate_players"],
              "route_data_included":False,
              "active_status_verified":False}} if role_priors else {})}
    return {"game_id":game["game_id"],"home":home,"away":away,
      "kickoff_utc":game["kickoff_utc"],"simulations":sims,
      "teams":team_summaries,"projections":outputs}


def run(db,season,week,out,sims=N,snap_files=None,
        captured_at_utc=None,only_future=False,personnel_evidence=None):
    conn=sqlite3.connect("file:"+str(Path(db).absolute())+"?mode=ro",uri=True)
    conn.row_factory=sqlite3.Row
    games=[dict(x) for x in conn.execute(
      "SELECT game_id,home_team,away_team,kickoff_utc FROM games WHERE season=? AND week=?",
      (season,week))]
    if not games:
        raise ValueError("no authenticated schedule rows for target week")
    captured=utc(captured_at_utc) if captured_at_utc is not None else None
    if only_future and captured is None:
        raise ValueError("future-only publication requires original capture timestamp")
    if personnel_evidence is not None and (captured is None or not only_future):
        raise ValueError("V6C personnel evidence requires a real future-only snapshot capture")
    forecasts=[]
    snap_files=list(snap_files or ())
    role_source_receipts=[]
    personnel_receipts=[]
    for game in games:
        kickoff=utc(game["kickoff_utc"])
        if only_future and kickoff<=captured:
            continue
        # Hard gate: a forward-captured forecast may NEVER see a prior game
        # whose 24h availability lag expires after the actual capture.
        cutoff=kickoff-timedelta(hours=2)
        if captured is not None:
            cutoff=min(cutoff,captured)
        history=asof_rows(conn,cutoff)
        by_team,players=game_history(history)
        league=league_rates(by_team)
        seed=SEED+int(hashlib.sha256(game["game_id"].encode()).hexdigest()[:8],16)
        if snap_files:
            from nfl_v6_snap_role_evidence import read_snap_roles
            source=read_snap_roles(snap_files,db,cutoff)
            role_source_receipts.append({
                "game_id":game["game_id"],
                "source_player_game_rows":len(source["by_player_game"]),
                "snap_publication_asof_verified":False,
                "source_files":source["snapshot_sources"]})
        else:
            source=None
        personnel=None
        if personnel_evidence is not None:
            from nfl_v6c_pregame_personnel import read_personnel
            base=Path(personnel_evidence)
            snapshot_path=base/(game["game_id"]+".json") if base.is_dir() else base
            if snapshot_path.exists():
                personnel=read_personnel(snapshot_path,game,cutoff,
                        captured_at_utc=captured.isoformat())
                personnel_receipts.append({k:v for k,v in personnel.items()
                    if k!="teams"})
            else:
                # Absence of evidence must not trigger a fabricated QB starter.
                personnel={"teams":{t:{"starter_id":None,"inactive_ids":set()}
                     for t in (game["home_team"],game["away_team"])}}
                personnel_receipts.append({"game_id":game["game_id"],
                    "status":"NO_SOURCE_ASSERTIONS_AVAILABLE"})
        forecasts.append(simulate_matchup(game,by_team,players,league,sims,seed,
             role_source=source,personnel=personnel))
    conn.close()
    payload={"version":(PERSONNEL_VERSION if personnel_evidence is not None
        else SNAP_VERSION if snap_files else VERSION),
      "season":season,"week":week,
      "snapshot_captured_at_utc":captured.isoformat() if captured else None,
      "forward_source_cutoff_enforced":captured is not None,
      "only_future_game_snapshots":bool(only_future),
      "status":"RESEARCH_SHADOW_NOT_PROMOTED","game_count":len(forecasts),
      "asof_policy":("MIN(CAPTURE_TIME,KICKOFF_MINUS_2H)_24H_DATA_LAG"
                     if captured is not None else
                     "KICKOFF_MINUS_2H_24H_DATA_AVAILABILITY_LAG"),
      "not_forward_calibrated":True,"active_status_verified":False,
      "sportsbook_inputs":False,"sims_per_game":sims,
      "snap_role_evidence": bool(snap_files),
      "historical_snap_source_timestamps_asof_not_verified":bool(snap_files),
      "snap_role_source_receipts":role_source_receipts,
      "personnel_source_receipts":personnel_receipts,
      "source_assertions_not_automatically_verified":personnel_evidence is not None,
      "games":forecasts}
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,indent=2)+"\n")
    return payload


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",default="nfl_models/nfl_model.sqlite")
    ap.add_argument("--season",type=int,required=True)
    ap.add_argument("--week",type=int,required=True)
    ap.add_argument("--sims",type=int,default=N)
    ap.add_argument("--out",default="/tmp/nfl_v6_joint_opportunity_shadow.json")
    ap.add_argument("--snap-files",nargs="*",default=[],
        help="Optional verified nflverse snap_counts CSV files; creates separate V6B research version")
    ap.add_argument("--captured-at-utc",default=None,
        help="Real immutable forecast capture clock; forbids future-known history")
    ap.add_argument("--only-future",action="store_true",
        help="Include only unstarted games when a real capture clock was supplied")
    ap.add_argument("--personnel-evidence",default=None,
        help="Directory of exact game-id source assertions; research-only V6C, future-only")
    args=ap.parse_args()
    result=run(args.db,args.season,args.week,args.out,args.sims,
               snap_files=args.snap_files,
               captured_at_utc=args.captured_at_utc,
               only_future=args.only_future,
               personnel_evidence=args.personnel_evidence)
    print(json.dumps({"version":result["version"],"games":len(result["games"]),
      "player_projections":sum(len(g["projections"]) for g in result["games"])}))
