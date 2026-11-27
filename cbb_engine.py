#!/usr/bin/env python3
"""CBB team/game intelligence v0: pregame-only, source-backed research.

This is NOT a promoted betting model. Two explicitly different forecasts are
tested: a simple recent-score baseline and an opponent-adjusted, pace + four
factors challenger. The engine cannot fabricate starting lineups/injuries.
Historical grading is day-by-day; games under evaluation never enter features.
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

SCHEMA = "CBB_PREGAME_INTELLIGENCE_RESEARCH_V0"
BASE_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
RELEASES = {
    "team": "espn_mens_college_basketball_team_boxscores/team_box_{}.parquet",
    "player": "espn_mens_college_basketball_player_boxscores/player_box_{}.parquet",
    "schedule": "espn_mens_college_basketball_schedules/mbb_schedule_{}.parquet",
}
PRIOR_GAMES = 2.0
REGULARIZATION_GAMES = 6.0
MIN_OBSERVATIONS = 3


def utc(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError("timestamps must be timezone aware")
    return t.tz_convert("UTC").to_pydatetime()


def finite(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def bounded(x, lo, hi):
    return max(lo, min(hi, x))


def json_ready(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (datetime, pd.Timestamp)): return utc(x).isoformat()
    if isinstance(x, dict): return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [json_ready(v) for v in x]
    if x is None or (not isinstance(x, (list, dict, tuple)) and pd.isna(x)): return None
    return x


def data_url(kind, season):
    return BASE_URL + RELEASES[kind].format(int(season))


def load_data(cache, seasons, include_player=False):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    output = {}
    needed = {"team": seasons, "schedule": seasons}
    if include_player: needed["player"] = [max(seasons)]
    for kind, years in needed.items():
        frames = []
        for season in years:
            path = cache / f"{kind}_{season}.parquet"
            if not path.exists() or path.stat().st_size < 100:
                import urllib.request
                request = urllib.request.Request(data_url(kind, season),
                                                 headers={"User-Agent": "CBB-Research/0.1"})
                with urllib.request.urlopen(request, timeout=90) as req:
                    payload = req.read()
                if len(payload) < 100:
                    raise RuntimeError(f"empty source {kind}/{season}")
                path.write_bytes(payload)
            df = pd.read_parquet(path)
            if df.empty: raise ValueError(f"empty {kind} dataset for {season}")
            df["source_season"] = int(season)
            frames.append(df)
        output[kind] = pd.concat(frames, ignore_index=True)
    return output


def possession(row):
    fga = finite(row.get("field_goals_attempted"))
    orb = finite(row.get("offensive_rebounds"))
    tov = finite(row.get("total_turnovers"))
    fta = finite(row.get("free_throws_attempted"))
    if any(x is None for x in (fga, orb, tov, fta)): return None
    n = fga - orb + tov + .475 * fta
    return n if 35 <= n <= 120 else None


def factors(row, opponent):
    fga = finite(row.get("field_goals_attempted"))
    fgm = finite(row.get("field_goals_made"))
    three = finite(row.get("three_point_field_goals_made"))
    fta = finite(row.get("free_throws_attempted"))
    ftm = finite(row.get("free_throws_made"))
    orb = finite(row.get("offensive_rebounds"))
    opp_drb = finite(opponent.get("defensive_rebounds"))
    tov = finite(row.get("total_turnovers"))
    if None in (fga, fgm, three, fta, ftm, orb, opp_drb, tov) or fga < 20:
        return None
    poss = possession(row)
    if poss is None: return None
    return {
        "efg": bounded((fgm + .5 * three) / fga, 0, .9),
        "tov": bounded(tov / max(fga + .475 * fta + tov, 1), 0, .5),
        "orb": bounded(orb / max(orb + opp_drb, 1), 0, .7),
        "ftr": bounded(fta / fga, 0, 1),
        "ftpct": bounded(ftm / max(fta, 1), .4, 1) if fta > 0 else .72,
        "three_rate": bounded(finite(row.get("three_point_field_goals_attempted"), 0) / fga, 0, .8),
    }


def games_from_frames(teams, schedule):
    """Only 2-sided completed games; fail closed on duplicates and nonsensical box."""
    sched = {}
    for _, row in schedule.iterrows():
        gid = str(row.get("game_id", row.get("id")))
        if gid in sched: continue
        sched[gid] = row.to_dict()
    games = []
    for gid, df in teams.groupby("game_id", sort=False):
        if len(df) != 2: continue
        r0, r1 = [r.to_dict() for _, r in df.iterrows()]
        if str(r0["team_id"]) != str(r1["opponent_team_id"]) or str(r1["team_id"]) != str(r0["opponent_team_id"]):
            continue
        q = sched.get(str(gid), {})
        if not q or q.get("status_type_completed") is not True:
            continue
        try: start = utc(q.get("game_date_time", q.get("date")))
        except Exception: continue
        if r0.get("team_home_away") == "home": home,away = r0,r1
        elif r1.get("team_home_away") == "home": home,away = r1,r0
        else: continue
        poss_h,poss_a = possession(home),possession(away)
        facts_h,facts_a = factors(home, away),factors(away,home)
        if None in (poss_h,poss_a,facts_h,facts_a): continue
        hp, ap = finite(home["team_score"]),finite(away["team_score"])
        if hp is None or ap is None: continue
        poss = (poss_h + poss_a)/2
        if not 40 <= poss <= 115: continue
        season = int(q.get("season", home.get("season", 0)))
        if season <= 0: continue
        games.append({
            "game_id": str(gid), "season": season, "start": start,
            "home": str(home["team_id"]), "away": str(away["team_id"]),
            "home_name": str(home.get("team_display_name")),
            "away_name": str(away.get("team_display_name")),
            "neutral": bool(q.get("neutral_site") is True),
            "home_pts": hp, "away_pts": ap, "poss": poss,
            "teams": {
                str(home["team_id"]): {"off": hp / poss*100, "def": ap / poss*100,
                                      "points": hp, "allowed": ap, "pace": poss,
                                      "f": facts_h, "opp": str(away["team_id"])},
                str(away["team_id"]): {"off": ap / poss*100, "def": hp / poss*100,
                                      "points": ap, "allowed": hp, "pace": poss,
                                      "f": facts_a, "opp": str(home["team_id"])},
            },
        })
    games.sort(key=lambda g:(g["start"],g["game_id"]))
    return games


def future_games(schedule, asof, days=3):
    asof=utc(asof)
    end=asof+timedelta(days=days)
    result=[]
    seen=set()
    for _,r in schedule.iterrows():
        try: start=utc(r["game_date_time"])
        except Exception: continue
        gid=str(r.get("game_id",r.get("id")))
        if gid in seen or not asof < start <= end or r.get("status_type_completed") is True:
            continue
        home,away=finite(r.get("home_id")),finite(r.get("away_id"))
        if home is None or away is None or home == away: continue
        seen.add(gid)
        result.append({
            "game_id":gid, "start":start, "season":int(r["season"]),
            "home":str(int(home)), "away":str(int(away)),
            "home_name":str(r.get("home_display_name",r.get("home_name"))),
            "away_name":str(r.get("away_display_name",r.get("away_name"))),
            "neutral":bool(r.get("neutral_site") is True),
            "status":str(r.get("status_type_state")),
        })
    return sorted(result,key=lambda x:(x["start"],x["game_id"]))


class Tracker:
    """Team + opponent ledgers; no future score can be accessed in predict()."""
    def __init__(self, season, previous=None):
        self.season=int(season)
        self.records=defaultdict(list)
        self.prior=defaultdict(list)
        for g in previous or []:
            if g["season"] < season:
                for team,rec in g["teams"].items():
                    self.prior[team].append(rec)
        self.latest=None
        self.count=0

    def add(self, game):
        if game["season"] != self.season: return
        if self.latest is not None and game["start"] < self.latest:
            raise ValueError("out-of-order game would leak future data")
        for team, rec in game["teams"].items():
            self.records[team].append(rec)
        self.count+=1
        self.latest=game["start"]

    def _history(self, team):
        current=self.records.get(team,[])
        last=current[-12:]
        # Previous roster is a very weak prior, vanishing after 12 current games.
        past=self.prior.get(team,[])[-12:]
        prior_w = PRIOR_GAMES * max(0, 1-len(current)/12)
        out=[]
        if past and prior_w:
            w=prior_w/len(past)
            out.extend((r,w) for r in past)
        out.extend((r,.87**(len(last)-i-1)) for i,r in enumerate(last))
        return out

    def ratings(self):
        weighted={t:self._history(t) for t in set(self.records)|set(self.prior)}
        points=[]
        for t,recs in weighted.items():
            for r,w in recs:
                if r in self.records.get(t,[]): points.append(r["off"])
        league=float(np.median(points)) if points else 100.
        if not 65 <= league <= 155:league=100.
        raw={}
        for t,recs in weighted.items():
            n=sum(w for _,w in recs)
            if not n: continue
            vals={}
            for k in ("off","def","pace","points","allowed"):
                baseline=league if k in ("off","def") else 69 if k=="pace" else 70
                pseudo=REGULARIZATION_GAMES if k in ("off","def") else 3
                vals[k]=(sum(w*r[k] for r,w in recs)+pseudo*baseline)/(n+pseudo)
            for k,base in (("efg",.50),("tov",.17),("orb",.29),("ftr",.32),("ftpct",.72),("three_rate",.36)):
                vals[k]=(sum(w*r["f"][k] for r,w in recs)+3*base)/(n+3)
            vals["n"]=len(self.records.get(t,[]))
            vals["opps"]=[(r["opp"],w) for r,w in recs]
            raw[t]=vals
        # Two synchronous opponent-strength adjustment passes, anchored to league.
        offense={t:v["off"] for t,v in raw.items()}
        defense={t:v["def"] for t,v in raw.items()}
        for _ in range(2):
            new_o={};new_d={}
            for t,v in raw.items():
                pairs=v["opps"]
                z=sum(w for _,w in pairs)
                opp_def=sum(w*(defense.get(opp,league)-league) for opp,w in pairs)/z if z else 0
                opp_off=sum(w*(offense.get(opp,league)-league) for opp,w in pairs)/z if z else 0
                new_o[t]=bounded(v["off"]-.65*opp_def,55,150)
                new_d[t]=bounded(v["def"]-.65*opp_off,55,150)
            offense,defense=new_o,new_d
        for t in raw:
            raw[t]["adj_off"]=offense[t]
            raw[t]["adj_def"]=defense[t]
        return raw,league

    def predict(self, game):
        h,a=game["home"],game["away"]
        if h==a:raise ValueError("duplicate teams")
        ratings,lg=self.ratings()
        H,A=ratings.get(h),ratings.get(a)
        if H is None or A is None or H["n"] < MIN_OBSERVATIONS or A["n"] < MIN_OBSERVATIONS:
            return None
        # Actual game pace and four factors cannot appear anywhere here.
        neutral=game.get("neutral",False)
        home_edge=0 if neutral else 2.4
        pace=bounded(.5*(H["pace"]+A["pace"]),55,82)
        offh=(H["adj_off"]+A["adj_def"])/2
        offa=(A["adj_off"]+H["adj_def"])/2
        efficiency_home=pace*offh/100+home_edge/2
        efficiency_away=pace*offa/100-home_edge/2

        def matchup(off,defn):
            # Defended box-score factors are inferred via opponent's points
            # allowed, not opponent-specific shooting by shot location.
            efg=.5*(off["efg"]+(1-(defn["adj_def"]-70)/220)*.5)
            efg=bounded(efg,.39,.67)
            tov=bounded(off["tov"]+.08*(defn["adj_def"]-lg)/lg,.10,.25)
            orb=bounded(off["orb"],.18,.42)
            ftr=bounded(off["ftr"],.17,.52)
            ft=bounded(off["ftpct"],.58,.85)
            fga=pace*(1-tov)/(1-orb*(1-efg)+.475*ftr)
            return fga*(2*efg+ftr*ft)
        # Four-factor mechanism is separately measurable; challenger can lose.
        mech_h=matchup(H,A)+home_edge/2
        mech_a=matchup(A,H)-home_edge/2
        cand_h=.8*efficiency_home+.2*mech_h
        cand_a=.8*efficiency_away+.2*mech_a

        # Last/current scoring baselines only, no market lines or results.
        base_h=(H["points"]+A["allowed"])/2+home_edge/2
        base_a=(A["points"]+H["allowed"])/2-home_edge/2
        delta=cand_h-cand_a
        p=0.5*(1+math.erf(delta/(12.0*math.sqrt(2))))
        p=bounded(p,.01,.99)
        return {
            "game_id":game["game_id"], "start_utc":utc(game["start"]).isoformat(),
            "season":game["season"],"home":h,"away":a,
            "home_name":game.get("home_name"),"away_name":game.get("away_name"),
            "neutral_site":neutral,"observed_team_games":{h:H["n"],a:A["n"]},
            "baseline":{"home_points":round(base_h,2),"away_points":round(base_a,2),
                        "margin":round(base_h-base_a,2),"total":round(base_h+base_a,2)},
            "challenger":{"home_points":round(cand_h,2),"away_points":round(cand_a,2),
                          "margin":round(delta,2),"total":round(cand_h+cand_a,2),
                          "p_home_win_UNCALIBRATED":round(p,4),
                          "model_version":"cbb_opponent_adjusted_pace_four_factors_v0"},
            "evidence":{
                "modeled_possessions":round(pace,2),
                "offense_per100":{h:round(H["adj_off"],2),a:round(A["adj_off"],2)},
                "defense_allowed_per100":{h:round(H["adj_def"],2),a:round(A["adj_def"],2)},
                "efg_observed":{h:round(H["efg"],3),a:round(A["efg"],3)},
                "turnover_rate_observed":{h:round(H["tov"],3),a:round(A["tov"],3)},
                "off_rebound_rate_observed":{h:round(H["orb"],3),a:round(A["orb"],3)},
                "three_attempt_rate_observed":{h:round(H["three_rate"],3),a:round(A["three_rate"],3)},
                "source":"previous completed game box scores only",
                "injury_availability":"UNKNOWN_NOT_VERIFIED",
                "starting_lineups":"UNKNOWN_NOT_VERIFIED",
            },
            "published":False,"calibrated":False,"bet_eligible":False,
        }


def player_context(player_box, team, asof, max_rows=8):
    """Player-role intelligence with real recorded minutes, not inferred starts."""
    if player_box is None or player_box.empty:return []
    games=player_box[(player_box["team_id"].astype(str)==str(team)) &
                     (player_box["game_date_time"] < pd.Timestamp(asof))]
    if games.empty: return []
    out=[]
    for pid,group in games.groupby("athlete_id"):
        group=group.sort_values("game_date_time").tail(max_rows)
        played=group[(group["did_not_play"]!=True) & (group["minutes"].fillna(0)>0)]
        if len(played)<2:continue
        mins=played["minutes"].astype(float).to_numpy()
        last=min(3,len(mins))
        before=mins[-6:-3] if len(mins)>=6 else np.array([])
        recent=mins[-last:]
        use=((played["field_goals_attempted"].fillna(0)+
              .44*played["free_throws_attempted"].fillna(0)+
              played["turnovers"].fillna(0)).astype(float).to_numpy())
        out.append({
            "player":str(played.iloc[-1].get("athlete_display_name","")),
            "athlete_id":str(pid),"games_played_in_sample":len(played),
            "recent_minutes":round(float(recent.mean()),1),
            "previous_three_minutes":round(float(before.mean()),1) if len(before)==3 else None,
            "minutes_change":round(float(recent.mean()-before.mean()),1) if len(before)==3 else None,
            "recent_box_usage_events":round(float(use[-last:].mean()),1),
            "verified_starting_lineup":False,
            "injury_status":"UNKNOWN",
        })
    return sorted(out,key=lambda x:-x["recent_minutes"])[:8]


def historical_test(games, season, cutoff, last_day=None):
    """Chronological test: never see current day's outcomes before all predictions."""
    prior=[g for g in games if g["season"]<season]
    pool=[g for g in games if g["season"]==season]
    t=Tracker(season,prior)
    grouped=defaultdict(list)
    for g in pool: grouped[g["start"].date()].append(g)
    rows=[]
    for date in sorted(grouped):
        day=grouped[date]
        # Freeze all forecasts for a calendar day BEFORE any of its games.
        eligible= date >= cutoff and (last_day is None or date <= last_day)
        if eligible:
            for g in day:
                p=t.predict(g)
                if p is None:continue
                actual_margin=g["home_pts"]-g["away_pts"]
                actual_total=g["home_pts"]+g["away_pts"]
                rows.append({
                    "game_id":g["game_id"],"date":str(date),
                    "home":g["home"],"away":g["away"],
                    "actual_margin":actual_margin,"actual_total":actual_total,
                    "base_margin":p["baseline"]["margin"],"base_total":p["baseline"]["total"],
                    "chall_margin":p["challenger"]["margin"],"chall_total":p["challenger"]["total"],
                    "p_home":p["challenger"]["p_home_win_UNCALIBRATED"],
                    "actual_home_win":int(actual_margin>0),
                })
        for g in day: t.add(g)
    if not rows:return {"n":0,"status":"INSUFFICIENT_RESULTS","rows":[]}
    d=pd.DataFrame(rows)
    def mae(k,actual):return round(float(np.abs(d[k]-d[actual]).mean()),4)
    brier=round(float(np.mean((d["p_home"]-d["actual_home_win"])**2)),5)
    # A simple baseline probability is deliberately not claimed calibrated.
    def normal_p(v):
        return bounded(.5*(1+math.erf(v/(12*math.sqrt(2)))), .01,.99)
    base_brier=float(np.mean([(normal_p(x)-y)**2 for x,y in zip(d["base_margin"],d["actual_home_win"])]))
    return {
        "n":len(d),"status":"RESEARCH_ONLY_NOT_PROMOTED",
        "cutoff":str(cutoff),"last_day":str(last_day) if last_day else None,
        "baseline":{"margin_mae":mae("base_margin","actual_margin"),
                    "total_mae":mae("base_total","actual_total"),
                    "home_win_brier_UNCALIBRATED":round(base_brier,5)},
        "challenger":{"margin_mae":mae("chall_margin","actual_margin"),
                      "total_mae":mae("chall_total","actual_total"),
                      "home_win_brier_UNCALIBRATED":brier},
        "paired_margin_delta_baseline_minus_challenger":round(
            float((np.abs(d["base_margin"]-d["actual_margin"])-
                   np.abs(d["chall_margin"]-d["actual_margin"])).mean()),4),
        "paired_total_delta_baseline_minus_challenger":round(
            float((np.abs(d["base_total"]-d["actual_total"])-
                   np.abs(d["chall_total"]-d["actual_total"])).mean()),4),
        "rows":json_ready(rows),
    }


def make_slate(games,schedule,players,asof,days=3):
    asof=utc(asof)
    season=max(int(g["season"]) for g in games)
    track=Tracker(season,[g for g in games if g["season"]<season])
    for g in games:
        if g["season"]==season and g["start"]<asof:
            track.add(g)
    cards=[]
    for g in future_games(schedule[schedule["season"]==season],asof,days=days):
        p=track.predict(g)
        if p is None: continue
        p["rotation_from_last_completed_games"]={
            g["home"]:player_context(players,g["home"],asof),
            g["away"]:player_context(players,g["away"],asof),
        }
        cards.append(p)
    return {"schema":SCHEMA,"generated_at_utc":asof.isoformat(),
            "source_seasons":[season-1,season],
            "model":"RESEARCH_ONLY_NO_BOOKMAKER_AND_NO_LIVE_PUBLISH",
            "probability_status":"UNVALIDATED_NORMAL_APPROXIMATION",
            "n_upcoming_scheduled":len(future_games(schedule[schedule["season"]==season],asof,days=days)),
            "n_with_sufficient_team_data":len(cards),
            "games":cards}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",default="/tmp/cbb_data")
    ap.add_argument("--out",default="/tmp/cbb_v0")
    ap.add_argument("--asof",default=None)
    ap.add_argument("--days",type=int,default=3)
    args=ap.parse_args()
    asof=utc(args.asof or datetime.now(timezone.utc))
    season=asof.year+1 if asof.month>=7 else asof.year
    # Previous year supplies weak carryover; 2026 is untouched development season
    # for the preliminary 2027 model evaluation.
    loaded=load_data(args.cache,[season-2,season-1,season],include_player=True)
    games=games_from_frames(loaded["team"],loaded["schedule"])
    if not games:raise RuntimeError("NO_VALID_COMPLETED_CBB_GAMES")
    current=[g for g in games if g["season"]==season]
    if not current:raise RuntimeError("CURRENT_SEASON_TEAM_BOXSCORES_MISSING")
    completed=[g for g in games if g["start"]<asof]
    # Historical evaluation is on 2025-26 (2026 season), before current 2026-27.
    test=historical_test(completed,season-1,cutoff=datetime(season-2,1,15).date(),
                         last_day=datetime(season-2,3,31).date())
    slate=make_slate(completed,loaded["schedule"],loaded["player"],asof,args.days)
    directory=Path(args.out);directory.mkdir(parents=True,exist_ok=True)
    (directory/"backtest.json").write_text(json.dumps(json_ready(test),indent=2,allow_nan=False)+"\n")
    (directory/"slate.json").write_text(json.dumps(json_ready(slate),indent=2,allow_nan=False)+"\n")
    manifest={"schema":SCHEMA,"run_at":asof.isoformat(),
              "data_sources":{k:[data_url(k,s) for s in ([season] if k=="player" else [season-2,season-1,season])]
                              for k in ("team","player","schedule")},
              "completed_game_rows":len(completed),
              "current_season_game_rows":len(current),
              "backtest_n":test["n"],
              "slate_games":slate["n_with_sufficient_team_data"],
              "no_live_picks_changed":True,
              "source_arrival_not_historically_frozen":True}
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"backtest_n":test["n"],
                      "baseline":test.get("baseline"),
                      "challenger":test.get("challenger"),
                      "slate_count":len(slate["games"]),
                      "completed":len(completed)},sort_keys=True))


if __name__=="__main__":
    main()
