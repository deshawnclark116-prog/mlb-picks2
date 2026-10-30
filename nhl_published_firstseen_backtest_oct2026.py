#!/usr/bin/env python3
"""Independent actual-results audit of published NHL model picks, Oct 25-28 2026.

No sportsbook lines, hindsight re-fitting, or fabricated pregame snapshots.
Predictions come from the selected authentic Git commit FOR EACH DATE.
Actuals are official NHL schedule and gamecenter FINAL box scores, not model
output or a mutable predictions feed. Missing team/player/status is UNVERIFIED.
"""
import argparse
import collections
import datetime as dt
import json
import math
import pathlib
import subprocess
import time
import urllib.error
import urllib.request

UTC=dt.timezone.utc
FIRST_PUBLISHED={
 "2026-10-25":"9c1c1485ca946b4ca0449dc9f3edf8fdc6fba702",
 "2026-10-26":"b6254dfa9280cceb5e6104642643f3dd147f68b1",
 "2026-10-27":"70827c9fadc4ffbd78152881f75dbf71a093fd21",
 "2026-10-28":"eddbb88c4ede10b89619190f6d008945d17ed774",
}
ALIASES={
 "Anaheim Ducks":"ANA","Boston Bruins":"BOS",
 "Buffalo Sabres":"BUF","Calgary Flames":"CGY",
 "Carolina Hurricanes":"CAR","Chicago Blackhawks":"CHI",
 "Colorado Avalanche":"COL","Columbus Blue Jackets":"CBJ",
 "Dallas Stars":"DAL","Detroit Red Wings":"DET",
 "Edmonton Oilers":"EDM","Florida Panthers":"FLA",
 "Los Angeles Kings":"LAK","Minnesota Wild":"MIN",
 "Montreal Canadiens":"MTL","Montréal Canadiens":"MTL",
 "Nashville Predators":"NSH","New Jersey Devils":"NJD",
 "New York Islanders":"NYI","New York Rangers":"NYR",
 "Ottawa Senators":"OTT","Philadelphia Flyers":"PHI",
 "Pittsburgh Penguins":"PIT","San Jose Sharks":"SJS",
 "Seattle Kraken":"SEA","St. Louis Blues":"STL",
 "Tampa Bay Lightning":"TBL","Toronto Maple Leafs":"TOR",
 "Utah Mammoth":"UTA","Vancouver Canucks":"VAN",
 "Vegas Golden Knights":"VGK","Washington Capitals":"WSH",
 "Winnipeg Jets":"WPG",
}
ALIASES.update({v:v for v in ALIASES.values()})
ALIASES.update({"SJ":"SJS","LA":"LAK","LAS":"VGK","UTAH":"UTA","MON":"MTL"})
MARKETS=("points","points_early_season","moneyline","moneyline_early_season")
def parse_time(x):
    return dt.datetime.fromisoformat(str(x).replace("Z","+00:00"))
def franchise(x):
    return ALIASES.get(str(x or "").strip())
def run_git(*args):
    return subprocess.check_output(["git",*args],text=True).strip()
def frozen_predictions(snapshots=FIRST_PUBLISHED, reader=run_git):
    rows=[];sources=[]
    for day,ref in snapshots.items():
        raw=reader("show",f"{ref}:docs/nhl_predictions.json")
        data=json.loads(raw)
        published=parse_time(data["generated_at_utc"])
        commit=parse_time(reader("show","-s","--format=%cI",ref))
        if data["game_date"]!=day or published.date().isoformat()!=day:
            raise ValueError("NHL_SOURCE_GAME_DATE_OR_PUBLISHED_DATE_CHANGED:"+day)
        before=len(rows)
        seen=set()
        for p in data["picks"]:
            market=p.get("market")
            if market not in MARKETS:
                raise ValueError("UNKNOWN_MODEL_MARKET:"+str(market))
            tip=parse_time(p["kickoff_utc"])
            if tip-published<dt.timedelta(minutes=15) or tip-commit<dt.timedelta(minutes=15):
                # Such rows are explicitly excluded, not disguised as original pregame.
                rows.append({**p,"source_ref":ref,"original_generated_at_utc":data["generated_at_utc"],
                    "source_commit_at_utc":commit.isoformat(),"grade":"UNVERIFIED_AFTER_CUTOFF"})
                continue
            # Points current/early-season are ALTERNATE estimates for the same event.
            unique=(p["game_date"],str(p.get("player_id") or ""),
               tuple(sorted((franchise(p.get("team")),franchise(p.get("opponent"))))),
               "points" if market.startswith("points") else "moneyline")
            if None in unique[2] or unique in seen:
                rows.append({**p,"source_ref":ref,"original_generated_at_utc":data["generated_at_utc"],
                    "source_commit_at_utc":commit.isoformat(),"grade":"UNVERIFIED_DUPLICATE_OR_AMBIGUOUS"})
                continue
            seen.add(unique)
            rows.append({**p,"source_ref":ref,"original_generated_at_utc":data["generated_at_utc"],
                "source_commit_at_utc":commit.isoformat(),"grade":"PENDING"})
        sources.append({"day":day,"source_sha":ref,"generated_at":data["generated_at_utc"],
             "source_commit_at":commit.isoformat(),"original_rows":len(data["picks"]),
             "selected_original_rows":len(rows)-before})
    return rows,sources

def retrieve_json(url,timeout=16,retries=3):
    last=None
    for attempt in range(retries):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"PropEdge NHL source-first audit; research/1.0","Accept":"application/json"})
            with urllib.request.urlopen(req,timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError,TimeoutError,OSError,ValueError) as exc:
            last=exc
            if attempt+1<retries:time.sleep(1.0*(attempt+1))
    raise RuntimeError("OFFICIAL_NHL_SOURCE_UNAVAILABLE:"+url+": "+str(last))

def official_games(days,fetch=retrieve_json):
    out={}
    for day in days:
        payload=fetch("https://api-web.nhle.com/v1/schedule/"+day)
        candidates=[(g,g.get("gameDate") or w.get("date")) for w in payload.get("gameWeek",[])
                    for g in w.get("games",[])]
        matches=[g for g,game_day in candidates if game_day==day
                 and str(g.get("gameType"))=="2"]
        if not matches:
            # Official single-day score API avoids certain schedule-calendar
            # responses with no match for a completed prior date.
            day_score=fetch("https://api-web.nhle.com/v1/score/"+day)
            candidates=[(g,g.get("gameDate")) for g in day_score.get("games",[])]
            matches=[g for g,game_day in candidates if game_day==day
                     and str(g.get("gameType"))=="2"]
        if not matches:
            print("OFFICIAL_NHL_DAY_SOURCE_SHAPE",json.dumps({
               "date":day,"schedule_keys":list(payload),
               "schedule_days":list({w.get("date") for w in payload.get("gameWeek",[])}),
               "schedule_sample_games":[{"date":g.get("gameDate"),"type":g.get("gameType")}
                 for w in payload.get("gameWeek",[]) for g in w.get("games",[])][:8],
               "score_day_keys":list(day_score),
               "score_sample_games":[{"date":g.get("gameDate"),"type":g.get("gameType")}
                  for g in day_score.get("games",[])][:8],
             },default=str))
            raise RuntimeError("OFFICIAL_NHL_DAY_NO_REGULAR_SEASON_GAMES:"+day)
        for g in matches:
            home=franchise(g["homeTeam"]["abbrev"])
            away=franchise(g["awayTeam"]["abbrev"])
            if not home or not away or home==away:raise RuntimeError("BAD_OFFICIAL_NHL_TEAM_IDENTITY")
            key=(day,tuple(sorted((home,away))))
            if key in out:raise RuntimeError("DUPLICATE_NHL_OFFICIAL_GAME")
            if g.get("gameState") not in ("OFF","FINAL"):
                out[key]={"gradeable":False,"game_id":g["id"],"state":g.get("gameState")}
                continue
            full=fetch("https://api-web.nhle.com/v1/gamecenter/"+str(g["id"])+"/boxscore")
            if str(full.get("id"))!=str(g["id"]):
                raise RuntimeError("OFFICIAL_NHL_WRONG_GAME_ID")
            if full.get("gameState") not in ("OFF","FINAL"):
                out[key]={"gradeable":False,"game_id":g["id"],"state":full.get("gameState")}
                continue
            hs=full.get("homeTeam",{}).get("score")
            aas=full.get("awayTeam",{}).get("score")
            if not isinstance(hs,int) or not isinstance(aas,int) or hs==aas:
                raise RuntimeError("INVALID_FINAL_NHL_SCORE:"+str(g["id"]))
            players={}
            for side in ("homeTeam","awayTeam"):
                team=home if side=="homeTeam" else away
                sub=full.get("playerByGameStats",{}).get(side,{})
                for bucket in ("forwards","defense"):
                    for player in sub.get(bucket,[]):
                        pid=str(player.get("playerId") or "")
                        if not pid or (team,pid) in players:
                            raise RuntimeError("OFFICIAL_NHL_PLAYER_ID_COLLISION")
                        goals=player.get("goals");assists=player.get("assists")
                        if not isinstance(goals,int) or not isinstance(assists,int):
                            raise RuntimeError("OFFICIAL_NHL_POINTS_MISSING:"+pid)
                        players[(team,pid)]={"goals":goals,"assists":assists,"points":goals+assists,
                            "shots":player.get("sog",player.get("shots"))}
            if not players:
                raise RuntimeError("OFFICIAL_NHL_FINAL_HAS_NO_SKATER_BOX:"+str(g["id"]))
            out[key]={"gradeable":True,"game_id":g["id"],"home":home,"away":away,
                      "home_score":hs,"away_score":aas,"player_stats":players}
    return out

def grade_predictions(rows,games):
    graded=[]
    for p in rows:
        row={k:p.get(k) for k in ("source_ref","original_generated_at_utc","source_commit_at_utc",
             "season","game_date","kickoff_utc","player_id","player","team","opponent",
             "market","source_market","pick","model_prob")}
        if p["grade"]!="PENDING":
            row.update({"grade":p["grade"],"result":None,"official_game_id":None})
            graded.append(row);continue
        team=franchise(p.get("team"));opp=franchise(p.get("opponent"))
        key=(p["game_date"],tuple(sorted((team,opp))))
        official=games.get(key)
        if not official or not official["gradeable"]:
            row.update({"grade":"UNVERIFIED_OFFICIAL_FINAL_NOT_AVAILABLE","result":None,
                         "official_game_id":official.get("game_id") if official else None})
            graded.append(row);continue
        row["official_game_id"]=official["game_id"]
        if not isinstance(p.get("model_prob"),(float,int)) or not(0<=p["model_prob"]<=1):
            raise ValueError("BAD_FROZEN_MODEL_PROBABILITY")
        if p["market"].startswith("points"):
            if p["pick"]!="OVER 0.5" or not p.get("player_id"):
                row.update({"grade":"UNVERIFIED_UNSUPPORTED_POINT_EVENT","result":None})
            else:
                actual=official["player_stats"].get((team,str(p["player_id"])))
                if actual is None:
                    row.update({"grade":"UNVERIFIED_NOT_IN_FINAL_SKATER_BOX","result":None})
                else:
                    win=actual["points"]>=1
                    row.update({"grade":"HIT" if win else "MISS","result":int(win),
                        "actual_points":actual["points"],"actual_goals":actual["goals"],
                        "actual_assists":actual["assists"]})
        else:
            predicted_side=official["home"] if p.get("is_home")==1 else official["away"] if p.get("is_home")==0 else None
            if predicted_side!=team or opp not in (official["home"],official["away"]) or team==opp:
                row.update({"grade":"UNVERIFIED_MONEYLINE_IDENTITY_MISMATCH","result":None})
            else:
                winner=official["home"] if official["home_score"]>official["away_score"] else official["away"]
                win=team==winner
                row.update({"grade":"HIT" if win else "MISS","result":int(win),
                    "winner":winner,"final_score":{"home":official["home_score"],"away":official["away_score"]}})
        graded.append(row)
    return graded

def summarize(rows):
    def metrics(items):
        verified=[x for x in items if x["grade"] in ("HIT","MISS")]
        won=sum(x["grade"]=="HIT" for x in verified)
        q=[x["model_prob"] for x in verified]
        return {"published_n":len(items),"verified_n":len(verified),
                "hit_n":won,"miss_n":len(verified)-won,
                "unverified_n":len(items)-len(verified),
                "hit_rate":round(won/len(verified),6) if verified else None,
                "average_model_probability":round(sum(q)/len(q),6) if verified else None,
                "brier":round(sum((x["model_prob"]-x["result"])**2 for x in verified)/len(verified),6)
                if verified else None}
    bydate={}
    for day in sorted(FIRST_PUBLISHED):
        items=[x for x in rows if x["game_date"]==day]
        bydate[day]={"all_published":metrics(items),
                     "anytime_points":metrics([x for x in items if x["market"].startswith("points")]),
                     "moneyline":metrics([x for x in items if x["market"].startswith("moneyline")]),
                     "points_p_ge65":metrics([x for x in items if x["market"].startswith("points") and x.get("model_prob",0)>=.65])}
    categories={"all_published":rows,
      "anytime_points":[x for x in rows if x["market"].startswith("points")],
      "moneyline":[x for x in rows if x["market"].startswith("moneyline")],
      "points_p_ge65":[x for x in rows if x["market"].startswith("points") and x.get("model_prob",0)>=.65]}
    return {"all_days":{k:metrics(v) for k,v in categories.items()},"by_date":bydate,
            "unverified_reason_counts":dict(collections.Counter(x["grade"] for x in rows if x["grade"] not in ("HIT","MISS")))}

def audit(fetch=retrieve_json,reader=run_git):
    rows,sources=frozen_predictions(reader=reader)
    games=official_games(FIRST_PUBLISHED,fetch=fetch)
    graded=grade_predictions(rows,games)
    report={"schema":"NHL_FIRSTSEEN_OCT25_28_2026_SERVED_PICKS_OFFICIAL_ACTUAL_AUDIT_V1",
      "study_dates":["2026-10-25","2026-10-28"],
      "source":"PRETIP_PINNED_GITHUB_COMMITS_AND_INDEPENDENT_NHL_OFFICIAL_FINAL_BOXSCORES",
      "grade_policy":"IN_FINAL_SKATER_BOX_ONLY_NO_INVENTED_DNP_OR_AFTER_GAME_PREGAME_BETS",
      "no_sportsbook_odds_or_bet_profit_claims":True,
      "points_original_calibration_bootstrap_override_not_certified":True,
      "source_vintages":sources,
      "official_final_games":sum(1 for v in games.values() if v["gradeable"]),
      "summary":summarize(graded),"rows":graded}
    if report["summary"]["all_days"]["all_published"]["published_n"]!=len(graded):
        raise RuntimeError("PUBLISHED_LEDGER_COUNT_CONSERVATION_FAILURE")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    result=audit()
    p=pathlib.Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NHL_REAL_SERVED_PICKS_AUDIT",json.dumps({
      "snapshots":result["source_vintages"],
      "actual_games":result["official_final_games"],
      "verified":result["summary"]["all_days"],
      "by_market":{k:result["summary"]["all_days"][k] for k in ("anytime_points","moneyline","points_p_ge65")},
      "uncertain":result["summary"]["unverified_reason_counts"]}))
if __name__=="__main__":main()
