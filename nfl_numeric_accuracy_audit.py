"""Strict as-of numerical NFL forecast evaluation, never betting hit-rate grading.

First-seen, prekickoff model yard predictions only. Read-only against a
committed prediction ledger and the independently sourced official game DB.
Report MAE, signed error, median abs error, within-10/20 yards and a SAME-
PLAYER/SAME-GAME previous-3-games naive baseline. Never promote models.
"""
import argparse
import json
import math
import re
import random
import sqlite3
import statistics
import unicodedata
from collections import defaultdict
from datetime import datetime,timezone,timedelta
from pathlib import Path

from nfl_pregame_delivery import as_utc

MARKETS={"rushing_yards":"rushing_yards","receiving_yards":"receiving_yards"}


def normalize_name(value):
    s=unicodedata.normalize("NFKD",str(value or ""))
    s="".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^a-z ]","",re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b","",s)).split())


def load_ledger(path):
    rows=[]
    for lineno,line in enumerate(Path(path).read_text().splitlines(),1):
        if not line.strip():
            continue
        try:
            p=json.loads(line)
        except json.JSONDecodeError as err:
            raise ValueError(f"CORRUPT_LEDGER_LINE_{lineno}") from err
        if not isinstance(p,dict):
            raise ValueError(f"INVALID_LEDGER_RECORD_{lineno}")
        rows.append(p)
    return rows


def evaluate(ledger, db, through_week=None):
    """Postgame evaluation: frozen first-seen records cannot use postkickoff info."""
    con=sqlite3.connect(str(db))
    con.row_factory=sqlite3.Row
    try:
        cols={x["name"] for x in con.execute("PRAGMA table_info(games)")}
        if "kickoff_utc" not in cols:
            raise ValueError("GAME_KICKOFF_TIMES_REQUIRED")
        schedules={}
        for row in con.execute("SELECT season,week,home_team,away_team,kickoff_utc FROM games"):
            if row["kickoff_utc"]:
                schedules[(row["season"],row["week"],frozenset((row["home_team"],row["away_team"])))]=as_utc(row["kickoff_utc"])
        all_players=[dict(row) for row in con.execute(
            "SELECT season,week,player_id,player_name,team,opponent,rushing_yards,receiving_yards FROM player_games")]
    finally:
        con.close()
    actual_ids={}
    actual_names=defaultdict(list)
    history=defaultdict(list)
    for row in all_players:
        season,week=row["season"],row["week"]
        ident=(season,week,str(row["player_id"]),str(row["team"]))
        actual_ids.setdefault(ident,row)
        actual_names[(season,week,normalize_name(row["player_name"]),row["team"])].append(row)
        history[str(row["player_id"])].append(row)
    # Baseline only uses games completed earlier than the target week.
    for rows in history.values():
        rows.sort(key=lambda x:(x["season"],x["week"]))
    excluded=defaultdict(int);scored=[];first_seen=set()
    for p in ledger:
        market=p.get("market")
        if market not in MARKETS:
            excluded["not_numerical_yards"]+=1;continue
        if through_week is not None and int(p.get("week",0))>through_week:
            excluded["after_requested_week"]+=1;continue
        proj=p.get("projected_median")
        if not isinstance(proj,(int,float)) or not math.isfinite(proj) or proj<0:
            excluded["no_valid_first_seen_median"]+=1;continue
        season,week=p.get("season"),p.get("week")
        if not isinstance(season,int) or not isinstance(week,int):
            excluded["invalid_week"]+=1;continue
        key=(season,week,market,str(p.get("player_id")),str(p.get("team")))
        if key in first_seen:
            excluded["duplicate_first_seen_key"]+=1;continue
        first_seen.add(key)
        try:
            logged=as_utc(p.get("logged_at"))
        except (ValueError,TypeError):
            excluded["untrusted_log_time"]+=1;continue
        kickoff=schedules.get((season,week,frozenset((p.get("team"),p.get("opponent")))))
        if kickoff is None:
            excluded["missing_schedule_kickoff"]+=1;continue
        if logged>=kickoff:
            excluded["after_kickoff"]+=1;continue
        actual=actual_ids.get((season,week,str(p.get("player_id")),str(p.get("team"))))
        if actual is None:
            candidates=actual_names.get((season,week,normalize_name(p.get("player")),p.get("team")),[])
            if len(candidates)!=1:
                excluded["missing_or_ambiguous_actual"]+=1;continue
            actual=candidates[0]
        target=actual.get(MARKETS[market])
        if not isinstance(target,(int,float)) or not math.isfinite(target):
            excluded["not_yet_played_or_stat_missing"]+=1;continue
        # Baseline must have distinct actual games *strictly earlier* and same
        # player identity. Name fallback for actual matching is not permitted
        # for baseline: missing crosswalk is openly reported instead.
        previous=[
            a[MARKETS[market]] for a in history.get(str(actual["player_id"]),[])
            if (a["season"],a["week"])<(season,week)
            and (previous_kickoff:=schedules.get((a["season"],a["week"],frozenset((a["team"],a["opponent"]))))) is not None
            and previous_kickoff+timedelta(hours=4) <= logged
            and isinstance(a[MARKETS[market]],(int,float)) and math.isfinite(a[MARKETS[market]])
        ]
        baseline=(statistics.mean(previous[-3:])) if previous else None
        scored.append({
            "season":season,"week":week,"market":market,"player":p.get("player"),
            "team":p.get("team"),"opponent":p.get("opponent"),"model_source":p.get("model_source"),
            "logged_at":logged.isoformat(),"kickoff":kickoff.isoformat(),
            "projected_median":float(proj),"actual":float(target),
            "baseline_last3":baseline,"abs_error":abs(float(proj)-float(target)),
            "error":float(proj)-float(target)
        })
    def metrics(rows):
        if not rows:return {"n":0,"status":"NO_POSTGAME_ACTUALS"}
        errs=[r["abs_error"] for r in rows]
        paired=[r for r in rows if r["baseline_last3"] is not None]
        answer={
            "n":len(rows),"mae":round(statistics.mean(errs),2),
            "median_abs_error":round(statistics.median(errs),2),
            "signed_bias":round(statistics.mean(r["error"] for r in rows),2),
            "within_10yd_pct":round(100*sum(e<=10 for e in errs)/len(errs),1),
            "within_20yd_pct":round(100*sum(e<=20 for e in errs)/len(errs),1),
            "same_game_baseline_n":len(paired),
            "status":"INSUFFICIENT_BASELINE_PAIRS"
        }
        if paired:
            mae=statistics.mean(r["abs_error"] for r in paired)
            base=statistics.mean(abs(r["baseline_last3"]-r["actual"]) for r in paired)
            # Multiple players on the same NFL game share correlated outcomes.
            # Resample entire games, not individual player records, before
            # treating a 1-yard point improvement as a meaningful finding.
            games=defaultdict(list)
            for r in paired:
                key=(r["season"],r["week"],tuple(sorted((r["team"],r["opponent"]))))
                games[key].append(abs(r["baseline_last3"]-r["actual"])-r["abs_error"])
            cluster_values=list(games.values())
            ci=None
            if len(cluster_values)>=2:
                rng=random.Random(20261011)
                lifts=[]
                for _ in range(2000):
                    sampled=[cluster_values[rng.randrange(len(cluster_values))] for _ in cluster_values]
                    pooled=[d for group in sampled for d in group]
                    lifts.append(statistics.mean(pooled))
                lifts.sort()
                ci=[round(lifts[49],2),round(lifts[1949],2)]
            answer.update(paired_model_mae=round(mae,2),baseline_last3_mae=round(base,2),
                          improvement_vs_last3_yds=round(base-mae,2),
                          paired_game_clusters=len(cluster_values),
                          cluster_bootstrap_improvement_ci95_yds=ci,
                          lift_supported_by_cluster_ci=ci is not None and ci[0]>0,
                          status="BEATS_LAST3_OBSERVED" if mae<base else "NOT_BEATING_LAST3")
        return answer
    by_market={m:metrics([r for r in scored if r["market"]==m]) for m in MARKETS}
    by_week={str(w):metrics([r for r in scored if r["week"]==w]) for w in sorted({r["week"] for r in scored})}
    return {
        "schema":"nfl-first-seen-numerical-accuracy-v1",
        "research_only":True,"automatic_promotion":False,"sportsbook_hit_rate_not_used":True,
        "first_seen_pregame_only":True,"snapshot_policy":"only ledger records with projected_median and logged_at before kickoff",
        "eligible_sample":len(scored),"excluded":dict(sorted(excluded.items())),
        "overall":metrics(scored),"by_market":by_market,"by_week":by_week,
        "forecasts":scored
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",default="nfl_models/nfl_model.sqlite")
    ap.add_argument("--ledger",default="docs/nfl_picks_log.jsonl")
    ap.add_argument("--out",default=None)
    ap.add_argument("--through-week",type=int)
    args=ap.parse_args()
    report=evaluate(load_ledger(args.ledger),args.db,args.through_week)
    payload=json.dumps(report,indent=2,sort_keys=True)
    if args.out:
        path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(payload+"\n")
    print(json.dumps({"status":report["overall"]["status"],"overall":report["overall"],
                      "markets":report["by_market"],"excluded":report["excluded"]},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
