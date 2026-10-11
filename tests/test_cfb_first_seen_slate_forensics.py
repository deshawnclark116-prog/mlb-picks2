"""Regression locks reproducible grading and positively detects historical leakage.

The leakage test intentionally asserts that a historical CFB context feature
CHANGES when only target-game participation is changed. This documents the
existing failed as-of invariant; it is not a pass for model promotion.
"""
import json
import sqlite3
from datetime import datetime,timezone
import cfb_first_seen_slate_forensics as f
import cfb_context_v2 as cv2


def scored(market,player,model_prob,actual,result,kick="2026-10-10T19:30:00Z"):
    line={"moneyline":None,"rushing_yards":69.5,"passing_touchdowns":1.5}.get(market)
    player_id={"moneyline":None,"rushing_yards":"7","passing_touchdowns":"8"}.get(market)
    key="|".join(("2026","6","401123456",market,"GAME" if market=="moneyline" else player_id,
                  "ML" if market=="moneyline" else str(line)))
    return dict(canonical_key=key,market=market,player=player,player_id=player_id,
                game_id="401123456",season=2026,week=6,kickoff_utc=kick,
                line=line,model_prob=model_prob,pick=(player+" ML" if market=="moneyline" else
                       ("OVER " if market=="rushing_yards" else "UNDER ")+str(line)),
                result=result,actual=actual,team=player,opponent="Other",
                logged_at="2026-10-09T00:00:00Z")


def test_true_first_pregame_and_ungraded_rows_are_separate_from_losses():
    money=scored("moneyline","Oklahoma Sooners",.8426,"Texas Longhorns","miss")
    rush=scored("rushing_yards","Run Person",.72,59,"miss")
    future=scored("passing_touchdowns","QB Zero",.65,None,None)
    ledger=[]
    for r in (money,rush,future):
        x={k:v for k,v in r.items() if k not in ("result","actual","canonical_key")}
        ledger.append(x)
    ledger.append({**ledger[0],"model_prob":.99,
                   "logged_at":"2026-10-10T20:00:00Z"})  # postgame cannot count
    r=f.audit(ledger,[money,rush],2026,6,
              datetime(2026,10,10,4,tzinfo=timezone.utc),
              datetime(2026,10,11,4,tzinfo=timezone.utc))
    assert r["distinct_pregame_ledger"]==3
    assert r["graded_predictions"]==2
    assert r["not_graded_predictions"]==1
    assert r["overall"]["wins"]==0
    assert r["invalid_or_postkickoff_ledger_rows"]==1
    assert r["by_market"]["rushing_yards"]["numerical_point_forecasts_logged"]==0
    assert r["decision"]=="FORENSIC_ONLY_NO_MODEL_OR_CALIBRATION_CHANGES"


def context_db():
    c=sqlite3.connect(":memory:")
    c.execute("CREATE TABLE player_games(season INTEGER,week INTEGER,game_id TEXT,game_date TEXT,team TEXT,player_id TEXT,position TEXT,carries REAL,receptions REAL,pass_attempts REAL)")
    for week in (1,2,3,4):
        for player,carries in (("lead",11),("backup",3)):
            c.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?)",(
                2026,week,f"week{week}",f"2026-09-{week*7:02d}",
                "Demo U",player,"RB",carries,0,0))
    c.commit()
    return c


def test_existing_historical_context_illegally_uses_target_game_participants():
    con=context_db()
    before=cv2.build_context(con)[(2026,"backup",4)]
    # Same prior three weeks. Change ONLY the future/current week's
    # lead player availability and yards. As-of features must NOT move.
    con.execute("UPDATE player_games SET carries=0 WHERE week=4 AND player_id='lead'")
    after=cv2.build_context(con)[(2026,"backup",4)]
    assert before["vacated_car"]==0
    assert after["vacated_car"]>0
    assert before["vacated_car"]!=after["vacated_car"]
    assert before["carry_share_last3"]==after["carry_share_last3"]


def test_moneyline_training_features_missing_but_serving_injects_zeros():
    trained=cv2.add_context({"is_home":1.0},2025,None,6,"Demo U","Other U",{}, {})
    assert trained["vacated_car"] is None
    serving=cv2.ServingContext.__new__(cv2.ServingContext)
    serving.team_ctx={"Demo U":{"vacated_car":0.0,"vacated_rec":0.0,
                              "qb_change":0.0,"team_plays_last3":56.0}}
    serving.player_ctx={};serving.ratings={"Demo U":3.0,"Other U":0.0}
    live=serving.features({"is_home":1.0},None,"Demo U","Other U")
    assert live["vacated_car"]==0.0
    assert live["team_plays_last3"]==56.0
    assert trained["team_plays_last3"] is None
