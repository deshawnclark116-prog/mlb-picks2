"""Leakage and opportunity-first research acceptance; no model promotion implied."""
import sqlite3
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import nfl_opportunity_shadow as m

def history():
    return [
        (2025,1,"JAX",8.,32.,21.),
        (2025,2,"JAX",10.,45.,24.),
        (2025,3,"JAX",16.,88.,27.),
        (2025,4,"JAX",18.,117.,30.),
        (2025,5,"JAX",20.,110.,35.),
        (2025,6,"JAX",24.,135.,37.),
    ]

def features(h=None, team=None, opponent=None):
    h=h or history()
    if team is None:
        team=[(2025,w,30.+w) for w in range(1,7)]
    if opponent is None:
        opponent=[(120,540)]
    return m.forecast_player(h,team,opponent,[(1000,4500)],
                             season=2025,market="rushing_yards")

def test_real_opportunity_not_just_yardage_and_role_expanding():
    result=features()
    assert result
    assert result["role_direction"]=="EXPANDING"
    assert result["last3_opportunities"]>(8+10+16)/3
    assert result["expected_opportunities"]>0
    assert result["projected_yards"]>=0
    assert result["history_games"]==6
    assert result["opponent_efficiency_adjustment"]<=0.1
    assert result["no_verified_future_starter_status"] is True

def test_defense_uses_shrunk_earlier_efficiency_not_fabricated_props():
    b=features(opponent=[(200,600)])
    c=features(opponent=[(200,1800)])
    assert b["projected_yards"]<c["projected_yards"]
    assert b["opponent_efficiency_adjustment"]>=-0.10
    assert c["opponent_efficiency_adjustment"]<=0.10
    assert "model_prob" not in c and "pick" not in c

def test_no_future_season_team_or_weak_opportunity_eligible():
    assert m.forecast_player(history()[:2],[],[],[(100,500)],
                             season=2025,market="rushing_yards") is None
    assert m.forecast_player(history(),[],[],[(100,500)],
                             season=2025,market="rushing_yards") is None
    switched=history()[:]
    switched[-1]=(2025,6,"PHI",24.,135.,37.)
    assert features(h=switched) is None
    weak=[(2025,i,"JAX",1.,3.,30.) for i in range(1,5)]
    assert m.forecast_player(weak,[(2025,i,30) for i in range(1,5)],
        [],[(100,500)],season=2025,market="rushing_yards") is None

def fake_db():
    c=sqlite3.connect(":memory:")
    c.executescript("""
    CREATE TABLE games (game_id TEXT, season INTEGER,week INTEGER,home_team TEXT,
                        away_team TEXT,kickoff_utc TEXT);
    CREATE TABLE player_games (
        player_id TEXT,player_name TEXT,position TEXT,team TEXT,opponent TEXT,
        season INTEGER,week INTEGER,game_id TEXT,season_type TEXT,
        carries REAL,rushing_yards REAL,targets REAL,receiving_yards REAL
    );
    """)
    rows=[]
    for season in (2024,2025,2026):
        for week in range(1,8):
            for team,opp in (("JAX","PHI"),("PHI","JAX")):
                for p,pos,mult in (("star","RB",1.0),("backup","RB",0.3),
                                   ("wideout","WR",0.85)):
                    vol=12*mult+week*0.6
                    yds=vol*4.5
                    target=8*mult+week*0.3
                    rec=target*9
                    # All team games use one shared actual game ID,
                    # distinct across season/week, as in nflverse.
                    rows.append((team+"_"+p,team+" "+p,pos,team,opp,season,week,
                                 f"{season}_{week:02}_PHI_JAX","REG",
                                 vol,yds,target,rec))
    c.executemany("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",rows)
    game_rows=[]
    for season in (2024,2025,2026):
        for week in range(1,8):
            kickoff=(datetime(season,9,1,tzinfo=timezone.utc)+timedelta(weeks=week-1))
            game_rows.append((f"{season}_{week:02}_PHI_JAX",season,week,
                              "JAX","PHI",kickoff.isoformat()))
    c.executemany("INSERT INTO games VALUES(?,?,?,?,?,?)",game_rows)
    c.commit()
    return c

def test_week_strict_asof_no_future_leakage_across_current_game():
    conn=fake_db()
    before,eligible=m.walkforward(conn,seasons=(2025,))
    week3=[r for r in before if r["season"]==2025 and r["week"]==3]
    # Change week three outcomes by x100. Projected week 3 CANNOT move
    # because it has seen only weeks <=2.
    conn.execute("UPDATE player_games SET rushing_yards=rushing_yards*100,"
                 "receiving_yards=receiving_yards*100 WHERE season=2025 AND week=3")
    after,_=m.walkforward(conn,seasons=(2025,))
    later=[r for r in after if r["season"]==2025 and r["week"]==3]
    assert [x["shadow"] for x in week3]==[x["shadow"] for x in later]
    assert [x["actual"] for x in week3]!=[x["actual"] for x in later]
    assert eligible[(2025,"rushing_yards")]>0

def test_shadow_report_never_self_promotes_or_claims_starter_status():
    con=fake_db()
    rows,eligible=m.walkforward(con)
    out=m.report(rows,eligible)
    assert out["promotion_status"]=="RESEARCH_ONLY_NOT_ELIGIBLE_FOR_PRODUCTION"
    assert out["uses_target_week_data_in_features"] is False
    assert "selection bias" in " ".join(out["evidence_limitations"]).lower()
    assert out["cohorts"]["2025_rushing_yards"]["n"]>0
    assert out["cohorts"]["2026_receiving_yards"]["n"]>0
    assert out["cohorts"]["2025_rushing_yards"]["game_cluster_ci_lift"] is not None
    assert "model_prob" not in rows[0]

def test_locked_champion_comparison_exact_firstseen_and_no_time_leakage(tmp_path):
    db=tmp_path/"nfl.sqlite"
    fixture=fake_db()
    remote=sqlite3.connect(db)
    fixture.backup(remote)
    remote.close()
    rows,eligible=m.walkforward(fixture,seasons=(2026,))
    target=next(r for r in rows if r["season"]==2026
                and r["week"]==4 and r["player"]=="JAX star"
                and r["market"]=="rushing_yards")
    p={"season":2026,"week":4,"market":"rushing_yards",
       "player_id":"JAX_star","player":"JAX star","team":"JAX",
       "opponent":"PHI","projected_median":target["shadow"]+15,
       "model_source":"v2_context",
       "logged_at":"2026-09-21T20:00:00Z"}
    ledger=tmp_path/"ledger.jsonl"
    ledger.write_text(json.dumps(p)+"\n")
    out=m.compare_first_seen(rows,str(db),str(ledger))
    assert out["matched"]==1
    assert out["incumbent_first_seen_eligible"]==1
    assert out["overall"]["shadow_lift_against_incumbent"]>0
    assert out["status"]=="RESEARCH_ONLY_NO_PROMOTION"
    # The prior-week game finished 2026-09-15; an earlier forecast
    # cannot use the shadow's post-September-15 team-opponent context.
    p["logged_at"]="2026-09-08T10:00:00Z"
    ledger.write_text(json.dumps(p)+"\n")
    out=m.compare_first_seen(rows,str(db),str(ledger))
    assert out["matched"]==0
    assert out["exclusions"]["prior_week_data_unavailable_at_forecast_time"]==1
