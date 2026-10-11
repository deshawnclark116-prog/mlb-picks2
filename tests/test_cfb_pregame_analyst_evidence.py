import json
import sqlite3
from datetime import datetime,timezone
import cfb_pregame_analyst_evidence as a
import cfb_espn_live_foundation_a as source

NOW=datetime(2026,10,11,14,0,tzinfo=timezone.utc)
BOARD={"season":2026,"week":5,"generated_at_utc":"2026-10-11T13:30:00Z","picks":[
  {"player_id":"101","player":"Starter Runner","team":"Hawks","opponent":"Bears",
   "market":"rushing_yards","pick":"OVER 69.5","model_prob":.83,
   "kickoff_utc":"2026-10-11T20:00:00Z"},
  {"player_id":"102","player":"Rare Backup","team":"Hawks","opponent":"Bears",
   "market":"rushing_yards","pick":"UNDER 69.5","model_prob":.98,
   "kickoff_utc":"2026-10-11T20:00:00Z"},
  {"player_id":"103","player":"QB One","team":"Hawks","opponent":"Bears",
   "market":"passing_touchdowns","pick":"OVER 1.5","model_prob":.77,
   "kickoff_utc":"2026-10-11T20:00:00Z"},
  {"player_id":"104","player":"Doubtful Receiver","team":"Bears","opponent":"Hawks",
   "market":"anytime_touchdowns","pick":"OVER 0.5","model_prob":.62,
   "kickoff_utc":"2026-10-11T20:00:00Z"},
  {"player_id":None,"player":"Hawks","team":"Hawks","opponent":"Bears",
   "market":"moneyline","pick":"Hawks ML","model_prob":.85,
   "kickoff_utc":"2026-10-11T20:00:00Z"}
]}

def make_db():
    c=sqlite3.connect(":memory:")
    c.executescript("""
    CREATE TABLE games(game_id TEXT,season INT,week INT,kickoff_utc TEXT,
                       home_team TEXT,away_team TEXT,home_points INT,away_points INT);
    CREATE TABLE player_games(player_id TEXT,game_id TEXT,team TEXT,carries INT,
                     pass_attempts INT,receptions INT);
    CREATE TABLE current_roster(team TEXT,player_name TEXT,position TEXT,season INT,
       espn_athlete_id TEXT,roster_status_raw TEXT,
       roster_injuries_json TEXT,roster_captured_at_utc TEXT);
    """)
    for w in range(1,6):
        gid=f"game-{w}"
        c.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?,?)",(gid,2026,w,
            f"2026-09-{1+w:02}T20:00:00Z" if w<5 else "2026-10-11T20:00:00Z",
            "Hawks","Bears",21 if w<5 else None,7 if w<5 else None))
        for pid,team,ca,pa,rec in (("101","Hawks",15,0,0),("102","Hawks",0,0,0),
                                    ("103","Hawks",1,27,0),("104","Bears",0,0,7)):
            c.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?)",
                   (pid,gid,team,999 if w==5 else ca,
                    999 if w==5 else pa,999 if w==5 else rec))
    for team,name,pos,pid,state,inj in [
       ("Hawks","Starter Runner","RB","101","active",[]),
       ("Hawks","Rare Backup","RB","102","active",[]),
       ("Hawks","QB One","QB","103",None,[]),
       ("Bears","Doubtful Receiver","WR","104",None,[{"status":"Questionable"}])]:
        c.execute("INSERT INTO current_roster VALUES(?,?,?,?,?,?,?,?)",(
          team,name,pos,2026,pid,state,json.dumps(inj),"2026-10-11T12:00:00Z"))
    c.commit()
    return c

def test_individual_role_and_injury_flags_are_source_grounded_not_probability():
    doc=a.make_snapshot(make_db(),BOARD,NOW)
    rows={r["player"]:r for r in doc["player_evidence"]}
    assert doc["evidence_count"]==4
    assert doc["has_source_roster_athlete_ids"] is True
    assert doc["markets_book_verified"] is False
    runner=rows["Starter Runner"]
    assert runner["recent_usage"]["mean_opportunities_in_observed_games"]==15
    assert runner["recent_usage"]["role_ready_from_prior_games"]
    assert runner["recent_usage"]["mean_share_when_observed"]<1
    assert runner["evidence_warnings"]==[]
    low=rows["Rare Backup"]
    assert low["published_model_prob"]==.98
    assert "LOW_RECENT_INVOLVEMENT_OBSERVATION" in low["evidence_warnings"]
    assert low["fully_verified_active_starter"] is False
    doubtful=rows["Doubtful Receiver"]
    assert doubtful["roster"]["has_explicit_injury_concern"] is True
    assert "ROSTER_IDENTITY_OR_AVAILABILITY_UNCERTAIN" in doubtful["evidence_warnings"]
    assert doubtful["recent_usage"]["metric"]=="receptions"
    assert not doubtful["roster"]["availability_certified"]

def test_absent_boxscore_is_unknown_not_a_fake_zero():
    c=make_db()
    c.execute("DELETE FROM player_games WHERE game_id='game-3' AND player_id='101'")
    d=a.make_snapshot(c,BOARD,NOW)
    r=next(x for x in d["player_evidence"] if x["player"]=="Starter Runner")
    assert r["recent_usage"]["unknown_or_ambiguous_appearances"]==1
    assert any(x["player_opportunities"] is None for x in r["recent_usage"]["last_three_completed_team_games"])
    assert "MISSING_BOXSCORE_IS_NOT_A_CONFIRMED_DNP" in r["evidence_warnings"]
    assert "THREE_GAME_ROLE_COVERAGE_INCOMPLETE" in r["evidence_warnings"]

def test_target_week_data_cannot_leak_into_pregame_receipts():
    c=make_db()
    before=a.make_snapshot(c,BOARD,NOW)
    c.execute("UPDATE player_games SET carries=100000 WHERE game_id='game-5'")
    after=a.make_snapshot(c,BOARD,NOW)
    assert before==after

def test_espn_roster_absence_fail_closed_and_no_starter_inference():
    c=make_db()
    c.execute("DROP TABLE current_roster")
    d=a.make_snapshot(c,BOARD,NOW)
    assert not d["has_source_roster_athlete_ids"]
    assert all("ROSTER_IDENTITY_OR_AVAILABILITY_UNCERTAIN" in x["evidence_warnings"]
           for x in d["player_evidence"])
    assert all(not x["fully_verified_active_starter"] for x in d["player_evidence"])

def test_espn_source_parser_keeps_actual_ids_status_and_injuries_without_new_calls(monkeypatch):
    called=[]
    def get(url):
        called.append(url)
        return {"athletes":[{"items":[{
          "id":"104","position":{"abbreviation":"WR"},
          "displayName":"Doubtful Receiver","status":{"name":"questionable"},
          "injuries":[{"status":"Questionable","date":"2026-10-10"}]
        }]}]}
    monkeypatch.setattr(source,"get",get)
    monkeypatch.setattr(source.time,"sleep",lambda x:None)
    cache=source.RosterCache()
    assert cache.get("404",2026)=={"104":"WR"}
    evidence=cache.get_evidence("404",2026)
    assert evidence["104"]["roster_status_raw"]=="questionable"
    assert json.loads(evidence["104"]["roster_injuries_json"])[0]["status"]=="Questionable"
    assert evidence["104"]["roster_captured_at_utc"]
    assert len(called)==1
    assert cache.get_names("404",2026)["104"]==("Doubtful Receiver","WR")
    assert len(called)==1
