import sqlite3
from datetime import datetime, timezone
import pytest
import nfl_analyst_role_preflight as m

NOW=datetime(2026,10,11,4,30,tzinfo=timezone.utc)
FORECAST={
    "season":2026,"week":5,"generated_at_utc":"2026-10-11T02:00:00Z",
    "scheduled_games":[{"home_team":"JAX","away_team":"PHI",
                        "kickoff_utc":"2026-10-11T13:30:00Z"}],
    "picks":[
      {"market":"rushing_yards","team":"JAX","opponent":"PHI",
       "player":"First Runner","games_played":3,
       "confidence":"HIGH","projected_median":10.0},
      {"market":"receiving_yards","team":"PHI","opponent":"JAX",
       "player":"Old Receiver","games_played":6,
       "confidence":"MEDIUM","projected_median":40.0},
    ]
}
class R:
    def __init__(self,d=None,fail=False):self.d=d or {};self.fail=fail
    def raise_for_status(self):
        if self.fail:raise ValueError("roster_unavailable")
    def json(self):return self.d

def active(url,timeout):
    team="JAX" if "/jax/" in url else "PHI"
    return R({"athletes":[{"position":"offense","items":[
        {"displayName":"First Runner" if team=="JAX" else "Old Receiver","injuries":[]}]}]})

def db():
    c=sqlite3.connect(":memory:")
    c.executescript("""
      CREATE TABLE games(game_id TEXT,season INT,week INT,home_team TEXT,away_team TEXT);
      CREATE TABLE player_games(player_id TEXT,player_name TEXT,team TEXT,
          season INT,week INT,game_id TEXT,season_type TEXT,
          carries REAL,targets REAL);
    """)
    for week in range(1,6):
        gid=f"2026_{week:02}_PHI_JAX"
        c.execute("INSERT INTO games VALUES(?,?,?,?,?)",(gid,2026,week,"JAX","PHI"))
        for team in ("JAX","PHI"):
            who="First Runner" if team=="JAX" else "Old Receiver"
            metric=(12+week,4) if team=="JAX" else (0,9+week)
            c.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?)",
                      (team+"-id",who,team,2026,week,gid,"REG",*metric))
    c.commit()
    return c

def test_pregame_roster_is_not_starter_declaration_and_boxscores_are_lagged():
    c=db()
    res=m.build(c,FORECAST,"JAX","PHI",NOW,active)
    assert len(res["player_market_evidence"])==2
    assert res["sources_have_verified_starting_lineups"] is False
    first=res["player_market_evidence"][0]
    assert first["roster_evidence"]["status"]=="ROSTER_LISTED_NOT_STARTER_VERIFIED"
    assert first["roster_evidence"]["has_verified_depth_chart_starter_designation"] is False
    assert first["usage_evidence"]["observed_stat_rows"]==3
    assert first["usage_evidence"]["opportunities_per_observed_game"]==14.0
    assert first["eligible_for_automatic_official_bet"] is False
    assert first["published_history_n"]==3
    assert "HIGH_STABILITY_IS_NOT_PROVEN_ROLE" in first["warnings"]
    # Altering an *upcoming* game may not change any pregame role estimate.
    c.execute("UPDATE player_games SET carries=999 WHERE week=5 AND team='JAX'")
    after=m.build(c,FORECAST,"JAX","PHI",NOW,active)
    assert after["player_market_evidence"][0]["usage_evidence"]==first["usage_evidence"]

def test_unavailable_roster_does_not_infer_active_starter():
    def risky(url,timeout):
        return R({"athletes":[{"position":"injuredReserveOrOut","items":[
           {"displayName":"First Runner","injuries":[{"status":"Out"}]}]}]})
    r=m.build(db(),FORECAST,"JAX","PHI",NOW,risky)
    assert r["player_market_evidence"][0]["roster_evidence"]["status"]=="LISTED_UNAVAILABLE"
    assert "CURRENT_AVAILABILITY_NOT_VERIFIED_OR_RISK" in r["player_market_evidence"][0]["warnings"]

def test_espn_outage_fails_closed_no_positive_roster_assumption():
    def failed(url,timeout): return R(fail=True)
    r=m.build(db(),FORECAST,"JAX","PHI",NOW,failed)
    assert r["roster_source_status"]["JAX"]=="ROSTER_SOURCE_UNAVAILABLE"
    assert all(x["roster_evidence"]["status"]=="ROSTER_SOURCE_UNAVAILABLE"
               for x in r["player_market_evidence"])

def test_missing_boxscore_row_is_unknown_not_zero_or_confirmed_dnp():
    con=db()
    con.execute("DELETE FROM player_games WHERE player_name='First Runner' AND week=3")
    r=m.build(con,FORECAST,"JAX","PHI",NOW,active)["player_market_evidence"][0]
    assert r["usage_evidence"]["unknown_stat_appearances"]==1
    assert r["usage_evidence"]["recent_games"][-2]["opportunities_when_observed"] is None
    assert r["usage_evidence"]["status"]=="ROLE_PARTICIPATION_NOT_ESTABLISHED"
    assert "ROLE_OPPORTUNITY_NOT_ESTABLISHED" in r["warnings"]

def test_expired_or_wrong_game_rejects_pregame_evidence():
    with pytest.raises(ValueError,match="EXACT_SCHEDULED_MATCH"):
        m.build(db(),FORECAST,"TB","DAL",NOW,active)
    with pytest.raises(ValueError,match="GAME_ALREADY_STARTED"):
        m.build(db(),FORECAST,"JAX","PHI",
                datetime(2026,10,11,13,30,tzinfo=timezone.utc),active)
