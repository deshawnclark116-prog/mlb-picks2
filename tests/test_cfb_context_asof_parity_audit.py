"""Reproduce an actual postgame-training information leak, not a tuned result."""
import sqlite3
import cfb_context_v2
import cfb_context_asof_parity_audit as integrity


def db():
    con=sqlite3.connect(":memory:")
    con.executescript("""
        CREATE TABLE player_games (
          season INTEGER,week INTEGER,game_id TEXT,game_date TEXT,
          team TEXT,player_id TEXT,position TEXT,
          carries REAL,receptions REAL,pass_attempts REAL
        );
        CREATE TABLE games (
          season INTEGER,week INTEGER,home_team TEXT,away_team TEXT,
          home_points REAL,away_points REAL,neutral_site INTEGER
        );
    """)
    for week in (1,2,3,4,5):
        game=f"2025_{week:02d}_TEST"
        con.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?)",
                    (2025,week,"TEAM","OPP",21+week,14,0))
        # A is the established runner through week 3 but *does not*
        # appear in target Week 4. A new player B appears Week 4.
        if week<=3:
            team=[("A","RB",16,0,0),("QB","QB",0,0,30),
                  ("R","WR",0,6,0)]
        else:
            team=[("B","RB",8,0,0),("QB","QB",0,0,28),
                  ("R","WR",0,6,0)]
        for pid,pos,c,r,p in team:
            con.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (2025,week,game,f"2025-09-{week:02}","TEAM",pid,pos,c,r,p))
        con.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (2025,week,game,f"2025-09-{week:02}","OPP","O","RB",20,0,0))
    con.commit()
    return con


def test_target_game_future_absence_leaks_into_legacy_training():
    con=db()
    old=cfb_context_v2.build_context(con)
    # This week-4 value is created using the *actual week-4* participation,
    # which is impossible to know at a week-4 pregame forecast cutoff.
    assert old[(2025,"B",4)]["vacated_car"]>0.5
    clean=integrity.PriorWeekUsage(con).features(2025,4,"TEAM","B")
    assert clean["vacated_car"] is None
    assert clean["vacated_rec"] is None
    assert clean["qb_change"] is None
    assert clean["carry_share_last3"]==0
    assert clean["team_plays_last3"]>0


def test_same_target_week_results_cannot_modify_pregame_features():
    con=db()
    before=integrity.PriorWeekUsage(con).features(2025,4,"TEAM","B")
    con.execute("UPDATE player_games SET carries=999,receptions=500,pass_attempts=900 "
                "WHERE season=2025 AND week=4")
    after=integrity.PriorWeekUsage(con).features(2025,4,"TEAM","B")
    assert before==after
    # The historical production builder DOES change target-game-derived
    # absence if B's own participation is changed.
    old=cfb_context_v2.build_context(con)
    assert old[(2025,"B",4)]["vacated_car"]!=before["vacated_car"]


def test_earlier_target_week_may_not_see_later_weeks():
    con=db()
    before=integrity.PriorWeekUsage(con).features(2025,3,"TEAM","A")
    con.execute("UPDATE player_games SET carries=999 WHERE week>3")
    assert before==integrity.PriorWeekUsage(con).features(2025,3,"TEAM","A")


def test_team_rating_must_be_target_week_asof_not_final_season_rating():
    con=db()
    first=cfb_context_v2.build_ratings(con)
    before=integrity.checked_asof_rating(first,2025,4,"TEAM","OPP")
    con.execute("UPDATE games SET home_points=1000,away_points=0 WHERE week>=4")
    after=cfb_context_v2.build_ratings(con)
    assert integrity.checked_asof_rating(after,2025,4,"TEAM","OPP")==before
    # Latest frozen-season rating has learned the target and future results.
    assert cfb_context_v2.build_ratings.latest[2025]["TEAM"] != before["team_rating"]


def test_audit_counts_nonzero_target_game_features_without_model_change():
    con=db()
    report=integrity.audit(con,season=2025)
    assert report["number_of_evaluated_historical_player_weeks"]>0
    assert report["number_of_nonzero_target_game_derived_features"]["vacated_car"]>0
    assert report["status"]=="RESEARCH_ONLY_NO_MODEL_PROMOTION"
    assert report["zero_predictive_code_changes"] is True
    assert report["asof_safe_values_for_unverified_pregame_absence"]["vacated_car"] is None
