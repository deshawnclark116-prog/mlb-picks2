"""Schedule truth from NFL/current team primary sources, not stale fixture feed."""
import csv
import io
import sqlite3
from datetime import datetime, timezone

import pytest

from nfl_verified_kickoff import (
    VERSION,REVISION,fix_fixture_clock,reconcile_db_schedule_rows
)
from nfl_game_day_contract import validate
from nfl_shadow_schedule import sanitize


def shadow_csv():
    buf=io.StringIO()
    writer=csv.DictWriter(buf,fieldnames=[
        "game_id","season","game_type","week","gameday","gametime",
        "away_team","home_team","home_rest","away_rest",
        "home_coach","away_coach","result","spread_line"])
    writer.writeheader()
    writer.writerow(dict(game_id="2026_09_LV_SF",season=2026,
            game_type="REG",week=9,gameday="2026-11-08",
            gametime="16:05",away_team="LV",home_team="SF",
            home_rest=7,away_rest=7,home_coach="Shanahan",
            away_coach="Kubiak",result="",spread_line="-3.5"))
    writer.writerow(dict(game_id="2026_09_ARI_SEA",season=2026,
            game_type="REG",week=9,gameday="2026-11-08",
            gametime="16:25",away_team="ARI",home_team="SEA",
            home_rest=7,away_rest=7,home_coach="Coach",
            away_coach="Coach",result="",spread_line="-2.5"))
    return buf.getvalue().encode()


def test_only_exact_verified_future_nfl_fixture_is_corrected_and_raw_bytes_unmodified():
    raw=shadow_csv()
    before=bytes(raw)
    sanitized,raw_sha,transform=sanitize(
        raw,now=datetime(2026,11,7,22,0,tzinfo=timezone.utc))
    assert before==raw
    assert "spread_line" not in sanitized.decode()
    assert VERSION in transform
    rows={r["game_id"]:r for r in csv.DictReader(io.StringIO(sanitized.decode()))}
    assert rows["2026_09_LV_SF"]["gametime"]=="16:25"
    assert rows["2026_09_ARI_SEA"]["gametime"]=="16:25"
    assert rows["2026_09_LV_SF"]["result"]==""
    assert raw_sha==__import__('hashlib').sha256(raw).hexdigest()
    # Source verification must be byte-deterministic for an immutable CAS.
    assert sanitize(raw,now=datetime(2026,11,7,22,0,tzinfo=timezone.utc))==(sanitized,raw_sha,transform)


def test_direct_kickoff_metadata_is_corrected_only_for_the_verified_game():
    rows=[("SF","LV","2026-11-08T21:05:00Z"),
          ("SEA","ARI","2026-11-08T21:25:00Z")]
    corrected,proof=reconcile_db_schedule_rows(rows,2026,9)
    assert corrected==[
        ("SF","LV","2026-11-08T21:25:00Z"),
        ("SEA","ARI","2026-11-08T21:25:00Z")
    ]
    assert len(proof)==1
    assert proof[0]["primary_sources"]==REVISION["primary_sources"]
    assert proof[0]["already_correct"] is False
    same,recheck=reconcile_db_schedule_rows(corrected,2026,9)
    assert same==corrected and recheck[0]["already_correct"] is True
    untouched,e=reconcile_db_schedule_rows(rows,2025,9)
    assert untouched==rows and not e


def test_refuse_future_conflicting_kickoff_without_a_new_verified_source():
    row={"season":2026,"week":9,"away_team":"LV","home_team":"SF",
         "gameday":"2026-11-08","gametime":"16:30"}
    with pytest.raises(ValueError,match="THIRD_CONFLICTING_SOURCE_TIME"):
        fix_fixture_clock(row)
    with pytest.raises(ValueError,match="THIRD_CONFLICTING_SOURCE_TIME"):
        reconcile_db_schedule_rows([("SF","LV","2026-11-08T21:30:00Z")],2026,9)
    with pytest.raises(ValueError,match="CONFLICTING_GAME_DATE"):
        fix_fixture_clock({**row,"gameday":"2026-11-09","gametime":"16:05"})


def test_serving_validation_uses_identical_source_corrected_schedule(tmp_path):
    db=tmp_path/"nfl.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE games (home_team TEXT, away_team TEXT, kickoff_utc TEXT, season INTEGER,week INTEGER)")
        conn.execute("INSERT INTO games VALUES (?,?,?,?,?)",
                     ("SF","LV","2026-11-08T21:05:00Z",2026,9))
    doc={
        "season":2026,"week":9,"generated_at_utc":"2026-11-07T22:30:00Z",
        "scheduled_games":[{"home_team":"SF","away_team":"LV",
                            "kickoff_utc":"2026-11-08T21:25:00Z"}],
        "picks":[]
    }
    now=datetime(2026,11,7,23,0,tzinfo=timezone.utc)
    approved=validate(doc,db,now)
    assert approved["status"]=="VALID_PREGAME_PUBLICATION"
    assert approved["future"]==1
    stale={**doc,"scheduled_games":[{
        "home_team":"SF","away_team":"LV","kickoff_utc":"2026-11-08T21:05:00Z"}]}
    with pytest.raises(ValueError,match="PUBLISHED_SCHEDULE_DIFFERS_FROM_FOUNDATION"):
        validate(stale,db,now)


# Week 11, Nov 22, 2026. The official NFL announced a swap on Nov 9:
# PIT@PHI 4:25 -> 1:00 ET, JAX@NYG 1:00 -> 4:25 ET.
def week11_shadow_csv():
    buf=io.StringIO()
    fields=["game_id","season","game_type","week","gameday","gametime",
            "away_team","home_team","home_rest","away_rest",
            "home_coach","away_coach","result","spread_line"]
    w=csv.DictWriter(buf,fieldnames=fields)
    w.writeheader()
    for away,home,clock in [("PIT","PHI","16:25"),("JAX","NYG","13:00"),("BAL","CAR","13:00")]:
        w.writerow(dict(game_id=f"2026_11_{away}_{home}",season=2026,
          game_type="REG",week=11,gameday="2026-11-22",
          gametime=clock,away_team=away,home_team=home,
          home_rest=7,away_rest=7,home_coach="Coach",
          away_coach="Coach",result="",spread_line="-3.5"))
    return buf.getvalue().encode()


def test_week11_official_swapped_games_corrected_in_serving_and_shadow_without_touching_raw():
    from nfl_verified_kickoff import WEEK11_REVISIONS
    from nfl_phase1d_schedule import parse_schedule, forecast_cutoff
    from datetime import timedelta
    raw=week11_shadow_csv()
    original=bytes(raw)
    sanitized,sha,transformation=sanitize(
        raw,now=datetime(2026,11,21,23,0,tzinfo=timezone.utc))
    assert raw==original
    assert sha==__import__("hashlib").sha256(raw).hexdigest()
    assert "spread_line" not in sanitized.decode()
    assert VERSION in transformation
    schedule=parse_schedule(sanitized)
    phi=schedule["2026_11_PIT_PHI"]
    nyg=schedule["2026_11_JAX_NYG"]
    bal=schedule["2026_11_BAL_CAR"]
    assert phi["kick"].isoformat()=="2026-11-22T18:00:00+00:00"
    assert nyg["kick"].isoformat()=="2026-11-22T21:25:00+00:00"
    assert bal["kick"].isoformat()=="2026-11-22T18:00:00+00:00"
    # Fresh scheduler keys and deadlines follow the VERIFIED current kickoff.
    assert forecast_cutoff(phi["kick"],"T90").isoformat()=="2026-11-22T16:30:00+00:00"
    assert forecast_cutoff(nyg["kick"],"T90").isoformat()=="2026-11-22T19:55:00+00:00"
    assert forecast_cutoff(phi["kick"],"T24").isoformat()=="2026-11-21T18:00:00+00:00"
    # A previously captured 4:25 PHI cutoff is NOT the 1 PM game's T24.
    assert forecast_cutoff(phi["kick"],"T24") != (
        datetime(2026,11,22,21,25,tzinfo=timezone.utc)-timedelta(days=1))
    assert {r["home_team"] for r in WEEK11_REVISIONS}=={"PHI","NYG"}
    # Sanitization is byte-deterministic and never modifies original provider data.
    assert sanitize(raw,now=datetime(2026,11,21,23,0,tzinfo=timezone.utc))==(sanitized,sha,transformation)


def test_week11_serving_reconciles_both_flexes_exactly_and_is_idempotent():
    from nfl_verified_kickoff import WEEK11_REVISIONS
    stale=[("PHI","PIT","2026-11-22T21:25:00Z"),
           ("NYG","JAX","2026-11-22T18:00:00Z"),
           ("CAR","BAL","2026-11-22T18:00:00Z")]
    corrected,proof=reconcile_db_schedule_rows(stale,2026,11)
    assert corrected==[
        ("PHI","PIT","2026-11-22T18:00:00Z"),
        ("NYG","JAX","2026-11-22T21:25:00Z"),
        ("CAR","BAL","2026-11-22T18:00:00Z")]
    assert len(proof)==2
    assert all(not p["already_correct"] for p in proof)
    assert all("nfl.com/schedules/2026/by-week/week-11" in p["primary_sources"][1] for p in proof)
    same,proof2=reconcile_db_schedule_rows(corrected,2026,11)
    assert same==corrected and len(proof2)==2
    assert all(p["already_correct"] for p in proof2)
    for unexpected_week in [10,12]:
        untouched,evidence=reconcile_db_schedule_rows(stale,2026,unexpected_week)
        assert untouched==stale and evidence==[]
    assert len(WEEK11_REVISIONS)==2


def test_week11_third_unverified_time_fails_closed_without_guessing():
    row={"season":2026,"week":11,"home_team":"PHI","away_team":"PIT",
         "gameday":"2026-11-22","gametime":"15:00"}
    with pytest.raises(ValueError,match="THIRD_CONFLICTING_SOURCE_TIME"):
        fix_fixture_clock(row)
    with pytest.raises(ValueError,match="THIRD_CONFLICTING_SOURCE_TIME"):
        reconcile_db_schedule_rows([("NYG","JAX","2026-11-22T20:05:00Z")],2026,11)
    with pytest.raises(ValueError,match="CONFLICTING_GAME_DATE"):
        fix_fixture_clock({**row,"gameday":"2026-11-23","gametime":"16:25"})
    # Team identity and week are part of the proof; no generic blanket clock fix.
    untouched,proof=fix_fixture_clock({**row,"away_team":"ARI"})
    assert untouched["gametime"]=="15:00" and proof is None


def test_week11_publication_validator_requires_league_true_flexes(tmp_path):
    db=tmp_path/"nfl.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE games (home_team TEXT, away_team TEXT, kickoff_utc TEXT, season INTEGER,week INTEGER)")
        con.executemany("INSERT INTO games VALUES (?,?,?,?,?)",[
            ("PHI","PIT","2026-11-22T21:25:00Z",2026,11),
            ("NYG","JAX","2026-11-22T18:00:00Z",2026,11)])
    manifest=[
        {"home_team":"PHI","away_team":"PIT","kickoff_utc":"2026-11-22T18:00:00Z"},
        {"home_team":"NYG","away_team":"JAX","kickoff_utc":"2026-11-22T21:25:00Z"}]
    now=datetime(2026,11,21,23,0,tzinfo=timezone.utc)
    doc={"season":2026,"week":11,"generated_at_utc":"2026-11-21T22:58:00Z",
         "scheduled_games":manifest,"picks":[]}
    checked=validate(doc,db,now)
    assert checked["status"]=="VALID_PREGAME_PUBLICATION"
    assert checked["scheduled"]==2 and checked["future"]==2
    for original,rejected_time in [(0,"2026-11-22T21:25:00Z"),
                                   (1,"2026-11-22T18:00:00Z")]:
        bad=[dict(row) for row in manifest]
        bad[original]["kickoff_utc"]=rejected_time
        with pytest.raises(ValueError,match="PUBLISHED_SCHEDULE_DIFFERS_FROM_FOUNDATION"):
            validate({**doc,"scheduled_games":bad},db,now)
    # The same strict schedule check rejects a player still tied to the
    # old kickoff even when her/his projected yardage is numerically valid.
    bad_pick={"team":"PHI","opponent":"PIT","player":"Test Receiver",
              "player_id":"test001","market":"receiving_yards",
              "kickoff_utc":"2026-11-22T21:25:00Z",
              "projected_low":20.0,"projected_median":50.0,"projected_high":80.0}
    with pytest.raises(ValueError,match="PICK_KICKOFF_MISMATCH"):
        validate({**doc,"picks":[bad_pick]},db,now)
