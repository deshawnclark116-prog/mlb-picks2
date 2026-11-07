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
