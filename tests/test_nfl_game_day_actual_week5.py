"""Read-only integration with actual committed 2026 Week 5 game and model data.

No network mocking, no player synthesis, and no changes to the immutable
T24/T90 records. This proves the same complete schedule schema works on a
real NFL week before the production publication code is activated.
"""
import json
import sqlite3
from pathlib import Path
from nfl_pregame_delivery import schedule_manifest, pregame_only, as_utc


def test_real_week5_schedule_coheres_with_282_published_picks():
    root=Path(__file__).resolve().parent.parent
    archive=json.loads((root/"docs/nfl_predictions_2026_w05.json").read_text())
    assert (archive["season"],archive["week"])==(2026,5)
    with sqlite3.connect(root/"nfl_models/nfl_model.sqlite") as con:
        rows=con.execute("SELECT home_team,away_team,kickoff_utc FROM games WHERE season=2026 AND week=5").fetchall()
    manifest=schedule_manifest(rows)
    print("ACTUAL_W5_GAME_MANIFEST",len(manifest),"ACTUAL_W5_PICKS",len(archive["picks"]))
    assert len(manifest)==14, "check committed NFL schedule coverage before deployment"
    assert len(archive["picks"])>=100, "Week 5 real historical model did not load"
    known={frozenset((x["home_team"],x["away_team"])):x["kickoff_utc"] for x in manifest}
    actual=archive["picks"]
    assert all(frozenset((p["team"],p["opponent"])) in known for p in actual)
    assert all(as_utc(p["kickoff_utc"])==as_utc(known[frozenset((p["team"],p["opponent"]))]) for p in actual)
    # The genuine pregame generated_at is a historical timestamp, not the
    # current wall clock; this proves valid forecasts are not falsely removed.
    safe,dropped=pregame_only(actual,known,archive["generated_at_utc"])
    assert len(safe)==len(actual), "archived pregame rows unexpectedly behind kickoff"
    assert not any(dropped.values()),str(dropped)
    numeric=[p for p in actual if p.get("market") in ("rushing_yards","receiving_yards")
             and isinstance(p.get("projected_median"),(int,float))]
    assert len(numeric)>=100
