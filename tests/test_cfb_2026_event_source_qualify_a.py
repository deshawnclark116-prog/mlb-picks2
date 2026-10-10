"""Source-integrity tests: fake data only; never produce actual forecasts."""
import csv
import json
import sqlite3
from datetime import datetime, timezone, timedelta

import pytest

import cfb_2026_event_source_qualify_a as Q


AT = datetime(2026, 10, 10, 14, 30, tzinfo=timezone.utc)


def setup(tmp_path):
    model, event = sqlite3.connect(":memory:"), sqlite3.connect(":memory:")
    model.row_factory, event.row_factory = sqlite3.Row, sqlite3.Row
    model.executescript("""
        CREATE TABLE games (game_id TEXT, season INT, week INT, kickoff_utc TEXT,
             home_team TEXT, away_team TEXT, home_points INT, away_points INT);
        CREATE TABLE schedule_snapshot (game_id TEXT, season INT, espn_status TEXT);
        CREATE TABLE player_games (player_id TEXT, game_id TEXT, season INT, week INT,
             position TEXT, carries INT, rushing_yards INT,
             pass_attempts INT, passing_touchdowns INT, passing_yards INT,
             receptions INT, receiving_yards INT);
    """)
    event.executescript("""
        CREATE TABLE rush_carries (player_id TEXT, game_id TEXT, week INT,
             season INT, team TEXT, yards INT);
        CREATE TABLE pass_attempts_log (player_id TEXT, game_id TEXT, week INT,
             season INT, team TEXT, yards INT, is_touchdown INT);
        CREATE TABLE recv_catches (player_id TEXT, game_id TEXT, week INT,
             season INT, team TEXT, yards INT);
    """)
    for week in (1, 2, 3):
        gid = str(1000 + week)
        model.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?)",
                      (gid, 2026, week, f"2026-09-{week:02d}T16:00:00Z", "A", "B", 24, 14))
        model.execute("INSERT INTO schedule_snapshot VALUES (?,?,?)", (gid, 2026, "STATUS_FINAL"))
        model.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      ("RB", gid, 2026, week, "RB", 2, 11, 0, 0, 0, 0, 0))
        model.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      ("QB", gid, 2026, week, "QB", 0, 0, 2, 1, 16, 0, 0))
        model.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      ("WR", gid, 2026, week, "WR", 0, 0, 0, 0, 0, 1, 16))
        event.executemany("INSERT INTO rush_carries VALUES (?,?,?,?,?,?)",
                          [("RB", gid, week, 2026, "A", 4),
                           ("RB", gid, week, 2026, "A", 7)])
        event.executemany("INSERT INTO pass_attempts_log VALUES (?,?,?,?,?,?,?)",
                          [("QB", gid, week, 2026, "A", 16, 1),
                           ("QB", gid, week, 2026, "A", 0, 0)])
        event.execute("INSERT INTO recv_catches VALUES (?,?,?,?,?,?)",
                      ("WR", gid, week, 2026, "A", 16))
    model.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?)",
                  ("FUTURE", 2026, 6, "2026-10-10T19:00:00Z", "A", "B", None, None))
    model.execute("INSERT INTO schedule_snapshot VALUES (?,?,?)",
                  ("FUTURE", 2026, "STATUS_SCHEDULED"))
    model.commit()
    event.commit()
    csvfile = tmp_path / "schedules_2026.csv"
    with csvfile.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game_id", "home_division", "away_division",
                                          "season_type", "season", "start_date"])
        w.writeheader()
        for week in (1, 2, 3):
            w.writerow({"game_id": str(1000 + week), "home_division": "fbs",
                        "away_division": "fbs", "season_type": "regular", "season": 2026,
                        "start_date": f"2026-09-{week:02d}T16:00:00Z"})
        w.writerow({"game_id": "FUTURE", "home_division": "fbs",
                    "away_division": "fbs", "season_type": "regular", "season": 2026,
                    "start_date": "2026-10-10T19:00:00Z"})
        w.writerow({"game_id": "FCS_ONLY", "home_division": "fcs",
                    "away_division": "fcs", "season_type": "regular", "season": 2026,
                    "start_date": "2026-09-01T16:00:00Z"})
    playerfile = tmp_path / "player_stats_2026.csv"
    playerfile.write_bytes(b"example player play rows represented in in-memory event db\n")
    return model, event, csvfile, playerfile


def pkt(market, player):
    return {"season": 2026, "week": 6, "market": market,
            "game_id": "FUTURE", "player_id": player, "team": "A",
            "kickoff_utc": "2026-10-10T19:00:00Z"}


def test_independent_all_four_markets_match_and_no_backfilled_claim(tmp_path):
    model, event, schedule, player = setup(tmp_path)
    rows = [pkt("rushing_yards", "RB"), pkt("passing_touchdowns", "QB"),
            pkt("rushing_yards", "MISSING")]
    report = Q.qualify(model, event, schedule, player, rows, received_at=AT)
    assert report["source_games_total"] == 4
    assert report["espn_confirmed_final_games"] == 3
    assert report["overlapping_completed_games"] == 3
    for market in Q.EVENT_TABLES:
        section = report["comparisons"][market]
        assert section["event_missing_player_games"] == 0
        assert section["event_mismatched_player_games"] == 0
        assert section["event_matched_player_games"] == 3
    assert report["operational_source_status"] == "MATCHED_OVERLAP_NOT_HISTORICALLY_VINTAGED"
    assert report["potential_future_picks_only"]["rushing_yards"] == {
        "picks": 2, "source_eligible": 1
    }
    assert report["potential_future_picks_only"]["passing_touchdowns"]["source_eligible"] == 1
    assert report["historical_source_publication_vintage_verified"] is False
    assert report["promoted"] is False
    assert report["eligible_to_overwrite_original_pregame_forecasts"] is False
    assert len(report["raw_source_files"]["player_stats"]["sha256"]) == 64


def test_refuse_2026_event_changes_after_qualifying_snapshot(tmp_path):
    model, event, schedule, player = setup(tmp_path)
    event.execute("UPDATE pass_attempts_log SET is_touchdown=0 WHERE game_id='1001'")
    event.commit()
    report = Q.qualify(model, event, schedule, player, [], received_at=AT)
    assert report["operational_source_status"] == "EVENT_RECONCILIATION_MISMATCH"
    assert report["comparisons"]["passing_touchdowns"]["event_mismatched_player_games"] == 1
    assert report["comparisons"]["passing_yards"]["event_mismatched_player_games"] == 0


def test_missing_events_are_not_zero_volume_or_certified(tmp_path):
    model, event, schedule, player = setup(tmp_path)
    event.execute("DELETE FROM rush_carries WHERE game_id='1001'")
    event.commit()
    report = Q.qualify(model, event, schedule, player, [], received_at=AT)
    assert report["operational_source_status"] == "EVENT_RECONCILIATION_MISMATCH"
    assert report["comparisons"]["rushing_yards"]["event_missing_player_games"] == 1


def test_wrong_team_same_athlete_id_does_not_qualify_future_projection(tmp_path):
    model, event, schedule, player = setup(tmp_path)
    pick = pkt("rushing_yards", "RB")
    pick["team"] = "B"  # same player ID, wrong team; previous code included A's history
    report = Q.qualify(model, event, schedule, player, [pick], received_at=AT)
    assert report["potential_future_picks_only"]["rushing_yards"]["source_eligible"] == 0


def test_source_schedule_kickoff_disagreement_blocks_qualification(tmp_path):
    model, event, schedule, player = setup(tmp_path)
    model.execute("UPDATE games SET kickoff_utc='2026-09-01T18:15:00Z' WHERE game_id='1001'")
    model.commit()
    report = Q.qualify(model, event, schedule, player, [], received_at=AT)
    assert report["schedule_kickoff_disagreements_over_2min"] == 1
    assert report["operational_source_status"] == "EVENT_RECONCILIATION_MISMATCH"
    assert report["readiness_to_publish_forecasts"] is False


def test_no_retroactive_historical_pregame_eligibility(tmp_path):
    model, event, schedule, player = setup(tmp_path)
    p = pkt("rushing_yards", "RB")
    p["week"] = 4
    p["kickoff_utc"] = "2026-10-01T19:00:00Z"
    report = Q.qualify(model, event, schedule, player, [p], received_at=AT)
    assert report["potential_future_picks_only"]["rushing_yards"] == {
        "picks": 1, "source_eligible": 0
    }
    assert report["historic_forecast_replay"].startswith("BLOCKED")


def test_no_false_final_or_future_event_source(tmp_path):
    model, event, schedule, player = setup(tmp_path)
    model.execute("UPDATE schedule_snapshot SET espn_status='STATUS_IN_PROGRESS' WHERE game_id='1003'")
    model.commit()
    report = Q.qualify(model, event, schedule, player,
                       [pkt("rushing_yards", "RB")], received_at=AT)
    assert report["espn_confirmed_final_games"] == 2
    assert report["potential_future_picks_only"]["rushing_yards"]["source_eligible"] == 0


def test_duplicate_player_teams_in_single_game_fail_closed(tmp_path):
    model, event, schedule, player = setup(tmp_path)
    event.execute("UPDATE rush_carries SET team='B' WHERE rowid=(SELECT MIN(rowid) FROM rush_carries)")
    event.commit()
    with pytest.raises(Q.QualificationError, match="CROSS_TEAM_SAME_PLAYER_GAME_EVENTS"):
        Q.qualify(model, event, schedule, player, [], received_at=AT)


def test_fbs_only_source_scope_and_no_duplicate_game_ids(tmp_path):
    model, event, schedule, player = setup(tmp_path)
    games, kicks = Q.source_games(schedule)
    assert "FCS_ONLY" not in games
    assert games == {"1001", "1002", "1003", "FUTURE"}
    with schedule.open("a") as f:
        f.write("1001,fbs,fbs,regular,2026,2026-09-01T16:00:00Z\n")
    with pytest.raises(Q.QualificationError, match="DUPLICATE_SOURCE_GAME"):
        Q.source_games(schedule)


def test_missing_or_naive_clock_refused():
    with pytest.raises(Q.QualificationError, match="NAIVE_RECEIPT_CLOCK"):
        Q.utc(datetime(2026, 10, 10, 14, 30))
    with pytest.raises(Q.QualificationError, match="INVALID_TIME"):
        Q.utc("not-a-timestamp")
