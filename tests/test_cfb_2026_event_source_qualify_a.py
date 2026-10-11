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
        CREATE TABLE schedule_snapshot (game_id TEXT, season INT, espn_status TEXT,
             espn_state TEXT DEFAULT 'post');
        CREATE TABLE player_games (player_id TEXT, game_id TEXT, season INT, week INT,
             position TEXT, carries INT, rushing_yards INT,
             pass_attempts INT, passing_touchdowns INT, passing_yards INT,
             receptions INT, receiving_yards INT, team TEXT DEFAULT 'A');
    """)
    event.executescript("""
        CREATE TABLE rush_carries (player_id TEXT, game_id TEXT, week INT,
             season INT, team TEXT, yards INT, carry_index INTEGER DEFAULT 0);
        CREATE TABLE pass_attempts_log (player_id TEXT, game_id TEXT, week INT,
             season INT, team TEXT, yards INT, is_touchdown INT,
             attempt_index INTEGER DEFAULT 0);
        CREATE TABLE recv_catches (player_id TEXT, game_id TEXT, week INT,
             season INT, team TEXT, yards INT);
    """)
    for week in (1, 2, 3):
        gid = str(1000 + week)
        model.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?)",
                      (gid, 2026, week, f"2026-09-{week:02d}T16:00:00Z", "A", "B", 24, 14))
        model.execute("INSERT INTO schedule_snapshot (game_id,season,espn_status) VALUES (?,?,?)", (gid, 2026, "STATUS_FINAL"))
        model.execute("INSERT INTO player_games (player_id, game_id, season, week, position, carries, rushing_yards, pass_attempts, passing_touchdowns, passing_yards, receptions, receiving_yards) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      ("RB", gid, 2026, week, "RB", 2, 11, 0, 0, 0, 0, 0))
        model.execute("INSERT INTO player_games (player_id, game_id, season, week, position, carries, rushing_yards, pass_attempts, passing_touchdowns, passing_yards, receptions, receiving_yards) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      ("QB", gid, 2026, week, "QB", 0, 0, 2, 1, 16, 0, 0))
        model.execute("INSERT INTO player_games (player_id, game_id, season, week, position, carries, rushing_yards, pass_attempts, passing_touchdowns, passing_yards, receptions, receiving_yards) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      ("WR", gid, 2026, week, "WR", 0, 0, 0, 0, 0, 1, 16))
        event.executemany("INSERT INTO rush_carries (player_id,game_id,week,season,team,yards) VALUES (?,?,?,?,?,?)",
                          [("RB", gid, week, 2026, "A", 4),
                           ("RB", gid, week, 2026, "A", 7)])
        event.executemany("INSERT INTO pass_attempts_log (player_id,game_id,week,season,team,yards,is_touchdown) VALUES (?,?,?,?,?,?,?)",
                          [("QB", gid, week, 2026, "A", 16, 1),
                           ("QB", gid, week, 2026, "A", 0, 0)])
        event.execute("INSERT INTO recv_catches VALUES (?,?,?,?,?,?)",
                      ("WR", gid, week, 2026, "A", 16))
    model.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?)",
                  ("FUTURE", 2026, 6, "2026-10-10T19:00:00Z", "A", "B", None, None))
    model.execute("INSERT INTO schedule_snapshot (game_id,season,espn_status) VALUES (?,?,?)",
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
    assert report["passing_td_event_semantics_certified"] is True
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


def test_turnover_team_tag_flip_not_falsely_treated_as_athlete_transfer(tmp_path):
    model, event, schedule, player = setup(tmp_path)
    event.execute("UPDATE rush_carries SET team='B' WHERE rowid=(SELECT MIN(rowid) FROM rush_carries)")
    event.commit()
    report = Q.qualify(model, event, schedule, player,
                       [pkt("rushing_yards", "RB")], received_at=AT)
    assert report["operational_source_status"] == "MIXED_SOURCE_TEAM_TAGS_WITH_ESPN_PARITY"
    market = report["comparisons"]["rushing_yards"]
    assert market["ambiguous_player_game_source_team_groups"] == 1
    assert market["event_missing_player_games"] == 0
    assert market["event_matched_player_games"] == 3
    assert market["mixed_team_tag_groups_with_exact_ESPN_stat_parity"] == 1
    assert set(market["ambiguous_source_team_samples"][0]["source_teams"].split(",")) == {"A", "B"}
    # Current roster/home-owner truth, not raw source's possession/defense tag.
    assert report["potential_future_picks_only"]["rushing_yards"]["source_eligible"] == 1
    assert report["readiness_to_publish_forecasts"] is False


def test_mixed_source_label_and_incorrect_stats_remains_hard_reconciliation_failure(tmp_path):
    model, event, schedule, player = setup(tmp_path)
    event.execute("UPDATE rush_carries SET team='B', yards=500 WHERE rowid=(SELECT MIN(rowid) FROM rush_carries)")
    event.commit()
    report = Q.qualify(model, event, schedule, player, [], received_at=AT)
    market = report["comparisons"]["rushing_yards"]
    assert market["mixed_team_tag_groups_failing_ESPN_stat_parity"] == 1
    assert market["event_mismatched_player_games"] == 1
    assert report["operational_source_status"] == "EVENT_RECONCILIATION_MISMATCH"


def test_fbs_only_source_scope_and_no_duplicate_game_ids(tmp_path):
    model, event, schedule, player = setup(tmp_path)
    games, kicks = Q.source_games(schedule)
    assert "FCS_ONLY" not in games
    assert games == {"1001", "1002", "1003", "FUTURE"}
    with schedule.open("a") as f:
        f.write("1001,fbs,fbs,regular,2026,2026-09-01T16:00:00Z\n")
    with pytest.raises(Q.QualificationError, match="DUPLICATE_SOURCE_GAME"):
        Q.source_games(schedule)


def shadow_ready_fixture(tmp_path):
    import cfb_2026_event_shadow_forecast_a as SH
    model, event, schedule, player = setup(tmp_path)
    for week in (1, 2, 3):
        gid = str(1000 + week)
        event.execute("DELETE FROM rush_carries WHERE game_id=?", (gid,))
        event.execute("DELETE FROM pass_attempts_log WHERE game_id=?", (gid,))
        event.executemany("INSERT INTO rush_carries (player_id,game_id,week,season,team,yards) VALUES (?,?,?,?,?,?)",
                          [("RB", gid, week, 2026, "A", 5)] * 15)
        event.executemany("INSERT INTO pass_attempts_log (player_id,game_id,week,season,team,yards,is_touchdown) VALUES (?,?,?,?,?,?,?)",
                          [("QB", gid, week, 2026, "A", 3, int(i < 2))
                           for i in range(20)])
    model.execute("UPDATE player_games SET carries=15,rushing_yards=75 WHERE position='RB'")
    model.execute("UPDATE player_games SET pass_attempts=20,passing_yards=60,passing_touchdowns=2 WHERE position='QB'")
    model.execute("UPDATE schedule_snapshot SET espn_state='pre' WHERE game_id='FUTURE'")
    model.commit()
    event.commit()
    board = {"picks": [
        {**pkt("rushing_yards", "RB"), "line": 69.5},
        {**pkt("passing_touchdowns", "QB"), "line": 1.5},
    ]}
    qa = Q.qualify(model, event, schedule, player, board["picks"], received_at=AT)
    return SH, model, event, schedule, board, qa


def test_genuine_future_pre_game_shadow_emits_real_yard_and_touchdown_point_forecasts(tmp_path):
    SH, model, event, schedule, board, qa = shadow_ready_fixture(tmp_path)
    when = AT + timedelta(minutes=5)
    out = SH.shadow(model, event, board, qa, schedule, generated_at=when,
                    n_simulations=200)
    assert out["status"] == SH.RESEARCH_STATUS
    assert out["shadow_predictions"] == 2
    assert out["historical_backfills"] == 0
    market = {r["market"]: r for r in out["shadow"]}
    rushing = market["rushing_yards"]
    assert rushing["projected_mean"] == 75.0
    assert rushing["projected_median"] == 75.0
    assert rushing["p10"] == 75.0 and rushing["p90"] == 75.0
    assert rushing["p_over_fixed_line"] == 1
    hist = rushing["empirical_outcome_histogram"]
    assert sum(bin["count"] for bin in hist) == 200
    assert hist == [{"stat_total": 75, "count": 200}]
    assert rushing["sample_mean_unrounded"] == 75
    assert rushing["empirical_outcome_histogram_schema"] == "INTEGER_STAT_TOTAL_AND_FREQUENCY_V1"
    passing = market["passing_touchdowns"]
    assert 0 <= passing["projected_mean"] <= 20
    assert passing["n_simulations"] == 200
    assert passing["prior_verified_same_team_game_ids"] == ["1001", "1002", "1003"]
    assert all(r["research_only"] and not r["historical_original_prediction"] for r in out["shadow"])
    assert SH.shadow(model, event, board, qa, schedule, generated_at=when,
                     n_simulations=200) == out  # deterministic and repeatable


def test_no_future_shadow_if_ran_at_or_after_kickoff(tmp_path):
    SH, model, event, schedule, board, qa = shadow_ready_fixture(tmp_path)
    late = datetime(2026, 10, 10, 19, 0, tzinfo=timezone.utc)
    out = SH.shadow(model, event, board, qa, schedule, generated_at=late,
                    n_simulations=200)
    assert out["shadow_predictions"] == 0
    assert out["excluded"]["NOT_GENUINELY_PREGAME"] == 2


def test_incorrect_source_play_totals_cannot_enter_shadow_pool(tmp_path):
    SH, model, event, schedule, board, qa = shadow_ready_fixture(tmp_path)
    event.execute("UPDATE rush_carries SET yards=500 WHERE rowid=(SELECT MIN(rowid) FROM rush_carries)")
    event.commit()
    out = SH.shadow(model, event, board, qa, schedule,
                    generated_at=AT + timedelta(minutes=5), n_simulations=200)
    assert out["shadow_predictions"] == 1
    assert out["shadow"][0]["market"] == "passing_touchdowns"
    assert out["excluded"]["LATEST_3_GAMES_NOT_FULLY_RECONCILED"] == 1


def test_undetectable_pass_TD_source_must_not_simulate_zero_touchdown_forecasts(tmp_path):
    SH, model, event, schedule, board, qa = shadow_ready_fixture(tmp_path)
    event.execute("UPDATE pass_attempts_log SET is_touchdown=0")
    event.commit()
    qa = Q.qualify(model, event, schedule, tmp_path / "player_stats_2026.csv",
                   board["picks"], received_at=AT)
    assert qa["passing_td_event_semantics_certified"] is False
    assert qa["passing_td_event_semantics_evidence"]["espn_positive_passing_td_player_games"] == 3
    assert qa["passing_td_event_semantics_evidence"]["event_positive_touchdown_indicators"] == 0
    result = SH.shadow(model, event, board, qa, schedule,
                       generated_at=AT + timedelta(minutes=5), n_simulations=200)
    assert result["shadow_predictions"] == 1  # verified rushing still permitted
    assert result["shadow"][0]["market"] == "rushing_yards"
    assert result["excluded"]["PASSING_TD_EVENT_LABELS_UNVALIDATED"] == 1


def test_shadow_cannot_skip_broken_recent_game_and_use_older_matches(tmp_path):
    SH, model, event, schedule, board, qa = shadow_ready_fixture(tmp_path)
    model.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?)",
                  ("1004", 2026, 4, "2026-09-04T16:00:00Z", "A", "B", 20, 7))
    model.execute("INSERT INTO schedule_snapshot (game_id,season,espn_status) VALUES (?,?,?)",
                  ("1004", 2026, "STATUS_FINAL"))
    model.execute("INSERT INTO player_games (player_id,game_id,season,week,position,carries,rushing_yards,pass_attempts,passing_touchdowns,passing_yards,receptions,receiving_yards) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  ("RB", "1004", 2026, 4, "RB", 15, 75, 0, 0, 0, 0, 0))
    event.executemany(
        "INSERT INTO rush_carries (player_id,game_id,week,season,team,yards) VALUES (?,?,?,?,?,?)",
        [("RB", "1004", 4, 2026, "A", 5)] * 15)
    # W3 breaks the latest sequence: W4 is good, W3 is bad, W2/W1 good.
    event.execute(
        "UPDATE rush_carries SET yards=500 WHERE rowid="
        "(SELECT MIN(rowid) FROM rush_carries WHERE game_id='1003')")
    event.commit()
    model.commit()
    with schedule.open("a") as fp:
        fp.write("1004,fbs,fbs,regular,2026,2026-09-04T16:00:00Z\n")
    qa = Q.qualify(model, event, schedule, tmp_path / "player_stats_2026.csv",
                   board["picks"], received_at=AT)
    out = SH.shadow(model, event, board, qa, schedule,
                    generated_at=AT + timedelta(minutes=5), n_simulations=200)
    assert out["excluded"]["LATEST_3_GAMES_NOT_FULLY_RECONCILED"] == 1
    assert not any(x["market"] == "rushing_yards" for x in out["shadow"])


def test_future_source_attestation_and_clock_are_required(tmp_path):
    SH, model, event, schedule, board, qa = shadow_ready_fixture(tmp_path)
    qa["captured_at_utc"] = "2026-10-10T22:00:00Z"
    with pytest.raises(Q.QualificationError, match="SOURCE_ATTESTED_IN_THE_FUTURE"):
        SH.shadow(model, event, board, qa, schedule, generated_at=AT,
                  n_simulations=200)


def test_shadow_rejects_source_file_mutated_after_audit(tmp_path):
    SH, model, event, schedule, board, qa = shadow_ready_fixture(tmp_path)
    with schedule.open("a") as fp:
        fp.write("tampered-source-after-hash\n")
    with pytest.raises(Q.QualificationError, match="SOURCE_SCHEDULE_HASH_MISMATCH"):
        SH.shadow(model, event, board, qa, schedule,
                  generated_at=AT + timedelta(minutes=5), n_simulations=200)


def test_shadow_rejects_changed_original_play_level_source_bytes(tmp_path):
    SH, model, event, schedule, board, qa = shadow_ready_fixture(tmp_path)
    source_file = tmp_path / "player_stats_2026.csv"
    source_file.write_text("tampered!")
    with pytest.raises(Q.QualificationError, match="SOURCE_PLAYER_EVENTS_HASH_MISMATCH"):
        SH.shadow(model, event, board, qa, schedule, player_csv=source_file,
                  generated_at=AT + timedelta(minutes=5), n_simulations=200)


def test_missing_or_naive_clock_refused():
    with pytest.raises(Q.QualificationError, match="NAIVE_RECEIPT_CLOCK"):
        Q.utc(datetime(2026, 10, 10, 14, 30))
    with pytest.raises(Q.QualificationError, match="INVALID_TIME"):
        Q.utc("not-a-timestamp")
