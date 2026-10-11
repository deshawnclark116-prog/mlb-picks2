"""Immutable CFB shadow future moneyline checks, independent of live picks."""
from datetime import datetime, timezone
import json
import sqlite3

import pytest

import cfb_asof_forward_moneyline as m


class FakeScorer:
    def __init__(self):
        self.features = []

    def probability(self, market, base, season, week, team, opp, player):
        self.features.append((base, season, week, team, opp, player))
        return {"model_probability": 0.78 if team == "H" else 0.31}


def fixtures():
    con = sqlite3.connect(":memory:")
    con.execute("""CREATE TABLE games (
        game_id TEXT, season INTEGER, week INTEGER, game_date TEXT,
        home_team TEXT, away_team TEXT,
        home_points REAL, away_points REAL, neutral_site INTEGER)""")
    for week in (1, 2, 3):
        con.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)",
                    (str(week), 2026, week, f"2026-09-{week:02d}",
                     "H", "A", 40, 14, 0))
    con.commit()
    kickoff = "2026-10-15T20:00:00+00:00"
    board = {"generated_at_utc": "2026-10-11T10:00:00Z",
             "season": 2026, "week": 7, "picks": [
                 {"market": "moneyline", "season": 2026, "week": 7,
                  "game_id": "game_7", "team": "H", "opponent": "A",
                  "is_home": 1, "is_neutral_site": 0, "kickoff_utc": kickoff}]}
    origin = {"market": "moneyline", "season": 2026, "week": 7,
              "game_id": "game_7", "team": "H", "opponent": "A",
              "model_source": "context_v2", "model_prob": .69,
              "logged_at": "2026-10-11T10:05:00Z",
              "kickoff_utc": kickoff}
    clock = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
    return con, board, origin, clock


def test_exact_first_pregame_moneyline_and_pair_reproduced():
    con, board, origin, clock = fixtures()
    scorer = FakeScorer()
    output, reasons = m.build_first_pregame_rows(
        con, board, [origin], scorer, captured_at=clock,
        model_digest="frozen_models", board_digest="original_board")
    assert reasons == {}
    assert len(output) == 1
    x = output[0]
    assert x["challenger_home_pre_pair_probability"] == .78
    assert x["challenger_away_pre_pair_probability"] == .31
    assert x["challenger_home_final_probability"] == pytest.approx(.78/1.09)
    assert x["challenger_probability_for_incumbent_side"] == pytest.approx(.78/1.09)
    assert x["challenger_changed_side"] is False
    assert x["first_incumbent_logged_at_utc"] < x["first_challenger_logged_at_utc"]
    assert x["betting_edge_certified"] is False
    assert x["official_prediction"] is False
    assert len(scorer.features) == 2
    assert all(f[1] == 2026 and f[2] == 7 and f[5] is None
               for f in scorer.features)


def test_same_week_target_games_and_future_results_never_change_features():
    con, board, origin, clock = fixtures()
    original = m.before_target_week_team_states(con, 2026, 7)
    first = FakeScorer()
    m.build_first_pregame_rows(con, board, [origin], first, captured_at=clock,
                               model_digest="s", board_digest="s")
    con.execute("""INSERT INTO games VALUES
        ('already_played_week7', 2026, 7, '2026-10-11',
         'H','A',0,120,0)""")
    con.execute("""INSERT INTO games VALUES
        ('a_future_week', 2026, 8, '2026-10-22',
         'H','A',0,120,0)""")
    con.commit()
    after = m.before_target_week_team_states(con, 2026, 7)
    second = FakeScorer()
    m.build_first_pregame_rows(con, board, [origin], second, captured_at=clock,
                               model_digest="s", board_digest="s")
    assert original == after
    assert first.features == second.features


def test_earliest_original_forecast_is_immutable():
    con, board, original, clock = fixtures()
    second = dict(original, logged_at="2026-10-12T10:00:00Z",
                  team="A", opponent="H", model_prob=.93)
    a = m.canonical_incumbents([second, original])
    assert a[(2026, 7, "game_7")] == original
    fake = FakeScorer()
    output, _ = m.build_first_pregame_rows(
        con, board, [second, original], fake, captured_at=clock,
        model_digest="s", board_digest="s")
    assert len(output) == 1
    assert output[0]["original_incumbent_first_probability"] == .69


def test_never_backdate_or_publish_post_kickoff():
    con, board, original, clock = fixtures()
    late = datetime(2026, 10, 16, tzinfo=timezone.utc)
    records, skip = m.build_first_pregame_rows(
        con, board, [original], FakeScorer(), captured_at=late,
        model_digest="s", board_digest="s")
    assert records == []
    assert skip["not_provably_before_kickoff"] == 1
    future_board = dict(board, generated_at_utc="2026-10-18T00:00:00Z")
    with pytest.raises(ValueError, match="board did not exist"):
        m.build_first_pregame_rows(
            con, future_board, [original], FakeScorer(), captured_at=clock,
            model_digest="s", board_digest="s")


def test_original_missing_wrong_game_and_inconsistent_home_denied():
    con, board, original, clock = fixtures()
    rows, reasons = m.build_first_pregame_rows(
        con, board, [], FakeScorer(), captured_at=clock,
        model_digest="s", board_digest="s")
    assert not rows
    assert reasons["missing_immutable_incumbent_first_pregame"] == 1
    new = dict(board, picks=[dict(board["picks"][0], is_home=None)])
    rows, reasons = m.build_first_pregame_rows(
        con, new, [original], FakeScorer(), captured_at=clock,
        model_digest="s", board_digest="s")
    assert not rows
    assert reasons["unverified_home_neutral_context"] == 1


def test_first_seen_append_cannot_rewrite_or_forge_duplicate(tmp_path):
    path = tmp_path / "immutable.jsonl"
    example = {"schema": m.SCHEMA, "season": 2026, "week": 7,
               "game_id": "game_7", "challenger_selected_confidence": .65}
    assert m.append_only(path, [example]) == (1, 1)
    different = dict(example, challenger_selected_confidence=.99)
    assert m.append_only(path, [different]) == (0, 1)
    assert json.loads(path.read_text()) == example
    with path.open("a") as handle:
        handle.write("{corrupted\n")
    with pytest.raises(ValueError, match="corrupt"):
        m.append_only(path, [example])


def test_fake_model_manifest_not_certified(tmp_path):
    with pytest.raises(ValueError, match="missing frozen challenger"):
        m.check_model_snapshot(tmp_path)
