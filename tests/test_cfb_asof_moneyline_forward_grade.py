"""Paired outcome scoring is strictly retrospective on frozen pregame rows."""
import json
import sqlite3
from datetime import datetime, timezone

import pytest

import cfb_asof_moneyline_forward_grade as g
import cfb_asof_forward_moneyline as m
from test_cfb_asof_forward_moneyline import fixtures, FakeScorer


def shadow_record():
    con, board, original, clock = fixtures()
    rows, skips = m.build_first_pregame_rows(
        con, board, [original], FakeScorer(), captured_at=clock,
        model_digest="frozen", board_digest="board")
    assert not skips and len(rows) == 1
    return con, rows[0]


def test_one_same_game_both_models_scored_on_same_result():
    con, shadow = shadow_record()
    game = {"game_id": "game_7", "season": 2026, "week": 7,
            "home_team": "H", "away_team": "A",
            "home_points": 27, "away_points": 14}
    graded, status = g.evaluate([shadow], [game], {})
    assert status == {}
    assert len(graded) == 1
    x = graded[0]
    assert x["incumbent_correct"] is True
    assert x["challenger_correct"] is True
    assert x["incumbent_home_probability"] == .69
    assert x["challenger_home_probability"] == pytest.approx(.78 / 1.09)
    assert x["challenger_brier"] < x["incumbent_brier"]
    assert x["challenger_logloss"] < x["incumbent_logloss"]


def test_unfinished_and_cancelled_never_scored_as_loss():
    con, shadow = shadow_record()
    game = {"game_id": "game_7", "season": 2026, "week": 7,
            "home_team": "H", "away_team": "A",
            "home_points": None, "away_points": None}
    graded, status = g.evaluate([shadow], [game], {})
    assert not graded and status["not_final"] == 1
    graded, status = g.evaluate([shadow], [game],
                                {"game_7":"STATUS_CANCELED"})
    assert not graded and status["cancelled_or_excluded"] == 1


def test_future_kickoff_and_immutable_duplicate_cannot_be_backfilled():
    con, shadow = shadow_record()
    gdata = {"game_id": "game_7", "season": 2026, "week": 7,
             "home_team": "H", "away_team": "A",
             "home_points": 20, "away_points": 10}
    counterfeit = dict(shadow,
                       first_challenger_logged_at_utc="2026-10-15T20:01:00+00:00")
    with pytest.raises(ValueError, match="post-kickoff"):
        g.evaluate([counterfeit], [gdata], {})
    with pytest.raises(ValueError, match="duplicate"):
        g.evaluate([shadow, shadow], [gdata], {})
    wrong = dict(shadow, team_home="unrelated")
    with pytest.raises(ValueError, match="incompatible"):
        g.evaluate([wrong], [gdata], {})


def test_shadow_grading_is_future_only_no_2025_backtest_promote():
    con, shadow = shadow_record()
    out = g.audit([shadow], con)
    assert out["original_shadows_seen"] == 1
    assert out["matched_final_games"] == 0
    assert out["overall"]["n"] == 0
    assert out["promotion_decision"] == "NONE_NOT_YET_PROVEN"
    assert out["paired_game_bootstrap"]["available"] is False


def test_paired_bootstrap_uses_games_not_duplicated_team_sides():
    rows = []
    for i in range(16):
        rows.append({"brier_gain_positive_helps_challenger": .01+i*.001,
                     "logloss_gain_positive_helps_challenger": .02+i*.001})
    result = g.paired_game_bootstrap(rows)
    assert result["resampling_unit"] == "one_game"
    assert result["brier_gain_ci95"][0] > 0
    assert result["logloss_gain_ci95"][0] > 0
    assert g.paired_game_bootstrap(rows[:2])["available"] is False


def test_mixed_model_versions_must_not_claim_single_calibration_gain():
    con, row = shadow_record()
    con.execute("""INSERT INTO games VALUES (
        'game_7',2026,7,'2026-10-15','H','A',30,10,0)""")
    con.execute("""INSERT INTO games VALUES (
        'game_8',2026,7,'2026-10-16','H','A',15,34,0)""")
    con.commit()
    r2=dict(row, game_id='game_8',
            frozen_challenger_model_sha256='a_different_trained_model_version')
    evaluated=g.audit([row,r2],con)
    assert evaluated["matched_final_games"] == 2
    assert evaluated["one_frozen_challenger_model"] is False
    assert evaluated["distinct_challenger_model_versions"] == 2
    assert evaluated["paired_game_bootstrap"]["available"] is False
    assert evaluated["overall"]["status"] == 'NOT_COMPARABLE_MULTIPLE_CHALLENGER_VERSIONS'
