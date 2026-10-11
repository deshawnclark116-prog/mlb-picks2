"""Prove count-forecast errors use ONLY frozen pregame shots and exact official boxscores."""
import copy
import pytest
import nhl_v2_firstseen_forward_score as s

START="2026-10-11T23:00:00Z"


def f(row="hash1",player=847000001,pred=2.7):
    return {"record_type":"FORECAST","row_hash":row,
            "forecast_id":"forecast-"+row,"game_id":2026020001,
            "player_id":player,"team":"BOS","scheduled_start":START,
            "generated_at":"2026-10-11T21:29:00Z",
            "cutoff_at":"2026-10-11T21:30:00Z","forecast_horizon":"T90",
            "expected_sog":pred,"engine_version":"nhl-v2-sog-b2-1.1",
            "meaningful_expected_participant":True,
            "comparators":{"simple_prior10_mean":2.3,
                           "human_frozen_mean":3.1}}


def g(fc,actual=3,played=True):
    return {"record_type":"GRADE","forecast_row_hash":fc["row_hash"],
            "forecast_id":fc["forecast_id"],"game_id":fc["game_id"],
            "player_id":fc["player_id"],
            "forecast_horizon":fc["forecast_horizon"],
            "actual_sog":actual,"played":played,"game_state":"OFF",
            "diagnostics_postgame_only":{"actual_toi_seconds":820}}


def test_matched_real_count_mae_and_comparator_identical_rows():
    p=f()
    out=s.report([p],[g(p,actual=3)])
    assert out["first_pregame_official_grade_rows"]==1
    core=out["by_horizon"]["T90"]["meaningful_expected_participants"]
    assert core["mae_expected_shot_count"]==.3
    assert core["bias_projected_minus_actual"]==-0.3
    assert core["prior_10_game_average"]["n"]==1
    assert core["prior_10_game_average"]["mae_gain_positive_favors_model"]==.4
    assert core["frozen_human_baseline"]["same_rows_comparator_mae"]==.1
    assert out["official_betting_picks"]==0


def test_shot_count_2point7_is_not_probability():
    p=f(pred=2.7)
    assert s.report([p],[g(p)])["first_pregame_official_grade_rows"]==1
    p["expected_sog"]=-.5
    with pytest.raises(ValueError,match="expected shot"):
        s.report([p],[g(p)])


def test_ungrounded_pregame_and_fake_final_are_never_scored():
    p=f()
    late=copy.deepcopy(p)
    late["generated_at"]="2026-10-11T22:01:00Z"
    with pytest.raises(ValueError,match="post-cutoff"):
        s.report([late],[g(late)])
    bad=g(p)
    bad["game_state"]="FUT"
    with pytest.raises(ValueError,match="official-final"):
        s.report([p],[bad])


def test_missing_and_uncertified_official_outcomes_do_not_become_zeros():
    p=f()
    out=s.report([p],[])
    assert out["first_pregame_official_grade_rows"]==0
    assert out["excluded"]["pending_or_never_graded_forecast_rows"]==1
    assert out["by_horizon"]["T90"]["meaningful_expected_participants"]["n"]==0
    bad=g(p,actual=1,played=False)
    with pytest.raises(ValueError,match="nonparticipant"):
        s.report([p],[bad])


def test_duplicate_player_game_model_row_not_scored_even_with_grade():
    p=f()
    duplicate=f(row="different-frozen-row")
    with pytest.raises(ValueError,match="ambiguous duplicate"):
        s.report([p,duplicate],[g(p)])


def test_forged_grade_does_not_match_model():
    p=f()
    grade=g(p)
    grade["forecast_row_hash"]="fake-irrelevant"
    with pytest.raises(ValueError,match="does not link"):
        s.report([p],[grade])


def test_no_hindsight_played_only_as_headline():
    p=f(pred=2.7)
    q=f(row="hash2",player=847000002,pred=1.7)
    out=s.report([p,q],[g(p,actual=3),g(q,actual=0,played=False)])
    assert out["by_horizon"]["T90"]["meaningful_expected_participants"]["n"]==2
    assert out["by_horizon"]["T90"]["meaningful_and_actually_played_diagnostic_only"]["n"]==1
    assert out["by_horizon"]["T90"]["meaningful_expected_participants"]["mae_expected_shot_count"]==1.0
