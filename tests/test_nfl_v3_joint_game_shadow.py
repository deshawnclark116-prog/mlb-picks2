"""Structural contract for joint-game shadow and frozen Phase1C separation."""
import json
from pathlib import Path
import numpy as np
import pytest

from nfl_v3_joint_game_shadow import (
    VERSION,ShadowParameters,joint_game_scripts,
)


def game(r,d):
    return {"types":{"carry":{"mu":r},"qb_att":{"mu":d},"target":{"mu":d*.75}}}


def test_original_phase1c_run_game_accepts_script_schema_and_is_unmodified():
    from nfl_phase1c_sim import run_game
    assert "script=None" in run_game.__code__.co_varnames or "script" in run_game.__code__.co_varnames
    # The fixed frozen runner is deliberately still calling the incumbent
    # with script=None. This research module cannot mutate it.
    src=(Path(__file__).resolve().parents[1]/"nfl_phase1d_runner.py").read_text()
    assert "seed, hz, None, b.qb_adjust" in src


def test_injectable_joint_game_preserves_every_team_opportunity():
    before={"ATL":game(27.5,34.2),"CIN":game(24.4,39.1)}
    untouched=json.dumps(before,sort_keys=True)
    result,proof=joint_game_scripts(before,"ATL","CIN",draws=12000,seed=313)
    assert proof["schema"]==VERSION
    assert proof["model_impact"]=="SHADOW_ONLY_NOT_PRODUCTION"
    assert proof["unvalidated_hyperparameters"] is True
    assert proof["has_real_team_strength_or_final_score_data"] is False
    assert proof["sportsbook_inputs_used"] is False
    assert proof["draws"]==12000
    assert json.dumps(before,sort_keys=True)==untouched
    for t in ("ATL","CIN"):
        assert result[t]["R"].shape==(12000,)
        assert result[t]["D"].shape==(12000,)
        assert result[t]["R"].dtype.kind in "iu"
        assert result[t]["D"].dtype.kind in "iu"
        assert np.all(result[t]["R"]>=0) and np.all(result[t]["D"]>=0)
        assert 15<result[t]["R"].mean()+result[t]["D"].mean()<110


def test_common_pace_creates_positive_between_team_correlation_without_invented_score():
    inputs={"ATL":game(27,35),"CIN":game(26,35)}
    a,p=joint_game_scripts(inputs,"ATL","CIN",draws=25000,seed=310)
    assert p["estimated_joint_total_plays_correlation"] > .18
    x=a["ATL"]["R"]+a["ATL"]["D"]
    y=a["CIN"]["R"]+a["CIN"]["D"]
    assert np.corrcoef(x,y)[0,1]>.18
    # Turning common pace completely off removes between-team dependence,
    # even though the complementary run/pass split still shares a hypothetical
    # score regime (a share, not a total-volume factor).
    b,_=joint_game_scripts(inputs,"ATL","CIN",draws=25000,seed=310,
                          parameters=ShadowParameters(pace_sigma=0))
    assert abs(np.corrcoef(b["ATL"]["R"]+b["ATL"]["D"],
                           b["CIN"]["R"]+b["CIN"]["D"])[0,1])<.045


def test_hypothetical_leading_team_reallocates_toward_rush_without_changing_totals():
    inputs={"ATL":game(25,35),"CIN":game(25,35)}
    a,_=joint_game_scripts(inputs,"ATL","CIN",draws=25000,seed=5,
                           parameters=ShadowParameters(run_share_shift=.15))
    b,_=joint_game_scripts(inputs,"ATL","CIN",draws=25000,seed=5,
                           parameters=ShadowParameters(run_share_shift=0))
    assert a["ATL"]["R"].shape == b["ATL"]["R"].shape
    assert abs(float(a["ATL"]["R"].mean() + a["ATL"]["D"].mean()) -
               float(b["ATL"]["R"].mean() + b["ATL"]["D"].mean())) < 1
    # Unlike the frozen independent baseline, each club's pass/rush mixture
    # now responds to a common pregame scenario. No implied winner.
    assert abs(float(a["ATL"]["R"].mean())-float(b["ATL"]["R"].mean()))<4


def test_fixed_seed_stability_and_no_nan_vectors():
    inputs={"ATL":game(25,35),"CIN":game(20,42)}
    a,p=joint_game_scripts(inputs,"ATL","CIN",draws=512,seed=63)
    b,q=joint_game_scripts(inputs,"ATL","CIN",draws=512,seed=63)
    assert p==q
    assert all(np.array_equal(a[t][x],b[t][x]) for t in a for x in ("R","D"))


@pytest.mark.parametrize("data,params",[
    ({"ATL":game(0,32),"CIN":game(22,35)},ShadowParameters()),
    ({"ATL":game(float("nan"),32),"CIN":game(22,35)},ShadowParameters()),
    ({"ATL":game(15,999),"CIN":game(22,35)},ShadowParameters()),
    ({"ATL":game(15,32)},ShadowParameters()),
    ({"ATL":game(15,32),"CIN":game(22,35)},ShadowParameters(pace_sigma=-1)),
    ({"ATL":game(15,32),"CIN":game(22,35)},ShadowParameters(run_share_shift=.2)),
])
def test_invalid_or_incomplete_pretip_input_refuses_scenario(data,params):
    with pytest.raises(ValueError):
        joint_game_scripts(data,"ATL","CIN",draws=100,seed=8,parameters=params)
