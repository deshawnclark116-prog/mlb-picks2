"""Falsified positive-tempo architecture replaced with explicitly unpromoted competition test."""
from pathlib import Path
import json
import numpy as np
import pytest

from nfl_v3_possession_competition_shadow import (
    VERSION,Scenario,possession_competition_scripts
)


def inputs(r,d):
    return {"types":{"carry":{"mu":r},"qb_att":{"mu":d},
                     "target":{"mu":d*.67}}}


def fixture():
    return {"ATL":inputs(27.,34.),"CIN":inputs(24.,39.)}


def test_one_real_game_pool_conserves_both_teams_and_market_volume():
    before=json.dumps(fixture(),sort_keys=True)
    x,proof=possession_competition_scripts(fixture(),"ATL","CIN",draws=25000,seed=20261107)
    assert json.dumps(fixture(),sort_keys=True)==before
    assert proof["schema"]==VERSION
    assert proof["status"]=="SHADOW_RESEARCH_NOT_DEPLOYED"
    assert not proof["production_promotion_allowed"]
    assert proof["old_positive_common_pace_branch"]=="FROZEN_FAILED_EMPIRICAL_SUPPORT"
    assert not proof["team_strength_or_winner_inferred"]
    assert not proof["bookmaker_data_used"]
    for t in ("ATL","CIN"):
        assert x[t]["R"].dtype.kind in "iu"
        assert x[t]["D"].dtype.kind in "iu"
        assert np.all(x[t]["R"]>=0)
        assert np.all(x[t]["D"]>=0)
        assert abs(float(x[t]["R"].mean())-fixture()[t]["types"]["carry"]["mu"])<2
        assert abs(float(x[t]["D"].mean())-fixture()[t]["types"]["qb_att"]["mu"])<2
    combined=x["ATL"]["R"]+x["ATL"]["D"]+x["CIN"]["R"]+x["CIN"]["D"]
    assert 115<float(combined.mean())<135
    assert -0.7<proof["simulated_team_total_proxy_correlation"]<-.2


def test_competition_not_a_free_positive_correlation_coupling():
    x,p=possession_competition_scripts(fixture(),"ATL","CIN",draws=20000,seed=13)
    a=x["ATL"]["R"]+x["ATL"]["D"]
    b=x["CIN"]["R"]+x["CIN"]["D"]
    assert np.corrcoef(a,b)[0,1]<-.2
    assert p["simulated_team_total_proxy_correlation"]<-.2


def test_same_seed_replays_world_for_world_and_code_path_not_production():
    x,p=possession_competition_scripts(fixture(),"ATL","CIN",draws=800,seed=2026)
    y,q=possession_competition_scripts(fixture(),"ATL","CIN",draws=800,seed=2026)
    assert p==q
    assert all(np.array_equal(x[t][k],y[t][k]) for t in x for k in ("R","D"))
    from nfl_phase1_forecast import CODE_FILES
    assert "nfl_v3_possession_competition_shadow.py" not in CODE_FILES
    src=(Path(__file__).resolve().parents[1]/"nfl_phase1d_runner.py").read_text()
    assert "seed, hz, None, b.qb_adjust" in src


@pytest.mark.parametrize("bad_inputs,teams,params",[
    ({"ATL":inputs(0,35),"CIN":inputs(24,39)},("ATL","CIN"),Scenario()),
    ({"ATL":inputs(24,float("nan")),"CIN":inputs(24,39)},("ATL","CIN"),Scenario()),
    (fixture(),("ATL","ATL"),Scenario()),
    (fixture(),("ATL","CIN"),Scenario(shared_total_sigma=-.1)),
    (fixture(),("ATL","CIN"),Scenario(possession_concentration=.5)),
    (fixture(),("ATL","CIN"),Scenario(run_share_shift=.99)),
])
def test_unsupported_source_or_unvalidated_hyperparameters_fail_closed(bad_inputs,teams,params):
    with pytest.raises(ValueError):
        possession_competition_scripts(bad_inputs,*teams,draws=100,seed=1,parameters=params)
