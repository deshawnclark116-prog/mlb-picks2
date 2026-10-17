"""Live simulator starter-role experiment must conserve team allocations."""
import copy
import numpy as np
import pytest
import nfl_v2_full_event_qb_starter_challenger as M

def fixture():
    return {"PHI":{"types":{
       "qb_att":{"ids":["HURTS","MCKEE","MILTON"],
                 "P1":np.array([.75,.22,.08]),
                 "pact24":np.array([.0057,.1852,.0796]),
                 "pact90":np.array([.005,.2,.08])},
       "carry":{"ids":["HURTS","MCKEE","PIERCE"],
                "P1":np.array([.16,.02,.72]),
                "pact24":np.array([.0057,.1852,.9237]),
                "pact90":np.array([.0,.22,.99])},
       "rz_carry":{"ids":["HURTS","MCKEE","PIERCE"],
                   "P1":np.array([.3,.04,.66]),
                   "pact24":np.array([.0057,.1852,.9237]),
                   "pact90":np.array([.0,.22,.99])},
       "target":{"ids":["SMITH","BROWN"],
                 "P1":np.array([.6,.4]),
                 "pact24":np.array([.96,.9]),"pact90":np.array([.9,.9])}
       }},
        "CAR":{"types":{"qb_att":{"ids":["YOUNG"],
                         "P1":np.array([.92]),
                         "pact24":np.array([.98]),"pact90":np.array([.98])}}}}

def test_full_game_qb_starter_role_transfer_not_talent_transfer():
    a=fixture();original=copy.deepcopy(a)
    b,proof=M.role_override_games(a,team="PHI",starter_id="MCKEE",
              qb_ids=["HURTS","MCKEE","MILTON"],horizon="T24")
    assert proof["original_starter_play_probability"]==.1852
    assert proof["position_prior_source"].startswith("highest")
    np.testing.assert_allclose(b["PHI"]["types"]["qb_att"]["P1"],[0,.75,0])
    np.testing.assert_allclose(b["PHI"]["types"]["qb_att"]["pact24"],[0,1,0])
    np.testing.assert_allclose(b["PHI"]["types"]["carry"]["pact24"],[0,1,.9237])
    np.testing.assert_allclose(b["PHI"]["types"]["rz_carry"]["pact24"],[0,1,.9237])
    assert b["PHI"]["types"]["carry"]["P1"][0]==.16
    np.testing.assert_array_equal(b["PHI"]["types"]["target"]["pact24"],
                                  original["PHI"]["types"]["target"]["pact24"])
    np.testing.assert_array_equal(b["CAR"]["types"]["qb_att"]["P1"],
                                  original["CAR"]["types"]["qb_att"]["P1"])
    for name in ("qb_att","carry","rz_carry","target"):
        for key in original["PHI"]["types"][name]:
            if isinstance(original["PHI"]["types"][name][key],np.ndarray):
                np.testing.assert_array_equal(original["PHI"]["types"][name][key],
                                             a["PHI"]["types"][name][key])

def test_qb_same_league_dropbacks_conserved_under_fixed_outside_bucket():
    from nfl_phase1c_sim import alloc_given
    before=fixture()
    after,_=M.role_override_games(before,team="PHI",starter_id="MCKEE",
                  qb_ids=["HURTS","MCKEE","MILTON"],horizon="T24")
    q=after["PHI"]["types"]["qb_att"]
    ns=2000
    counts,other,active=alloc_given(q["pact24"],q["P1"],.04,None,
             np.full(ns,30,dtype=np.int64),np.random.default_rng(42))
    assert np.all(counts.sum(1)+other==30)
    assert np.all(counts[:,0]==0) and np.all(counts[:,2]==0)
    assert np.all(active[:,1]) and not active[:,0].any()
    assert counts[:,1].mean()>25  # role now owns team's 30-pass budget

def test_wrong_id_and_invalid_propensity_rejected_without_model_mutation():
    d=fixture()
    with pytest.raises(ValueError,match="STARTER_NOT_EXACT"):
        M.role_override_games(d,team="PHI",starter_id="WRONG",
                             qb_ids=["HURTS","MCKEE","MILTON"],horizon="T24")
    d["PHI"]["types"]["qb_att"]["P1"][:]=0
    with pytest.raises(ValueError,match="INVALID_FROZEN_QB_ROLE"):
        M.role_override_games(d,team="PHI",starter_id="MCKEE",
                             qb_ids=["HURTS","MCKEE","MILTON"],horizon="T24")

def test_t90_starter_condition_and_roster_lineage():
    d=fixture()
    o,p=M.role_override_games(d,team="PHI",starter_id="MCKEE",
                         qb_ids=["HURTS","MCKEE","MILTON"],horizon="T90")
    np.testing.assert_allclose(o["PHI"]["types"]["qb_att"]["pact90"],[0,1,0])
    np.testing.assert_allclose(o["PHI"]["types"]["carry"]["pact90"],[0,1,.99])
    assert p["horizon"]=="T90"
    assert p["QB_efficiency_models_not_substituted"] is True
