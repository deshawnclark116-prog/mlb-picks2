"""Official NFL Out and T90 inactive must never consume simulated opportunities."""
import numpy as np

from nfl_phase1_forecast import apply_snapshot_rules
from nfl_phase1c_sim import alloc_given


def game():
    return {"types":{
        "carry":{
            "ids":["rb_out","rb_questionable","rb_active","rb_otherteam"],
            "pact24":np.array([.25,.55,.95,.8]),
            "pact90":np.array([.25,.55,.95,.8]),
            "pact_lookup":np.array([.25,.55,.95,.8]),
        },
        "target":{
            "ids":["rb_out","rb_active"],
            "pact24":np.array([.82,.94]),
            "pact90":np.array([.82,.94]),
            "pact_lookup":np.array([.82,.94]),
        }
    }}


def injury(player,team="ATL",week="9",year="2026",status="Out"):
    return {"gsis_id":player,"team":team,"week":week,
            "season":year,"report_status":status}


def roster(player,team="ATL",week="9",year="2026",status="INA"):
    return {"gsis_id":player,"team":team,"week":week,
            "season":year,"status":status}


def test_own_week_reported_out_is_zero_even_when_prior_is_below_half():
    g=game()
    before=g["types"]["carry"]["pact24"].copy()
    logs=[]
    revised=apply_snapshot_rules(
        g,"T24",[
            injury("rb_out",status=" OUT "),
            injury("rb_questionable",status="Questionable"),
            injury("rb_otherteam",team="CIN",status="Out"),
            injury("rb_active",week="8",status="Out")
        ],[],2026,9,"ATL",logs)
    c=revised["types"]["carry"]
    assert c["pact24"].tolist()==[0,.55,.95,.8]
    assert c["pact_lookup"].tolist()==[0,.55,.95,.8]
    assert revised["types"]["target"]["pact24"].tolist()==[0,.94]
    assert revised["types"]["target"]["pact_lookup"].tolist()==[0,.94]
    assert any(x["rule"]=="verified_weekly_injury_report_out" and
               x["player"]=="rb_out" for x in logs)
    assert g["types"]["carry"]["pact24"].tolist()==before.tolist()


def test_t90_inactive_and_explicit_out_take_precedence_over_uncertain_statuses():
    revised=apply_snapshot_rules(
        game(),"T90",[
            injury("rb_out",status="Out"),
            injury("rb_questionable",status="Questionable"),
        ],[
            roster("rb_questionable"),
            roster("rb_active",team="CIN"),
            roster("rb_otherteam",week="8"),
        ],2026,9,"ATL",[])
    c=revised["types"]["carry"]
    assert c["pact90"].tolist()==[0,0,.95,.8]
    assert c["pact_lookup"].tolist()==[0,0,.95,.8]


def test_verified_out_never_gets_volume_across_10000_worlds():
    g=apply_snapshot_rules(
        game(),"T24",[injury("rb_out")],[],2026,9,"ATL",[])
    p=g["types"]["carry"]["pact24"]
    rng=np.random.default_rng(20261107)
    named,outside,active=alloc_given(
        p,np.array([.5,.3,.15,.05]),.1,40.0,
        np.repeat(30,10000),rng)
    assert named.shape==(10000,4)
    assert not active[:,0].any()
    assert not named[:,0].any()
    assert np.array_equal(named.sum(1)+outside,np.repeat(30,10000))
    assert (named[:,1].sum()+named[:,2].sum()+named[:,3].sum())>0


def test_without_fresh_source_evidence_preserve_independent_model_probabilities():
    g=game()
    for horizon in ("T24","T90"):
        result=apply_snapshot_rules(g,horizon,[],[],2026,9,"ATL",[])
        key="pact24" if horizon=="T24" else "pact90"
        for t in g["types"]:
            assert np.array_equal(result["types"][t][key],g["types"][t][key])
            assert np.array_equal(result["types"][t]["pact_lookup"],
                                  g["types"][t]["pact_lookup"])


def test_stale_injury_reports_or_wrong_team_must_never_zero_player():
    rows=[injury("rb_out",team="CIN"),injury("rb_out",week="8"),
          injury("rb_out",year="2025"),injury("rb_out",status="Doubtful")]
    result=apply_snapshot_rules(game(),"T24",rows,[],2026,9,"ATL",[])
    assert result["types"]["carry"]["pact24"][0]==.25


def test_snapshot_zeroed_status_baseline_tracks_most_recent_verified_out():
    result=apply_snapshot_rules(game(),"T24",[injury("rb_out")],
                                [],2026,9,"ATL",[])
    for type_name in ("carry","target"):
        rows=result["types"][type_name]
        assert rows["pact_lookup"][0]==rows["pact24"][0]==0.0
