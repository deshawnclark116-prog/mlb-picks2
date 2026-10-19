"""Only genuine pre-kickoff statistical *research* projections can reach UI."""
import copy
import pytest
import cfb_player_stat_frontend_bridge as P

RUN={
    "id":38053736857,
    "name":P.ORIGIN_WORKFLOW,
    "status":"completed","conclusion":"success",
    "head_sha":"d"*40,
    "created_at":"2026-10-10T12:55:14Z",
    "run_started_at":"2026-10-10T12:55:14Z",
    "updated_at":"2026-10-10T12:58:44Z",
}
AT="2026-10-10T16:00:00Z"
BASE_AT="2026-10-10T12:58:35.307131Z"
MC_AT="2026-10-10T12:58:37.923624Z"

def fixture():
    base={
        "schema":P.BASE_SCHEMA,"status":"FROZEN_PREGAME_NAIVE_POINT_BASELINE_ONLY",
        "historical_forecasts_backfilled":0,"production_model_modified":False,
        "generated_at_utc":BASE_AT,"baseline_rows":2,"forecasts":[{
            "game_id":"G1","player_id":"RB1","market":"rushing_yards",
            "season":2026,"week":6,"player":"Rush Player","team":"A",
            "opponent":"B","kickoff_utc":AT,"position":"RB",
            "point_mean_last3":70.0,"point_median_last3":70.0,
            "last3_game_ids":["G-1","G-2","G-3"],
            "last3_values":[60,70,80],
            "research_generated_at_utc":BASE_AT,
            "board_generated_at_utc":"2026-10-10T08:31:31Z",
        },{
            "game_id":"G1","player_id":"QB1","market":"passing_touchdowns",
            "season":2026,"week":6,"player":"Throw Player","team":"A",
            "opponent":"B","kickoff_utc":AT,"position":"QB",
            "point_mean_last3":1.0,"point_median_last3":1.0,
            "last3_game_ids":["G-1","G-2","G-3"],
            "last3_values":[0,1,2],
            "research_generated_at_utc":BASE_AT,
            "board_generated_at_utc":"2026-10-10T08:31:31Z",
        }],
    }
    mc={
        "schema":P.MC_SCHEMA,"status":"NEW_PREKICKOFF_RESEARCH_ONLY_NOT_PRODUCTION",
        "historical_backfills":0,"forward_original_ledger_modified":False,
        "generated_at_utc":MC_AT,
        "shadow_predictions":1,
        "shadow":[{
            "game_id":"G1","player_id":"RB1","market":"rushing_yards",
            "team":"A","kickoff_utc":AT,"forecast_generated_at_utc":MC_AT,
            "projected_mean":70.0,"projected_median":60.0,"p10":50,"p90":90,
            "original_classifier_line":69.5,"p_over_fixed_line":0.5,
            "sample_mean_unrounded":70.0,"n_simulations":200,
            "empirical_outcome_histogram_schema":"INTEGER_STAT_TOTAL_AND_FREQUENCY_V1",
            "empirical_outcome_histogram":[{"stat_total":50,"count":100},
                                            {"stat_total":90,"count":100}],
            "source_sha256":"e"*64,
        }]
    }
    return base,mc,copy.deepcopy(RUN)


def go(batch):
    return P.convert(*batch,original_run_id=38053736857)


def test_genuine_stat_projection_and_baseline_is_not_calibrated_model():
    d=go(fixture())
    assert d["forecast_count"]==2 and d["monte_carlo_count"]==1
    assert d["official_new_betting_picks"]==0
    assert d["research_only"] is True and d["is_live_predictions_feed"] is False
    r=next(x for x in d["forecasts"] if x["market"]=="rushing_yards")
    assert r["baseline"]["mean"]==70
    assert r["monte_carlo"]["mean"]==70
    assert r["monte_carlo"]["p10"]==50 and r["monte_carlo"]["p90"]==90
    assert "probability" not in r and "pick" not in r and "line" not in r
    qb=next(x for x in d["forecasts"] if x["market"]=="passing_touchdowns")
    assert qb["baseline"]["mean"]==1 and qb["monte_carlo"] is None


@pytest.mark.parametrize("modify,reason",[
    (lambda b,s,m:b["forecasts"][0].update(point_mean_last3=99),"RETROACTIVE_BASELINE_REWRITE"),
    (lambda b,s,m:s["shadow"][0].update(projected_mean=250),"SIMULATED_DISTRIBUTION_NOT_CERTIFIED"),
    (lambda b,s,m:s["shadow"][0].update(p_over_fixed_line=0.99),"SIMULATED_DISTRIBUTION_NOT_CERTIFIED"),
    (lambda b,s,m:s["shadow"][0].update(n_simulations=500),"SIMULATED_DISTRIBUTION_NOT_CERTIFIED"),
    (lambda b,s,m:s["shadow"][0].update(team="B"),"MONTE_CARLO_IDENTITY_OR_KICKOFF_DIFFERENCE"),
    (lambda b,s,m:s["shadow"][0].update(player_id="another"),"DUPLICATE_OR_UNMATCHED_MONTE_CARLO"),
    (lambda b,s,m:s["shadow"][0].update(forecast_generated_at_utc="2026-10-10T17:00:00Z"),"MONTE_CARLO_NOT_ORIGINALLY_PREGAME"),
    (lambda b,s,m:b["forecasts"][0].update(kickoff_utc="2026-10-10T12:00:00Z"),"BASELINE_NOT_GENERATED_BEFORE_KICKOFF"),
    (lambda b,s,m:m.update(conclusion="failure"),"UNVERIFIED_ORIGINAL_RESEARCH_RUN"),
    (lambda b,s,m:m.update(id=999),"UNVERIFIED_ORIGINAL_RESEARCH_RUN"),
    (lambda b,s,m:b.update(historical_forecasts_backfilled=4),"RETROACTIVE_OR_PRODUCTION_DATA"),
])
def test_source_tampering_blocks_frontend_publication(modify,reason):
    case=fixture()
    modify(*case)
    with pytest.raises(P.PublicationBlocked,match=reason):
        go(case)


def test_duplicate_identity_does_not_inflate_frontend():
    b,s,m=fixture()
    b["forecasts"].append(copy.deepcopy(b["forecasts"][0]))
    b["baseline_rows"]+=1
    with pytest.raises(P.PublicationBlocked,match="DUPLICATE_PLAYER_GAME_MARKET"):
        go((b,s,m))


def test_publication_works_without_monte_carlo_but_never_invents_intervals():
    b,s,m=fixture()
    s["shadow"]=[]
    s["shadow_predictions"]=0
    result=go((b,s,m))
    assert result["forecast_count"]==2
    assert result["monte_carlo_count"]==0
    assert all(x["monte_carlo"] is None for x in result["forecasts"])
