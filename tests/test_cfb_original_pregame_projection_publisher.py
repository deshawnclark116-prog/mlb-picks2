"""Fail-closed release tests: original Week6 frozen research into app-safe counts."""
import copy
import hashlib
from collections import Counter

import pytest

import cfb_original_pregame_projection_publisher as P


def fixture():
    original={
      "schema":P.BASE,"status":"FROZEN_PREGAME_NAIVE_POINT_BASELINE_ONLY",
      "historical_forecasts_backfilled":0,"production_model_modified":False,
      "generated_at_utc":"2026-10-10T12:58:35.307131Z","baseline_rows":170,
      "forecasts":[],
    }
    shadow={
      "schema":P.MC,"status":"NEW_PREKICKOFF_RESEARCH_ONLY_NOT_PRODUCTION",
      "historical_backfills":0,"forward_original_ledger_modified":False,
      "generated_at_utc":"2026-10-10T12:58:37.923624Z","shadow_predictions":15,
      "shadow":[],
    }
    origin={"id":P.RUN_DEFAULT,"name":P.ORIGIN_NAME,"conclusion":"success",
            "status":"completed","head_sha":"a"*40,
            "created_at":"2026-10-10T12:52:00Z",
            "run_started_at":"2026-10-10T12:53:00Z",
            "updated_at":"2026-10-10T13:05:00Z"}
    for i in range(170):
        market="rushing_yards" if i<58 else "passing_touchdowns" if i<134 else "anytime_touchdowns"
        gid="G"+str(i//4)
        pid="P"+str(i)
        mean=70 if market=="rushing_yards" else 2
        vals=[mean,mean,mean]
        original["forecasts"].append({
            "game_id":gid,"player_id":pid,"player":"Test Player "+pid,
            "team":"Team A","opponent":"Team B",
            "market":market,"season":2026,"week":6,
            "kickoff_utc":"2026-10-10T19:00:00Z",
            "research_generated_at_utc":original["generated_at_utc"],
            "board_generated_at_utc":"2026-10-10T08:31:31Z",
            "last3_values":vals,"last3_game_ids":["OLD1","OLD2","OLD3"],
            "point_mean_last3":mean,"point_median_last3":mean,
        })
        if i>=15:
            continue
        shadow["shadow"].append({
            "game_id":gid,"player_id":pid,"team":"Team A","market":market,
            "kickoff_utc":"2026-10-10T19:00:00Z",
            "forecast_generated_at_utc":shadow["generated_at_utc"],
            "projected_mean":73.0,"projected_median":73.0,"p10":73,"p90":73,
            "n_simulations":4000,
            "empirical_outcome_histogram_schema":"INTEGER_STAT_TOTAL_AND_FREQUENCY_V1",
            "empirical_outcome_histogram":[{"stat_total":73,"count":4000}],
            "research_only":True,"historical_original_prediction":False,
        })
    return original,shadow,origin


def build(b,s,o):
    return P.build(b,s,o,original_artifact_sha256="e"*64,shadow_sha256="f"*64)


def test_real_frozen_week6_population_all_counts_not_legacy_odds():
    b,s,o=fixture();result=build(b,s,o)
    assert result["n_projections"]==170
    assert result["n_empirical_simulations"]==15
    assert result["by_market"]=={"rushing_yards":58,"passing_touchdowns":76,"anytime_touchdowns":36}
    assert result["official_betting_picks"]==0
    assert result["research_only"] is True
    assert result["original_pregame_run_id"]==P.RUN_DEFAULT
    assert result["no_historical_backfills"]
    models=Counter(r["model_type"] for r in result["projections"])
    assert models=={"EMPIRICAL_PER_PLAY_MC_RESEARCH":15,"NAIVE_LAST3_BOX_SCORE_BASELINE":155}
    for p in result["projections"]:
        assert p["mean"]>=0 and "line" not in p and "hit_probability" not in p
        assert p["sportsbook_line_verified"] is False
    assert result["projections"][0]["model_type"]=="EMPIRICAL_PER_PLAY_MC_RESEARCH"
    assert result["projections"][0]["mean"]==73
    assert result["projections"][0]["baseline_mean"]==70
    assert result["projections"][0]["p10"]==73


def test_empty_or_changed_publication_population_never_claims_170():
    b,s,o=fixture()
    b["forecasts"].pop()
    b["baseline_rows"]=169
    with pytest.raises(P.PublicationError,match="FROZEN_OCT10_POPULATION_MISMATCH"):
        build(b,s,o)


def test_postgame_or_late_generate_rejected_no_backfill():
    b,s,o=fixture()
    b["forecasts"][0]["kickoff_utc"]="2026-10-10T12:00:00Z"
    with pytest.raises(P.PublicationError,match="BASELINE_NOT_GENERATED_BEFORE_KICKOFF"):
        build(b,s,o)


def test_mutated_last3_mean_rejected():
    b,s,o=fixture()
    b["forecasts"][50]["point_mean_last3"]=96
    with pytest.raises(P.PublicationError,match="BASELINE_MEAN_CHANGED"):
        build(b,s,o)


def test_simulator_histogram_missing_forbidden():
    b,s,o=fixture()
    s["shadow"][0]["empirical_outcome_histogram"]=[]
    with pytest.raises(P.PublicationError,match="SIMULATED_DISTRIBUTION_NOT_FROZEN"):
        build(b,s,o)


def test_simulated_mean_tampered_from_original_draws():
    b,s,o=fixture()
    s["shadow"][0]["projected_mean"]=80
    with pytest.raises(P.PublicationError,match="SIMULATED_MEAN_NOT_MATCHING"):
        build(b,s,o)


def test_ambiguous_same_player_game_market_not_silently_deduplicated():
    b,s,o=fixture()
    b["forecasts"][1]["game_id"]=b["forecasts"][0]["game_id"]
    b["forecasts"][1]["player_id"]=b["forecasts"][0]["player_id"]
    with pytest.raises(P.PublicationError,match="DUPLICATE_ORIGINAL_BASELINE"):
        build(b,s,o)


def test_late_unrelated_or_failed_run_rejected():
    b,s,o=fixture()
    o["conclusion"]="failure"
    with pytest.raises(P.PublicationError,match="ORIGINAL_GITHUB_ACTIONS_RUN_NOT_VERIFIED"):
        build(b,s,o)
    o["conclusion"]="success";o["created_at"]="2026-10-10T14:00:00Z"
    with pytest.raises(P.PublicationError,match="ORIGINAL_RESEARCH_ARTIFACT_NOT_DURING_ACTIONS_RUN"):
        build(b,s,o)


def test_passing_td_not_mislabeled_as_validated_monte_carlo():
    b,s,o=fixture()
    doc=build(b,s,o)
    tds=[p for p in doc["projections"] if p["market"]=="passing_touchdowns"]
    assert len(tds)==76 and all(x["model_type"]=="NAIVE_LAST3_BOX_SCORE_BASELINE" for x in tds)
    assert all(x["p10"] is None and x["simulations"] is None for x in tds)


def test_no_origins_mutated():
    b,s,o=fixture()
    original=(copy.deepcopy(b),copy.deepcopy(s),copy.deepcopy(o))
    build(b,s,o)
    assert (b,s,o)==original
