"""NHL real-count postgame forensics remain strict, paired and non-promotional."""
from copy import deepcopy
import pytest
import nhl_v2_count_miss_forensics as R

START="2026-10-11T23:00:00Z"
def fixture(pid=8471001,h="T90",pred=2.4,actual=3,played=True,gid=2026020011):
    fc={
        "record_type":"FORECAST","row_hash":f"frozen-{gid}-{pid}-{h}",
        "forecast_id":f"forecast-{gid}-{pid}-{h}",
        "game_id":gid,"player_id":pid,"team":"BOS",
        "engine_version":"nhl-v2-sog-b2-1.1",
        "scheduled_start":START,"forecast_horizon":h,
        "generated_at":"2026-10-11T21:28:00Z" if h=="T90" else "2026-10-11T22:28:00Z",
        "cutoff_at":"2026-10-11T21:30:00Z" if h=="T90" else "2026-10-11T22:30:00Z",
        "expected_sog":pred,
        "meaningful_expected_participant":True,
        "availability_state":"IN_ROSTER_SPOTS_OBSERVED_NOT_CERTIFIED",
        "availability_confidence":"NOT_CERTIFIED",
        "comparators":{"simple_prior10_mean":2.0,"human_frozen_mean":2.8},
        "receipt":{"player_name":"Test Forward","position":"F",
                   "recent_participation":{"team_games_since_appearance":0},
                   "toi_seconds":{"recent3":1000},
                   "sog_per60_app10":8.5}
    }
    gr={
        "record_type":"GRADE","forecast_row_hash":fc["row_hash"],
        "forecast_id":fc["forecast_id"],"game_id":gid,
        "player_id":pid,"forecast_horizon":h,
        "actual_sog":actual,"played":played,
        "game_state":"FINAL",
        "diagnostics_postgame_only":{"actual_toi_seconds":1200 if played else None,
                                    "actual_pp_toi_seconds":60 if played else None}
    }
    return fc,gr


def test_same_player_horizon_pair_and_real_count_not_probability():
    f1,g1=fixture(h="T90",pred=2.4)
    f2,g2=fixture(h="T30",pred=2.8)
    result=R.investigate([f1,f2],[g1,g2])
    pair=result["same_player_same_game_across_horizons"]["T30_vs_T90"]
    assert pair["matched_player_games"]==1
    assert pair["earlier_mae_same_players"]==.6
    assert pair["later_mae_same_players"]==.2
    assert pair["mae_gain_positive_means_later_improved"]==.4
    assert pair["game_block_bootstrap_gain"]["available"] is False
    assert result["by_horizon"]["T30"]["meaningful_expected_population"]["n"]==1
    assert result["betting_picks"]==0
    assert result["thresholds_tuned"] is False


def test_no_postgame_ice_time_leak_into_forecast_and_bucketing_is_diagnostic():
    a,ga=fixture()
    b,gb=fixture(pid=8471002,actual=0,played=False)
    c,gc=fixture(pid=8471003,actual=0,played=True)
    gc["diagnostics_postgame_only"]["actual_toi_seconds"]=400
    frozen=deepcopy([a,b,c])
    result=R.investigate([a,b,c],[ga,gb,gc])
    bucket=result["by_horizon"]["T90"]["postgame_deployment_DIAGNOSTIC_ONLY"]
    assert bucket["ACTUAL_TOI_75_TO_125PCT_OF_RECENT3"]["n"]==1
    assert bucket["DID_NOT_PLAY_POSTGAME"]["n"]==1
    assert bucket["ACTUAL_TOI_LESS_THAN_75PCT_OF_RECENT3"]["n"]==1
    assert [a,b,c]==frozen
    assert result["by_horizon"]["T90"]["meaningful_expected_population"]["n"]==3
    assert result["by_horizon"]["T90"]["sensitivity_actual_played_only_POSTGAME_DIAGNOSTIC"]["n"]==2


def test_exact_same_player_prior10_comparison():
    a,ga=fixture(pred=2.4)
    result=R.investigate([a],[ga])
    b=result["by_horizon"]["T90"]["prior10_matched_baseline"]
    assert b["n"]==1
    assert b["model_mae_same_rows"]==.6
    assert b["baseline_mae_same_rows"]==1.0
    assert b["mae_gain_positive_favors_model"]==.4


def test_no_future_information_false_results_or_duplicate_identity():
    a,ga=fixture()
    a["generated_at"]="2026-10-11T23:01:00Z"
    with pytest.raises(ValueError,match="post-cutoff"):
        R.investigate([a],[ga])
    a,ga=fixture()
    b,gb=fixture()
    b["row_hash"]="a_second_immutable_row"
    b["forecast_id"]="a_second_identifier"
    gb["forecast_row_hash"]=b["row_hash"]
    gb["forecast_id"]=b["forecast_id"]
    with pytest.raises(ValueError,match="ambiguous"):
        R.investigate([a,b],[ga,gb])


def test_game_cluster_resampling_refuses_small_sample():
    records=[{"game_id":str(i),"gain":.05} for i in range(7)]
    assert R._game_ci(records,k="gain")["available"] is False
    records.append({"game_id":"8","gain":-.1})
    ci=R._game_ci(records,k="gain")
    assert ci["available"] is True
    assert ci["unit"]=="game"
    assert ci["games"]==8
    assert ci["ci95"][0]<=ci["ci95"][1]


def test_no_hindsight_lines_and_no_fake_model_version_mix():
    a,ga=fixture()
    b,gb=fixture(pid=8471002)
    b["engine_version"]="retrains-on-known-2026-results"
    with pytest.raises(ValueError,match="different frozen versions"):
        R.investigate([a,b],[ga,gb])
