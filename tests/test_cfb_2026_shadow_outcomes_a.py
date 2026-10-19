"""Independent paper-only CFB statistical-forecast outcome grading invariants."""
import copy
import sqlite3

import pytest

import cfb_2026_shadow_outcomes_a as O


def fixture():
    db=sqlite3.connect(":memory:")
    db.row_factory=sqlite3.Row
    db.executescript("""
        CREATE TABLE games (game_id TEXT, season INT, week INT, kickoff_utc TEXT,
            home_team TEXT, away_team TEXT, home_points INT, away_points INT);
        CREATE TABLE schedule_snapshot (game_id TEXT, season INT, espn_status TEXT);
        CREATE TABLE player_games (player_id TEXT, team TEXT, game_id TEXT,
            season INT, week INT, position TEXT, rushing_yards INT, passing_touchdowns INT);
    """)
    db.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?,?)",
               ("G1",2026,6,"2026-10-10T16:00:00Z","A","B",24,13))
    db.execute("INSERT INTO schedule_snapshot VALUES(?,?,?)",("G1",2026,"STATUS_FINAL"))
    db.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?)",
               ("P1","A","G1",2026,6,"RB",85,None))
    row={
        "game_id":"G1","player_id":"P1","team":"A","market":"rushing_yards",
        "week":6,"kickoff_utc":"2026-10-10T16:00:00Z",
        "forecast_generated_at_utc":"2026-10-10T11:17:00Z",
        "projected_mean":80.0,"projected_median":75.0,
        "p10":40.0,"p90":110.0,
        "p_over_fixed_line":0.6,
        "original_classifier_line":69.5,
        "baseline_last3_mean":70.0,
        "baseline_last3_game_ids":["OLD1","OLD2","OLD3"],
        "prior_verified_same_team_game_ids":["OLD1","OLD2","OLD3"],
        "historical_original_prediction":False,"research_only":True,
    }
    paper={"schema":O.INPUT_SCHEMA,"status":"NEW_PREKICKOFF_RESEARCH_ONLY_NOT_PRODUCTION",
           "generated_at_utc":"2026-10-10T11:17:00Z","historical_backfills":0,
           "forward_original_ledger_modified":False,
           "shadow_predictions":1,"shadow":[row]}
    return paper,db


def test_exact_real_stat_errors_and_original_baseline_comparator():
    paper,db=fixture()
    r=O.audit(paper,db,artifact_sha256="original-paper-artifact-sha256")
    assert r["graded_completed"]==1
    assert r["ungraded_count"]==0
    assert r["overall"]["mean_projection_mae"]==5
    assert r["overall"]["median_projection_mae"]==10
    assert r["overall"]["mean_projection_bias_pred_minus_actual"]==-5
    assert r["overall"]["baseline_last3_mean_mae_same_eligible_rows"]==15
    assert r["overall"]["challenger_mean_mae_on_baseline_comparable_rows"]==5
    assert r["overall"]["p10_p90_empirical_coverage"]==1
    assert r["overall"]["p10_p90_mean_width"]==70
    assert r["overall"]["prob_over_line_brier"]==0.16
    assert r["input_artifact_sha256"]=="original-paper-artifact-sha256"
    assert r["no_model_fitting_or_backfill"]


def test_missing_original_baseline_never_filled_in_after_game():
    paper,db=fixture()
    for key in ("baseline_last3_mean","baseline_last3_game_ids"):
        paper["shadow"][0].pop(key)
    r=O.audit(paper,db)
    assert r["overall"]["original_pregame_baseline_n"]==0
    assert r["overall"]["baseline_last3_mean_mae_same_eligible_rows"] is None
    assert r["overall"]["mean_projection_mae"]==5


def test_original_frozen_histogram_scored_with_exact_crps_without_resimulating():
    paper,db=fixture()
    row=paper["shadow"][0]
    row.update({
        "projected_mean":50.0,
        "projected_median":50.0,
        "p10":0.0,
        "p90":100.0,
        "p_over_fixed_line":0.5,
        "sample_mean_unrounded":50.0,
        "n_simulations":2,
        "empirical_outcome_histogram_schema":"INTEGER_STAT_TOTAL_AND_FREQUENCY_V1",
        "empirical_outcome_histogram":[{"stat_total":0,"count":1},
                                       {"stat_total":100,"count":1}],
    })
    result=O.audit(paper,db)
    assert result["graded_completed"]==1
    # Two equally likely 0/100 simulated rushing totals, actual 85:
    # E|X-85| = 50; half pairwise correction = 25; CRPS = 25.
    assert result["overall"]["frozen_exact_distribution_crps_n"]==1
    assert result["overall"]["frozen_exact_distribution_mean_crps"]==25
    assert result["overall"]["frozen_last3_point_baseline_mean_crps_same_population"]==15
    assert result["graded_rows"][0]["frozen_histogram_crps"]==25


def test_missing_original_sample_histogram_never_recreated_after_game():
    paper,db=fixture()
    result=O.audit(paper,db)
    assert result["overall"]["frozen_exact_distribution_crps_n"]==0
    assert result["overall"]["frozen_exact_distribution_mean_crps"] is None


@pytest.mark.parametrize("change,reason",[
    (lambda row: row.update(empirical_outcome_histogram=[
        {"stat_total":0,"count":2}]),"FROZEN_HISTOGRAM_MEAN_TAMPERED"),
    (lambda row: row.update(empirical_outcome_histogram=[
        {"stat_total":100,"count":1},{"stat_total":0,"count":1}]),"INVALID_OR_UNSORTED"),
    (lambda row: row.update(empirical_outcome_histogram=[
        {"stat_total":0,"count":1}]),"HISTOGRAM_SAMPLE_COUNT_MISMATCH"),
    (lambda row: row.update(p_over_fixed_line=0.8),"REPORTED_PROBABILITY"),
])
def test_histogram_data_cannot_change_at_grade_time(change,reason):
    paper,db=fixture()
    row=paper["shadow"][0]
    row.update({
        "projected_mean":50.0,"projected_median":50.0,
        "p10":0.0,"p90":100.0,"p_over_fixed_line":0.5,
        "sample_mean_unrounded":50.0,"n_simulations":2,
        "empirical_outcome_histogram_schema":"INTEGER_STAT_TOTAL_AND_FREQUENCY_V1",
        "empirical_outcome_histogram":[{"stat_total":0,"count":1},
                                       {"stat_total":100,"count":1}],
    })
    change(row)
    with pytest.raises(O.ShadowOutcomeError,match=reason):
        O.audit(paper,db)


def test_unfinished_game_is_never_graded():
    paper,db=fixture()
    db.execute("UPDATE schedule_snapshot SET espn_status='STATUS_IN_PROGRESS'")
    r=O.audit(paper,db)
    assert r["graded_completed"]==0
    assert r["ungraded_reasons"]=={"GAME_NOT_OFFICIALLY_FINAL":1}
    assert r["overall"]["n"]==0


def test_no_player_line_after_final_is_not_assumed_zero():
    paper,db=fixture()
    db.execute("DELETE FROM player_games")
    r=O.audit(paper,db)
    assert r["graded_completed"]==0
    assert r["ungraded_reasons"]=={"MISSING_FINISHED_PLAYER_STAT_LINE":1}


@pytest.mark.parametrize("field,value,msg",[
    ("forecast_generated_at_utc","2026-10-10T16:01:00Z","POSTKICKOFF"),
    ("kickoff_utc","2026-10-10T15:00:00Z","ORIGINAL_AND_OUTCOME_KICKOFF"),
    ("team","B","ACTUAL_PLAYER_TEAM"),
    ("projected_mean",float("nan"),"INVALID_PROJECTED_MEAN"),
    ("p10",120.0,"INVALID_DISTRIBUTION_QUANTILES"),
    ("p_over_fixed_line",1.5,"INVALID_MODEL_PROBABILITY"),
])
def test_reject_tampering_to_original_frozen_research_record(field,value,msg):
    paper,db=fixture()
    paper["shadow"][0][field]=value
    if field=="forecast_generated_at_utc":
        paper["generated_at_utc"]=value
    with pytest.raises(O.ShadowOutcomeError,match=msg):
        O.audit(paper,db)


def test_duplicate_player_game_market_refused_not_double_counted():
    paper,db=fixture()
    paper["shadow"].append(copy.deepcopy(paper["shadow"][0]))
    paper["shadow_predictions"]=2
    with pytest.raises(O.ShadowOutcomeError,match="DUPLICATE_FORECAST_ID"):
        O.audit(paper,db)


def source_run(**changes):
    base={
        "id":38047614744,
        "name":"CFB 2026 event data qualification (research, no serving)",
        "head_sha":"a"*40,
        "status":"completed","conclusion":"success",
        "created_at":"2026-10-10T11:05:00Z",
        "run_started_at":"2026-10-10T11:10:00Z",
        "updated_at":"2026-10-10T11:20:00Z",
        "html_url":"https://github.com/example/repo/actions/runs/38047614744",
    }
    base.update(changes)
    return base


def test_original_actions_run_proves_forecast_was_truly_pregame():
    paper,db=fixture()
    result=O.audit(paper,db,source_run_metadata=source_run())
    assert result["actions_run_temporal_provenance_verified"] is True
    assert result["source_actions_run"]["head_sha"]=="a"*40
    assert result["source_actions_run"]["run_id"]==38047614744


@pytest.mark.parametrize("tamper,reason",[
    ({"conclusion":"failure"},"UNVERIFIED_OR_UNSUCCESSFUL"),
    ({"name":"Unrelated Workflow"},"UNVERIFIED_OR_UNSUCCESSFUL"),
    ({"created_at":"2026-10-10T17:00:00Z",
      "run_started_at":"2026-10-10T17:05:00Z",
      "updated_at":"2026-10-10T17:15:00Z"},
     "ORIGINAL_ACTIONS_RUN_TIMELINE_INVALID"),
    ({"created_at":"2026-10-10T10:00:00Z",
      "run_started_at":"2026-10-10T10:05:00Z",
      "updated_at":"2026-10-10T10:15:00Z"},
     "ORIGINAL_ACTIONS_RUN_TIMELINE_INVALID"),
])
def test_forged_or_unrelated_origin_action_cannot_grade(tamper,reason):
    paper,db=fixture()
    with pytest.raises(O.ShadowOutcomeError,match=reason):
        O.audit(paper,db,source_run_metadata=source_run(**tamper))


def test_fake_pregame_json_when_actions_started_after_kickoff_is_rejected():
    paper,db=fixture()
    # A run that started after kickoff cannot generate this artifact for
    # research validity even when its JSON filename claims pregame timing.
    later=source_run(created_at="2026-10-10T17:00:00Z",
                     run_started_at="2026-10-10T17:05:00Z",
                     updated_at="2026-10-10T17:15:00Z")
    paper["generated_at_utc"]="2026-10-10T17:10:00Z"
    paper["shadow"][0]["forecast_generated_at_utc"]="2026-10-10T17:10:00Z"
    with pytest.raises(O.ShadowOutcomeError,match="ACTIONS_RUN_DID_NOT_START_BEFORE_KICKOFF"):
        O.audit(paper,db,source_run_metadata=later)


def test_unverified_generation_cannot_be_counted_as_outcome():
    paper,db=fixture()
    paper["status"]="PRODUCTION_PICKS"
    with pytest.raises(O.ShadowOutcomeError,match="UNVERIFIED_PAPER"):
        O.audit(paper,db)
    paper["status"]="NEW_PREKICKOFF_RESEARCH_ONLY_NOT_PRODUCTION"
    paper["historical_backfills"]=1
    with pytest.raises(O.ShadowOutcomeError,match="FROZEN_EVIDENCE"):
        O.audit(paper,db)
