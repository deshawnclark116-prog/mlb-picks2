"""No official V2 bet invented; exact W/L on actual scored player + archive."""
import nfl_v2_bookline_win_loss_audit as A


def sample():
    v2={"schema":"NFL_V2_FIRST_SEEN_REAL_FINAL_NUMERIC_SCORECARD_V1",
        "all_graded_rows":[
            {"week":5,"team":"ATL","game_id":"2026_05_BAL_ATL",
             "horizon":"T90","player":"jahan dotson","market":"rec_yds",
             "median":21,"actual":54,"cutoff_utc":"2026-10-11T22:50:00Z",
             "kickoff_utc":"2026-10-12T00:20:00Z"}]}
    logged=[{"season":2026,"week":5,"team":"ATL","opponent":"BAL",
             "market":"receiving_yards","player":"Jahan Dotson",
             "book":"draftkings","line":15.5,"odds":-120,
             "pick":"UNDER 15.5","logged_at":"2026-10-06T00:47:00+00:00",
             "projected_median":11}]
    original={"results":[{"season":2026,"week":5,"team":"ATL","opponent":"BAL",
              "player":"Jahan Dotson","market":"receiving_yards",
              "pick":"UNDER 15.5","actual":54}]}
    schedule=[{"home_team":"ATL","away_team":"BAL",
               "kickoff_utc":"2026-10-12T00:20:00Z"}]
    return v2,logged,original,schedule


def test_original_v2_vs_book_differs_from_original_legacy_pick():
    d=A.build(*sample())
    assert d["v2_official_bet_record_count"]==0
    assert d["v2_hypothetical_bookline_grading"]["T90"]["wins"]==1
    assert d["v2_hypothetical_bookline_grading"]["T90"]["losses"]==0
    assert d["original_legacy_serving_booked_picks_not_v2"]["summary"]["wins"]==0
    assert d["original_legacy_serving_booked_picks_not_v2"]["summary"]["losses"]==1
    r=d["v2_hypothetical_player_results"][0]
    assert r["v2_predicted_median"]==21 and r["actual"]==54
    assert r["hypothetical_side_from_v2_median"]=="OVER"
    assert r["research_only_not_original_v2_bet"] is True


def test_late_bookline_never_joined_and_never_backdates():
    v2,logged,old,sched=sample()
    logged[0]["logged_at"]="2026-10-12T00:00:00Z"
    d=A.build(v2,logged,old,sched)
    assert d["v2_hypothetical_bookline_grading"]["T90"]["book_comparisons"]==0
    assert d["no_line_no_grading_count"]["BOOK_QUOTE_AFTER_V2_CUTOFF"]==1
    assert d["original_legacy_serving_booked_picks_not_v2"]["summary"]["wins"]==0


def test_no_duplicate_identity_or_empty_model_edge():
    v2,logged,old,sched=sample()
    v2["all_graded_rows"][0]["median"]=15.5
    d=A.build(v2,logged,old,sched)
    assert d["v2_hypothetical_bookline_grading"]["T90"]["book_comparisons"]==0
    assert d["no_line_no_grading_count"]["NO_MODEL_VS_LINE_DIFFERENCE"]==1
    v2["all_graded_rows"][0]["median"]=21
    v2["all_graded_rows"].append(dict(v2["all_graded_rows"][0]))
    d=A.build(v2,logged,old,sched)
    assert d["no_line_no_grading_count"]["NO_UNIQUE_REAL_V2_PLAYER_GAME_T90"]==1


def test_push_recorded_not_graded_as_winner():
    assert A.grade("OVER",16,16)=="PUSH"
    assert A.grade("UNDER",16,16)=="PUSH"
    assert A.grade("UNDER",24.5,16)=="WIN"
    assert A.grade("OVER",24.5,16)=="LOSS"


def test_no_offered_mainline_no_forced_fake_bet():
    v2,logged,old,sched=sample()
    logged[0].pop("book")
    d=A.build(v2,logged,old,sched)
    assert d["v2_hypothetical_bookline_grading"]["T90"]["book_comparisons"]==0
    assert d["original_legacy_serving_booked_picks_not_v2"]["summary"]["book_comparisons"]==0
