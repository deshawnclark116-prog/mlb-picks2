from nhl_sog_serving_safety_audit import audit

def test_fixed_line_under_flood_never_becomes_verified_bet():
    board={"picks":[{"market":"shots_on_goal","game_id":"g","player_id":str(i),
                    "pick":"UNDER 2.5","line":2.5} for i in range(200)]}
    r=audit(board)
    assert r["total_raw_sog_entries"]==200
    assert r["actionable_verified_picks"]==0
    assert r["findings"]["unverified_line_or_availability"]==200

def test_conflicting_early_season_and_regular_predictions_flagged():
    p={"game_id":"g","player_id":"p","line":2.5,
       "sportsbook_line_verified":True,"bookmaker":"FD",
       "line_source_timestamp_utc":"2026-10-10T12:00:00Z",
       "active_roster_verified":True,"player_status_verified":True}
    r=audit({"picks":[dict(p,market="shots_on_goal",pick="UNDER 2.5"),
                      dict(p,market="shots_on_goal_early_season",pick="OVER 2.5")]})
    assert r["findings"]["duplicate_player_game"]==1
    assert r["findings"]["conflicting_side"]==1
    assert r["players"][0]["status"]=="RESEARCH_ONLY"

def test_even_verified_line_is_not_a_calibrated_model_or_value_proof():
    p={"market":"shots_on_goal","game_id":"g","player_id":"p","pick":"OVER 3.5",
       "line":3.5,"sportsbook_line_verified":True,"bookmaker":"FD",
       "line_source_timestamp_utc":"2026-10-10T12:00:00Z",
       "active_roster_verified":True,"player_status_verified":True}
    r=audit({"picks":[p]})
    assert r["players"][0]["status"]=="NEEDS_MODEL_VALIDATION"
    assert r["actionable_verified_picks"]==0
