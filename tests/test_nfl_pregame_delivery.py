"""Live NFL game-day safety: the slate comes from schedule, not eligible player count."""
from datetime import datetime, timezone, timedelta
import pytest
from nfl_pregame_delivery import as_utc, pregame_only, schedule_manifest

K1="2026-10-11T13:30:00Z"  # London morning game
K2="2026-10-11T17:00:00Z"  # East Coast early window
K3="2026-10-11T20:25:00Z"  # late window
R1={"player":"Player A","team":"PHI","opponent":"JAX","market":"rushing_yards","projected_median":70}
R2={"player":"Player B","team":"CLE","opponent":"NYJ","market":"receiving_yards","projected_median":50}
R3={"player":"Player C","team":"DET","opponent":"ARI","market":"rushing_yards","projected_median":90}
KICKS={frozenset(("PHI","JAX")):K1,frozenset(("CLE","NYJ")):K2,frozenset(("DET","ARI")):K3}

def test_all_scheduled_games_survive_even_with_no_eligible_players():
    rows=[("JAX","PHI",K1),("NYJ","CLE",K2),("ARI","DET",K3)]
    s=schedule_manifest(rows)
    assert len(s)==3
    assert [r["kickoff_utc"] for r in s]==[K1,K2,K3]
    assert s[0]["home_team"]=="JAX"
    assert s[-1]["away_team"]=="DET"

def test_started_games_removed_before_logging_while_future_games_remain():
    now=datetime(2026,10,11,18,tzinfo=timezone.utc)
    safe,rejected=pregame_only([R1,R2,R3],KICKS,now)
    assert [r["player"] for r in safe]==["Player C"]
    assert safe[0]["kickoff_utc"]==K3
    assert rejected=={"kickoff_passed":2,"unresolved_schedule":0,"invalid_kickoff":0}
    assert "kickoff_utc" not in R3, "do not mutate research or original picks"

def test_deadline_strict_at_exact_start_of_kickoff():
    safe,rejected=pregame_only([R1,R2],KICKS,K2)
    assert safe==[]
    assert rejected["kickoff_passed"]==2

def test_invalid_and_unknown_kickoffs_are_quarantined():
    kicks={**KICKS,frozenset(("LA","BUF")):"not-a-timestamp"}
    rows=[R3,{"team":"GB","opponent":"CHI"}, {"team":"LA","opponent":"BUF"}]
    safe,rejected=pregame_only(rows,kicks,"2026-10-11T15:00:00Z")
    assert len(safe)==1
    assert rejected["unresolved_schedule"]==1
    assert rejected["invalid_kickoff"]==1

def test_schedule_fails_closed_on_duplicates_and_missing_times():
    with pytest.raises(ValueError,match="duplicate"):
        schedule_manifest([("JAX","PHI",K1),("PHI","JAX",K1)])
    with pytest.raises(ValueError,match="incomplete"):
        schedule_manifest([("JAX","PHI",None)])

def test_timezones_consistent_in_kickoff_comparisons():
    safe,_=pregame_only([R2],KICKS,"2026-10-11T12:59:59-04:00")
    assert len(safe)==1
    assert as_utc(safe[0]["kickoff_utc"])==datetime(2026,10,11,17,tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        as_utc("2026-10-11T13:30:00")

def test_first_game_after_1pm_is_not_hidden_just_because_london_started():
    now="2026-10-11T14:00:00Z"
    safe,rejected=pregame_only([R1,R2,R3],KICKS,now)
    assert [x["player"] for x in safe]==["Player B","Player C"]
    assert rejected["kickoff_passed"]==1
