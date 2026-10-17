"""The public NFL page must never silently hide a QB-room contradiction."""
from datetime import datetime,timezone,timedelta
import pytest
import nfl_v2_qb_role_conflict_audit as A

NOW=datetime(2026,10,17,23,30,tzinfo=timezone.utc)

def board():
    fields=["game_id","player","team","opp","pos","outcome","p_active",
            "exp_opp","role_share","mean","median"]
    return {
        "schema":"nfl-new-engine-shadow-v1","status":{"publication_state":"PUBLISHED"},
        "row_fields":fields,
        "games":[
          {"game_id":"2026_06_CAR_PHI","home":"PHI","away":"CAR",
           "kickoff_utc":"2026-10-18T17:00:00Z",
           "horizons":{"T24":{"cutoff":"2026-10-17T17:00:00Z"}}},
          {"game_id":"2026_06_CHI_ATL","home":"ATL","away":"CHI",
           "kickoff_utc":"2026-10-18T17:00:00Z",
           "horizons":{"T24":{"cutoff":"2026-10-17T17:00:00Z"}}}],
        "forecasts":{"T24":[
          ["2026_06_CAR_PHI","Tanner McKee","PHI","CAR","QB","pass_yds",.1852,5,.5,38,0],
          ["2026_06_CAR_PHI","Jalen Hurts","PHI","CAR","QB","pass_yds",.0057,0,.2,0,0],
          ["2026_06_CHI_ATL","Caleb Williams","CHI","ATL","QB","pass_yds",.8122,13,.7,94,75],
          ["2026_06_CHI_ATL","Tyson Bagent","CHI","ATL","QB","pass_yds",.7202,11,.5,85,62],
          ["2026_06_CHI_ATL","Case Keenum","CHI","ATL","QB","pass_yds",.5603,8,.3,57,13],
          ["2026_06_CHI_ATL","Michael Penix","ATL","CHI","QB","pass_yds",.98,23,.8,156,150],
        ],"T90":[]}
    }

def test_flags_the_real_mckee_zero_yard_and_bears_three_qb_conflicts():
    d=A.inspect(board(),now=NOW)
    assert d["teams_horizons_with_qb_projections"]==3
    assert d["flagged_team_horizons"]==2
    flags={r["team"]:r["flags"] for r in d["role_conflicts"]}
    assert flags["PHI"]==["NO_QB_WITH_MAJORITY_PLAY_PROBABILITY"]
    assert "MULTIPLE_QBS_PROJECTED_FOR_SUBSTANTIVE_PASSING" in flags["CHI"]
    assert "ATL" not in flags
    assert d["original_forecasts_not_changed"] and d["official_bets_created"]==0


def test_no_trick_postgame_predictions_allowed_in_pregame_report():
    d=A.inspect(board(),now=NOW+timedelta(days=2))
    assert d["teams_horizons_with_qb_projections"]==0
    assert d["flagged_team_horizons"]==0


def test_garbled_roster_probabilities_fail_closed():
    v=board()
    v["forecasts"]["T24"][0][6]=float("nan")
    with pytest.raises(ValueError,match="NONFINITE"):
        A.inspect(v,now=NOW)


def test_no_auto_starter_inferred_from_three_QB_uncertainty():
    d=A.inspect(board(),now=NOW)
    for item in d["role_conflicts"]:
        assert item["automatic_starter_selection"] is False
        assert item["source_evidence_type"].startswith("PUBLISHED_ORIGINAL")


def test_nonpublished_forecasts_not_silently_accepted():
    v=board();v["status"]["publication_state"]="UNVERIFIED"
    with pytest.raises(ValueError,match="UNVERIFIED_OR_UNPUBLISHED"):
        A.inspect(v,now=NOW)
