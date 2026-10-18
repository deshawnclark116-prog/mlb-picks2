"""Research QB1 role evidence must be conditional and timestamped."""
from datetime import datetime,timezone,timedelta
import copy,pytest
import nfl_v2_depthchart_role_ingest as I

NOW=datetime(2026,10,17,23,50,tzinfo=timezone.utc)
KICK="2026-10-18T17:00:00Z"


def board():
    return {
        "schema":"nfl-new-engine-shadow-v1",
        "status":{"publication_state":"PUBLISHED"},
        "row_fields":["game_id","team","player","pos","outcome","median","p_active"],
        "games":[{"game_id":"2026_06_CHI_ATL","home":"ATL","away":"CHI",
                  "kickoff_utc":KICK,
                  "horizons":{"T24":{"cutoff":"2026-10-17T17:00:00Z"}}}],
        "forecasts":{"T24":[
            ["2026_06_CHI_ATL","CHI","Caleb Williams","QB","pass_yds",75,.8122],
            ["2026_06_CHI_ATL","CHI","Tyson Bagent","QB","pass_yds",62,.7202],
            ["2026_06_CHI_ATL","CHI","Case Keenum","QB","pass_yds",13,.5603],
            ["2026_06_CHI_ATL","ATL","Michael Penix","QB","pass_yds",151,.9892]],
            "T90":[]}
        }


def chart():
    return {"depthCharts":[{"name":"Offense","positions":{"quarterback":{
        "position":{"abbreviation":"QB"},"athletes":[
            {"rank":1,"athlete":{"displayName":"Caleb Williams","id":"1"}},
            {"rank":2,"athlete":{"displayName":"Tyson Bagent","id":"2"}},
            {"rank":3,"athlete":{"displayName":"Case Keenum","id":"3"}}]}}}]}


def roster(availability=None):
    item={"displayName":"Caleb Williams","fullName":"Caleb Williams"}
    if availability:item["injuries"]=[{"status":availability}]
    return {"athletes":[{"position":"Quarterback","items":[item]}]}


class Response:
    def __init__(self,x):self.x=x
    def raise_for_status(self):pass
    def json(self):return self.x


def getter(url,timeout):
    return Response(chart() if url.endswith("depthcharts") else roster())


def test_qb1_with_three_median_qb_conflicts_qualifies_only_conditional_scenario():
    d=I.capture(board(),now=NOW,getter=getter)
    assert d["not_official_starter_confirmation"]
    assert d["team_data"]["CHI"]["status"]=="CONDITIONAL_QB1_ROLE_CANDIDATE"
    assert d["source_errors"].get("CHI") is None
    r=I.provisional_targets(board(),d,now=NOW)
    assert len(r)==1
    assert r[0]["official_verified_starter"]=="Caleb Williams"
    assert r[0]["not_game_day_verified_starter"] is True
    assert r[0]["source_tier"].startswith("PROVISIONAL")
    assert r[0]["source_was_known_at_original_cutoff"] is False


def test_current_roster_out_or_questionable_must_not_be_selected():
    for label in ("Out","Questionable","Injured Reserve"):
        def get(url,timeout,label=label):
            return Response(chart() if url.endswith("depthcharts") else roster(label))
        d=I.capture(board(),now=NOW,getter=get)
        assert d["team_data"]["CHI"]["status"]=="DEPTH_CHART_CANNOT_RESOLVE_STARTER"
        assert I.provisional_targets(board(),d,now=NOW)==[]


def test_unknown_player_identity_no_automatic_role_transfer():
    d=I.capture(board(),now=NOW,getter=getter)
    d["team_data"]["CHI"]["primary"]["name"]="Not In Game"
    assert I.provisional_targets(board(),d,now=NOW)==[]


def test_data_must_be_timed_before_actual_scenario():
    d=I.capture(board(),now=NOW,getter=getter)
    d["generated_at_utc"]=(NOW+timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError,match="STALE_OR_FUTURE"):
        I.provisional_targets(board(),d,now=NOW)


def test_missing_espn_results_fail_closed_without_fabricated_starter():
    def down(url,timeout):
        raise ValueError("ESPNSOURCE")
    d=I.capture(board(),now=NOW,getter=down)
    assert set(d["source_errors"])=={"ATL","CHI"}
    assert I.provisional_targets(board(),d,now=NOW)==[]


def test_no_pregame_QB1_scenario_if_original_is_already_coherent():
    b=board()
    b["forecasts"]["T24"][1][5]=0
    b["forecasts"]["T24"][2][5]=0
    d=I.capture(b,now=NOW,getter=getter)
    assert I.provisional_targets(b,d,now=NOW)==[]


def test_rank1_depthchart_not_unique_never_autoselected():
    c=chart()
    c["depthCharts"][0]["positions"]["quarterback"]["athletes"].append(
        {"rank":1,"athlete":{"displayName":"Tyson Bagent","id":"2"}})
    def get(url,timeout):
        return Response(c if url.endswith("depthcharts") else roster())
    d=I.capture(board(),now=NOW,getter=get)
    assert d["team_data"]["CHI"]["status"]=="NO_UNAMBIGUOUS_DEPTH_CHART_QB1"
    assert I.provisional_targets(board(),d,now=NOW)==[]
