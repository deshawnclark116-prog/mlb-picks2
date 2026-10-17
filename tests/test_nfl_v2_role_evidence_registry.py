"""Automatic candidate filtering for all NFL teams, not only Philadelphia."""
from datetime import datetime,timezone,timedelta
import copy
import pytest
import nfl_v2_role_evidence_registry as R

NOW=datetime(2026,10,17,23,48,tzinfo=timezone.utc)
KICK="2026-10-18T17:00:00Z"
CUT="2026-10-17T17:00:00Z"
URL="https://www.philadelphiaeagles.com/news/panthers-vs-eagles-injury-report-2026-nfl-week-6-saquon-barkley-devonta-smith"


def board():
    f=["game_id","player","team","opp","pos","outcome","p_active","exp_opp","mean","median","id"]
    rows=[
      ["2026_06_CAR_PHI","Tanner McKee","PHI","CAR","QB","pass_yds",.1852,5,39,0,"a"],
      ["2026_06_CAR_PHI","Jalen Hurts","PHI","CAR","QB","pass_yds",.0057,0,0,0,"b"],
      ["2026_06_CAR_PHI","Bryce Young","CAR","PHI","QB","pass_yds",.98,30,240,229,"c"]
    ]
    return {"schema":"nfl-new-engine-shadow-v1",
            "status":{"publication_state":"PUBLISHED"},"row_fields":f,
            "games":[{"game_id":"2026_06_CAR_PHI","home":"PHI","away":"CAR",
                      "kickoff_utc":KICK,"horizons":{"T24":{"cutoff":CUT}}}],
            "forecasts":{"T24":rows,"T90":[]}}


def evidence(published="2026-10-16T20:00:00Z"):
    return {"schema":R.LEGACY,"assertions":[{
        "game_id":"2026_06_CAR_PHI","team":"PHI","opponent":"CAR",
        "kickoff_utc":KICK,"verified_starter":"Tanner McKee",
        "out_players":["Jalen Hurts"],
        "official_sources":[{"url":URL,"published_at_utc":published,
                             "supports":["Tanner McKee will start",
                                         "Jalen Hurts ruled out"]}]
    }]}


def test_exact_original_snapshot_displays_true_starter_conflict():
    rows=R.targets(board(),evidence(),now=NOW)
    assert len(rows)==1
    x=rows[0]
    assert x["original_model_p_active"]==.1852
    assert x["team"]=="PHI" and x["opponent"]=="CAR"
    assert x["source_was_known_at_original_cutoff"] is True
    assert x["must_not_relabel_as_original_T24_T90"] is True


def test_official_starter_news_after_original_cutoff_makes_new_scenario_only():
    x=R.targets(board(),evidence("2026-10-17T21:00:00Z"),now=NOW)[0]
    assert x["source_was_known_at_original_cutoff"] is False
    assert x["new_pregame_scenario_generated_after_original_cutoff"] is True


def test_multi_team_games_each_select_their_own_qb_and_no_role_guessing():
    b=board()
    b["forecasts"]["T24"][2][6]=.12
    e=evidence()
    e["schema"]=R.SCHEMA
    e["assertions"].append({
        "game_id":"2026_06_CAR_PHI","team":"CAR","opponent":"PHI",
        "kickoff_utc":KICK,"verified_starter":"Bryce Young",
        "out_players":["Jalen Hurts"],
        "official_sources":[{"url":"https://www.panthers.com/news/example",
          "published_at_utc":"2026-10-17T20:00:00Z",
          "supports":["Bryce Young will start","Jalen Hurts out"]}]
    })
    # Hurts is not a CAR QB. This cannot be falsely accepted.
    with pytest.raises(ValueError,match="MISSING_OR_AMBIGUOUS_OUT_QB_SNAPSHOT"):
        R.targets(b,e,now=NOW)


def test_conflicting_starter_assertions_same_team_fail_closed():
    e=evidence();e["assertions"].append(copy.deepcopy(e["assertions"][0]))
    with pytest.raises(ValueError,match="CONFLICTING_VERIFIED"):
        R.targets(board(),e,now=NOW)


def test_untrusted_url_and_missing_source_support_fail_closed():
    e=evidence();e["assertions"][0]["official_sources"][0]["url"]="https://philadelphiaeagles.com.evil.com/news/mckee"
    with pytest.raises(ValueError,match="UNTRUSTED"):
        R.targets(board(),e,now=NOW)
    e=evidence();e["assertions"][0]["official_sources"][0]["supports"]=["Tanner McKee maybe"]
    with pytest.raises(ValueError,match="STARTER_NOT_AFFIRMED"):
        R.targets(board(),e,now=NOW)


def test_game_already_started_prevents_scenario_not_a_postgame_backtest():
    with pytest.raises(ValueError,match="KICKOFF_CHANGED_OR_PASSED"):
        R.targets(board(),evidence(),now=datetime(2026,10,18,18,tzinfo=timezone.utc))


def test_news_not_published_by_actual_generation_time_refused():
    with pytest.raises(ValueError,match="POST_GAME_OR_FUTURE"):
        R.targets(board(),evidence("2026-10-18T00:00:00Z"),now=NOW)


def test_without_verified_source_has_zero_starter_overrides():
    e={"schema":R.SCHEMA,"assertions":[]}
    assert R.targets(board(),e,now=NOW)==[]
