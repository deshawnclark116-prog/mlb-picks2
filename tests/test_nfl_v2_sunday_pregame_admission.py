"""Freeze-respecting Sunday NFL V2 role and QB forensic safety gate."""
from datetime import datetime,timezone,timedelta
import pytest
import nfl_v2_sunday_pregame_admission as A

NOW=datetime(2026,10,17,22,0,tzinfo=timezone.utc)
K="2026-10-18T17:00:00Z"
C="2026-10-17T17:00:00Z"
FIELDS=["game_id","player","team","opp","pos","outcome","p_active",
        "mean","median","p10","p90","unc","unc_reasons"]
GAME="2026_06_CAR_PHI"


def row(player,team,opp,pos,oc,p,median):
    return [GAME,player,team,opp,pos,oc,p,median+2,median,0,median*2,.3,[]]


def shadow(*,weak_philly=True,with_qb=True):
    rows=[
       row("Panthers QB","CAR","PHI","QB","pass_yds",.96,224),
       row("Panthers WR","CAR","PHI","WR","rec_yds",.98,57),
       row("Eagles WR","PHI","CAR","WR","rec_yds",.99,66)]
    if with_qb:
        if weak_philly:rows += [row("Jalen Hurts","PHI","CAR","QB","pass_yds",.0057,0)]
        else:rows += [row("Philadelphia QB","PHI","CAR","QB","pass_yds",.96,250)]
    return {"row_fields":FIELDS,"status":{
        "readiness_ready":True,"publication_state":"PUBLISHED",
        "verification_errors":[],"season":2026,"week":6,
        "model_versions":["locked_phase1c"]},
        "games":[{"game_id":GAME,"home":"PHI","away":"CAR","kickoff_utc":K,
                  "horizons":{"T24":{"cutoff":C,"n_records":len(rows)}}}],
        "forecasts":{"T24":rows,"T90":[]},
        "lines_meta":{"generated_at_utc":"2026-10-08T18:00:00Z"},
        "lines":[{"player":"Panthers WR","book":"draftkings",
                  "line":50.5,"over_price":-115,"under_price":-115}]}


def role():
    return {"schema":"nfl-role-evidence-coverage-v1",
        "captured_at_utc":"2026-10-17T21:00:00Z",
        "player_market_evidence":[
         {"player":"Panthers WR","team":"CAR","opponent":"PHI",
          "market":"receiving_yards","roster_status":"ROSTER_LISTED_NOT_STARTER_VERIFIED",
          "role_evidence_complete_for_showcase":True},
         {"player":"Eagles WR","team":"PHI","opponent":"CAR",
          "market":"receiving_yards","roster_status":"ROSTER_LISTED_NOT_STARTER_VERIFIED",
          "role_evidence_complete_for_showcase":True}
        ]}


def test_genuinely_missing_philly_starter_qb_blocks_team_even_if_wr_median_high():
    result=A.audit(shadow(),role(),now=NOW)
    assert result["summary"]["future_games"]==1
    assert result["summary"]["games_with_qb_coherence_blockers"]==1
    assert result["games"][0]["qb_safety"]["PHI"]["state"]=="BLOCK_NO_PLAUSIBLE_MODEL_QUARTERBACK"
    assert result["games"][0]["qb_safety"]["CAR"]["state"]=="UNIQUE_MODEL_QB_CANDIDATE_NOT_VERIFIED_STARTER"
    assert not [x for x in result["research_only_role_screened_players"] if x["team"]=="PHI"]
    assert len(result["research_only_role_screened_players"])==1
    assert result["official_bets_emitted"]==0
    assert result["archived_book_lines_visible"]==1
    assert result["archived_lines_verified_still_offered"] is False


def test_multiple_plausible_qbs_block_an_offense():
    b=shadow(weak_philly=False)
    b["forecasts"]["T24"].append(row("Backup QB","PHI","CAR","QB","pass_yds",.75,70))
    b["games"][0]["horizons"]["T24"]["n_records"]+=1
    a=A.audit(b,role(),now=NOW)
    assert a["games"][0]["qb_safety"]["PHI"]["state"]=="BLOCK_COMPETING_MODEL_QB_ROLES"


def test_low_qb_model_volume_is_warning_not_automatic_bet():
    b=shadow(weak_philly=False)
    b["forecasts"]["T24"][3][8]=75
    a=A.audit(b,role(),now=NOW)
    assert a["games"][0]["qb_safety"]["PHI"]["state"]=="BLOCK_IMPLAUSIBLY_LOW_MODEL_QB_PASS_VOLUME"
    assert a["official_bet_candidates"]==[]


def test_stale_roster_and_outdated_quotes_never_become_official_bets():
    s=role();s["captured_at_utc"]="2026-10-13T21:00:00Z"
    a=A.audit(shadow(weak_philly=False),s,now=NOW)
    assert a["roster_fresh"] is False
    assert not a["research_only_role_screened_players"]
    assert a["risk_counts"]["CURRENT_ROSTER_ROLE_CAPTURE_MISSING_OR_STALE"]==2
    assert a["fresh_provider_issued_book_lines_count"]==0


def test_bad_operational_publication_or_missing_rows_fail_closed():
    v=shadow();v["status"]["verification_errors"]=["LEDGER_BROKEN"]
    with pytest.raises(ValueError,match="UNVERIFIED"):A.audit(v,role(),now=NOW)
    v=shadow();v["games"][0]["horizons"]["T24"]["n_records"]+=1
    a=A.audit(v,role(),now=NOW)
    assert a["summary"]["future_games_with_pregame_model"]==0
    assert a["unverified_or_missing_forecast_reasons"]["MISMATCHED_V2_GAME_RECORD_COUNT"]==1


def test_past_games_not_retroactively_certified_using_future_info():
    a=A.audit(shadow(weak_philly=False),role(),
              now=datetime(2026,10,19,18,tzinfo=timezone.utc))
    assert a["summary"]["future_games"]==0
    assert a["official_bets_emitted"]==0


def test_legal_t90_preferred_when_original_forecast_is_already_published():
    b=shadow(weak_philly=False)
    b["games"][0]["horizons"]["T90"]={
        "cutoff":"2026-10-18T15:30:00Z","n_records":len(b["forecasts"]["T24"])}
    b["forecasts"]["T90"]=b["forecasts"]["T24"]
    now=datetime(2026,10,18,16,tzinfo=timezone.utc)
    s=role();s["captured_at_utc"]="2026-10-18T15:45:00Z"
    a=A.audit(b,s,now=now)
    assert a["games"][0]["best_already_published_horizon"]=="T90"
    assert a["official_bets_emitted"]==0


def test_normalize_player_name_and_missing_probability_fail_closed():
    assert A.namekey("De'Vaughn Vele Jr.")=="devaughn vele"
    r=shadow(weak_philly=False)
    r["forecasts"]["T24"][1][6]=None
    a=A.audit(r,role(),now=NOW)
    assert a["risk_counts"]["MODEL_PARTICIPATION_UNCERTAIN"]==1
