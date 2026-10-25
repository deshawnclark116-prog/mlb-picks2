"""Smoke-test tiny original-locked V2 NHL count feed. Never refit/make goals."""
import copy
import pytest
from nhl_compact_delivery import make_compact

FIELDS=["game_id","player_id","team","opponent","player","game_date","scheduled_start",
"forecast_horizon","generated_at","cutoff_at","meaningful_candidate",
"mean","median","original_row_hash","p_ge1","p_ge2","p_ge3","p_ge4","p_ge5"]
def source():
 def row(date,id,h,mean):
  return ["2026020021",id,"DAL","NYI","Synthetic",date,
     "2026-10-26T00:00:00Z",h,
     "2026-10-25T16:00:00Z","2026-10-25T17:00:00Z",True,
     mean,round(mean),"a"*64,.9,.8,.7,.5,.3]
 return {"schema":"nhl-v2-sog-nfl-style-shadow-v1",
  "status":"RESEARCH_ONLY_NOT_VALIDATED_FOR_BETTING",
  "verified_original_hash_chain":True,
  "verified_frozen_code_and_model_lock":True,
  "not_a_betting_board":True,"dressed_lineups_verified":False,
  "actual_player_book_lines_verified":False,
  "historical_corrected_probabilities_were_not_published_pregame":True,
  "row_fields":FIELDS,
  "horizons":["T24H","T90","T30"],
  "dates":["2026-10-24","2026-10-25"],
  "forecasts":{
    "T24H":[row("2026-10-25","8476001","T24H",3.0),
            row("2026-10-25","8476002","T24H",2.5)],
    "T90":[row("2026-10-25","8476001","T90",3.1)],
    "T30":[row("2026-10-25","8476001","T30",3.2)]
  },
  "total_valid_forecasts":4,
  "source_snapshot_commit_sha":"a"*40,
  "source_ledger_sha256":"b"*64}
def test_small_daily_immutable_count_delivery_prefers_actual_freshest_pretip():
 d=source();old=copy.deepcopy(d);c=make_compact(d)
 assert d==old
 assert c["compact_delivery_schema"]=="NHL_V2_ORIGINAL_COUNT_BEST_HORIZON_DATE_DELIVERY_V1"
 assert c["total_valid_forecasts"]==2
 assert len(c["forecasts"]["T30"])==1
 assert len(c["forecasts"]["T24H"])==1
 assert c["forecasts"]["T30"][0][FIELDS.index("mean")]==3.2
 assert c["forecasts"]["T30"][0][FIELDS.index("p_ge3")]==.7
 assert c["source_ledger_sha256"]==d["source_ledger_sha256"]
 assert c["not_a_betting_board"] and not c["dressed_lineups_verified"]
 assert c["historical_corrected_probabilities_were_not_published_pregame"]
def test_late_and_duplicated_originals_fail_closed():
 d=source()
 d["forecasts"]["T30"][0][FIELDS.index("cutoff_at")]="2026-10-26T01:00:00Z"
 with pytest.raises(ValueError,match="NOT_ORIGINAL_PREGAME"):make_compact(d)
 d=source()
 d["forecasts"]["T30"].append(copy.deepcopy(d["forecasts"]["T30"][0]))
 with pytest.raises(ValueError,match="DUPLICATE_SOURCE"):make_compact(d)
def test_source_cannot_make_goals_or_claim_calibrated_probabilities():
 d=source();d["dressed_lineups_verified"]=True
 with pytest.raises(ValueError,match="NOT_LOCKED_RESEARCH"):make_compact(d)
 d=source()
 x=make_compact(d)
 assert x["no_goal_market_in_this_shot_only_artifact"] is True
 assert x["model_probabilities_prospectively_calibrated"] is False
