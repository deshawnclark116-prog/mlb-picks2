"""CFB immutable first-source evidence, no future injuries or hindsight."""
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest
import cfb_personnel_first_seen as ledger

G='2026-10-11T18:06:07+00:00'
S='2026-10-11T18:06:16Z'
KO='2026-10-13T23:00Z'
NOW=datetime(2026,10,11,18,7,tzinfo=timezone.utc)

def board():
 return {"season":2026,"week":7,"generated_at_utc":G,
     "picks":[{"season":2026,"week":7,"game_id":"401871067",
      "team":"Delaware Blue Hens","opponent":"Middle Tennessee Blue Raiders",
      "market":"passing_touchdowns","player":"Nick Minicucci","player_id":"5153846",
      "model_prob":.6407,"pick":"UNDER 1.5","line":1.5,
      "model_source":"context_v2","kickoff_utc":KO}]}

def observation():
 return {
  "game_id":"401871067","season":2026,"week":7,"kickoff_utc":KO,
  "team":"Delaware Blue Hens","opponent":"Middle Tennessee Blue Raiders",
  "market":"passing_touchdowns","player":"Nick Minicucci","player_id":"5153846",
  "published_model_prob":.6407,"published_model_side":"UNDER 1.5",
  "published_line_is_sportsbook_verified":False,
  "roster":{"status":"ROSTER_LISTED_AVAILABILITY_UNKNOWN",
            "availability_certified":False,
            "source_status":"Active","source_injury_details":[],
            "roster_checked_utc":"2026-10-11T18:01:32+00:00"},
  "recent_usage":{"metric":"pass_attempts",
      "snapshot_is_not_starting_lineup_certification":True,
      "mean_opportunities_in_observed_games":20,
      "unknown_or_ambiguous_appearances":1},
  "evidence_warnings":["THREE_GAME_ROLE_COVERAGE_INCOMPLETE"],
  "fully_verified_active_starter":False,"calibrated_market_edge_established":False}

def snapshot():
 return {"schema":"cfb-analyst-evidence-v1",
  "status":"RESEARCH_ONLY_DO_NOT_AUTO_PROMOTE","model_changed":False,
  "markets_book_verified":False,"is_current_injury_and_starter_feed_complete":False,
  "season":2026,"week":7,"board_generated_at_utc":G,
  "evidence_captured_at_utc":S,
  "player_evidence":[observation()]}

def test_first_genuine_asof_receipt_wins_and_never_changes(tmp_path):
 path=tmp_path/"first.jsonl"
 first=ledger.capture(snapshot(),board(),path,clock=NOW)
 assert first["newly_frozen"]==1
 saved=path.read_bytes()
 later=snapshot()
 later["player_evidence"][0]["roster"]["source_status"]="Questionable"
 later["evidence_captured_at_utc"]="2026-10-11T18:07:22Z"
 second=ledger.capture(later,board(),path,clock=datetime(2026,10,11,18,8,tzinfo=timezone.utc))
 assert second["newly_frozen"]==0
 assert path.read_bytes()==saved
 result=json.loads(path.read_text())
 assert result["original_research_line_NOT_book"]==1.5
 assert result["original_model_probability"]==.6407
 assert result["verified_active_starter"] is False
 assert result["sportsbook_line_verified"] is False
 assert result["betting_edge_validated"] is False
 assert result["source_quality_flags"]==["THREE_GAME_ROLE_COVERAGE_INCOMPLETE"]
 assert len(result["source_receipt_sha256"])==64

def test_forecast_version_source_and_lookahead_must_match(tmp_path):
 bad=snapshot()
 bad["board_generated_at_utc"]="2026-10-11T17:00Z"
 with pytest.raises(ValueError,match="generation mismatch"):
  ledger.capture(bad,board(),tmp_path/"no.jsonl",clock=NOW)
 assert not (tmp_path/"no.jsonl").exists()
 after=datetime(2026,10,13,23,1,tzinfo=timezone.utc)
 x=ledger.capture(snapshot(),board(),tmp_path/"late.jsonl",clock=after)
 assert x["newly_frozen"]==0 and x["rejection_reasons"]["not_genuinely_pregame"]==1

def test_after_capture_roster_is_never_used_as_pregame_source(tmp_path):
 bad=snapshot()
 bad["player_evidence"][0]["roster"]["roster_checked_utc"]="2026-10-11T19:00:00Z"
 x=ledger.capture(bad,board(),tmp_path/"r.jsonl",clock=NOW)
 assert x["newly_frozen"]==0
 assert x["rejection_reasons"]["post_capture_or_game_roster_source"]==1

def test_false_active_starter_or_book_qualification_is_rejected(tmp_path):
 bad=snapshot()
 bad["player_evidence"][0]["published_line_is_sportsbook_verified"]=True
 x=ledger.capture(bad,board(),tmp_path/"r.jsonl",clock=NOW)
 assert x["newly_frozen"]==0
 assert x["rejection_reasons"]["false_status_or_market_certification"]==1

def test_distinct_game_player_market_not_merged(tmp_path):
 b=board();s=snapshot()
 copy=dict(b["picks"][0])
 copy.update(player="Second Player",player_id="99",pick="OVER 1.5",
             model_prob=.51)
 b["picks"].append(copy)
 new=dict(s["player_evidence"][0])
 new.update(player="Second Player",player_id="99",
            published_model_side="OVER 1.5",published_model_prob=.51)
 s["player_evidence"].append(new)
 x=ledger.capture(s,b,tmp_path/"first.jsonl",clock=NOW)
 assert x["newly_frozen"]==2
 assert len((tmp_path/"first.jsonl").read_text().splitlines())==2

def test_corrupt_existing_ledger_prevents_unsafe_rewrite(tmp_path):
 path=tmp_path/"broken.jsonl"
 path.write_text("{not valid}\n")
 with pytest.raises(ValueError,match="corrupted immutable receipt"):
  ledger.capture(snapshot(),board(),path,clock=NOW)
 assert path.read_text()=="{not valid}\n"

def test_real_current_artifact_if_present_is_never_backfilled(tmp_path):
 p=Path("docs/cfb_pregame_evidence.json")
 b=Path("docs/cfb_predictions.json")
 if not p.exists() or not b.exists():pytest.skip("Repo artifacts unavailable")
 snap=json.loads(p.read_text());board_doc=json.loads(b.read_text())
 # The current time may already be after many games. Never backfill.
 at=datetime.now(timezone.utc)
 capture=ledger.capture(snap,board_doc,tmp_path/"real.jsonl",clock=at)
 assert capture["newly_frozen"]<=capture["eligible_pregame"]
 assert capture["rejection_reasons"].get("false_status_or_market_certification",0)==0
