"""Contract tests: do not grade imaginary NHL betting picks or leaked postgame data."""
import copy
import json
import unittest
from nhl_published_firstseen_backtest_oct2026 import (
    frozen_predictions,official_games,grade_predictions,summarize,franchise,
)
REF="a"*40
DAY="2026-10-25"
TIP="2026-10-25T19:00:00+00:00"
PUBLISHED="2026-10-25T17:07:44+00:00"
def orig():
 return [
 {"market":"points","player_id":8479314,"player":"Skater Alpha",
  "team":"FLA","opponent":"CHI","season":2026,"game_date":DAY,
  "kickoff_utc":TIP,"pick":"OVER 0.5","model_prob":.72},
 {"market":"points","player_id":8484144,"player":"Skater Bravo",
  "team":"CHI","opponent":"FLA","season":2026,"game_date":DAY,
  "kickoff_utc":TIP,"pick":"OVER 0.5","model_prob":.68},
 {"market":"points","player_id":8477495,"player":"Skater Scratched",
  "team":"FLA","opponent":"CHI","season":2026,"game_date":DAY,
  "kickoff_utc":TIP,"pick":"OVER 0.5","model_prob":.71},
 {"market":"moneyline","player_id":None,"player":"Florida Panthers",
  "team":"Florida Panthers","opponent":"Chicago Blackhawks",
  "is_home":0,"season":2026,"game_date":DAY,
  "kickoff_utc":TIP,"pick":"Florida Panthers ML","model_prob":.62},
 ]
def fake_official(url):
 if "/schedule/" in url:
  return {"gameWeek":[{"games":[{"gameDate":DAY,"gameType":2,
   "id":2026020001,"gameState":"OFF",
   "homeTeam":{"abbrev":"CHI"},"awayTeam":{"abbrev":"FLA"}}]}]}
 return {"id":2026020001,"gameState":"OFF",
  "homeTeam":{"abbrev":"CHI","score":2},
  "awayTeam":{"abbrev":"FLA","score":3},
  "playerByGameStats":{
   "awayTeam":{"forwards":[{"playerId":8479314,"goals":1,"assists":1},
                           {"playerId":99,"goals":0,"assists":0}],"defense":[]},
   "homeTeam":{"forwards":[{"playerId":8484144,"goals":0,"assists":0}],
               "defense":[]}}}
def as_rows():
 return [{**p,"source_ref":REF,"original_generated_at_utc":PUBLISHED,
  "source_commit_at_utc":PUBLISHED,"grade":"PENDING"} for p in orig()]
class Research(unittest.TestCase):
 def test_original_team_and_player_exact_matching(self):
  assert franchise("Florida Panthers")=="FLA"
  assert franchise("CHI")=="CHI"
  assert franchise("Utah Mammoth")=="UTA"
  result=grade_predictions(as_rows(),official_games([DAY],fake_official))
  self.assertEqual([r["grade"] for r in result],
    ["HIT","MISS","UNVERIFIED_NOT_IN_FINAL_SKATER_BOX","HIT"])
  self.assertEqual(result[0]["actual_points"],2)
  self.assertEqual(result[1]["actual_points"],0)
  self.assertEqual(result[3]["winner"],"FLA")
  s=summarize(result)["all_days"]
  self.assertEqual(s["verified_n"],3)
  self.assertEqual(s["unverified_n"],1)
  self.assertEqual(s["hit_n"],2)
  self.assertEqual(s["miss_n"],1)
 def test_no_finish_or_wrong_moneyside_cannot_be_gamed(self):
  unfinished=lambda url: {"gameWeek":[{"games":[{
      "gameDate":DAY,"gameType":2,"id":2026020001,
      "gameState":"FUT","homeTeam":{"abbrev":"CHI"},"awayTeam":{"abbrev":"FLA"}
      }]}]} if "/schedule/" in url else {}
  output=grade_predictions(as_rows(),official_games([DAY],unfinished))
  self.assertTrue(all(x["grade"]=="UNVERIFIED_OFFICIAL_FINAL_NOT_AVAILABLE" for x in output))
  bad=copy.deepcopy(as_rows());bad[-1]["is_home"]=1
  output=grade_predictions(bad,official_games([DAY],fake_official))
  self.assertEqual(output[-1]["grade"],"UNVERIFIED_MONEYLINE_IDENTITY_MISMATCH")
 def test_historical_commit_date_after_tip_does_not_forge_firstseen(self):
  import nhl_published_firstseen_backtest_oct2026 as mod
  d={"game_date":DAY,"generated_at_utc":PUBLISHED,"picks":orig()}
  def source(cmd,spec,*more):
   if cmd=="show" and spec=="-s":return "2026-10-25T20:00:00+00:00"
   return json.dumps(d)
  rows,_=frozen_predictions({DAY:REF},source)
  self.assertEqual(len(rows),4)
  self.assertTrue(all(x["grade"]=="UNVERIFIED_AFTER_CUTOFF" for x in rows))
 def test_duplicate_market_generation_not_counted_as_two_hits(self):
  r=orig()
  r.append({**r[0],"market":"points_early_season","model_prob":.99})
  d={"game_date":DAY,"generated_at_utc":PUBLISHED,"picks":r}
  def source(cmd,spec,*more):
   if cmd=="show" and spec=="-s":return PUBLISHED
   return json.dumps(d)
  rows,_=frozen_predictions({DAY:REF},source)
  self.assertEqual(sum(x["grade"]=="PENDING" for x in rows),4)
  self.assertEqual(sum(x["grade"]=="UNVERIFIED_DUPLICATE_OR_AMBIGUOUS" for x in rows),1)
 def test_mutated_missing_official_player_id_quarantined_and_no_unearned_grade(self):
  bad=copy.deepcopy(as_rows())
  bad[0]["player_id"]=999999
  scored=grade_predictions(bad,official_games([DAY],fake_official))
  self.assertEqual(scored[0]["grade"],"UNVERIFIED_NOT_IN_FINAL_SKATER_BOX")
if __name__=="__main__":unittest.main()
