"""Source-exact NFL numeric firstseen/outcome and repeated-miss tests."""
from datetime import datetime,timezone,timedelta
import json,sqlite3,tempfile,unittest
from pathlib import Path

from nfl_numeric_forward_grade import (
    SCHEMA,MARKET_COLUMNS,grade,first_seen_records
)

def event(week,market="receiving_yards",player="Test Receiver",pid="ESPN7",
          team="KC",logged=True,median=30,relative="BEFORE"):
    day=datetime(2026,9,8,tzinfo=timezone.utc)+timedelta(weeks=week)
    kickoff=day.strftime("%Y-%m-%dT18:00:00Z")
    logged_at=day.strftime("%Y-%m-%dT16:00:00Z") if relative=="BEFORE" else kickoff
    p={
        "season":2026,"week":week,"market":market,
        "player_id":pid,"player":player,"team":team,
        "opponent":"BUF","kickoff_utc":kickoff,
        "projected_median":median,"projected_mean":median+.5,
        "model_source":"v2_context",
        "projection_probability_source":"UNAVAILABLE_V2_CONTEXT_NO_MATCHED_DISTRIBUTION"
    }
    if logged:p["logged_at"]=logged_at
    return p

class NumericForwardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.ledger=self.root/"frozen.jsonl"
        self.con=sqlite3.connect(":memory:")
        self.con.executescript("""
          CREATE TABLE player_games(
            player_id TEXT,player_name TEXT,team TEXT,season INT,week INT,
            season_type TEXT,rushing_yards REAL,receiving_yards REAL,
            passing_yards REAL,receptions REAL,carries REAL,completions REAL
          );""")
        self.now=datetime(2026,11,22,15,0,tzinfo=timezone.utc)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def write(self,events):
        self.ledger.write_text("\n".join(json.dumps(x) for x in events)+"\n")

    def record(self,week,actual,player="Test Receiver",pid="GSIS7",market="receiving_yards"):
        vals={"rushing_yards":None,"receiving_yards":None,"passing_yards":None,
              "receptions":None,"carries":None,"completions":None}
        vals[MARKET_COLUMNS[market]]=actual
        self.con.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (pid,player,"KC",2026,week,"REG",
                          *(vals[MARKET_COLUMNS[k]] for k in MARKET_COLUMNS)))
        self.con.commit()

    def test_original_firstseen_is_frozen_and_after_kickoff_never_sneaks_in(self):
        records=[event(1),event(1),event(2,relative="AFTER"),event(3,logged=False)]
        self.write(records)
        original=self.ledger.read_bytes()
        loaded,skipped=first_seen_records(self.ledger,self.now)
        self.assertEqual(len(loaded),1)
        self.assertEqual(skipped.get("DUPLICATE_FIRST_SEEN_SKIPPED"),1)
        self.assertEqual(skipped.get("AFTER_KICKOFF_OR_FUTURE_FIRST_SEEN"),1)
        self.assertEqual(skipped.get("UNVERIFIED_OR_MISSING_FIRST_SEEN_TIME"),1)
        self.assertEqual(self.ledger.read_bytes(),original)

    def test_grade_three_repeated_actual_misses_but_not_missing_games(self):
        self.write([event(1),event(2),event(3),event(4)])
        for week,actual in ((1,10),(2,13),(3,8)):
            self.record(week,actual)
        doc=grade(self.ledger,self.con,self.now)
        self.assertEqual(doc["schema"],SCHEMA)
        self.assertEqual(doc["first_seen_count"],4)
        self.assertEqual(doc["graded_count"],3)
        self.assertEqual(doc["metrics_by_market"]["receiving_yards"]["median_mae"],round((20+17+22)/3,3))
        self.assertEqual(doc["ungraded"].get("NO_COMPLETED_PLAYER_GAME_OR_UNRESOLVED_ID"),1)
        self.assertEqual(len(doc["recent_repeated_overprojections"]),1)
        warning=doc["recent_repeated_overprojections"][0]
        self.assertEqual(warning["last3_median_errors"],[20,17,22])
        self.assertEqual(warning["overprojected_by_10_count"],3)
        self.assertFalse(doc["model_automatic_learning"])
        self.assertEqual(doc["results"][0]["match_basis"],"UNIQUE_SAME_TEAM_NAME_FALLBACK")
        # Source file is NEVER rewritten by grading.
        self.assertEqual(self.ledger.read_text().count("\n"),4)

    def test_no_warning_with_only_two_graded_games_and_no_zero_for_unplayed(self):
        self.write([event(1),event(2),event(3)])
        self.record(1,0)
        self.record(2,0)
        doc=grade(self.ledger,self.con,self.now)
        self.assertEqual(doc["graded_count"],2)
        self.assertEqual(doc["recent_repeated_overprojections"],[])
        self.assertEqual(doc["results"][0]["actual"],0)
        self.assertEqual(doc["ungraded"]["NO_COMPLETED_PLAYER_GAME_OR_UNRESOLVED_ID"],1)

    def test_market_separation_and_repeated_misses_do_not_borrow_sportbook_probs(self):
        data=[event(1),event(2),event(3),
              event(1,market="passing_yards",player="Test QB",pid="qb7",median=230),
              event(1,market="rushing_yards",player="Test Runner",pid="rb7",median=75)]
        self.write(data)
        for week in (1,2,3):self.record(week,10)
        self.record(1,150,player="Test QB",pid="qb7",market="passing_yards")
        self.record(1,42,player="Test Runner",pid="rb7",market="rushing_yards")
        results=grade(self.ledger,self.con,self.now)
        self.assertEqual(results["graded_count"],5)
        self.assertEqual(set(results["metrics_by_market"]),
                        {"passing_yards","receiving_yards","rushing_yards"})
        self.assertEqual(len(results["recent_repeated_overprojections"]),1)
        self.assertEqual(results["recent_repeated_overprojections"][0]["market"],"receiving_yards")
        self.assertTrue(results["no_sportsbook_prices"])

    def test_true_negative_yardage_is_real_and_not_mislabeled_unavailable(self):
        self.write([event(1,market="rushing_yards",player="Test Runner",
                          pid="RB100",median=15)])
        self.record(1,-3,player="Test Runner",pid="RB100",market="rushing_yards")
        result=grade(self.ledger,self.con,self.now)
        self.assertEqual(result["graded_count"],1)
        self.assertEqual(result["metrics_by_market"]["rushing_yards"]["median_mae"],18)
        self.assertEqual(result["results"][0]["actual"],-3)
        self.assertNotIn("INVALID_NEGATIVE_ACTUAL",result["ungraded"])

    def test_real_gsis_id_matches_only_correct_team(self):
        self.write([event(1,pid="GSIS7"),event(1,pid="GSIS8",player="Test Receiver",team="BUF")])
        self.record(1,9,pid="GSIS7")
        doc=grade(self.ledger,self.con,self.now)
        self.assertEqual(doc["graded_count"],1)
        self.assertEqual(doc["results"][0]["match_basis"],"EXACT_GSIS_ID")
        self.assertEqual(doc["ungraded"]["NO_COMPLETED_PLAYER_GAME_OR_UNRESOLVED_ID"],1)

if __name__=="__main__":
    unittest.main()
