import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import nfl_precision_forward as F


class NflPrecisionFirstseenTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 8, 17, 45, tzinfo=timezone.utc)
        self.kickoff = "2026-11-08T18:00:00Z"
        self.board = {
            "generated_at_utc": "2026-11-08T17:41:51Z", "season": 2026, "week": 9,
            "picks": [self.pick("Precise", ".02", .47, .70),
                      self.pick("Volatile", ".03", .53, .15)],
        }
        self.role = {
            "schema": "nfl-role-evidence-coverage-v1",
            "status": "PREGAME_RESEARCH_ROLE_SNAPSHOT",
            "forecast_generated_at_utc": self.board["generated_at_utc"],
            "captured_at_utc": "2026-11-08T17:41:52Z",
            "season": 2026, "week": 9,
            "player_market_evidence": [
                self.evidence(p) for p in self.board["picks"]],
        }

    def pick(self, player, pid, reach, band):
        return dict(market="receiving_yards", player_id=pid, player=player,
            team="CLE", opponent="NO", season=2026, week=9, kickoff_utc=self.kickoff,
            projected_median=40, projected_mean=43, projected_low=3,
            projected_high=95, whole_projection=40,
            projection_reach_probability=reach,
            projection_within_10_probability=band, projection_simulation_n=10000,
            projection_probability_source=F.SOURCE, model_source="blended_sim",
            games_played=6, line=35.5, model_prob=.94)

    def evidence(self, p):
        return dict(player_id=p["player_id"],team=p["team"],opponent=p["opponent"],
            market=p["market"],kickoff_utc=p["kickoff_utc"],
            published_model_median=p["projected_median"],
            published_model_source=p["model_source"],
            role_evidence_complete_for_showcase=True,observed_stat_rows=3,
            roster_status="ROSTER_LISTED_NOT_STARTER_VERIFIED",warnings=[],
            has_verified_starting_role=False,
            has_calibrated_book_bet_probability=False)

    def test_precision_band_beats_near_fifty_percent_median(self):
        r=F.capture_rows(self.board,self.role,self.now)
        self.assertEqual([p["player"] for p in r], ["Precise", "Volatile"])
        self.assertEqual([p["frontend_research_rank"] for p in r],[1,2])
        self.assertTrue(all(p["official_active_status_verified"] is False for p in r))

    def test_source_timing_and_role_warnings_fail_closed(self):
        self.assertEqual(F.capture_rows(self.board,self.role,
                         datetime(2026,11,8,18,0,tzinfo=timezone.utc)), [])
        altered=json.loads(json.dumps(self.role))
        altered["player_market_evidence"][0]["warnings"]=["INJURY_UNCERTAIN"]
        self.assertEqual([p["player"] for p in F.capture_rows(
            self.board,altered,self.now)], ["Volatile"])
        altered["forecast_generated_at_utc"]="2026-11-08T16:40:00Z"
        with self.assertRaises(ValueError):
            F.capture_rows(self.board,altered,self.now)

    def test_append_only_repeated_revisions_do_not_overwrite_firstseen(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=Path(td)/"firstseen.jsonl"
            first=F.capture(self.board,self.role,self.now,ledger)
            self.assertEqual(first["new_firstseen"],2)
            self.board["picks"][0]["projection_within_10_probability"]=.01
            second=F.capture(self.board,self.role,self.now,ledger)
            self.assertEqual(second["new_firstseen"],0)
            lines=[json.loads(s) for s in ledger.read_text().splitlines()]
            self.assertEqual(lines[0]["projection_within_10_probability"],.70)

    def test_postgame_grader_never_uses_another_week_or_invents_dnp(self):
        with tempfile.TemporaryDirectory() as td:
            database=Path(td)/"nfl.sqlite"
            con=sqlite3.connect(database)
            con.execute("CREATE TABLE player_games (season INTEGER, week INTEGER, player_id TEXT, player_name TEXT, team TEXT, opponent TEXT, rushing_yards REAL, receiving_yards REAL)")
            con.execute("INSERT INTO player_games VALUES(2026,9,'.02','Precise','CLE','NO',0,45)")
            con.execute("INSERT INTO player_games VALUES(2026,8,'.02','Precise','CLE','PHI',0,0)")
            con.commit();con.close()
            row=F.capture_rows(self.board,self.role,self.now)
            results=F.grade(row,database)
            self.assertEqual(results["summary"]["n_graded"],1)
            self.assertEqual(results["summary"]["n_unresolved"],1)
            self.assertEqual(results["summary"]["observed_within_10_rate"],1.0)
            self.assertEqual(results["summary"]["mean_abs_error_yards"],5.0)
            self.assertEqual(results["summary"]["n_graded_name_crosswalk"],0)


    def test_espn_to_gsis_fallback_requires_unique_name_and_correct_teams(self):
        with tempfile.TemporaryDirectory() as td:
            database=Path(td)/"nfl.sqlite"
            con=sqlite3.connect(database)
            con.execute("CREATE TABLE player_games (season INTEGER, week INTEGER, player_id TEXT, player_name TEXT, team TEXT, opponent TEXT, rushing_yards REAL, receiving_yards REAL)")
            con.execute("INSERT INTO player_games VALUES(2026,9,'00-01234','Precise','CLE','NO',0,45)")
            con.execute("INSERT INTO player_games VALUES(2026,9,'00-09999','Precise','PIT','BAL',0,99)")
            con.commit();con.close()
            row=F.capture_rows(self.board,self.role,self.now)
            result=F.grade(row,database)
            self.assertEqual(result["summary"]["n_graded"],1)
            self.assertEqual(result["summary"]["n_graded_name_crosswalk"],1)
            self.assertEqual([x["actual"] for x in result["results"] if x["result"]=="GRADED"],[45])



if __name__=="__main__":
    unittest.main()
