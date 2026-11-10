"""Strict Phase1E frozen-pregame numerical grading regression tests."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
import nfl_phase1e_forward_numeric_grade as G

GID="2026_04_BUF_MIN"
KICK="2026-10-04T17:00:00Z"
CUT="2026-10-04T15:30:00Z"
PRE="2026-10-04T15:27:00Z"
NAME=f"2026_wk04_T90_{GID}.jsonl"


def p1(pid,market="rec_yds",name="Player",median=55,content="snap1"):
    return {"game_id":GID,"horizon":"T90","season":2026,"week":4,
      "team":"BUF","opponent":"MIN","player_id":pid,"player_name":name,
      "outcome":market,"median":median,"mean":median+3,
      "quantiles":{"p10":10,"p90":95},
      "p_active":.93,"expected_opportunities":8.5,
      "kickoff":KICK,"cutoff":CUT,
      "input_snapshots":{"retrieval_ts":PRE,"content_id":content,
                         "time_travel":False}}


def v2(pid,market="rec_yds",point=60,content="snap1"):
    return {"game_id":GID,"horizon":"T90","season":2026,"week":4,
      "team":"BUF","opponent":"MIN","player_id":pid,"outcome":market,
      "kind":"v2_forward_forecast","eligibility":"eligible",
      "kickoff":KICK,"cutoff":CUT,"as_of_retrieval_ts":PRE,
      "input_hashes":{"content_id":content},
      "v2_frozen_comparator":{"point":point}}


class ForwardAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for sub in ("forecasts","v2forecasts"):
            (self.root/sub/"batches").mkdir(parents=True)
        self.db=self.root/"truth.db"
        con=sqlite3.connect(self.db)
        con.executescript("""
          CREATE TABLE games(game_id TEXT, home_team TEXT, away_team TEXT,
                             home_score INTEGER, away_score INTEGER);
          CREATE TABLE player_games(game_id TEXT, team TEXT, player_id TEXT,
                  season_type TEXT, receiving_yards REAL, rushing_yards REAL,
                  passing_yards REAL);
        """)
        con.execute("INSERT INTO games VALUES (?,?,?,?,?)",
                    (GID,"BUF","MIN",17,7))
        con.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?,?)",
                    (GID,"BUF","p1","REG",58,0,0))
        con.commit();con.close()

    def write(self,first,second):
        head={"_batch":NAME[:-6],"game_id":GID,
              "generated_at":"2026-10-04T15:42:00Z"}
        for folder,records in (("forecasts",[head]+first),("v2forecasts",second)):
            p=self.root/folder/"batches"/NAME
            p.write_text("\n".join(json.dumps(x) for x in records)+"\n")

    def test_paired_real_outcome_only_never_dnp_as_zero(self):
        self.write([p1("p1",name="Star"),p1("p2",name="Missing")],
                   [v2("p1"),v2("p2")])
        result=G.grade(self.root,self.db,weeks={4})
        self.assertEqual(result["matched_rows"],1)
        self.assertEqual(result["excluded_counts"]
            ["unverified_player_participation_or_identity"],1)
        row=result["complete_player_by_player"][0]
        self.assertEqual(row["actual"],58.0)
        self.assertEqual(row["phase1_median_abs_error"],3)
        self.assertEqual(row["frozen_v2_abs_error"],2)
        self.assertEqual(result["entire_paired_cohort"]["n"],1)
        self.assertEqual(result["pregame_projected_50plus_cohort"]["n"],1)

    def test_source_mismatch_refused_not_silently_graded(self):
        self.write([p1("p1",content="snap1")],[v2("p1",content="snap2")])
        with self.assertRaisesRegex(ValueError,"DIFFERENT_PREGAME_SNAPSHOT"):
            G.grade(self.root,self.db,weeks={4})

    def test_late_capture_refused_even_if_forecast_wins(self):
        a=p1("p1")
        a["input_snapshots"]["retrieval_ts"]="2026-10-04T16:00:00Z"
        self.write([a],[v2("p1")])
        with self.assertRaisesRegex(ValueError,"AFTER_ORIGINAL_CUTOFF"):
            G.grade(self.root,self.db,weeks={4})

    def test_time_travel_pretending_to_be_live_is_refused(self):
        a=p1("p1")
        a["input_snapshots"]["time_travel"]=True
        self.write([a],[v2("p1")])
        with self.assertRaisesRegex(ValueError,"NOT_A_GENUINE_LIVE_CAPTURE"):
            G.grade(self.root,self.db,weeks={4})

    def test_incoherent_game_and_postgame_generation_refused(self):
        a=p1("p1")
        a["game_id"]="wrong"
        self.write([a],[v2("p1")])
        with self.assertRaisesRegex(ValueError,"GAME_HORIZON"):
            G.grade(self.root,self.db,weeks={4})
        self.write([p1("p1")],[v2("p1")])
        f=self.root/"forecasts"/"batches"/NAME
        allrows=f.read_text().splitlines()
        hdr=json.loads(allrows[0]);hdr["generated_at"]="2026-10-04T18:00:00Z"
        allrows[0]=json.dumps(hdr);f.write_text("\n".join(allrows)+"\n")
        with self.assertRaisesRegex(ValueError,"AFTER_KICKOFF"):
            G.grade(self.root,self.db,weeks={4})

    def test_duplicate_pregame_player_identity_refused(self):
        self.write([p1("p1"),p1("p1")],[v2("p1")])
        with self.assertRaisesRegex(ValueError,"DUPLICATE_ORIGINAL_ID"):
            G.grade(self.root,self.db,weeks={4})


if __name__=="__main__":
    unittest.main()
