#!/usr/bin/env python3
import datetime as dt
import unittest
import tennis_firstseen_production_backtest as A

P1="Alexander Zverev"
P2="Botic Van De Zandschulp"
START="2026-09-11T20:00:00Z"
LOGGED="2026-09-11T17:00:00Z"

def comp(p1=P1,p2=P2,winner=P1,date=START,score=((6,6),(4,3))):
    names=[p1,p2]
    return {"id":"1001","date":date,
     "status":{"type":{"state":"post","completed":True,"description":"Final"}},
     "competitors":[{"winner":n==winner,"athlete":{"displayName":n},
       "linescores":[{"value":x,"winner":n==winner}
                     for i,x in enumerate(score[j])]}
       for j,n in enumerate(names)]}

def pick(market,choice,**kw):
    return {"tour":"atp","espn_date":"20260911","tourney":"US Open","competition_id":"1001",
      "player1":P1,"player2":P2,"market":market,"pick":choice,"model_prob":.55,
      "logged_at":LOGGED,"start_time_utc":START,**kw}

class TennisTruthTests(unittest.TestCase):
    def test_moneyline_requires_correct_actual_winner_and_pregame(self):
        p=pick("moneyline",P1+" ML")
        r,err=A.grade(p,comp())
        self.assertIsNone(err)
        self.assertEqual(r["result"],"WIN")
        self.assertEqual(r["winner"],P1)
        p["pick"]=P2+" ML"
        self.assertEqual(A.grade(p,comp())[0]["result"],"LOSS")
        p["logged_at"]=START
        self.assertEqual(A.grade(p,comp())[1],"NO_PROOF_PICK_PUBLISHED_BEFORE_SCHEDULED_START")
    def test_competition_id_is_not_sufficient_when_espn_changes_players(self):
        p=pick("moneyline",P1+" ML")
        # Historical source actually has at least 3 such collisions.
        r,why=A.grade(p,comp(p1="Federico Cina",p2="Nuno Borges",winner="Federico Cina"))
        self.assertIsNone(r)
        self.assertEqual(why,"ESPN_FINAL_PLAYERS_NOT_ORIGINAL_PICK_PLAYERS")
    def test_total_games_exact_line_push_and_provenance(self):
        p=pick("total_games","OVER 19.0",line=19.0,model_source="model_projected_line")
        r,why=A.grade(p,comp())
        self.assertIsNone(why)
        self.assertEqual(r["actual_games"],19)
        self.assertEqual(r["result"],"PUSH")
        self.assertEqual(r["bet_evidence"],"model_generated_not_a_book_bet")
        p["pick"]="UNDER 18.5";p["line"]=18.5
        self.assertEqual(A.grade(p,comp())[0]["result"],"LOSS")
        p["pick"]="OVER 18.5"
        self.assertEqual(A.grade(p,comp())[0]["result"],"WIN")
        p["line"]=17.5
        self.assertEqual(A.grade(p,comp())[1],"ORIGINAL_TOTALS_LINE_MISMATCH")
    def test_book_count_only_given_quoted_book_and_real_line(self):
        p=pick("total_games","OVER 18.5",line=18.5,odds=1.92,book="bovada",
               model_source="real_book_line")
        self.assertEqual(A.grade(p,comp())[0]["bet_evidence"],
                         "archived_book_quote_not_executable_certified")
        p.pop("book")
        self.assertEqual(A.grade(p,comp())[0]["bet_evidence"],"unverified_line_not_bettable")
    def test_exact_score_is_NOT_a_book_and_requires_both_sets(self):
        p=pick("set_betting",P1+" 2-0")
        r,err=A.grade(p,comp())
        self.assertIsNone(err)
        self.assertEqual(r["bet_evidence"],"not_applicable")
        self.assertEqual(r["result"],"WIN")
        p["pick"]=P1+" 2-1"
        self.assertEqual(A.grade(p,comp())[0]["result"],"LOSS")
    def test_retirments_and_walkovers_arent_wins(self):
        p=pick("moneyline",P1+" ML")
        for status in ("Walkover","Retired","Abandoned"):
            c=comp()
            c["status"]["type"]["description"]=status
            self.assertEqual(A.grade(p,c)[1],"FINAL_INCOMPLETE_RETIREMENT_OR_WALKOVER")
    def test_duplicate_final_id_is_quarantined(self):
        data={"events":[{"groupings":[{"grouping":{"displayName":"Men's Singles"},
           "competitions":[comp(),comp()]}]}]}
        self.assertIsNone(A.finalized_by_id(data,"atp")["1001"])
    def test_unfinished_or_wrong_group_never_graded(self):
        c=comp()
        c["status"]["type"]["completed"]=False
        data={"events":[{"groupings":[{"grouping":{"displayName":"Men's Singles"},
           "competitions":[c]}]}]}
        self.assertEqual(A.finalized_by_id(data,"atp"),{})
        self.assertEqual(A.finalized_by_id(data,"wta"),{})
    def test_summary_excludes_unfinished_and_preserves_all_three_markets(self):
        ps=[pick("moneyline",P1+" ML"),pick("total_games","OVER 18.5",line=18.5),
            pick("set_betting",P1+" 2-0")]
        ps[0]["competition_id"]="1001"
        result=A.audit(ps,{("atp","20260911"):{"1001":comp()}})
        self.assertEqual(result["scored"],3)
        self.assertEqual(result["by_market"]["moneyline"]["wins"],1)
        self.assertEqual(result["by_market"]["total_games"]["wins"],1)
        self.assertEqual(result["by_market"]["set_betting"]["wins"],1)
        self.assertEqual(result["actual_verified_book_bets"],0)

if __name__=="__main__":
    unittest.main()
