#!/usr/bin/env python3
"""Live tennis builder moneyline-only gate; no paid odds call, no leakage."""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tennis_serving_builder_a as S

class MoneylineOnlyServing(unittest.TestCase):
    def test_no_other_market_generated_or_bookmaker_endpoint_ever_called(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            db=root/"matches.sqlite"
            sqlite3.connect(db).close()
            out=root/"predictions.json"
            hist=root/"tennis_picks_log.jsonl"
            clock="2026-10-19T10:00Z"
            events=[
                {"p1_name":"Player Alpha","p2_name":"Player Beta",
                 "tourney_name":"Test Invitational","best_of":3,
                 "competition_id":"game1","start_time":clock,"state":"pre"},
                {"p1_name":"Player Gamma","p2_name":"Player Delta",
                 "tourney_name":"Test Invitational","best_of":3,
                 "competition_id":"game2","start_time":clock,"state":"in"}
            ]
            state={"overall_elo":{"A":1520.0,"B":1480.0,"C":1520.0,"D":1480.0},
                   "name_index":{}}
            def nameid(_,n):
                return {"Player Alpha":"A","Player Beta":"B","Player Gamma":"C","Player Delta":"D"}[n]
            def elo(_,player,surface):return {"A":(1520,45),"B":(1480,45),"C":(1520,45),"D":(1480,45)}[player]
            opts=["tennis_serving_builder_a.py","--db",str(db),"--out",str(out),"--date","20261019","--tours","atp"]
            with patch.object(sys,"argv",opts),\
                 patch.object(S,"build_state",return_value=state),\
                 patch.object(S,"fetch_espn_schedule",return_value={"events":[]}),\
                 patch.object(S,"extract_actionable_matches",return_value=events),\
                 patch.object(S,"resolve_player",side_effect=nameid),\
                 patch.object(S,"resolve_surface",return_value=("Hard",True)),\
                 patch.object(S,"combined_elo",side_effect=elo),\
                 patch.object(S,"PICKS_LOG_PATH",hist),\
                 patch.object(S,"fetch_totals_odds",side_effect=AssertionError("SHOULD NEVER PURCHASE TOTAL ODDS")):
                self.assertEqual(S.main(),0)
            doc=json.loads(out.read_text())
            self.assertEqual(doc["published_markets"],["moneyline"])
            self.assertEqual(doc["discarded_market_odds_requests"],0)
            self.assertFalse(doc["moneyline_market_odds_requested"])
            self.assertEqual(len(doc["picks"]),1)
            p=doc["picks"][0]
            self.assertEqual(p["market"],"moneyline")
            self.assertEqual(p["pick"],"Player Alpha ML")
            self.assertEqual(p["prediction_status"],"UNVALIDATED_MONEYLINE_RESEARCH_ONLY")
            self.assertTrue(p["ungraded"])
            self.assertFalse(doc["markets"]["moneyline"]["bettable"])
            self.assertNotIn("odds",p)
            self.assertEqual(len(hist.read_text().splitlines()),1)
            with patch.object(S,"PICKS_LOG_PATH",hist):
                self.assertEqual(S.load_logged_pick_keys(hist),{("game1","moneyline")})

    def test_existing_legacy_logged_markets_do_not_get_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"historical.jsonl"
            old={"competition_id":"historical_match","market":"total_games","pick":"OVER 22.5"}
            path.write_text(json.dumps(old)+"\n")
            keys=S.load_logged_pick_keys(path)
            new={"competition_id":"new_match","market":"moneyline","pick":"Alpha ML"}
            self.assertEqual(S.append_new_picks_to_log(path,keys,[new]),1)
            entries=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(entries[0],old)
            self.assertEqual(entries[1]["market"],"moneyline")

if __name__=="__main__":unittest.main()
