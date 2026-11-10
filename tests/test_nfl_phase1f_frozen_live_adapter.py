"""Phase1F extension must preserve all frozen Phase1E code and stored forecasts."""
import os
import tempfile
import unittest
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import nfl_phase1d_runner as RN
import nfl_phase1e_live as LIVE
import nfl_phase1f_shadow as F


class FrozenSafetyTests(unittest.TestCase):
    def test_frozen_code_identity_exactly_equal_to_existing_locked_identity(self):
        import nfl_phase1e_ops as OPS
        self.assertEqual(OPS.code_identity(),
            "b77dc75b8587a359ae481b1c24cbaee0633915b20a79f63028a8582241bb079d")

    def test_no_flag_runs_only_original_frozen_runner(self):
        a=LIVE.LiveRunner.__new__(LIVE.LiveRunner)
        orig=[{"game_id":"GAME","status":"FORECAST_SUCCESS"}]
        with patch.dict(os.environ,{"NFL_PHASE1F_CHALLENGER":"0"}), \
             patch.object(RN.Runner,"run_context",return_value=orig) as baseline, \
             patch.object(F,"run_group") as alt:
            logs=a.run_context("T90",[],2026,11,"test",None)
        self.assertEqual(logs,orig)
        self.assertNotIn("phase1f",logs[0])
        baseline.assert_called_once()
        alt.assert_not_called()

    def test_separate_run_uses_only_baseline_successes(self):
        a=LIVE.LiveRunner.__new__(LIVE.LiveRunner)
        orig=[{"game_id":"GOOD","status":"FORECAST_SUCCESS"},
              {"game_id":"BAD","status":"SAFE_EXPLICIT_FAILURE"}]
        with patch.dict(os.environ,{"NFL_PHASE1F_CHALLENGER":"1"}), \
             patch.object(RN.Runner,"run_context",return_value=orig), \
             patch.object(F,"run_group",return_value={
               "GOOD":{"game_id":"GOOD","status":"PHASE1F_REAL_PREGAME_RESEARCH_SAVED"}}
                 ) as alt:
            logs=a.run_context("T90",[],2026,11,"test",None,
                               only_games={"GOOD","BAD"})
        self.assertEqual(logs[0]["phase1f"]["status"],
                         "PHASE1F_REAL_PREGAME_RESEARCH_SAVED")
        self.assertNotIn("phase1f",logs[1])
        self.assertEqual(alt.call_args.kwargs["only_games"],{"GOOD"})

    def test_coach_source_annotation_never_mutates_frozen_units(self):
        key=(2026,11,"BUF")
        kick=datetime(2026,11,12,tzinfo=timezone.utc)
        row={"s":2026,"w":10,"kick":kick}
        original={"key":key,"team_hist":[row]}
        class D:
            game={(2026,10,"BUF"):{"coach":"Example coach"}}
        with patch.object(F.P1,"Data",return_value=D()), \
             patch.object(F.P1,"build",return_value=[original]):
            ctx=F.asof_context("/tmp/no-file-needed",[key])
        self.assertEqual(ctx[key]["team_hist"][0]["coach"],"Example coach")
        self.assertNotIn("coach",row)
        self.assertIsNot(ctx[key],original)


if __name__=="__main__":
    unittest.main()
