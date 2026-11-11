"""Phase1G is isolated from the original frozen predictor and frontend."""
import os
import unittest
from unittest.mock import patch
import nfl_phase1d_runner as RN
import nfl_phase1e_live as LIVE
import nfl_phase1g_shadow as GSHADOW


class Phase1GAdapterTests(unittest.TestCase):
    def test_exact_unchanged_frozen_prefit_code_identity(self):
        import nfl_phase1e_ops as OPS
        self.assertEqual(OPS.code_identity(),
            "b77dc75b8587a359ae481b1c24cbaee0633915b20a79f63028a8582241bb079d")

    def test_no_new_flag_does_not_run_challenger(self):
        a=LIVE.LiveRunner.__new__(LIVE.LiveRunner)
        base=[{"game_id":"GOOD","status":"FORECAST_SUCCESS"}]
        with patch.dict(os.environ,{"NFL_PHASE1G_CHALLENGER":"0",
                                    "NFL_PHASE1F_CHALLENGER":"0"}), \
             patch.object(RN.Runner,"run_context",return_value=base) as original, \
             patch.object(GSHADOW,"run_group") as alt:
            got=a.run_context("T90",[],2026,11,"test",None)
        self.assertEqual(got,base)
        self.assertNotIn("phase1g",got[0])
        original.assert_called_once()
        alt.assert_not_called()

    def test_capture_only_after_original_forecast_success(self):
        a=LIVE.LiveRunner.__new__(LIVE.LiveRunner)
        base=[{"game_id":"GOOD","status":"FORECAST_SUCCESS"},
              {"game_id":"BAD","status":"SAFE_EXPLICIT_FAILURE"}]
        with patch.dict(os.environ,{"NFL_PHASE1G_CHALLENGER":"1",
                                    "NFL_PHASE1F_CHALLENGER":"0"}), \
             patch.object(RN.Runner,"run_context",return_value=base), \
             patch.object(GSHADOW,"run_group",return_value={
               "GOOD":{"game_id":"GOOD","status":"PHASE1G_REAL_PREGAME_RESEARCH_SAVED"}
                }) as alt:
            got=a.run_context("T90",[],2026,11,"test",None)
        self.assertEqual(got[0]["phase1g"]["status"],
                         "PHASE1G_REAL_PREGAME_RESEARCH_SAVED")
        self.assertNotIn("phase1g",got[1])
        self.assertEqual(alt.call_args.kwargs["only_games"],{"GOOD"})

    def test_research_shadow_failure_keeps_original_model_result(self):
        a=LIVE.LiveRunner.__new__(LIVE.LiveRunner)
        a.root="/tmp/test-not-persisted"
        base=[{"game_id":"GOOD","status":"FORECAST_SUCCESS"}]
        with patch.dict(os.environ,{"NFL_PHASE1G_CHALLENGER":"1",
                                    "NFL_PHASE1F_CHALLENGER":"0"}), \
             patch.object(RN.Runner,"run_context",return_value=base), \
             patch.object(GSHADOW,"run_group",side_effect=RuntimeError("test failure")), \
             patch("nfl_phase1d_cas.Ledger") as ledger:
            got=a.run_context("T90",[],2026,11,"test",None)
        self.assertEqual(got[0]["status"],"FORECAST_SUCCESS")
        self.assertIn("SAFE_EXPLICIT_FAILURE",got[0]["phase1g"]["status"])
        ledger.return_value.append_many.assert_called_once()


if __name__=="__main__":
    unittest.main()
