#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nfl_v2_phase1f_efficiency_forensic as F


def test_summary_prefers_lower_hybrid_mae():
    rows=[
      {"phase1e_abs_error":10.0,"human_eff_swap_abs_error":5.0,
       "phase1b_efficiency_abs_error":1.0,"human_efficiency_abs_error":0.5,
       "oracle_opportunity_phase1b_eff_abs_error":8.0,"oracle_opportunity_human_eff_abs_error":4.0,
       "oracle_efficiency_abs_error":3.0},
      {"phase1e_abs_error":20.0,"human_eff_swap_abs_error":15.0,
       "phase1b_efficiency_abs_error":2.0,"human_efficiency_abs_error":1.5,
       "oracle_opportunity_phase1b_eff_abs_error":12.0,"oracle_opportunity_human_eff_abs_error":9.0,
       "oracle_efficiency_abs_error":6.0},
    ]
    s=F.summarize(rows,"rec_yds")
    assert s["phase1e_mae"]==15.0
    assert s["human_eff_swap_mae"]==10.0
    assert s["human_eff_swap_minus_phase1e_mae"]==-5.0
    assert s["hybrid_row_wins"]==2


def test_actual_efficiency_zero_opportunity_is_not_invented():
    assert F.mean([None,None]) is None


if __name__=="__main__":
    test_summary_prefers_lower_hybrid_mae()
    test_actual_efficiency_zero_opportunity_is_not_invented()
    print("NFL V2 Phase 1F efficiency forensic tests: PASS")
