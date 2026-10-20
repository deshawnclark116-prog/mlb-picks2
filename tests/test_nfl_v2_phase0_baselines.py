#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_v2_phase0_baselines as B


def main():
    hist = {"p": [
        {"season":2026,"week":1,"team":"A","rushing_yards":30.0,"carries":10.0},
        {"season":2026,"week":2,"team":"A","rushing_yards":50.0,"carries":10.0},
        {"season":2026,"week":3,"team":"A","rushing_yards":40.0,"carries":10.0},
        # Same target week must NEVER enter the baseline.
        {"season":2026,"week":4,"team":"A","rushing_yards":999.0,"carries":99.0},
    ]}
    row={"season":2026,"week":4,"player_id":"p","outcome":"rush_yds"}
    p=B.baseline_predictions(row,hist)
    assert abs(p["recent3_mean"]-40.0)<1e-12
    assert abs(p["recent3_median"]-40.0)<1e-12
    assert abs(p["recent5_mean"]-40.0)<1e-12
    assert abs(p["workload3_x_efficiency8"]-40.0)<1e-12
    assert p["ewma8_decay0.70"] < 50

    audit={"rows":[{
        "clean_meaningful":True,"forecast_id":"f","player":"P","player_id":"p","position":"RB",
        "team":"A","opponent":"B","outcome":"rush_yds","actual":44.0,"median":60.0,
        "season":2026,"week":4
    }]}
    rep=B.compare(audit,hist)
    x=rep["by_outcome"]["rush_yds"]["candidates"]["recent3_mean"]
    assert x["n"]==1
    assert x["baseline_mae"]==4.0
    assert x["v1_mae_same_rows"]==16.0
    assert x["baseline_minus_v1_mae"]==-12.0
    assert x["baseline_row_wins"]==1
    assert rep["sportsbook_inputs_used"] is False
    print("NFL V2 Phase 0 baseline tests: PASS")


if __name__=="__main__":
    main()
