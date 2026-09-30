"""
NFL_PHASE1D_FREEZE_ARTIFACTS  (Phase 1D)  -- the Phase 1A artifact bundle of the freeze candidate, and the determinism proof of the walk-forward refit

Fits the Phase 1A artifacts by the walk-forward algorithm with every completed regular-season week through 2026 week 3 (the last burned week) as training data (target week = 2026 week 4),
TWICE, from scratch. The two bundles must have the identical sha256. The bundle is saved under nfl_models/nfl_player_outcome_phase1d/artifacts_freeze_fit_2026wk3/ (a manifest with the
sha256 of every artifact file plus the files themselves).

  python -u nfl_phase1d_freeze_artifacts.py [--data-dir /tmp/nflcsv]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import nfl_phase1_data as P1
import nfl_phase1_forecast as FC
import nfl_phase1d_p1a as P

REPO = Path(__file__).resolve().parent
OUT = REPO / "nfl_models" / "nfl_player_outcome_phase1d"
TARGET = (2026, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/tmp/nflcsv")
    a = ap.parse_args()
    t0 = time.time()
    D = P1.Data(a.data_dir)
    completed = {k for k in D.stat}
    assert max((s, w) for (s, w, _) in completed) <= (2026, 3), "data beyond the last burned week"
    U = P1.build(D, depth_universe=True)
    win = P.walk_forward_windows(completed, TARGET)
    print("windows", win.detail["last_training_week"], len(win.detail["train_weeks"]), len(win.detail["valid_weeks"]), f"({time.time() - t0:.0f}s)", flush=True)
    shas, arts = [], []
    for i in range(2):
        t1 = time.time()
        art = P.fit_artifacts(U, win, code=FC.file_hashes(), audit_week=TARGET,
                              extra_meta={"purpose": "FREEZE CANDIDATE bundle: Phase 1A refit on every completed week through 2026 wk3 (all burned data)", "target_week": list(TARGET),
                                          "algorithm": "walk-forward weekly refit v1 (nfl_phase1d_p1a.fit_artifacts)"})
        shas.append(art.manifest()["bundle_sha256"]); arts.append(art)
        print(f"fit {i + 1}: {shas[-1]} ({time.time() - t1:.0f}s)", flush=True)
    d = OUT / "artifacts_freeze_fit_2026wk3"
    shutil.rmtree(d, ignore_errors=True)
    man = arts[0].save(d)
    (OUT / "freeze_fit_determinism.json").write_text(json.dumps({"fit_1_bundle_sha256": shas[0], "fit_2_bundle_sha256": shas[1], "identical_bundle_sha256": shas[0] == shas[1], "bundle_sha256": shas[0],
                                                                "training_window": win.detail, "seconds": round(time.time() - t0), "files": man["files"]}, indent=1))
    print("identical:", shas[0] == shas[1])


if __name__ == "__main__":
    main()
