"""Export the AD_HOC_PREGAME Phase 1 + frozen-v2 stores to docs/nfl_phase1_experimental.json (separate from docs/nfl_predictions.json, which is never touched)."""
import argparse
import json
import shutil
from pathlib import Path

import nfl_phase1_store as ST
import nfl_phase1e_v2 as V2

REPO = Path(__file__).resolve().parent


def q(rec, k):
    return rec["quantiles"].get(k)


def export(root, out_json, evidence_dir):
    root = Path(root)
    f = [r for r in ST.Store(root, "forecasts").all_records() if r["horizon"] == "AD_HOC_PREGAME"]
    v = [r for r in ST.Store(root, "v2forecasts").all_records() if r["horizon"] == "AD_HOC_PREGAME"]
    for r in f:                                                       # match on the same key space; Phase 1 outcome names rush_yds / rec_yds are the v2 outcomes
        pass
    pairs, p_only, v_only = V2.match(f, v)
    v2by = {(p["key"]): p for p in pairs}
    rows = []
    for r in sorted(f, key=lambda r: (r["team"], r["player_name"], r["outcome"])):
        row = {"player": r["player_name"], "player_id": r["player_id"], "team": r["team"], "position": r["position"], "outcome": r["outcome"], "p_active": r["p_active"], "mean": r["mean"], "median": r["median"],
               "sd": r["sd"], "p10": q(r, "p10"), "p90": q(r, "p90"), "p_zero": r["p_zero"], "p_ge1": r.get("event_probability_ge1")}
        k = (r["game_id"], r["player_id"], r["outcome"], r["horizon"], r["cutoff"])
        if k in v2by:
            c = v2by[k]["v2_frozen_comparator"]
            row["v2_frozen"] = {"point": round(c["point_floored"], 2), "p10": round(c["quantiles"][9], 2), "p90": round(c["quantiles"][89], 2)}
            pr = v2by[k]["v2_live_production"]
            row["v2_production_point"] = round(pr["point"], 2) if pr else None
        rows.append(row)
    r0 = f[0]
    meta = {"label": "EXPERIMENTAL / AD_HOC_PREGAME", "disclaimer": "New Phase 1 player-outcome engine, shadow only. Operational run at N=100,000 draws (R11 is still a BLOCKER). NOT a T24/T90 forecast, NOT freeze evidence, NOT scored, not a betting recommendation. Production NFL picks (nfl.html) are unchanged.",
            "game_id": r0["game_id"], "away": r0["opponent"] if r0["team"] != r0["opponent"] and r0["team"] == "CLE" else r0["team"], "home": "CLE", "kickoff_utc": r0["kickoff"], "retrieval_ts_utc": r0["cutoff"],
            "information_cutoff": "real retrieval timestamp (all inputs retrieved at or before this instant)", "model_version": r0["model_version"], "n_draws": r0["simulation"]["n_draws"],
            "snapshot_set_id": r0["input_snapshots"].get("snapshot_set_id") if isinstance(r0.get("input_snapshots"), dict) else None, "n_phase1_records": len(f), "n_v2_records": len(v),
            "n_v2_eligible": sum(x["eligibility"] == "eligible" for x in v), "n_phase1_v2_pairs": len(pairs), "v2_comparator_version": V2.COMPARATOR_VERSION,
            "outcomes_note": "mean/median/p10/p90 are the calibrated simulated distribution of the stat (inactive players count as 0). v2 is the frozen incumbent comparator (conditional on playing)."}
    Path(out_json).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    ev = Path(evidence_dir)
    for kind in ("forecasts", "v2forecasts", "baselines"):
        d = root / kind / "batches"
        if d.exists():
            shutil.copytree(d, ev / kind, dirs_exist_ok=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", default=str(REPO / "docs" / "nfl_phase1_experimental.json")); ap.add_argument("--evidence", required=True)
    a = ap.parse_args()
    print(json.dumps(export(a.root, a.out, a.evidence), indent=1))
