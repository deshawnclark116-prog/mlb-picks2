"""
NFL_PHASE1_EVALUATE  (Phase 1A, shadow research)

Runs the full Phase 1A development evaluation and writes
nfl_models/nfl_player_outcome_phase1a/{report.json, selected_architecture.json,
rejected_candidates.json, component_tables.md}.

All scores are DEVELOPMENT (2025 + 2026 wk1-3, burned) - never holdout results.
No production files are read or written. No sportsbook numbers are read.

  python -u nfl_phase1_evaluate.py --data-dir /tmp/nfl_data
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

import nfl_phase1_availability as A
import nfl_phase1_common as C
import nfl_phase1_data as P
import nfl_phase1_opportunity as O
import nfl_phase1_role_state as R
import nfl_phase1_team_environment as TE

OUT = Path(__file__).resolve().parent / "nfl_models" / "nfl_player_outcome_phase1a"


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [strip(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def build_pipeline(data_dir, cache=None):
    """Returns everything downstream needs. Deterministic."""
    if cache and Path(cache).exists():
        D = None
        U = pickle.load(open(cache, "rb"))
    else:
        D = P.Data(data_dir)
        U = P.build(D)
        if cache:
            pickle.dump(U, open(cache, "wb"))
    return D, U


def run(data_dir, cache=None, write=True, stage_cache=None):
    t0 = time.time()
    C.FIT_AUDIT.clear()
    D, U = build_pipeline(data_dir, cache)
    print(f"units {len(U)} ({time.time() - t0:.0f}s)", flush=True)
    if stage_cache and Path(stage_cache).exists():
        st = pickle.load(open(stage_cache, "rb"))
        U, te_rep, fitted, te_rows, av_rep, off, dfn, role_rep, role_sel, rows_by_type = (st[k] for k in (
            "U", "te_rep", "fitted", "te_rows", "av_rep", "off", "dfn", "role_rep", "role_sel", "rows_by_type"))
        C.FIT_AUDIT[:] = st["audit"]
        print("stages loaded from stage cache (development speed-up)", flush=True)
    else:
        te_rep, fitted, te_rows = TE.evaluate(U)
        print(f"team env done ({time.time() - t0:.0f}s)", flush=True)
        av_rep, off, dfn = A.evaluate(U)
        print(f"availability done ({time.time() - t0:.0f}s)", flush=True)
        role_rep, role_sel, rows_by_type = R.evaluate(U)
        print(f"role state done ({time.time() - t0:.0f}s)", flush=True)
        if stage_cache:
            pickle.dump({"U": U, "te_rep": te_rep, "fitted": fitted, "te_rows": te_rows, "av_rep": av_rep, "off": off, "dfn": dfn,
                         "role_rep": role_rep, "role_sel": role_sel, "rows_by_type": rows_by_type, "audit": list(C.FIT_AUDIT)},
                        open(stage_cache, "wb"), protocol=4)
    te_pred = TE.predict(fitted, te_rows)
    te_b0 = {(r["s"], r["w"], r["team"], k): (C.safe(r[f"b0_{k}"], r[f"lg_{k}"]), fitted[k]["k_b0"]) for r in te_rows for k in TE.TARGETS}
    tab_off, base_off, _ = A.status_table([r for r in off if C.TRAIN(r["s"], r["w"]) or C.VALID(r["s"], r["w"])],
                                          np.array([r["y"] for r in off if C.TRAIN(r["s"], r["w"]) or C.VALID(r["s"], r["w"])]))
    tab_def, base_def, _ = A.status_table([r for r in dfn if C.TRAIN(r["s"], r["w"]) or C.VALID(r["s"], r["w"])],
                                          np.array([r["y"] for r in dfn if C.TRAIN(r["s"], r["w"]) or C.VALID(r["s"], r["w"])]))

    def pact_lookup(ref):
        st = A._status(ref.get("inj"))
        return tab_def.get(st, base_def) if "pfr" in ref else tab_off.get(st, base_off)

    pact_sel = lambda ref: ref["p_active_T24"]
    opp_rep, frames_all = O.evaluate(rows_by_type, U, te_pred, te_b0, pact_sel, pact_lookup)
    sf = O.structural_vs_fallback(opp_rep, frames_all)
    print(f"opportunity done ({time.time() - t0:.0f}s)", flush=True)
    report = {"phase": "1A (shadow)", "label": C.DEV_LABEL, "protocol_version": "1.1",
              "as_of_assumptions": P.AS_OF_ASSUMPTIONS,
              "team_environment": te_rep, "availability": av_rep, "role_state": role_rep,
              "opportunity": strip(opp_rep), "route_structural_vs_target_allocation": sf,
              "fit_audit": C.FIT_AUDIT}
    selected = {
        "protocol": "nfl_player_outcome_phase1_protocol.json v1.1",
        "selection_data": "burned development data only (train 2023 + 2024 wk1-12, validation 2024 wk13-18, development 2025 + 2026 wk1-3)",
        "availability": {"T24": {"offense": av_rep["offense"]["T24"]["selected"], "defense": av_rep["defense"]["T24"]["selected"]},
                         "T90": {"offense": av_rep["offense"]["T90"]["selected"], "defense": av_rep["defense"]["T90"]["selected"]},
                         "conditional_snap_share": {"offense": av_rep["offense_conditional_snap_share"]["selected"],
                                                    "defense": av_rep["defense_conditional_snap_share"]["selected"]},
                         "three_state_label": "not used (no source labels 'limited')"},
        "team_environment": {k: v["selected"] for k, v in te_rep["targets"].items()},
        "role_state": role_sel,
        "opportunity": {k: {"selected_config": v["selected_config"], "dirichlet_alpha": v["dirichlet_alpha"]} for k, v in opp_rep["types"].items()},
        "official_forward_receiving_path": sf.get("official_forward_path"),
    }
    rejected = {"team_environment": {}, "availability": {}, "role_state": {}, "opportunity": {}}
    for k, v in te_rep["targets"].items():
        rejected["team_environment"][k] = [{"candidate": n, "combined_mae": m["combined"]["mae"], "combined_crps": m["combined"]["crps"],
                                            "vs_B0": m["vs_B0"]} for n, m in v["candidates"].items() if n != v["selected"]]
    for side in ("offense", "defense"):
        for tag in ("T24", "T90"):
            R_ = av_rep[side][tag]
            rejected["availability"][f"{side}_{tag}"] = [{"candidate": n, "logloss": R_[n]["combined"]["logloss"], "vs_B0": R_[n]["vs_B0_logloss"]}
                                                          for n in ("B0_status_lookup", "C1_logistic", "C2_xgb") if n != R_["selected"]]
    for k, v in role_rep["types"].items():
        rejected["role_state"][k] = [{"candidate": n, "combined_mae": m["combined"]["mae"], "vs_R0": m["vs_R0"]}
                                     for n, m in v["models"].items() if n != v["selected"]]
    for k, v in opp_rep["types"].items():
        rejected["opportunity"][k] = {"components_not_earning_place": [c for c, e in v["components_earning_place"].items() if not e],
                                      "leave_one_out": {c: {"crps_improvement_of_full": m["full_vs_without"]["crps_improvement"],
                                                            "p": m["full_vs_without"]["p_crps_not_better"]} for c, m in v["leave_one_out"].items()}}
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "report.json").write_text(json.dumps(strip(report), indent=1, default=float))
        (OUT / "selected_architecture.json").write_text(json.dumps(strip(selected), indent=1, default=float))
        (OUT / "rejected_candidates.json").write_text(json.dumps(strip(rejected), indent=1, default=float))
    return report, selected, rejected, {"U": U, "D": D, "te_pred": te_pred, "te_rows": te_rows, "fitted": fitted, "off": off, "dfn": dfn,
                                        "rows_by_type": rows_by_type, "opp_rep": opp_rep, "frames_all": frames_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache", default=None, help="optional pickle of replay units (development speed-up only)")
    ap.add_argument("--stage-cache", default=None, help="optional pickle of availability/role/team stages (development speed-up only)")
    args = ap.parse_args()
    print("NFL_PHASE1_EVALUATE (development; burned data; shadow)\n" + "=" * 54)
    report, selected, rejected, _ = run(args.data_dir, args.cache, stage_cache=args.stage_cache)
    print(json.dumps(selected, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
