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


def tables_md(report, selected):
    """Markdown component tables generated straight from report.json (no hand-typed numbers)."""
    L = ["# Phase 1A component tables (DEVELOPMENT: 2025 + 2026 wk1-3, burned; never holdout results)", ""]
    te = report["team_environment"]["targets"]
    L += ["## Team environment (negative-binomial distributions; baseline = shrunk current/prior team average)", "",
          "| target | selected | base MAE | sel MAE | base CRPS | sel CRPS | 2025 base->sel MAE | 2026 wk1-3 base->sel MAE | p(MAE) | p(CRPS) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in te.items():
        c = v["candidates"]; b = c["B0_blend"]; sc = c[v["selected"]]
        L.append(f"| {k} | {v['selected']} | {b['combined']['mae']:.3f} | {sc['combined']['mae']:.3f} | {b['combined']['crps']:.3f} | {sc['combined']['crps']:.3f} | "
                 f"{b['2025']['mae']:.3f} -> {sc['2025']['mae']:.3f} | {b['2026_wk1_3']['mae']:.3f} -> {sc['2026_wk1_3']['mae']:.3f} | {sc['vs_B0']['p_mae_not_better']} | {sc['vs_B0']['p_crps_not_better']} |")
    av = report["availability"]
    L += ["", "## Availability P(active) (baseline = status lookup)", "", "| side | time | selected | base logloss / Brier / ECE | selected logloss / Brier / ECE | 2025 ll base->sel | 2026 ll base->sel |", "|---|---|---|---|---|---|---|"]
    for side in ("offense", "defense"):
        for tag in ("T24", "T90"):
            R_ = av[side][tag]; b = R_["B0_status_lookup"]; sc = R_[R_["selected"]]
            L.append(f"| {side} | {tag} | {R_['selected']} | {b['combined']['logloss']:.4f} / {b['combined']['brier']:.4f} / {b['combined']['ece']:.4f} | "
                     f"{sc['combined']['logloss']:.4f} / {sc['combined']['brier']:.4f} / {sc['combined']['ece']:.4f} | {b['2025']['logloss']:.4f} -> {sc['2025']['logloss']:.4f} | {b['2026_wk1_3']['logloss']:.4f} -> {sc['2026_wk1_3']['logloss']:.4f} |")
    L += ["", "## Role-state share models (next-game share given playing; MAE; baseline R0 = last-8 mean)", "",
          "| share type | n | R0 last8 | R1 ewma | R2 kalman | R3 bocpd | R4 hmm | R5 learned | selected | R5 vs R0 p | 3-class role-change logloss: freq / R5 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, v in report["role_state"]["types"].items():
        m = v["models"]; rc = v["role_change_logloss"]
        L.append(f"| {t} | {v['n_dev']} | " + " | ".join(f"{m[k]['combined']['mae']:.4f}" for k in ("R0", "R1", "R2", "R3", "R4", "R5")) +
                 f" | {v['selected']} | {m['R5']['vs_R0']['p_not_better']} | {rc['B0_class_frequency']:.4f} / {rc.get('R5', float('nan')):.4f} |")
    L += ["", "## Opportunity allocation (player counts, pregame universe, non-participants = 0; selected chain vs dumb baseline)", "",
          "| type | n | base MAE | sel MAE | base CRPS | sel CRPS | base cov80 | sel cov80 | 2025 CRPS base->sel | 2026 wk1-3 CRPS base->sel | p(MAE) | p(CRPS) | selected components |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in report["opportunity"]["types"].items():
        a0, sm = v["steps"]["A0"], v["selected_metrics"]
        g26 = lambda m: m.get("2026_wk1_3", {}).get("crps", "n/a")
        comps = ",".join(c for c, x in v["selected_config"].items() if x)
        L.append(f"| {k} | {a0['n']} | {a0['mae']:.3f} | {sm['mae']:.3f} | {a0['crps']:.3f} | {sm['crps']:.3f} | {a0['cov80']} | {sm['cov80']} | "
                 f"{a0['2025']['crps']} -> {sm['2025']['crps']} | {g26(a0)} -> {g26(sm)} | {sm['vs_A0']['p_mae_not_better']} | {sm['vs_A0']['p_crps_not_better']} | {comps} |")
    L += ["", "## Leave-one-component-out (improvement in CRPS of the full chain over the chain without the component; + = component helps)", "",
          "| type | coherent | role | avail | team | dispersion |", "|---|---|---|---|---|---|"]
    for k, v in report["opportunity"]["types"].items():
        lo = v["leave_one_out"]
        cell = lambda c: (f"{lo[c]['full_vs_without']['crps_improvement']:+.4f} (p={lo[c]['full_vs_without']['p_crps_not_better']})" if c in lo else "n/a")
        L.append(f"| {k} | " + " | ".join(cell(c) for c in ("coherent", "role", "avail", "team", "dispersion")) + " |")
    return "\n".join(L) + "\n"


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


def families_kept(role_rep, alpha=0.10):
    """Role-state feature families that earned their place (leave-one-family-out, week-block bootstrap p < alpha).
    usage_only always stays. Selected on burned development data."""
    out = {}
    for t, v in role_rep["types"].items():
        kept = ["usage_only"]
        for fam, x in v["learned_leave_one_family_out"].items():
            if x["family_contribution"] > 0 and x["p_family_not_helpful"] < alpha:
                kept.append(fam)
        out[t] = kept
    return out


def role_change_choice(role_rep):
    """Which model supplies P(contract/stable/expand) per share type: lowest development logloss, and it must beat the
    class-frequency baseline; otherwise no model earns it (point estimates, not bootstrapped)."""
    out = {}
    for t, v in role_rep["types"].items():
        rc = v["role_change_logloss"]
        cand = {k: rc[k] for k in ("R2", "R3", "R4", "R5") if k in rc}
        best = min(cand, key=cand.get)
        out[t] = {"model": best if cand[best] < rc["B0_class_frequency"] else "class_frequency (no model earns it)",
                  "logloss": {**{k: round(x, 4) for k, x in cand.items()}, "class_frequency": round(rc["B0_class_frequency"], 4)}}
    return out


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
    fam_kept = families_kept(role_rep)
    opp_rep, frames_all = O.evaluate(rows_by_type, U, te_pred, te_b0, pact_sel, pact_lookup, fam_kept)
    sf = O.structural_vs_fallback(opp_rep, frames_all)
    elig = O.eligible_universe_check(opp_rep)
    print(f"opportunity done ({time.time() - t0:.0f}s)", flush=True)
    report = {"phase": "1A (shadow)", "label": C.DEV_LABEL, "protocol_version": "1.1",
              "as_of_assumptions": P.AS_OF_ASSUMPTIONS,
              "team_environment": te_rep, "availability": av_rep, "role_state": role_rep,
              "opportunity": strip(opp_rep), "route_structural_vs_target_allocation": sf, "protocol_eligible_universe_check": elig,
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
        "role_state": {"share_model": role_sel, "families_kept_by_share_type": fam_kept,
                       "expand_stable_contract_probabilities": role_change_choice(role_rep),
                       "depth_chart_family": "supported by the 2025-split ablation for carry, rb_carry, qb_rush, target, qb_att, rz_carry, rz_target; "
                                             "not for route; not applicable to defense (report.json role_state[...].depth_chart_ablation). "
                                             "Depth snapshots exist only from 2025, so the family enters the walk-forward fit on 2025+ rows and cannot be in the "
                                             "2023-24-trained development split reported here."},
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
        (OUT / "component_tables.md").write_text(tables_md(strip(report), selected))
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
