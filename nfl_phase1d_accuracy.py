"""
NFL_PHASE1D_ACCURACY  (Phase 1D)  -- accuracy numbers that can never be read without their universe

Why: "72% of rushing rows within +/-10 yards" was measured on the EXPANDED pregame universe, where most candidates have a true outcome of 0 and are forecast
as ~0. That number is NOT "72% accurate forecasting of active players". Every within-tolerance figure produced here therefore carries the name of the
universe it was measured on, and `assert_labeled` refuses any structure or table that has a bare "accuracy" figure.

Slices (each reported separately, for every offensive continuous outcome and receptions):
  A  full pregame universe                          every Phase 1A candidate row (non-participants have outcome 0)                       [primary-style]
  B  participated                                   DIAGNOSTIC ONLY: candidates who actually played (uses the outcome of the game)
  C  positive actual opportunity                    DIAGNOSTIC ONLY: carries>0 (rush), targets>0 (receiving), attempts>0 (passing)
  D  protocol-eligible, stable role                 protocol prior-usage eligibility (as-of) and role shift <= median
  E  protocol-eligible, volatile role               protocol prior-usage eligibility and role shift >= 75th percentile
  H  protocol-eligible, all role states             (D + E + the middle band) = the pre-registered primary scoring universe of Amendment G
  F  P(active) >= 0.90                              pregame probability threshold
  G  expected-opportunity tiers                     fixed tiers of the simulated mean opportunity count of the row

  python -u nfl_phase1d_accuracy.py --scratch DIR --out-dir nfl_models/nfl_player_outcome_phase1d [--joint joint_depth_N1000.pkl]
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import nfl_phase1_data as P1
import nfl_phase1b_evaluate as EB
import nfl_phase1c_metrics as MT

OUTCOMES = {"rush_yds": ("carry", "rush_att"), "rec_yds": ("target", "targets"), "rec": ("target", "targets"), "pass_yds": ("qb_att", "pass_att")}
TOL = {"rush_yds": (5, 10, 15, 20, 25, 30, 35, 40), "rec_yds": (5, 10, 15, 20, 25, 30, 35, 40), "pass_yds": (10, 20, 30, 40, 50, 60, 75, 100, 125, 150), "rec": (0, 1, 2, 3)}
ELIG = {"rush_yds": (("RB", "FB"), "car", 5.0), "rec_yds": (("WR", "TE", "RB", "FB"), "tgt", 3.0), "rec": (("WR", "TE", "RB", "FB"), "tgt", 3.0), "pass_yds": (("QB",), "att", 15.0)}
TIERS = {"rush_att": ((0, 1), (1, 5), (5, 10), (10, 1e9)), "targets": ((0, 1), (1, 3), (3, 6), (6, 1e9)), "pass_att": ((0, 15), (15, 25), (25, 1e9))}
UNIVERSE = {
    "A": "A_full_pregame_universe(all Phase 1A candidate rows; non-participants have outcome 0)",
    "B": "B_participated_players_only(DIAGNOSTIC: selects on the game outcome)",
    "C": "C_positive_actual_opportunity_only(DIAGNOSTIC: selects on the game outcome)",
    "D": "D_protocol_eligible_stable_role",
    "E": "E_protocol_eligible_volatile_role",
    "H": "H_protocol_eligible_all_role_states(primary scoring universe of Amendment G)",
    "F": "F_forecast_P(active)>=0.90",
}


class UnlabeledAccuracy(AssertionError):
    pass


def assert_labeled(obj, path="$"):
    """No bare accuracy figure anywhere: a `within` block must sit next to a `universe` string; a key called 'accuracy' is forbidden."""
    if isinstance(obj, dict):
        if "accuracy" in obj or "accuracy_curve" in obj:
            raise UnlabeledAccuracy(f"{path}: generic 'accuracy' field emitted (must be 'within_tolerance' with a universe label)")
        if any(str(k).startswith("within_tolerance") for k in obj) and not isinstance(obj.get("universe"), str):
            raise UnlabeledAccuracy(f"{path}: within_tolerance without a universe label")
        for k, v in obj.items():
            assert_labeled(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_labeled(v, f"{path}[{i}]")


def assert_md_labeled(text):
    """Every markdown table whose header carries within-tolerance columns must have a `universe` column, and every such row must start with a universe name."""
    header_ok = False
    for i, line in enumerate(text.splitlines()):
        if line.startswith("|") and "---" not in line:
            if "within" in line.lower():
                if "universe" not in line.lower():
                    raise UnlabeledAccuracy(f"markdown line {i}: within-tolerance header without a universe column: {line[:100]}")
                header_ok = True
            elif header_ok:
                first = line.split("|")[1].strip()
                if not first[:2] in ("A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_"):
                    raise UnlabeledAccuracy(f"markdown line {i}: within-tolerance row without a universe name: {line[:100]}")
        elif not line.startswith("|"):
            header_ok = False


def prior_usage(U, want):
    out = {}
    for u in U:
        s, w, team = u["key"]
        if (s, w) not in want:
            continue
        for r in u["players"]:
            h = r["hist"][-3:]
            m = lambda k: float(np.mean([g[k] for g in h])) if h else 0.0
            out[(s, w, team, r["gid"])] = {"pos": r["pos"], "n_hist": r["n_hist"], "car": m("car"), "tgt": m("tgt"), "att": m("att")}
    return out


def cell(pred, y, tol, universe):
    err = np.abs(pred - y)
    return {"universe": universe, "n": int(len(y)), "within_tolerance": {str(t): round(float((err <= t + 1e-9).mean()), 4) for t in tol},
            "mae_median": round(float(err.mean()), 4), "median_ae": round(float(np.median(err)), 4), "share_of_rows_with_true_outcome_zero": round(float((y == 0).mean()), 4)}


def run(scratch, out_dir, joint="joint_depth_N1000.pkl", data_dir="/tmp/nflcsv"):
    S = Path(scratch)
    dd = pickle.load(open(S / joint, "rb"))
    pack = pickle.load(open(S / "p1a_inputs_depth.pkl", "rb"))
    D = P1.Data(data_dir)
    ACT = EB.load_actuals(data_dir)
    U = P1.build(D, depth_universe=True)
    dev = {(s, w) for (s, w, _) in pack["games"]}
    prior = prior_usage(U, dev)
    res = {"label": "DEVELOPMENT (2025 + 2026 wk1-3; burned, not holdout)", "n_draws": int(next(iter(dd["T24"].values()))[1].shape[1]),
           "rule": "every within-tolerance figure names its universe; A is NOT 'accuracy of forecasting active players'", "outcomes": {}}
    for name, (tname, opp) in OUTCOMES.items():
        res["outcomes"][name] = {}
        for hz in ("T24", "T90"):
            keys, Sx = dd[hz][name]
            Sx = Sx.astype(np.float64)
            y = np.array([ACT.get((s, w, g), {}).get(name, 0.0) for (s, w, t, g) in keys])
            med = np.median(Sx, 1)
            pk = "pact24" if hz == "T24" else "pact90"
            look = {}
            for (s, w, tm), g in pack["games"].items():
                t = g["types"].get(tname)
                if t:
                    for j, gid in enumerate(t["ids"]):
                        v = (float(t[pk][j]), abs(float(t["P1"][j]) - float(t["P0"][j])))
                        cur = look.get((s, w, tm, gid))
                        look[(s, w, tm, gid)] = v if cur is None else (max(cur[0], v[0]), max(cur[1], v[1]))
            pact = np.array([look.get(k, (1.0, 0.0))[0] for k in keys]); shift = np.array([look.get(k, (1.0, 0.0))[1] for k in keys])
            k2, S2 = dd[hz][opp]; m2 = {k: i for i, k in enumerate(k2)}
            exp_opp = np.array([S2[m2[k]].mean() if k in m2 else np.nan for k in keys])
            played = np.array([D.played_off((k[0], k[1], k[2]), k[3]) for k in keys])
            pp = [D.pp.get((k[0], k[1], k[3]), {}) for k in keys]
            pos_opp = np.array([{"rush_yds": p.get("car", 0), "rec_yds": p.get("tgt", 0), "rec": p.get("tgt", 0), "pass_yds": p.get("att", 0)}[name] > 0 for p in pp])
            poss, ukey, thr = ELIG[name]
            elig = np.array([(prior.get(k) is not None and prior[k]["pos"] in poss and prior[k]["n_hist"] >= 3 and prior[k][ukey] >= thr) for k in keys])
            e_shift = shift[elig]
            stable = elig & (shift <= np.median(e_shift)) if elig.sum() else elig
            volatile = elig & (shift >= np.quantile(e_shift, 0.75)) if elig.sum() else elig
            slices = {"A": np.ones(len(keys), bool), "B": played, "C": pos_opp, "D": stable, "E": volatile, "H": elig, "F": pact >= 0.90}
            for per, f in (("combined", lambda k: True), ("2025", lambda k: k[0] == 2025), ("2026_wk1_3", lambda k: k[0] == 2026)):
                pm = np.array([f(k) for k in keys])
                blk = {}
                for sk, m in slices.items():
                    mm = m & pm
                    if mm.sum() >= 30:
                        blk[UNIVERSE[sk]] = cell(med[mm], y[mm], TOL[name], UNIVERSE[sk])
                for lo, hi in TIERS[opp]:
                    mm = pm & (exp_opp >= lo) & (exp_opp < hi)
                    if mm.sum() >= 30:
                        lab = f"G_expected_{opp}_tier[{lo:g},{hi:g})"
                        blk[lab] = cell(med[mm], y[mm], TOL[name], lab)
                res["outcomes"][name].setdefault(hz, {})[per] = blk
    assert_labeled(res)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "accuracy_slices.json").write_text(json.dumps(res, indent=1))
    md = to_markdown(res)
    assert_md_labeled(md)
    (out / "accuracy_slices.md").write_text(md)
    return res


def to_markdown(res):
    L = ["# Accuracy slices (Phase 1D) - every within-tolerance figure names its universe", "",
         f"{res['label']}. Simulated draws per row: {res['n_draws']}. Row A is the expanded pregame universe: most of its rows have a true outcome of 0 and are forecast near 0, so a "
         "large within-tolerance share there does NOT mean the forecasts of active players are that accurate; compare with rows D / E / H / F. B and C select on the game outcome and are "
         "diagnostics only.", ""]
    for name, per_hz in res["outcomes"].items():
        for hz, per in per_hz.items():
            if hz != "T24":
                continue
            L += [f"## {name}, {hz}, combined 2025 + 2026 wk1-3", ""]
            blk = per["combined"]
            tol = list(next(iter(blk.values()))["within_tolerance"])
            L += ["| universe | n | share of rows with true outcome 0 | MAE(median) | " + " | ".join(f"within +/-{t}" for t in tol) + " |", "|---|---|---|---|" + "---|" * len(tol)]
            for u, c in blk.items():
                L.append(f"| {u} | {c['n']} | {c['share_of_rows_with_true_outcome_zero']} | {c['mae_median']} | " + " | ".join(str(c["within_tolerance"][t]) for t in tol) + " |")
            L.append("")
    return "\n".join(L) + "\n"


def relabel_legacy(obj, parent=None):
    """Phase 1C curve files predate the labeling rule: rename generic fields and attach the universe (= the enclosing subset name) to every cell."""
    LEG = {"all_eligible": "A_full_pregame_universe(all Phase 1A candidate rows; non-participants have outcome 0)"}
    REN = {"accuracy_curve_median": "within_tolerance_median", "accuracy_curve_mean_rounded": "within_tolerance_mean_rounded", "accuracy_curve": "within_tolerance"}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[REN.get(k, k)] = relabel_legacy(v, k)
        if any(k.startswith("within_tolerance") for k in out) and "universe" not in out:
            out["universe"] = LEG.get(parent, parent) if parent else "unnamed"
        return {(LEG.get(k, k) if isinstance(v, dict) else k): v for k, v in out.items()}
    if isinstance(obj, list):
        return [relabel_legacy(v, parent) for v in obj]
    return obj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--joint", default="joint_depth_N1000.pkl")
    ap.add_argument("--data-dir", default="/tmp/nflcsv")
    a = ap.parse_args()
    r = run(a.scratch, a.out_dir, a.joint, a.data_dir)
    print("written; outcomes:", list(r["outcomes"]))
