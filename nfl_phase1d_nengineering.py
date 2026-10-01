"""
NFL_PHASE1D_NENGINEERING  (Phase 1D)  -- is Monte Carlo error at N=25,000 negligible relative to football-model error?

Criteria and the burned evaluation set were written to nfl_models/nfl_player_outcome_phase1d/n_engineering_criteria.json BEFORE this ran.
N is NOT selected from football scores: actual outcomes enter only as the yardstick of model error (RMSE of the mean forecast) and for the CRPS
comparison of two simulation sizes on identical rows.

  python -u nfl_phase1d_nengineering.py --scratch DIR --adj FILES... --out FILE
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

import nfl_phase1_data as P1
import nfl_phase1_forecast as FC
import nfl_phase1c_dryrun as DR
import nfl_phase1b_evaluate as EB
import nfl_phase1c_evaluate as EV
import nfl_phase1c_sim as SM

REPO = Path(__file__).resolve().parent
CRIT = REPO / "nfl_models" / "nfl_player_outcome_phase1d" / "n_engineering_criteria.json"
OUTS = ("rush_yds", "rec_yds", "rec", "pass_yds", "atd")
TAILS = {"rush_yds": ("P(rush_yds>=75)", 75.0), "rec_yds": ("P(rec_yds>=100)", 100.0), "pass_yds": ("P(pass_yds>=300)", 300.0), "atd": ("P(anytime_td>=1)", 1.0)}


def crps_draws(S, y):
    Ss = np.sort(S, 1)
    m = Ss.shape[1]
    t1 = np.abs(Ss - y[:, None]).mean(1)
    i = np.arange(1, m + 1)
    t2 = np.sum((2 * i - m - 1)[None, :] * Ss, axis=1) / (m * m)
    return t1 - t2


def summarize(S, name):
    S = (S >= 1).astype(np.float64) if name == "atd" else S
    out = {"mean": S.mean(1), "median": np.median(S, 1), "p10": np.quantile(S, 0.10, 1), "p90": np.quantile(S, 0.90, 1), "sd": S.std(1)}
    if name in TAILS:
        out["tail"] = (S >= TAILS[name][1]).mean(1)
    return out, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--adj", nargs="+", required=True)
    ap.add_argument("--data-dir", default="/tmp/nflcsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None, help="pickle file: completed games are saved after each game and skipped on restart (the container can be rebooted); no effect on results")
    ap.add_argument("--N", type=int, default=None, help="candidate N (Phase 1E escalation); criteria / weeks / seeds / reference unchanged")
    a = ap.parse_args()
    crit = json.load(open(CRIT))
    N, NREF = (a.N or crit["fixed_N"]), crit["reference_N"]
    S0 = Path(a.scratch)
    cal = json.load(open(EV.OUT / "calibration_depth.json"))["final_maps"]
    bundle = DR.make_bundle(S0, a.adj, N, calibration=cal, records_file="records_depth.pkl")
    pack = pickle.load(open(S0 / "p1a_inputs_depth.pkl", "rb"))
    D = P1.Data(a.data_dir)
    ACT = EB.load_actuals(a.data_dir)
    rows = {o: {"y": [], "runs": {}} for o in OUTS}
    t0 = time.time()
    plan = [("ref", NREF, 9001)] + [(f"r{i}", N, 100 + i) for i in range(1, 5)]
    done_games, elapsed0 = set(), 0.0
    if a.checkpoint and Path(a.checkpoint).exists():
        ck = pickle.load(open(a.checkpoint, "rb"))
        rows, done_games, elapsed0 = ck["rows"], ck["done"], ck["elapsed"]
        t0 -= elapsed0
        print(f"resumed from checkpoint: {len(done_games)} games done", flush=True)
    for (s, w) in crit["predetermined_burned_set"]["weeks"]:
        for (A, B) in DR.week_games(pack, D, s, w):
            if (s, w, A, B) in done_games:
                continue
            for tag, n, seed in plan:
                res, gs = SM.run_game(pack, A, B, s, w, bundle.eff, bundle.idx, bundle.defaults, bundle.C, n, seed, crit["predetermined_burned_set"]["horizon"], None, bundle.qb_adjust)
                per = SM.collect(res, gs, s, w)
                for o in OUTS:
                    keys, S = per[o]
                    S = FC.calibrate_draws(o, S.astype(np.float64), bundle.calibration)
                    sm, Sx = summarize(S, o)
                    if tag == "ref":
                        y = np.array([float((ACT.get((s, w, g), {}).get("rush_td", 0.0) + ACT.get((s, w, g), {}).get("rec_td", 0.0)) > 0) for (_, _, _, g) in keys]) if o == "atd" \
                            else np.array([ACT.get((s, w, g), {}).get(EV.ACT_KEY[o], 0.0) for (_, _, _, g) in keys])
                        rows[o]["y"].append(y)
                    sm["crps"] = crps_draws(Sx, rows[o]["y"][-1])
                    rows[o]["runs"].setdefault(tag, []).append(sm)
            done_games.add((s, w, A, B))
            if a.checkpoint:
                pickle.dump({"rows": rows, "done": done_games, "elapsed": time.time() - t0}, open(a.checkpoint + ".tmp", "wb"), protocol=4)
                os.replace(a.checkpoint + ".tmp", a.checkpoint)
            print(f"{s} wk{w} {A}-{B} done ({time.time() - t0:.0f}s)", flush=True)
    res = {"criteria_file": str(CRIT.relative_to(REPO)), "N": N, "reference_N": NREF, "label": "burned development rows; engineering audit only", "outcomes": {}, "verdict": {}}
    cat = lambda o, tag, k: np.concatenate([r[k] for r in rows[o]["runs"][tag]])
    for o in OUTS:
        y = np.concatenate(rows[o]["y"])
        n = len(y)
        rmse = float(np.sqrt(((cat(o, "r1", "mean") - y) ** 2).mean()))
        c1 = float((cat(o, "r1", "sd") / np.sqrt(N)).mean() / max(rmse, 1e-12))
        crps25 = float(np.mean([cat(o, f"r{i}", "crps").mean() for i in range(1, 5)])); crps100 = float(cat(o, "ref", "crps").mean())
        c2 = abs(crps25 - crps100) / max(crps100, 1e-12)
        entry = {"n_rows": int(n), "model_rmse_of_mean": rmse, "C1_mean_se_over_rmse": c1, "crps_25k_mean_of_4": crps25, "crps_100k_reference": crps100, "C2_rel_crps_diff": float(c2)}
        stab = {}
        ref25 = "r1"
        for stat in ("mean", "median", "p10", "p90"):
            d = np.concatenate([np.abs(cat(o, f"r{i}", stat) - cat(o, ref25, stat)) for i in (2, 3, 4)]) / max(rmse, 1e-12)
            stab[stat] = {"mean_abs_diff_over_rmse": float(d.mean()), "p95_abs_diff_over_rmse": float(np.quantile(d, 0.95))}
        entry["C4_repeat_stability"] = stab
        entry["C4_pass"] = all(v["mean_abs_diff_over_rmse"] <= 0.02 and v["p95_abs_diff_over_rmse"] <= 0.05 for v in stab.values()) if o != "atd" else None
        if o in TAILS:
            tails = np.stack([cat(o, f"r{i}", "tail") for i in (1, 2, 3, 4)])
            p = np.clip(tails.mean(0), 0, 1)
            entry["C3_max_analytic_se"] = float(np.sqrt(np.max(p * (1 - p)) / N))
            entry["C3_max_empirical_sd_across_repeats"] = float(tails.std(0, ddof=1).max())
            entry["C3_pass"] = entry["C3_max_analytic_se"] <= 0.005 and entry["C3_max_empirical_sd_across_repeats"] <= 0.005
        entry["C1_pass"] = c1 <= 0.01 if o != "atd" else None
        entry["C2_pass"] = c2 <= 0.005
        res["outcomes"][o] = entry
    core = [o for o in OUTS if o != "atd"]
    res["verdict"] = {"C1": all(res["outcomes"][o]["C1_pass"] for o in core), "C2": all(res["outcomes"][o]["C2_pass"] for o in OUTS),
                      "C3": all(res["outcomes"][o]["C3_pass"] for o in TAILS if o in res["outcomes"]), "C4": all(res["outcomes"][o]["C4_pass"] for o in core)}
    res["verdict"]["N_retained"] = all(res["verdict"].values())
    res["seconds"] = round(time.time() - t0)
    import resource
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    res["repeat_design"] = {"plan": [(t, n, sd) for t, n, sd in plan], "weeks": crit["predetermined_burned_set"]["weeks"]}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
