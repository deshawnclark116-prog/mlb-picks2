"""
NHL_GOALS_G1_RUN -- development (D1-D4), architecture selection, 2024 late-period confirmation, 2025 final historical confirmation for the skater GOALS head.
Protocol: nhl_models/nhl_outcome_engine/phase_goals_g1_protocol.json (committed first). 2024 / 2025 are never touched by `dev`; `confirm2024` refuses to run before the dev artifacts are committed; `confirm2025` only after 2024 passed.
  python nhl_goals_g1_run.py dev|confirm2024|confirm2025 --work DIR
"""
import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import nhl_goals_g1_data as GD
import nhl_goals_g1_models as GM
import nhl_sog_phase1a_data as D
import nhl_sog_phase1a_metrics as M
import nhl_sog_phase1b_probe as PB

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_outcome_engine"
PROTOCOL = OUT / "phase_goals_g1_protocol.json"
CODE = ["nhl_goals_g1_data.py", "nhl_goals_g1_models.py", "nhl_goals_g1_run.py", "tests/test_nhl_goals_g1.py"]
FOLDS = [("D1", [2018, 2019], 2020), ("D2", [2018, 2019, 2020], 2021), ("D3", [2018, 2019, 2020, 2021], 2022), ("D4", [2018, 2019, 2020, 2021, 2022], 2023)]
ARCHS = ("G0", "G1", "G2")
MODEL_CLASS = {"G0": GM.G0, "G1": GM.G1, "G2": GM.G2}
THRESH = {"P1_rel_crps": 0.005, "P2_nll": 0.005, "P3_ks": 0.02, "P4_5_slice": 0.05, "min_rows": 500,
          "C1_crps": 0.005, "C2_nll": 0.005, "C3_ks": 0.02, "C3_cov": 0.02, "C3_bias": 0.05, "C3_ece": 0.02, "C3_zero": 0.01, "C4_slice": 0.10}
TS_STATUS_OK, TS_STATUS_BAD = "GOALS_HISTORICAL_CHAMPION", "GOALS_HISTORICAL_CHAMPION_NOT_ESTABLISHED"


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def committed_and_clean(path):
    rel = str(Path(path).resolve().relative_to(REPO))
    return bool(git("log", "--format=%H", "-1", "--", rel)) and git("status", "--porcelain", "--", rel) == ""


def js(o):
    if isinstance(o, dict):
        return {str(k): js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [js(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(js(obj), indent=1, sort_keys=True) + "\n")


# ------------------------------------------------------------------ data
def load_goals_table(work):
    p = Path(work) / "goals_table.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    games, rows = D.load_frozen()
    tab = pickle.load(open(Path(work) / "table.pkl", "rb"))[0] if (Path(work) / "table.pkl").exists() else D.build_prediction_rows(games, rows)[0]
    scoring = GD.load_scoring()
    ext, rep = GD.build_goals_table(tab, rows, scoring)
    pickle.dump((ext, rep), open(p, "wb"), protocol=4)
    return ext, rep


def sel(tab, seasons):
    m = np.isin(tab["season"], seasons)
    return {k: v[m] for k, v in tab.items()}


def slice_masks(tab):
    return {"F": tab["POS_F"] == 1, "D": tab["POS_D"] == 1, "early_season(<=10 team games)": tab["N_CURRENT_SEASON_TEAM_GAMES_OBS"] <= 10, "established_season(>10)": tab["N_CURRENT_SEASON_TEAM_GAMES_OBS"] > 10}


def game_weeks(tab):
    uniq, idx = np.unique(tab["game_id"], return_index=True)
    return uniq, D.week_index(tab["start"][idx])


# ------------------------------------------------------------------ scoring one prediction
def score(model_id, pred, tab):
    y = tab["goals"]; g, pl = tab["game_id"], tab["player_id"]
    pmf = pred["pmf"]; mean = GM.pmf_mean(pmf)
    summ, rows = M.summarize("pmf", {"pmf": pmf}, y, model_id, g, pl, mean)
    ex = PB.calib_extras("pmf", {"pmf": pmf}, y, model_id, g, pl)
    summ["calibration"] = PB.calibration_report(ex, y)
    td = M.threshold_diag(rows_p_ge(pmf), y)
    summ["threshold_diagnostics"] = {k.replace("SOG", "goals"): v for k, v in td.items() if int(k.split(">=")[1].rstrip(")")) <= 3}
    summ["zero_goal_calibration"] = {"mean_pred_P0": float(pmf[:, 0].mean()), "observed_zero_rate": float((y == 0).mean()), "abs_gap": float(abs(pmf[:, 0].mean() - (y == 0).mean()))}
    summ["relative_mean_bias"] = float((mean.mean() - y.mean()) / y.mean())
    summ["coherence"] = GM.coherence_report(pmf)
    cls = GM.pos_class_of(tab)
    summ["by_class"] = {c: {"rows": int((cls == c).sum()), "mean_pred": float(mean[cls == c].mean()) if (cls == c).any() else None, "mean_obs": float(y[cls == c].mean()) if (cls == c).any() else None, "mean_pred_P0": float(pmf[cls == c, 0].mean()) if (cls == c).any() else None,
                         "observed_zero_rate": float((y[cls == c] == 0).mean()) if (cls == c).any() else None} for c in GM.CLASSES}
    summ["diagnostics"] = pred.get("diagnostics", {})
    if "implied_mean_sog" in pred:
        summ["implied_mean_sog"] = float(pred["implied_mean_sog"].mean())
        summ["observed_mean_sog"] = float(tab["sog"].mean())
        summ["implied_conversion_probability"] = float(mean.sum() / pred["implied_mean_sog"].sum())
        summ["observed_conversion_probability_played"] = float(tab["goals"][tab["played"] == 1].sum() / tab["eff_sog"][tab["played"] == 1].sum())
    if "posterior_sd" in pred:
        summ["posterior_conversion_sd"] = {"mean": float(pred["posterior_sd"].mean()), "quantiles_5_50_95": [float(v) for v in np.percentile(pred["posterior_sd"], [5, 50, 95])]}
        summ["prior_dominated_player_row_frequency"] = float(pred["prior_dominated"].mean())
    masks = slice_masks(tab)
    summ["slices_crps"] = {k: M.slice_crps(rows["crps"], m) for k, m in masks.items()}
    return summ, rows


def rows_p_ge(pmf):
    cdf = np.cumsum(pmf, axis=1)
    return np.stack([1.0 - cdf[:, j] for j in range(5)], axis=1)


# ------------------------------------------------------------------ pure decision rules (unit-tested)
def promotion_decision(*, rel_crps_improvement, boot_upper95, rel_nll, ks_delta, fd_ok, phase_ok):
    t = {"P1": {"pass": bool(rel_crps_improvement >= THRESH["P1_rel_crps"] or boot_upper95 < 0), "rel_crps_improvement": rel_crps_improvement, "boot_upper95": boot_upper95},
         "P2": {"pass": bool(rel_nll <= THRESH["P2_nll"]), "value": rel_nll}, "P3": {"pass": bool(ks_delta <= THRESH["P3_ks"]), "value": ks_delta},
         "P4": {"pass": bool(fd_ok)}, "P5": {"pass": bool(phase_ok)}}
    return t, all(v["pass"] for v in t.values())


def select_architecture(order, tests):
    """order = complexity order simplest first. tests[(challenger, incumbent)] -> bool (promotion). Sequential: challengers face the CURRENT incumbent."""
    incumbent, history = order[0], []
    for ch in order[1:]:
        ok = tests(ch, incumbent)
        history.append({"challenger": ch, "incumbent": incumbent, "promoted": bool(ok)})
        if ok:
            incumbent = ch
    return incumbent, history


def confirmation_gate(sel_s, comp_s, sel_rows, comp_rows, tab):
    """C1-C4 of the protocol: selected architecture vs the strongest development comparator."""
    masks = slice_masks(tab)
    ok_fd, d_fd = M.slice_gate(sel_rows["crps"], comp_rows["crps"], {k: masks[k] for k in ("F", "D")}, THRESH["min_rows"], THRESH["C4_slice"])
    ok_ph, d_ph = M.slice_gate(sel_rows["crps"], comp_rows["crps"], {k: masks[k] for k in masks if k not in ("F", "D")}, THRESH["min_rows"], THRESH["C4_slice"])
    g = {"C1_crps_not_materially_worse": {"pass": bool(sel_s["crps_macro_game"] <= comp_s["crps_macro_game"] * (1 + THRESH["C1_crps"])), "selected": sel_s["crps_macro_game"], "comparator": comp_s["crps_macro_game"], "relative": sel_s["crps_macro_game"] / comp_s["crps_macro_game"] - 1},
         "C2_nll_not_materially_worse": {"pass": bool(sel_s["nll_macro_game"] <= comp_s["nll_macro_game"] * (1 + THRESH["C2_nll"])), "selected": sel_s["nll_macro_game"], "comparator": comp_s["nll_macro_game"], "relative": sel_s["nll_macro_game"] / comp_s["nll_macro_game"] - 1}}
    ece1 = sel_s["threshold_diagnostics"]["P(goals>=1)"]["ece_10_equal_count_bins"]
    cal = {"pit_ks_delta": sel_s["pit_ks"] - comp_s["pit_ks"], "pit80_abs_error_delta": sel_s["calibration"]["80"]["randomized_pit_abs_error"] - comp_s["calibration"]["80"]["randomized_pit_abs_error"],
           "pit90_abs_error_delta": sel_s["calibration"]["90"]["randomized_pit_abs_error"] - comp_s["calibration"]["90"]["randomized_pit_abs_error"], "abs_relative_mean_bias": abs(sel_s["relative_mean_bias"]), "ece_P_goals_ge_1": ece1, "zero_goal_abs_gap": sel_s["zero_goal_calibration"]["abs_gap"]}
    cal_ok = cal["pit_ks_delta"] <= THRESH["C3_ks"] and cal["pit80_abs_error_delta"] <= THRESH["C3_cov"] and cal["pit90_abs_error_delta"] <= THRESH["C3_cov"] and cal["abs_relative_mean_bias"] <= THRESH["C3_bias"] and ece1 <= THRESH["C3_ece"] and cal["zero_goal_abs_gap"] <= THRESH["C3_zero"]
    g["C3_calibration_guards"] = {"pass": bool(cal_ok), **cal}
    g["C4_no_catastrophic_slice"] = {"pass": bool(ok_fd and ok_ph), "position": d_fd, "season_phase": d_ph}
    g["all_pass"] = all(v["pass"] for k, v in g.items() if k != "all_pass")
    return g


# ------------------------------------------------------------------ development
def run_dev(tab, log=print):
    t0 = time.time()
    folds = {}
    pooled = {a: {"crps": [], "game_crps": [], "games": [], "weeks": [], "masks": {}} for a in ARCHS}
    for fid, tr_s, va_s in FOLDS:
        tr, va = sel(tab, tr_s), sel(tab, [va_s])
        sog = GM.fit_sog(tr)
        models = {"G0": GM.G0().fit(tr, sog), "G1": GM.G1().fit(tr, sog), "G2": GM.G2().fit(tr)}
        uniq, wk = game_weeks(va)
        fold = {"train_seasons": tr_s, "validate_season": va_s, "n_train": int(len(tr["game_id"])), "n_validate": int(len(va["game_id"])), "models": {}, "artifacts": {a: models[a].artifact() for a in ARCHS}}
        for a in ARCHS:
            pred = models[a].predict(va)
            summ, rows = score(f"{a}_{fid}", pred, va)
            fold["models"][a] = summ
            pool = pooled[a]
            pool["crps"].append(rows["crps"]); pool["game_crps"].append(rows["crps_per_game"]); pool["games"].append(rows["games"]); pool["weeks"].append(wk)
            for k, m in slice_masks(va).items():
                pool["masks"].setdefault(k, []).append(m)
            pool.setdefault("pit_ks", []).append(summ["pit_ks"]); pool.setdefault("nll", []).append(summ["nll_macro_game"]); pool.setdefault("crps_m", []).append(summ["crps_macro_game"])
        folds[fid] = fold
        log(f"{fid} validate {va_s}: " + " ".join(f"{a} crps={fold['models'][a]['crps_macro_game']:.5f} nll={fold['models'][a]['nll_macro_game']:.5f}" for a in ARCHS))
    mean = {a: {"crps": float(np.mean(pooled[a]["crps_m"])), "nll": float(np.mean(pooled[a]["nll"])), "pit_ks": float(np.mean(pooled[a]["pit_ks"]))} for a in ARCHS}
    cache = {}

    def test(ch, inc):
        pc, pi = pooled[ch], pooled[inc]
        crps_ch, crps_inc = np.concatenate(pc["crps"]), np.concatenate(pi["crps"])
        delta = np.concatenate(pc["game_crps"]) - np.concatenate(pi["game_crps"])
        wk = np.concatenate(pc["weeks"])
        bs = M.bootstrap_report(delta, wk, float(np.mean(np.concatenate(pi["game_crps"]))))
        rel = (mean[inc]["crps"] - mean[ch]["crps"]) / mean[inc]["crps"]
        masks = {k: np.concatenate(v) for k, v in pc["masks"].items()}
        ok_fd, d_fd = M.slice_gate(crps_ch, crps_inc, {k: masks[k] for k in ("F", "D")}, THRESH["min_rows"], THRESH["P4_5_slice"])
        ok_ph, d_ph = M.slice_gate(crps_ch, crps_inc, {k: masks[k] for k in masks if k not in ("F", "D")}, THRESH["min_rows"], THRESH["P4_5_slice"])
        t, ok = promotion_decision(rel_crps_improvement=rel, boot_upper95=bs["one_sided_95_upper_bound"], rel_nll=(mean[ch]["nll"] - mean[inc]["nll"]) / mean[inc]["nll"], ks_delta=mean[ch]["pit_ks"] - mean[inc]["pit_ks"], fd_ok=ok_fd, phase_ok=ok_ph)
        cache[(ch, inc)] = {"criteria": t, "bootstrap_pooled_games": bs, "slices_position": d_fd, "slices_season_phase": d_ph, "promoted": ok}
        return ok
    selected, history = select_architecture(list(ARCHS), test)
    others = [a for a in ARCHS if a != selected]
    comparator = min(others, key=lambda a: mean[a]["crps"])
    lowest = min(ARCHS, key=lambda a: mean[a]["crps"])
    return {"folds": folds, "mean_over_folds": mean, "promotion_tests": {f"{c}_vs_{i}": v for (c, i), v in cache.items()}, "selection_history": history, "selected": selected, "strongest_development_comparator": comparator, "lowest_mean_crps_architecture": lowest,
            "selected_equals_lowest_crps": bool(selected == lowest), "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ confirmations
def refit_and_score(tab, train_seasons, test_season, names, tag):
    tr, te = sel(tab, train_seasons), sel(tab, [test_season])
    sog = GM.fit_sog(tr) if any(n in ("G0", "G1") for n in names) else None
    out, rows, arts = {}, {}, {}
    for n in names:
        m = GM.G2().fit(tr) if n == "G2" else MODEL_CLASS[n]().fit(tr, sog)
        out[n], rows[n] = score(f"{n}_{tag}", m.predict(te), te)
        arts[n] = m.artifact()
    return out, rows, arts, te, tr


def run_confirmation(tab, dev, train_seasons, test_season, tag):
    selected, comp = dev["selected"], dev["strongest_development_comparator"]
    names = [selected] if selected == comp else [selected, comp]
    summ, rows, arts, te, tr = refit_and_score(tab, train_seasons, test_season, names, tag)
    gate = confirmation_gate(summ[selected], summ[comp], rows[selected], rows[comp], te)
    return {"selected": selected, "comparator": comp, "targets_fit": f"{train_seasons[0]}-{train_seasons[-1]}", "target_scored": test_season, "n_train_rows": int(len(tr["game_id"])), "n_test_rows": int(len(te["game_id"])), "summaries": summ, "gate": gate, "passed": bool(gate["all_pass"]), "artifacts": arts}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["dev", "confirm2024", "confirm2025"]); ap.add_argument("--work", required=True)
    a = ap.parse_args()
    Path(a.work).mkdir(parents=True, exist_ok=True)
    assert committed_and_clean(PROTOCOL), "the protocol must be committed before any performance is computed"
    assert all(committed_and_clean(REPO / c) for c in CODE), "data / models / run / test code must be committed (clean) before any performance is computed"
    tab, rep = load_goals_table(a.work)
    protocol_sha = git("log", "--format=%H", "-1", "--", str(PROTOCOL.relative_to(REPO)))
    if a.cmd == "dev":
        assert not (OUT / "phase_goals_g1_dev_results.json").exists(), "development already scored"
        res = run_dev(tab)
        res.update({"protocol_sha": protocol_sha, "code_head": git("rev-parse", "HEAD"), "data_quality": rep, "feature_table_sha256": D.table_hash({k: v for k, v in tab.items() if k in D.FEATURES or k in ("game_id", "player_id", "goals")}), "exposure": "development years 2018-2023 only; 2024/2025 untouched"})
        write_json("phase_goals_g1_dev_results.json", {k: v for k, v in res.items()})
        sa = {"selected_architecture": res["selected"], "strongest_development_comparator": res["strongest_development_comparator"], "selection_history": res["selection_history"], "mean_over_folds": res["mean_over_folds"], "protocol_sha": protocol_sha,
              "frozen_rule": "refit the selected architecture (and the comparator, for the gate) on the confirmation training window with identical procedures; no new feature, no tuning, no calibration rescue", "rejected_architectures_frozen": [x for x in ARCHS if x != res["selected"]],
              "status": "FROZEN_BY_DEVELOPMENT_SELECTION; 2024 / 2025 may not change it"}
        write_json("phase_goals_g1_selected_architecture.json", sa)
        print(json.dumps(js({"selected": res["selected"], "comparator": res["strongest_development_comparator"], "mean_over_folds": res["mean_over_folds"], "history": res["selection_history"]}), indent=1))
    elif a.cmd == "confirm2024":
        assert committed_and_clean(OUT / "phase_goals_g1_dev_results.json") and committed_and_clean(OUT / "phase_goals_g1_selected_architecture.json"), "dev results + selected architecture must be committed before 2024 is scored"
        assert not (OUT / "phase_goals_g1_2024_confirmation.json").exists(), "2024 already scored once"
        dev = json.loads((OUT / "phase_goals_g1_dev_results.json").read_text())
        r = run_confirmation(tab, dev, list(range(2018, 2024)), 2024, "2024")
        r.update({"label": "LATE_PERIOD_CONFIRMATION_DESCRIPTIVELY_EXPOSED", "protocol_sha": protocol_sha, "scored_once": True, "on_failure": "RESEARCH / FROZEN_REJECTED; 2025 not scored" if not r["passed"] else "proceed to 2025 (after commit)"})
        write_json("phase_goals_g1_2024_confirmation.json", r)
        print(json.dumps(js({"selected": r["selected"], "comparator": r["comparator"], "passed": r["passed"], "gate": {k: v["pass"] for k, v in r["gate"].items() if isinstance(v, dict)}, "crps": {n: s["crps_macro_game"] for n, s in r["summaries"].items()}}), indent=1))
    else:
        c24 = OUT / "phase_goals_g1_2024_confirmation.json"
        assert committed_and_clean(c24), "2024 confirmation must be committed before 2025 is scored"
        assert json.loads(c24.read_text())["passed"] is True, "2024 did not pass: 2025 is never scored"
        assert not (OUT / "phase_goals_g1_2025_confirmation.json").exists(), "2025 already scored once"
        dev = json.loads((OUT / "phase_goals_g1_dev_results.json").read_text())
        r = run_confirmation(tab, dev, list(range(2018, 2025)), 2025, "2025")
        r.update({"label": "FINAL_HISTORICAL_CONFIRMATION_DESCRIPTIVELY_EXPOSED", "protocol_sha": protocol_sha, "scored_once": True, "status": TS_STATUS_OK if r["passed"] else TS_STATUS_BAD})
        write_json("phase_goals_g1_2025_confirmation.json", r)
        print(json.dumps(js({"status": r["status"], "gate": {k: v["pass"] for k, v in r["gate"].items() if isinstance(v, dict)}, "crps": {n: s["crps_macro_game"] for n, s in r["summaries"].items()}}), indent=1))


if __name__ == "__main__":
    main()
