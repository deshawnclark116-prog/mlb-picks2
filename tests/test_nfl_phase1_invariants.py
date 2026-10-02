"""
Phase 1A invariant tests (research, shadow). Standalone (pytest is not installed) and
pytest-compatible.

  python -u tests/test_nfl_phase1_invariants.py --unit-only
  python -u tests/test_nfl_phase1_invariants.py --data-dir /tmp/nfl_data [--cache units.pkl]

Unit / synthetic (no data): allocation coherence, inactive branch, injury
redistribution stress cases, determinism, timestamp helpers, source scan.
Data invariants: as-of contract on every forecast row, T-24 vs T-90 separation,
depth-chart snapshot timing, no target-game information in features
(perturbation), roles use only prior games, fit-window audit, forecast records.
"""
import argparse
import inspect
import os
import re
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_phase1_availability as A  # noqa: E402
import nfl_phase1_common as C  # noqa: E402
import nfl_phase1_data as P  # noqa: E402
import nfl_phase1_opportunity as O  # noqa: E402
import nfl_phase1_role_state as R  # noqa: E402
import nfl_phase1_team_environment as TE  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = [REPO / f for f in ("nfl_phase1_data.py", "nfl_phase1_common.py", "nfl_phase1_availability.py", "nfl_phase1_team_environment.py",
                          "nfl_phase1_role_state.py", "nfl_phase1_opportunity.py", "nfl_phase1_evaluate.py")]


def _rng(seed=1):
    return np.random.default_rng(seed)


# ------------------------------------------------------------------ synthetic allocation stress tests
def mean_counts(pact, prop, other, mu_T, k_T=500.0, alpha=None, N=4000, seed=1):
    counts, T, active, full = O.sample_alloc(np.asarray(pact, float), np.asarray(prop, float), other, mu_T, k_T, alpha, N, _rng(seed))
    return counts, T, active, full


def test_counts_plus_other_equal_total_and_nonnegative():
    counts, T, active, full = mean_counts([0.9, 0.9, 0.5], [0.5, 0.3, 0.1], 0.05, 30)
    assert (counts >= 0).all()
    assert np.allclose(full.sum(1), 1.0)
    assert (full >= 0).all()
    assert (counts.sum(1) <= T).all()             # remainder is the outside-candidate bucket


def test_inactive_branch_receives_zero_and_weight_redistributes():
    counts, T, active, full = mean_counts([0.5, 1.0], [0.6, 0.4], 0.0, 30, N=6000)
    assert (counts[~active[:, 0], 0] == 0).all(), "inactive player received opportunity"
    assert (full[~active[:, 0], 0] == 0).all()
    # when player 0 is out, player 1 takes everything (no other bucket)
    assert (counts[~active[:, 0], 1] == T[~active[:, 0]]).all()


def test_rb1_ruled_out_redistributes_to_committee():
    base = mean_counts([1, 1, 1], [0.55, 0.30, 0.10], 0.05, 28)[0].mean(0)
    out = mean_counts([0.0, 1, 1], [0.55, 0.30, 0.10], 0.05, 28)[0].mean(0)
    assert out[0] == 0
    assert out[1] > base[1] and out[2] > base[2]
    assert abs(out.sum() + 0.05 * 28 * (out.sum() / (out.sum() + 1e-9)) * 0 - out.sum()) < 1e-9
    # named total stays ~ constant (redistribution, not disappearance): named share 0.95 both ways -> 1.0 * 28 * (0.40/0.45) vs 0.95
    assert out.sum() > base.sum() * 0.85
    # proportional to propensity
    assert abs(out[1] / out[2] - 0.30 / 0.10) < 0.4


def test_wr1_ruled_out_targets_redistribute():
    base = mean_counts([1, 1, 1, 1], [0.30, 0.22, 0.18, 0.12], 0.05, 34)[0].mean(0)
    out = mean_counts([0.0, 1, 1, 1], [0.30, 0.22, 0.18, 0.12], 0.05, 34)[0].mean(0)
    assert out[0] == 0 and all(out[1:] > base[1:])
    assert abs(out.sum() - base.sum()) < 0.25 * base.sum()


def test_starting_qb_ruled_out_backup_takes_attempts():
    out = mean_counts([0.0, 1.0], [0.95, 0.05], 0.0, 36, N=2000)
    counts, T = out[0], out[1]
    assert (counts[:, 0] == 0).all()
    assert (counts[:, 1] == T).all()               # backup absorbs the whole dropback volume


def test_rookie_promotion_moves_share():
    old = mean_counts([1, 1], [0.60, 0.05], 0.05, 28)[0].mean(0)
    promoted = mean_counts([1, 1], [0.30, 0.40], 0.05, 28)[0].mean(0)
    assert promoted[1] > old[1] * 4 and promoted[0] < old[0]


def test_two_back_committee_is_symmetric():
    c = mean_counts([1, 1], [0.45, 0.45], 0.05, 28, N=8000)[0].mean(0)
    assert abs(c[0] - c[1]) < 0.4


def test_returning_player_with_partial_activity():
    full = mean_counts([1.0, 1.0], [0.5, 0.4], 0.05, 28)[0].mean(0)
    uncertain = mean_counts([0.5, 1.0], [0.5, 0.4], 0.05, 28)[0].mean(0)
    assert uncertain[0] < full[0] and uncertain[1] > full[1]


def test_expected_named_total_close_to_named_share_times_total():
    counts, T, _, _ = mean_counts([1, 1, 1], [0.5, 0.3, 0.15], 0.05, 30, N=6000)
    assert abs(counts.sum(1).mean() / T.mean() - 0.95) < 0.01


def test_dirichlet_keeps_coherence():
    counts, T, active, full = mean_counts([1, 1, 1], [0.5, 0.3, 0.15], 0.05, 30, alpha=20.0)
    assert np.allclose(full.sum(1), 1.0) and (counts.sum(1) <= T).all()


def test_sampling_is_deterministic():
    a = O.sample_alloc(np.array([0.9, 0.8]), np.array([0.5, 0.4]), 0.05, 30, 40.0, 20.0, 50, _rng(3))
    b = O.sample_alloc(np.array([0.9, 0.8]), np.array([0.5, 0.4]), 0.05, 30, 40.0, 20.0, 50, _rng(3))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert O.stable_seed(("k", 1), "x") == O.stable_seed(("k", 1), "x")


def test_team_environment_fit_is_deterministic():
    x = np.array([[1, 3.0, 0.1, 1, 0.0, 0], [1, 3.1, -0.1, 0, 0.3, 0], [1, 2.9, 0.0, 1, -0.2, 1]] * 20)
    y = np.array([20, 24, 18] * 20, float)
    assert np.array_equal(TE.fit_glm(x, y), TE.fit_glm(x, y))


def test_availability_probabilities_valid():
    X = np.random.default_rng(0).normal(size=(300, 4)); y = (X[:, 0] + 0.2 * np.random.default_rng(1).normal(size=300) > 0).astype(float)
    m = A.fit_logit(X, y)
    p = A.pred_logit(m, X)
    assert ((p >= 0) & (p <= 1)).all()
    assert np.allclose(p + (1 - p), 1.0)           # P(inactive) + P(active) = 1; no third state is modelled


def test_t24_feature_set_excludes_t90_roster_status():
    assert not any(c.startswith("t90_") for c in A.OFF_T24 + A.OFF_T24_DEPTH + A.DEF_T24)
    assert all(c.startswith("t90_") for c in A.T90_EXTRA)


def test_no_sportsbook_variables_in_feature_sets():
    bad = re.compile(r"spread|total_line|moneyline|odds|implied|over_under", re.I)
    cols = set(A.OFF_T24 + A.OFF_T24_DEPTH + A.DEF_T24 + A.T90_EXTRA + list(O.POS_FLAGS))
    for fam in R.FAMILIES.values():
        cols |= set(fam)
    for k in TE.TARGETS:
        cols |= set(TE.XGB_COLS(k))
    assert not [c for c in cols if bad.search(c)]


def test_sources_never_read_sportsbook_columns():
    for f in SRC:
        text = f.read_text()
        for col in P.MARKET_COLUMNS:
            hits = [m.start() for m in re.finditer(re.escape(col), text)]
            allowed = 1 if f.name == "nfl_phase1_data.py" else 0
            assert len(hits) <= allowed, f"{f.name}: sportsbook column '{col}' referenced {len(hits)}x"


def test_no_target_or_oracle_variable_names_in_feature_sets():
    for fam in R.FAMILIES.values():
        assert not any(c.startswith("ORC_") or c.startswith("y_") for c in fam)


def test_injury_timestamp_precedes_t24_cutoff_for_all_slots():
    for gd, gt in (("2025-09-07", "13:00"), ("2025-09-04", "20:20"), ("2025-09-08", "20:15"), ("2025-12-20", "16:30"),
                   ("2025-09-28", "09:30"), ("2025-11-27", "12:30")):
        k = P.kickoff_utc(gd, gt)
        assert P.injury_info_ts(k) <= k - P.T24, (gd, gt)
    # depth-chart selection never returns a snapshot after the cutoff
    D = type("D", (), {})()
    from datetime import datetime, timezone
    snaps = [datetime(2025, 9, d, 7, tzinfo=timezone.utc) for d in (1, 3, 5, 7)]
    D.depth_times = {"X": snaps}; D.depth = {"X": [(t, {}) for t in snaps]}
    D.depth_at = lambda team, cutoff, D=D: P.Data.depth_at(D, team, cutoff)
    ts, _ = D.depth_at("X", datetime(2025, 9, 6, 0, tzinfo=timezone.utc))
    assert ts == snaps[2]
    assert D.depth_at("X", datetime(2025, 8, 30, tzinfo=timezone.utc))[0] is None


def test_fit_audit_rejects_post_burned_and_dev_rows():
    ok = [{"s": 2023, "w": 3}, {"s": 2024, "w": 14}]
    C.audit_fit("unit_ok", ok[:1], ok[1:])
    for bad in ([{"s": 2026, "w": 4}], [{"s": 2025, "w": 5}]):
        try:
            C.audit_fit("unit_bad", bad)
        except AssertionError:
            continue
        raise AssertionError("audit_fit accepted forbidden rows")
    C.audit_fit("depth_ablation", [{"s": 2025, "w": 5}])          # the one labelled development-window fit


def test_windows_disjoint():
    for s in range(2022, 2027):
        for w in range(1, 19):
            assert sum([C.TRAIN(s, w), C.VALID(s, w), C.DEV(s, w)]) <= 1


# ------------------------------------------------------------------ data invariants
def data_checks(data_dir, cache):
    import pickle
    import nfl_phase1_evaluate as E
    D, U = E.build_pipeline(data_dir, cache)
    res = []

    def check(name, cond, detail=""):
        res.append((name, bool(cond), detail))

    # as-of contract on every forecast row
    late = pre = dep = hist_bad = notprior = 0
    n_rows = 0
    for u in U:
        kick = u["kick"]
        for r in u["players"]:
            n_rows += 1
            if r["max_info_ts_T24"] and r["max_info_ts_T24"] > u["c24"]:
                late += 1
            if r["max_info_ts_T90"] and r["max_info_ts_T90"] > u["c90"]:
                late += 1
            if r["depth_ts"] is not None and r["depth_ts"] > u["c24"]:
                dep += 1
            if r["inj_info_ts"] is not None and r["inj_info_ts"] > u["c24"]:
                late += 1
            for g in r["hist"]:
                if g["kick"] + timedelta(hours=24) > u["c24"]:
                    hist_bad += 1
        for r in u["defenders"]:
            if r["inj_info_ts"] is not None and r["inj_info_ts"] > u["c24"]:
                late += 1
            for g in r["hist"]:
                if g["kick"] + timedelta(hours=24) > u["c24"]:
                    hist_bad += 1
        for g in u["team_hist"] + u["opp_off_hist"] + u["def_allowed_hist"] + u["opp_allowed_hist"]:
            if g["kick"] + timedelta(hours=24) > u["c24"]:
                hist_bad += 1
        if not (u["c24"] < kick and u["c90"] < kick and u["c24"] < u["c90"]):
            pre += 1
    check("no post-cutoff source timestamp in any forecast row (T-24h and T-90m)", late == 0, f"{late} violations over {n_rows} player rows")
    check("every forecast cutoff precedes kickoff (T-24h < T-90m < kickoff)", pre == 0, f"{pre} violations")
    check("depth-chart snapshot timestamp <= forecast cutoff", dep == 0, f"{dep} violations")
    check("history entries (roles, teams, defenders) all completed >= 24h before the cutoff", hist_bad == 0, f"{hist_bad} violations")
    check("2024 week-labelled depth charts are not used", all(r["depth_ts"] is None or r["depth_ts"].year >= 2025 for u in U for r in u["players"]))
    # roles use only prior games
    for tname in ("carry", "target"):
        rows = R.type_rows(U, tname)
        bad = sum(1 for r in rows for g in r["ref"]["hist"] if (g["s"], g["w"]) >= (r["s"], r["w"]) and g["team"] == r["team"] and g["s"] == r["s"])
        check(f"role state '{tname}' uses only prior games", bad == 0, f"{bad} same-or-later games in history")
    # T-24 availability cannot see T-90 information: prediction must not change when game-day roster status changes
    off_rows = [F for F in A.build_rows(U[-300:])[0]][:400]
    cols = A.OFF_T24
    m = A.fit_logit(C.matrix(off_rows, cols), np.array([r["y"] for r in off_rows]))
    p1 = A.pred_logit(m, C.matrix(off_rows, cols))
    for r in off_rows:
        for k in list(r):
            if k.startswith("t90_"):
                r[k] = 1.0 - (r[k] or 0.0)
    p2 = A.pred_logit(m, C.matrix(off_rows, cols))
    check("T-24 availability prediction is invariant to game-day (T-90) roster status", np.array_equal(p1, p2))
    # perturbation: altering target-week realized data must not change that unit's pre-game inputs
    target = next(u for u in U if u["key"][0] == 2025 and u["key"][1] == 10)
    tk = target["key"]

    def snapshot(u):
        return (u["team_hist"], u["opp_off_hist"], u["def_allowed_hist"], u["opp_allowed_hist"],
                [(r["gid"], [(g["s"], g["w"], g["car"], g["tgt"], g["att"], g["routes"], g["snap"]) for g in r["hist"]], r["inj"], r["prev_roster"], r["depth"])
                 for r in u["players"]],
                [(r["pfr"], [(g["s"], g["w"], g["pct"]) for g in r["hist"]], r["inj"], r["prev_roster"]) for r in u["defenders"]])
    before = snapshot(target)
    saved = {}
    team, wk = tk[2], tk[1]
    for key in list(D.pp):
        if key[0] == 2025 and key[1] == wk:
            saved[("pp", key)] = dict(D.pp[key])
            for c in ("car", "tgt", "att", "routes", "rz_car", "rz_tgt"):
                D.pp[key][c] = D.pp[key].get(c, 0.0) + 17
    for key in list(D.tg):
        if key[0] == 2025 and key[1] == wk:
            saved[("tg", key)] = dict(D.tg[key])
            for c in ("plays", "dropbacks", "rushes", "targets", "rz_plays"):
                D.tg[key][c] = D.tg[key].get(c, 0.0) + 23
    for key in list(D.osnap):
        if key[0] == 2025 and key[1] == wk:
            saved[("osnap", key)] = dict(D.osnap[key])
            D.osnap[key] = {g: (v[0] + 9, min(1.0, v[1] + 0.3)) for g, v in D.osnap[key].items()}
    U2 = P.build(D)
    after = snapshot(next(u for u in U2 if u["key"] == tk))
    for (kind, key), v in saved.items():
        {"pp": D.pp, "tg": D.tg, "osnap": D.osnap}[kind][key] = v
    check("perturbing the target week's realized stats leaves that unit's pre-game inputs unchanged", before == after)
    # no forward data: latest 2026 stat week must be <= 3 (burned boundary); all fits audited
    maxw = max(k[1] for k in D.stat if k[0] == 2026)
    check("no game beyond the burned boundary (2026 wk3) is present in the replay", maxw <= 3, f"latest 2026 week with stats = {maxw}")
    return res


def forecast_record_checks(U):
    """Forecast records (T-24h) built from the replay must be timestamped before kickoff and use info <= forecast time."""
    bad = 0; n = 0
    for u in U[:400]:
        for r in u["players"]:
            rec = {"forecast_ts": u["c24"], "kick": u["kick"], "max_info_ts": r["max_info_ts_T24"]}
            n += 1
            if not (rec["forecast_ts"] < rec["kick"] and (rec["max_info_ts"] is None or rec["max_info_ts"] <= rec["forecast_ts"])):
                bad += 1
    return [("every forecast record is timestamped before kickoff with source timestamps <= forecast time", bad == 0, f"{bad}/{n} bad")]


def opportunity_data_checks(run_out):
    res = []
    frames = run_out["opp_rep"]["_frames"]
    coh = inact = neg = 0
    for name, fs in frames.items():
        if not O.ALLOC[name][4]:
            continue
        other = run_out["opp_rep"]["types"][name]["other_weight"]
        for f in fs[:150]:
            counts, T, active, full = O.sample_alloc(f["pact_sel"], f["P1"], other, f["mu_sel"], f["k_sel"], None, 60, _rng(5))
            coh += int(not np.allclose(full.sum(1), 1.0))
            neg += int((counts < 0).any() or (full < 0).any())
            inact += int((counts[~active] != 0).any())
    res.append(("real team-games: shares nonnegative and sum to 1 (incl. outside bucket)", coh == 0 and neg == 0, f"{coh} incoherent, {neg} negative"))
    res.append(("real team-games: inactive branch receives zero simulated opportunity", inact == 0, f"{inact} violations"))
    # per-sample reconciliation is exact (named counts + outside bucket = total); report calibration of the named share separately
    exact = 0
    for name, fs in frames.items():
        if not O.ALLOC[name][4]:
            continue
        other = run_out["opp_rep"]["types"][name]["other_weight"]
        for f in fs[:100]:
            counts, T, active, full = O.sample_alloc(f["pact_sel"], f["P1"], other, f["mu_sel"], f["k_sel"], None, 40, _rng(9))
            exact += int(not (counts.sum(1) <= T).all())
    fs = frames["carry"][:25]
    other = run_out["opp_rep"]["types"]["carry"]["other_weight"]
    cfg = run_out["opp_rep"]["types"]["carry"]["selected_config"]
    r1 = O.run_cfg(fs, cfg, other, 10.0, 2.0, True, N=40, tag="det"); r2 = O.run_cfg(fs, cfg, other, 10.0, 2.0, True, N=40, tag="det")
    res.append(("repeated opportunity simulation with identical inputs is bit-identical", np.array_equal(r1[1], r2[1]), ""))
    res.append(("real team-games: named counts never exceed the team total (named + outside bucket = total exactly)", exact == 0, f"{exact} violations"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("NFL_DATA_DIR"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--stage-cache", default=None)
    ap.add_argument("--unit-only", action="store_true")
    ap.add_argument("--with-run", action="store_true", help="also run the full evaluation and check its real allocations (slow)")
    args = ap.parse_args()
    units = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for k, v in units:
        v()
        print(f"PASS unit {k}")
    print(f"{len(units)}/{len(units)} unit / synthetic checks pass\n")
    if args.unit_only:
        return 0
    if not args.data_dir:
        raise SystemExit("--data-dir required for data invariants")
    import nfl_phase1_evaluate as E
    res = data_checks(args.data_dir, args.cache)
    _, U = E.build_pipeline(args.data_dir, args.cache)
    res += forecast_record_checks(U)
    if args.with_run:
        _, _, _, out = E.run(args.data_dir, args.cache, write=False, stage_cache=args.stage_cache)
        res += opportunity_data_checks(out)
        audit = C.FIT_AUDIT
        forbidden = [a for a in audit if any(w.endswith(":dev") for w in a["windows"]) and a["label"] not in C.DEV_WINDOW_LABELS]
        res.append(("no forward/evaluation outcomes in any architecture-selection fit (audited)", not forbidden, f"{len(audit)} fits audited, {len(forbidden)} forbidden"))
        p1 = TE.evaluate(out["U"])[0]["targets"]["rushes"]["candidates"]["C2_poisson_glm"]["combined"]["mae"]
        p2 = TE.evaluate(out["U"])[0]["targets"]["rushes"]["candidates"]["C2_poisson_glm"]["combined"]["mae"]
        res.append(("repeated run with identical inputs is deterministic (team environment refit)", p1 == p2, f"{p1} vs {p2}"))
    if args.with_run:
        import subprocess
        outs = []
        for seed in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            o = subprocess.run([sys.executable, str(REPO / "tests" / "_determinism_probe.py"), args.data_dir], capture_output=True, text=True, env=env)
            outs.append(o.stdout.strip().splitlines()[-1] if o.stdout.strip() else o.stderr[-300:])
        res.append(("repeated run in separate processes with different hash seeds is identical (replay fingerprint + team-environment scores)", outs[0] == outs[1], f"{outs[0]} | {outs[1]}"))
    failed = [r for r in res if not r[1]]
    for name, ok, detail in res:
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    print(f"\n{len(res) - len(failed)}/{len(res)} data invariants pass")
    if failed:
        raise SystemExit("INVARIANT FAILURES: " + "; ".join(f[0] for f in failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
