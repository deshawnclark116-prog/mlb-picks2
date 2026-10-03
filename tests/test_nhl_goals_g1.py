"""NHL goals head (G1 phase) tests: labels, leakage, effective-SOG rule, exact compound PMFs, hyperprior, selection / confirmation discipline. python tests/test_nhl_goals_g1.py"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tests"))
import nhl_engine_feasibility_data as FD  # noqa: E402
import nhl_goals_g1_data as GD  # noqa: E402
import nhl_goals_g1_models as GM  # noqa: E402
import nhl_goals_g1_run as R  # noqa: E402
import nhl_sog_phase1a_data as D  # noqa: E402
import nhl_sog_phase1a_metrics as M  # noqa: E402
import test_nhl_phase1a as T1A  # noqa: E402

OUT = REPO / "nhl_models" / "nhl_outcome_engine"
SPORTSBOOK = ("odds", "sportsbook", "bookmaker", "vegas", "juice", "vig", "implied_prob", "prop_line", "market_line", "closing_line", "projection")


# ------------------------------------------------------------------ synthetic fixtures
def synth_scoring(rows):
    out = {}
    for r in rows:
        goals = 1 if (r["player_id"] * 7 + r["game_id"]) % 5 == 0 else 0
        out[(r["game_id"], r["player_id"])] = (min(goals, r["sog"] + 1), r["sog"])
    return out


def synth_tab(seed=0, seasons=(2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025), games_per=70, per_game=14):
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ["game_id", "team_id", "player_id", "season", "start", "played", "sog", "goals", "eff_sog"] + D.FEATURES + GD.GOAL_HISTORY_FEATURES}
    gid = 0
    for si, s in enumerate(seasons):
        base = 1538352000 + si * 31536000
        for g in range(games_per):
            gid += 1
            for j in range(per_game):
                pos = rng.integers(0, 3)
                played = int(rng.random() < 0.8)
                mu = 1.0 + 0.8 * (pos == 0) + 0.2 * rng.random()
                sog = int(rng.poisson(mu)) * played
                goals = int(rng.binomial(sog, 0.11 if pos == 0 else 0.05)) if sog else 0
                cols["game_id"].append(gid); cols["team_id"].append(1 + (j % 2)); cols["player_id"].append(1000 + j + 20 * (g % 7)); cols["season"].append(s); cols["start"].append(base + g * 86400)
                cols["played"].append(played); cols["sog"].append(sog); cols["goals"].append(goals); cols["eff_sog"].append(max(sog, goals))
                for f in D.FEATURES:
                    cols[f].append({"POS_F": float(pos == 0), "POS_D": float(pos == 1), "POS_UNKNOWN": float(pos == 2)}.get(f, float(rng.normal()) if f not in D.BINARY else float(rng.integers(0, 2))))
                for f in GD.GOAL_HISTORY_FEATURES:
                    cols[f].append(float(rng.integers(0, 40)) if f != "GOALS_MEAN_APP10" else float(rng.random() * 0.4))
    tab = {k: np.array(v) for k, v in cols.items()}
    tab["N_CURRENT_SEASON_TEAM_GAMES_OBS"] = np.tile(np.arange(1, games_per + 1).repeat(per_game), len(seasons)).astype(float)
    tab["EFFSOG_SUM_CUM"] = np.maximum(tab["EFFSOG_SUM_CUM"], tab["GOALS_SUM_CUM"])
    return tab


# ------------------------------------------------------------------ labels / source rules (real frozen tables)
def test_goal_labels_join_one_to_one_and_nonparticipants_are_zero():
    games, rows = D.load_frozen()
    sc = GD.load_scoring()
    assert len(sc) == len(rows) == 397778 and all((r["game_id"], r["player_id"]) in sc for r in rows)
    assert all(sc[(r["game_id"], r["player_id"])][1] == r["sog"] for r in rows)                           # official shots == frozen sog on every row
    tab = {"game_id": np.array([rows[0]["game_id"], rows[0]["game_id"]]), "player_id": np.array([rows[0]["player_id"], 99999999]), "played": np.array([1, 0]), "sog": np.array([rows[0]["sog"], 0])}
    goals, mism = GD.attach_goals(tab, sc)
    assert goals[1] == 0 and mism == 0 and goals[0] == sc[(rows[0]["game_id"], rows[0]["player_id"])][0]


def test_goalie_goals_are_excluded_from_the_skater_head_and_shootout_goals_never_count():
    games, rows = D.load_frozen()
    assert not [r for r in rows if str(r["position"]).upper().startswith("G")]                             # skater universe only
    s0 = json.loads((OUT / "phase_scoring_s0_feasibility.json").read_text())["questions"]
    assert s0["S3"]["pbp_scorers_not_in_skater_table_classes"] == {"goalie_goal": 3}                        # goalie goals live outside the skater table
    tg = {r["gameId"]: r for r in FD.read_table("phase_team_game_t0_data/team_games_2023.jsonl.gz")}
    sc = {}
    for r in FD.read_table("phase_scoring_s0_data/skater_scoring_2023.jsonl.gz"):
        sc[(r["gameId"], r["teamAbbrev"])] = sc.get((r["gameId"], r["teamAbbrev"]), 0) + r["goals"]
    so = [g for g in tg.values() if g["lastPeriodType"] == "SO"]
    assert so
    ok = 0
    for g in so:
        h, a = sc.get((g["gameId"], g["homeAbbrev"]), 0), sc.get((g["gameId"], g["awayAbbrev"]), 0)
        win = g["homeScore"] > g["awayScore"]
        ok += (h == g["homeScore"] - (1 if win else 0)) and (a == g["awayScore"] - (0 if win else 1))
    assert ok == len(so)                                                                                    # skater goals exclude the shootout-deciding goal in every 2023 shootout game


def test_effective_sog_is_max_of_sog_and_goals_and_the_goal_target_is_never_altered():
    assert GD.effective_sog(np.array([0, 1, 5, 3]), np.array([1, 2, 0, 3])).tolist() == [1, 2, 5, 3]
    games, rows = D.load_frozen(); sc = GD.load_scoring()
    defect = [k for k, (g, s) in sc.items() if g > s]
    assert len(defect) == 30 and all(sc[k][0] == sc[k][1] + 1 for k in defect)                              # the S0 defect: exactly goals = sog + 1
    tab = {"game_id": np.array([k[0] for k in defect]), "player_id": np.array([k[1] for k in defect]), "played": np.ones(30, dtype=int), "sog": np.array([sc[k][1] for k in defect])}
    goals, _ = GD.attach_goals(tab, sc)
    assert (goals == np.array([sc[k][0] for k in defect])).all() and (goals > tab["sog"]).all()             # goal label kept exactly
    assert (GD.effective_sog(tab["sog"], goals) == goals).all()                                             # effective SOG repairs the denominator only


# ------------------------------------------------------------------ leakage / legality (shared-state rules, goal history)
def hist_world():
    games, rows = T1A.world()
    return games, rows, synth_scoring(rows)


def test_target_game_mutation_cannot_change_goal_history_or_shared_features():
    games, rows, sc = hist_world()
    tab, _ = T1A.build(games, rows)
    h0 = GD.history_features(tab, rows, sc)
    tgt = 2023020014
    sc2 = {k: ((v[0] + 3), v[1] + 9) if k[0] == tgt else v for k, v in sc.items()}
    rows2 = [({**r, "sog": r["sog"] + 9} if r["game_id"] == tgt else r) for r in rows]
    h1 = GD.history_features(tab, rows2, sc2)
    m = tab["game_id"] == tgt
    assert m.sum() > 0
    for k in GD.GOAL_HISTORY_FEATURES:
        assert np.array_equal(h0[k][m], h1[k][m], equal_nan=True), k
    T1A.test_target_row_mutation_leaves_every_feature_unchanged()


def test_goal_history_cutoff_legality_and_all_team_skill_vs_current_team_role():
    games, rows, sc = hist_world()
    tab, _ = T1A.build(games, rows)
    h = GD.history_features(tab, rows, sc)
    start = {r["game_id"]: D.epoch(r["game_start_utc"]) for r in rows}
    for i in range(len(tab["game_id"])):
        limit = int(tab["start"][i]) - D.CUTOFF_BACK_S
        prior = [r for r in rows if r["player_id"] == tab["player_id"][i] and start[r["game_id"]] <= limit]
        assert h["N_APPS_CUM"][i] == len(prior)
        assert h["GOALS_SUM_CUM"][i] == sum(sc[(r["game_id"], r["player_id"])][0] for r in prior)
        assert h["EFFSOG_SUM_CUM"][i] == sum(max(r["sog"], sc[(r["game_id"], r["player_id"])][0]) for r in prior)
        assert all(start[r["game_id"]] + 210 * 60 <= tab["start"][i] - 90 * 60 for r in prior)             # source_start + 210 min <= prediction cutoff (T90)
    x7 = (tab["player_id"] == 401) & (tab["game_id"] == 2023020010)
    assert x7.any() and h["N_APPS_CUM"][x7][0] > tab["N_ROLE_APPEARANCES_10"][x7][0]                       # traded player: all-team skill history exceeds current-team role history


# ------------------------------------------------------------------ exact compound PMFs
def test_g0_binomial_thinning_of_nb2_is_exactly_nb2_with_scaled_mean():
    mu, alpha, p = 3.1, 0.35, 0.09
    n = 1
    parts = list(GM.sog_pmf_chunks({"mu": np.array([mu]), "alpha": alpha}, n))
    sl, pm, sf = parts[0]
    K = pm.shape[1] - 1
    assert float(sf.max()) < 1e-10
    got = pm @ GM.binom_matrix(p, K)
    r = 1 / alpha
    want = stats.nbinom.pmf(np.arange(GM.GMAX + 1), r, r / (r + mu * p))
    assert np.allclose(got[0], want, atol=1e-10)


def test_g1_betabinomial_pmf_exactness_support_and_normalisation():
    a = np.array([0.7, 2.0, 11.0]); b = np.array([5.0, 9.0, 80.0]); K = 30
    bb = GM.betabinom_pmf_rows(a, b, K)
    for i in range(3):
        for s in (0, 1, 7, 30):
            want = stats.betabinom.pmf(np.arange(GM.GMAX + 1), s, a[i], b[i])
            assert np.allclose(bb[i, s], want, atol=1e-12)
            assert (bb[i, s, s + 1:] == 0).all()                                                           # conditional goals never exceed the sampled SOG
            assert abs(bb[i, s].sum() - 1.0) < 1e-9 or s > GM.GMAX
    sb = GM.binom_matrix(0.1, K)
    assert all(sb[s, s + 1:].sum() == 0 for s in range(K + 1))


def test_mixture_over_the_full_sog_pmf_sums_to_one_and_has_no_goals_above_sog_mass():
    mu = np.array([0.0001, 1.2, 4.5, 6.4]); params = {"mu": mu, "alpha": 0.15}   # realistic B2 range (fitted max mu 6.4, alpha 0.15)
    a = np.array([1.0, 3.0, 12.0, 30.0]); b = np.array([8.0, 30.0, 100.0, 170.0])
    pmf, diag = GM.compound_betabinom_pmf(params, a, b, 4)
    c = GM.coherence_report(pmf)
    assert c["min_probability"] >= 0 and c["finite"] and c["max_missing_tail_mass"] < 1e-9 and abs(c["max_row_sum"] - 1) < 1e-9 and diag["max_sog_survival_at_K"] < 1e-10
    assert abs(pmf[0, 0] - 1) < 1e-3                                                                        # almost no shots -> almost surely 0 goals
    # brute-force reference for one row
    s = np.arange(0, 400)
    ps = stats.nbinom.pmf(s, 1 / 0.15, (1 / 0.15) / (1 / 0.15 + mu[2]))
    ref = np.array([sum(ps[k] * stats.betabinom.pmf(y, k, a[2], b[2]) for k in s) for y in range(GM.GMAX + 1)])
    assert np.allclose(pmf[2], ref, atol=1e-8)
    g0 = GM.G0(); g0.p = {"F": 0.12, "D": 0.05, "U": 0.08}; g0.p_pool = 0.1; g0.sog = {"nb": {"alpha": 0.15}}
    # G0 through its own predict path on a minimal table with a stub SOG predictor
    tab = {"POS_F": np.array([1.0, 0, 0, 1.0]), "POS_D": np.array([0, 1.0, 0, 0]), "POS_UNKNOWN": np.array([0, 0, 1.0, 0])}
    orig = GM.sog_params
    try:
        GM.sog_params = lambda arch, t: (params, mu)
        g0.sog = {"nb": {"alpha": 0.15}, "fit": {}}
        g0.__class__.predict.__globals__["sog_params"] = GM.sog_params
        tab["game_id"] = np.arange(4)
        out = g0.predict(tab)
    finally:
        GM.sog_params = orig
    assert GM.coherence_report(out["pmf"])["max_missing_tail_mass"] < 1e-9


# ------------------------------------------------------------------ training-only estimation
def test_hyperprior_mle_recovers_truth_reports_diagnostics_and_uses_training_rows_only():
    rng = np.random.default_rng(3)
    p = rng.beta(2.0, 18.0, size=3000); n = rng.integers(20, 400, size=3000); g = rng.binomial(n, p)
    h = GM.fit_hyperprior(g, n)
    assert h["converged"] and not h["any_boundary_hit"] and abs(h["mean"] - 0.1) < 0.01 and 12 < h["concentration"] < 30
    assert set(h) >= {"a", "b", "converged", "n_iter", "neg_loglik", "boundary_hits", "any_boundary_hit", "n_players"}
    tab = synth_tab(1)
    tr = R.sel(tab, [2018, 2019])
    stub = {"nb": {"alpha": 0.3}, "fit": {}}
    base = GM.G1().fit(tr, stub).artifact()["hyperprior"]
    bad = {k: v.copy() for k, v in tab.items()}
    later = np.isin(bad["season"], [2020, 2021, 2022, 2023, 2024, 2025])
    bad["goals"][later] = 17; bad["eff_sog"][later] = 17; bad["sog"][later] = 0                            # corrupt EVERY non-training row
    tr2 = R.sel(bad, [2018, 2019])
    assert GM.G1().fit(tr2, stub).artifact()["hyperprior"] == base
    g0a = GM.G0().fit(tr, stub).artifact()["conversion_by_class"]; assert GM.G0().fit(tr2, stub).artifact()["conversion_by_class"] == g0a


# ------------------------------------------------------------------ decision rules / discipline
def test_promotion_requires_practical_or_statistical_gain_and_every_guard():
    ok = dict(rel_crps_improvement=0.006, boot_upper95=0.001, rel_nll=0.0, ks_delta=0.0, fd_ok=True, phase_ok=True)
    assert R.promotion_decision(**ok)[1] is True
    assert R.promotion_decision(**{**ok, "rel_crps_improvement": 0.004})[1] is False                         # below 0.5% and not statistically supported -> simpler wins
    assert R.promotion_decision(**{**ok, "rel_crps_improvement": 0.001, "boot_upper95": -1e-6})[1] is True    # statistical support alone suffices
    assert R.promotion_decision(**{**ok, "rel_nll": 0.006})[1] is False
    assert R.promotion_decision(**{**ok, "ks_delta": 0.021})[1] is False
    assert R.promotion_decision(**{**ok, "fd_ok": False})[1] is False and R.promotion_decision(**{**ok, "phase_ok": False})[1] is False
    inc, hist = R.select_architecture(["G0", "G1", "G2"], lambda ch, inc: False)
    assert inc == "G0" and [h["promoted"] for h in hist] == [False, False]
    inc, hist = R.select_architecture(["G0", "G1", "G2"], lambda ch, inc: ch == "G1")
    assert inc == "G1" and hist[1]["incumbent"] == "G1"                                                    # G2 faces the CURRENT incumbent
    t = json.loads((OUT / "phase_goals_g1_protocol.json").read_text())["selection_rule"]["promotion_criteria"]
    assert R.THRESH["P1_rel_crps"] == 0.005 and R.THRESH["P2_nll"] == 0.005 and R.THRESH["P3_ks"] == 0.02 and R.THRESH["P4_5_slice"] == 0.05 and "0.5%" in t["P1_practical_or_statistical"]


def test_development_uses_only_2018_2023_and_is_unaffected_by_2024_2025_rows():
    for fid, tr, va in R.FOLDS:
        assert all(2018 <= s <= 2022 for s in tr) and 2020 <= va <= 2023 and va > max(tr)
    assert [f[2] for f in R.FOLDS] == [2020, 2021, 2022, 2023]
    tab = synth_tab(5)
    a = R.run_dev(tab, log=lambda m: None)
    bad = {k: v.copy() for k, v in tab.items()}
    ex = np.isin(bad["season"], [2024, 2025]); bad["goals"][ex] = 19; bad["sog"][ex] = 33; bad["eff_sog"][ex] = 33
    b = R.run_dev(bad, log=lambda m: None)
    for k in ("mean_over_folds", "selected", "strongest_development_comparator", "selection_history"):
        assert a[k] == b[k], k
    assert a["selected"] in R.ARCHS and a["strongest_development_comparator"] != a["selected"]
    for fid in a["folds"]:
        for m in R.ARCHS:
            c = a["folds"][fid]["models"][m]["coherence"]
            assert c["min_probability"] >= 0 and c["max_missing_tail_mass"] < 1e-6


def test_confirmations_have_no_rescue_path_2024_once_and_2025_only_after_a_pass():
    import tempfile
    src = (REPO / "nhl_goals_g1_run.py").read_text()
    assert "never scored" in src and "already scored once" in src and "must be committed before 2024 is scored" in src and "2024 confirmation must be committed before 2025 is scored" in src
    dev = {"selected": "G0", "strongest_development_comparator": "G1"}
    tab = synth_tab(7)
    r = R.run_confirmation(tab, dev, list(range(2018, 2024)), 2024, "t")                                     # architecture comes ONLY from the dev dict
    assert r["selected"] == "G0" and r["comparator"] == "G1" and set(r["summaries"]) == {"G0", "G1"} and r["target_scored"] == 2024 and "all_pass" in r["gate"]
    saved = (R.OUT, R.committed_and_clean, R.load_goals_table, sys.argv)
    with tempfile.TemporaryDirectory() as t:
        R.OUT = Path(t); R.committed_and_clean = lambda p: True; R.load_goals_table = lambda w: (tab, {})
        try:
            (Path(t) / "phase_goals_g1_2024_confirmation.json").write_text(json.dumps({"passed": False}))
            sys.argv = ["x", "confirm2025", "--work", t]
            try:
                R.main(); raise SystemExit("2025 was allowed after a failed 2024")
            except AssertionError as e:
                assert "2025 is never scored" in str(e)
            (Path(t) / "phase_goals_g1_dev_results.json").write_text("{}"); (Path(t) / "phase_goals_g1_selected_architecture.json").write_text("{}")
            sys.argv = ["x", "confirm2024", "--work", t]
            try:
                R.main(); raise SystemExit("2024 re-scored")
            except AssertionError as e:
                assert "already scored once" in str(e)
        finally:
            R.OUT, R.committed_and_clean, R.load_goals_table, sys.argv = saved


# ------------------------------------------------------------------ metrics / determinism / no sportsbook
def test_deterministic_pit_and_calibration_metric_correctness():
    v1 = M.pit_v("G0_2020", [1, 2, 3], [4, 5, 6]); v2 = M.pit_v("G0_2020", [1, 2, 3], [4, 5, 6])
    assert (v1 == v2).all() and ((v1 >= 0) & (v1 < 1)).all() and not (v1 == M.pit_v("G1_2020", [1, 2, 3], [4, 5, 6])).all()
    pmf = np.array([[0.7, 0.2, 0.1, 0, 0, 0], [0.5, 0.3, 0.2, 0, 0, 0]])
    y = np.array([0, 2])
    s, rows = M.summarize("pmf", {"pmf": pmf}, y, "t", np.array([1, 2]), np.array([1, 2]), R.GM.pmf_mean(pmf))
    assert abs(rows["crps"][0] - M.crps_point(np.cumsum(pmf[0]), 0)) < 1e-12 and abs(rows["crps"][1] - M.crps_point(np.cumsum(pmf[1]), 2)) < 1e-12
    assert abs(rows["nll"][0] + np.log(0.7)) < 1e-12 and abs(rows["nll"][1] + np.log(0.2)) < 1e-12
    pge = R.rows_p_ge(pmf)
    assert np.allclose(pge[:, 0], [0.3, 0.5]) and np.allclose(pge[:, 1], [0.1, 0.2]) and np.allclose(pge[:, 2], [0.0, 0.0])
    td = M.threshold_diag(pge, y)
    assert abs(td["P(SOG>=1)"]["brier"] - np.mean((np.array([0.3, 0.5]) - np.array([0, 1])) ** 2)) < 1e-12
    sc = R.score("t", {"pmf": pmf}, {"goals": y, "game_id": np.array([1, 2]), "player_id": np.array([1, 2]), "POS_F": np.ones(2), "POS_D": np.zeros(2), "POS_UNKNOWN": np.zeros(2), "N_CURRENT_SEASON_TEAM_GAMES_OBS": np.array([5.0, 50.0]), "sog": y, "eff_sog": y, "played": np.ones(2, dtype=int)})[0]
    assert abs(sc["zero_goal_calibration"]["mean_pred_P0"] - 0.6) < 1e-12 and abs(sc["zero_goal_calibration"]["observed_zero_rate"] - 0.5) < 1e-12
    assert list(sc["threshold_diagnostics"]) == ["P(goals>=1)", "P(goals>=2)", "P(goals>=3)"]


def test_no_sportsbook_inputs_in_features_tables_protocol_or_code_paths():
    assert not [f for f in GD.G2_FEATURES if any(w in f.lower() for w in SPORTSBOOK)]
    tab = synth_tab(2)
    assert not [k for k in tab if any(w in k.lower() for w in SPORTSBOOK)]
    proto = json.loads((OUT / "phase_goals_g1_protocol.json").read_text())
    assert any("sportsbook" in x for x in proto["inputs_forbidden"]) and "no sportsbook inputs" in proto["prohibitions"]
    assert set(GD.G2_FEATURES) == set(D.FEATURES) | set(GD.GOAL_HISTORY_FEATURES) and len(GD.G2_FEATURES) == 37
    code = " ".join((REPO / f).read_text().lower() for f in ("nhl_goals_g1_data.py", "nhl_goals_g1_models.py"))
    assert "import requests" not in code and "urllib" not in code                                              # no network inputs of any kind


def test_protocol_registered_before_any_goals_code_or_results_and_matches_constants():
    import hashlib
    p = json.loads((OUT / "phase_goals_g1_protocol.json").read_text())
    body = {k: v for k, v in p.items() if k != "protocol_body_sha256"}
    assert hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == p["protocol_body_sha256"]
    assert p["status"] == "PREREGISTERED_BEFORE_ANY_GOALS_MODEL_PERFORMANCE"
    assert p["exposure_labels"]["2024"] == p["exposure_labels"]["2025"] == "DESCRIPTIVELY_EXPOSED_NOT_MODEL_SCORED" and "pristine" in p["exposure_labels"]["meaning"]
    assert p["development_folds"]["D4"] == {"train": [2018, 2019, 2020, 2021, 2022], "validate": 2023}
    assert p["distribution_coherence"]["support"].startswith("goals are integers in 0..20") and GD.GMAX == 20
    c3 = p["confirmation_2024"]["gate_vs_strongest_development_comparator"]["C3_calibration_guards"]
    assert "0.05" in c3 and "0.02" in c3 and "0.01" in c3 and R.THRESH["C3_bias"] == 0.05 and R.THRESH["C3_ece"] == 0.02 and R.THRESH["C3_zero"] == 0.01 and R.THRESH["C4_slice"] == 0.10
    first = lambda path: subprocess.run(["git", "log", "--format=%H", "--diff-filter=A", "--", path], cwd=REPO, capture_output=True, text=True).stdout.split()[-1]
    proto_c = first("nhl_models/nhl_outcome_engine/phase_goals_g1_protocol.json")
    dev = subprocess.run(["git", "log", "--format=%H", "--diff-filter=A", "--", "nhl_models/nhl_outcome_engine/phase_goals_g1_dev_results.json"], cwd=REPO, capture_output=True, text=True).stdout.split()
    if dev:
        for f in ("nhl_goals_g1_data.py", "nhl_goals_g1_models.py", "nhl_goals_g1_run.py"):
            assert subprocess.run(["git", "merge-base", "--is-ancestor", first(f), dev[-1]], cwd=REPO).returncode == 0
        assert subprocess.run(["git", "merge-base", "--is-ancestor", proto_c, dev[-1]], cwd=REPO).returncode == 0 and proto_c != dev[-1]


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
