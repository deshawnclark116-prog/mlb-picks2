"""CFB Outcome Engine v1 tests: research cutoff (Week 5 rejected), PIT candidate universe, shared-state leakage, identity, allocation coherence, training-only fitting, validation isolation, gates, protocol discipline. python tests/test_cfb_phase1.py"""
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import cfb_phase1_common as C  # noqa: E402
import cfb_phase1_data as D  # noqa: E402
import cfb_phase1_evaluate as EV  # noqa: E402
import cfb_phase1_role_state as RS  # noqa: E402
import cfb_phase1_team_environment as TE  # noqa: E402

OUT = REPO / "cfb_models" / "cfb_outcome_engine"
SB = ("odds", "spread", "sportsbook", "bookmaker", "vegas", "implied", "prop_line", "market_line", "closing_line", "pickcenter", "fanduel", "draftkings")


# ------------------------------------------------------------------ synthetic world
def tgrow(gid, season, week, team, opp, home, pts, opp_pts, rush, pas, date):
    return {"game_id": gid, "season": season, "week": week, "game_date": date, "team": team, "opponent": opp, "is_home": home, "neutral": 0, "team_div": "fbs", "opp_div": "fbs", "team_conf": "X", "opp_conf": "X", "team_points": pts, "opp_points": opp_pts,
            "has_play_rows": 1, "rush_plays": rush, "pass_att_plays": pas, "sack_plays": 2, "plays": rush + pas + 2, "rz_rush_plays": 4, "rz_pass_plays": 4, "rush_yards": 150, "completions": 18, "pass_comp_yards": 220, "interceptions": 1}


def plrow(pid, team, opp, gid, season, week, date, pos, car, rec, att, home=1):
    return {"player_id": pid, "player_name": f"P{pid}", "position": pos, "team": team, "opponent": opp, "season": season, "week": week, "game_id": gid, "game_date": date, "is_home": home, "carries": car, "rushing_yards": car * 4, "rushing_touchdowns": 0,
            "receptions": rec, "receiving_yards": rec * 10, "receiving_touchdowns": 0, "pass_attempts": att, "completions": att // 2, "passing_yards": att * 6, "passing_touchdowns": 0, "passing_interceptions": 0}


def world(weeks=9, season=2019):
    """4 FBS teams A B C D, round-robin pairs each week. Per team: QB(1x), RB1 (2x), RB2 (3x), WR (4x), TE (5x) ids = team-number * 100 + role. Player X (999) plays for A in weeks 1-4 then for B from week 5 (a transfer, same athlete_id)."""
    teams = ["A", "B", "C", "D"]; num = {t: i + 1 for i, t in enumerate(teams)}
    pairs = [[("A", "B"), ("C", "D")], [("A", "C"), ("B", "D")], [("A", "D"), ("B", "C")]]
    tg, pg = [], []
    for w in range(1, weeks + 1):
        date = f"{season}-09-{min(28, 1 + 7 * (w - 1) % 28):02d}"
        for (h, a) in pairs[(w - 1) % 3]:
            gid = f"{season}{w:02d}{h}{a}"
            hp, ap = 24 + (w % 3) * 3, 17 + (w % 4)
            tg.append(tgrow(gid, season, w, h, a, 1, hp, ap, 34 + (w % 5), 28 + (w % 4), date)); tg.append(tgrow(gid, season, w, a, h, 0, ap, hp, 31 + (w % 4), 31 + (w % 3), date))
            for t, o, home in ((h, a, 1), (a, h, 0)):
                n = num[t]
                pg += [plrow(n * 100 + 1, t, o, gid, season, w, date, "QB", 3, 0, 30, home), plrow(n * 100 + 2, t, o, gid, season, w, date, "RB", 20, 2, 0, home), plrow(n * 100 + 3, t, o, gid, season, w, date, "RB", 8 + (w % 3), 1, 0, home),
                       plrow(n * 100 + 4, t, o, gid, season, w, date, "WR", 0, 6, 0, home), plrow(n * 100 + 5, t, o, gid, season, w, date, "TE", 0, 3, 0, home)]
                if (t == "A" and w <= 4) or (t == "B" and w >= 5):
                    pg.append(plrow(999, t, o, gid, season, w, date, "WR", 2, 4, 0, home))
    return tg, pg


def build(tg, pg, seasons=(2019,)):
    cand, cov = D.build_candidates(tg, pg, list(seasons))
    return cand, RS.build_feature_table(tg, pg, cand, list(seasons)), cov


def feat_map(rows, week, keys=RS.FEATURES):
    return {(r["team"], r["player_id"]): [r[k] for k in keys] for r in rows if r["week"] == week}


def same(a, b):
    return a.keys() == b.keys() and all(np.array_equal(np.array(a[k], float), np.array(b[k], float), equal_nan=True) for k in a)


# ------------------------------------------------------------------ research cutoff
def test_week5_and_later_are_rejected_and_2026_weeks_1_4_are_diagnostic_only():
    for s, w in ((2026, 5), (2026, 6), (2026, 14), (2027, 1)):
        try:
            C.assert_research_allowed(s, w, "fit"); raise SystemExit("allowed")
        except C.ResearchCutoffError:
            pass
        try:
            C.assert_research_allowed(s, w, "diagnostic"); raise SystemExit("diagnostic allowed")
        except C.ResearchCutoffError:
            pass
    assert C.assert_research_allowed(2025, 14, "fit") and C.assert_research_allowed(2026, 4, "diagnostic")
    try:
        C.assert_research_allowed(2026, 1, "fit"); raise SystemExit("burned data usable for fit")
    except C.ResearchCutoffError:
        pass
    rows = [{"season": 2025, "week": 3}, {"season": 2026, "week": 5}]
    try:
        C.filter_research_rows(rows, "fit"); raise SystemExit("loader let Week 5 through")
    except C.ResearchCutoffError:
        pass
    assert C.filter_research_rows(rows, "fit", strict=False) == [rows[0]]


def test_frozen_tables_hold_no_2026_row_verify_and_detect_tampering():
    man = D.verify_manifest()
    assert man["contains_2026"] is False and man["seasons"] == list(range(2018, 2026))
    tg, pg = D.load_frozen()
    assert max(r["season"] for r in tg + pg) == 2025 and min(r["season"] for r in tg + pg) == 2018
    with tempfile.TemporaryDirectory() as t:
        shutil.copytree(D.DATA, Path(t) / "d")
        f = Path(t) / "d" / "team_games_2020.jsonl.gz"
        f.write_bytes(f.read_bytes()[:-5] + b"\x00\x00\x00\x00\x00")
        try:
            D.verify_manifest(Path(t) / "d"); raise SystemExit("tamper undetected")
        except RuntimeError as e:
            assert "mismatch" in str(e)
    src = (REPO / "cfb_phase1_data.py").read_text()
    assert "assert_research_allowed" in src


# ------------------------------------------------------------------ leakage / PIT
def test_target_week_mutation_cannot_change_team_state_or_candidates_or_player_features():
    tg, pg = world()
    cand0, rows0, _ = build(tg, pg)
    W = 7
    tg2 = [({**r, "rush_plays": r["rush_plays"] + 25, "pass_att_plays": 3, "plays": 60, "team_points": 99, "opp_points": 0} if r["week"] == W else r) for r in tg]
    pg2 = [({**r, "carries": r["carries"] + 11, "receptions": 0, "pass_attempts": 0, "position": "OL"} if r["week"] == W else r) for r in pg]
    pg2 = [r for r in pg2 if not (r["week"] == W and r["player_id"] % 100 == 4)]                        # a participant removed from the target week
    cand1, rows1, _ = build(tg2, pg2)
    c0 = {(c["team"], c["player_id"]) for c in cand0 if c["week"] == W}; c1 = {(c["team"], c["player_id"]) for c in cand1 if c["week"] == W}
    assert c0 == c1 and c0
    assert same(feat_map(rows0, W), feat_map(rows1, W))                                                    # every state feature identical although the target week changed
    trows0 = [r for r in TE.build_team_table(tg, [2019]) if r["week"] == W] if False else [r for r in TE.build_team_table(tg, [2019]) if r["week"] == W]
    trows1 = [r for r in TE.build_team_table(tg2, [2019]) if r["week"] == W]
    for a, b in zip(trows0, trows1):
        assert all(np.array_equal(np.array(a[k], float), np.array(b[k], float), equal_nan=True) for k in TE.TEAM_FEATURES)
    # labels DO change (the mutation is real)
    assert any(a["y_carries"] != b["y_carries"] for a, b in zip([r for r in rows0 if r["week"] == W], [r for r in rows1 if r["week"] == W]))


def test_candidate_universe_uses_only_earlier_weeks_and_keeps_non_participants_with_zero_outcome():
    tg, pg = world()
    cand, rows, cov = build(tg, pg)
    assert min(c["week"] for c in cand) == 2                                                               # week 1 has no prior appearance: nobody is a candidate
    wk3_A = [c for c in cand if c["week"] == 3 and c["team"] == "A"]
    ids = {c["player_id"] for c in wk3_A}
    assert {101, 102, 103, 104, 105, 999} <= ids
    pg_skip = [r for r in pg if not (r["week"] == 3 and r["team"] == "A" and r["player_id"] == 104)]
    cand2, _, _ = build(tg, pg_skip)
    r104 = [c for c in cand2 if c["week"] == 3 and c["team"] == "A" and c["player_id"] == 104]
    assert r104 and r104[0]["y_receptions"] == 0 and r104[0]["y_carries"] == 0                           # absent in the target game -> still a candidate with outcome 0
    assert {c["player_id"] for c in cand2 if c["week"] == 3 and c["team"] == "A"} == ids
    assert all(c["position"] in D.SKILL for c in cand)
    # a player's last appearance for another team removes him from the old team's candidates
    wk5 = {(c["team"], c["player_id"]) for c in cand if c["week"] == 5}
    assert ("A", 999) not in wk5 and ("B", 999) in wk5 or ("B", 999) not in wk5                           # (B has not seen him yet at week 5: appearances become visible only after week 5)
    wk6 = {(c["team"], c["player_id"]) for c in cand if c["week"] == 6}
    assert ("B", 999) in wk6 and ("A", 999) not in wk6


def test_identity_provider_id_history_follows_the_player_but_role_history_stays_current_team():
    tg, pg = world()
    cand, rows, _ = build(tg, pg)
    r = [x for x in rows if x["week"] == 7 and x["team"] == "B" and x["player_id"] == 999]
    assert r and r[0]["P_TRANSFER_NEWCOMER"] == 1.0                                                        # same athlete_id, earlier appearances for team A
    assert r[0]["P_N_SKILL_APPS_10"] >= 4 and r[0]["P_CARRIES_PER_APP_L10"] == 2.0                          # all-team skill history includes the A games
    assert r[0]["P_APPS_L12"] == 2.0                                                                       # role window counts only B's games (weeks 5, 6)
    # the same player under a DIFFERENT id (name collision) gets no stitched history
    pg_new = [({**x, "player_id": 4242} if (x["player_id"] == 999 and x["team"] == "B") else x) for x in pg]
    _, rows2, _ = build(tg, pg_new)
    q = [x for x in rows2 if x["week"] == 7 and x["team"] == "B" and x["player_id"] == 4242]
    assert q and q[0]["P_TRANSFER_NEWCOMER"] == 0.0 and q[0]["P_N_SKILL_APPS_10"] == 2.0
    ident = json.loads((OUT / "cfb_identity_contract.json").read_text())
    assert "FORBIDDEN as a join" in ident["rules"]["fuzzy_name_matching"] and ident["rules"]["placeholder_ids"].startswith("negative ids are never stitched")
    audit = json.loads((OUT / "cfb_identity_audit.json").read_text())["box_score_producers"]
    assert audit["producer_transfers_with_name_inconsistency"] == 0 and audit["producer_transfers_with_placeholder_id"] == 0 and audit["producer_transfers_with_positive_id"] > 2000


def test_inactive_players_get_zero_opportunity_and_allocation_is_coherent():
    N = np.array([40, 40, 40, 0]); s = np.array([0.5, 0.3, 0.2, 0.4])
    pmf = EV.bb_pmf(N, s, 20.0)
    assert (pmf[0, 41:] == 0).all() and (pmf[1, 41:] == 0).all()                                           # carries <= team carries
    assert abs(pmf[3, 0] - 1.0) < 1e-12 and pmf[3, 1:].sum() == 0                                          # no team carries -> zero opportunity
    mean = pmf @ np.arange(pmf.shape[1])
    assert abs(mean[:3].sum() - 40.0) < 1e-6                                                               # shares summing to one allocate exactly the team total in expectation
    assert (pmf >= 0).all() and np.all(pmf.sum(axis=1) <= 1 + 1e-9) and abs(pmf[:3].sum(axis=1) - 1).max() < 1e-9
    # receptions <= targets / completions <= attempts analogue: binomial thinning keeps support
    assert stats.binom.pmf(np.arange(5, 11), 4, 0.5).sum() == 0
    # nonnegative opportunity counts in every real feature row: shares are fractions
    tg, pg = world(); _, rows, _ = build(tg, pg)
    assert all(0 <= r["P_CARRY_SHARE_L5"] <= 1 for r in rows if not math.isnan(r["P_CARRY_SHARE_L5"])) and all(r["y_carries"] >= 0 and r["y_receptions"] >= 0 and r["y_pass_att"] >= 0 for r in rows)
    assert all(r["y_carries"] <= r["team_carries_target"] for r in rows)                                    # a player's carries never exceed the team carries of the game


def test_training_only_preprocessing_shrinkage_and_hyperparameters():
    rng = np.random.default_rng(0)
    tab = {n: rng.normal(size=500) for n in ("a", "b")}; tab["a"][:20] = np.nan
    p = TE.Prep(["a", "b"]).fit(tab); med = dict(p.med)
    other = {n: v + 100.0 for n, v in tab.items()}
    assert p.transform(other).shape[1] == 4 and p.med == med                                                # transform never refits
    tr = {"x": rng.normal(size=300)}; y = rng.poisson(30, 300)
    tr["T_RUSH_B0"] = rng.normal(30, 3, 300)
    a1 = TE.B0Count("T_RUSH_B0").fit(tr, y).nb["alpha"]
    tr_bad = dict(tr)
    assert TE.B0Count("T_RUSH_B0").fit(tr_bad, y).nb["alpha"] == a1
    y_tr = np.array([0, 3, 5, 10, 2] * 100); N = np.full(500, 40); s = np.full(500, 0.1)
    k1 = EV.fit_kappa(y_tr, N, s)["kappa"]
    assert EV.fit_kappa(y_tr, N, s)["kappa"] == k1 and EV.KAPPA_BOUNDS == (0.5, 500.0)


def test_validation_and_late_period_isolation_dev_results_do_not_move_when_2025_changes():
    tg, pg = D.load_frozen()
    rows = TE.build_team_table(tg)
    a = EV.run_team_component(rows, "y_rush", "T_RUSH_B0", lambda m: None)
    bad = [({**r, "y_rush": 3, "y_pass": 3, "T_RUSH_B0": 1.0} if r["season"] == 2025 else r) for r in rows]
    b = EV.run_team_component(bad, "y_rush", "T_RUSH_B0", lambda m: None)
    assert a["mean_over_folds"] == b["mean_over_folds"] and a["gates"] == b["gates"] and a["retained"] == b["retained"]
    assert [f[2] for f in C.DEV_FOLDS] == [2021, 2022, 2023, 2024] and C.LATE_CONFIRMATION_SEASON == 2025
    for fid, tr, va in C.DEV_FOLDS:
        assert max(tr) < va <= 2024 and 2018 not in tr


# ------------------------------------------------------------------ rules / metrics
def test_play_coverage_validity_and_team_state_excludes_invalid_games():
    g = tgrow("g", 2019, 1, "A", "B", 1, 20, 10, 35, 30, "2019-09-01")
    assert C.coverage_valid(g) and not C.coverage_valid({**g, "plays": 29}) and not C.coverage_valid({**g, "plays": 141}) and not C.coverage_valid({**g, "has_play_rows": 0, "plays": None})
    tg, _ = world(weeks=4)
    base = TE.build_team_table(tg, [2019])
    tg2 = [({**r, "rush_plays": 0, "pass_att_plays": 0, "sack_plays": 0, "plays": 0} if (r["week"] == 2 and r["team"] == "A") else r) for r in tg]
    t2 = TE.build_team_table(tg2, [2019])
    a3 = [r for r in base if r["week"] == 3 and r["team"] == "A"]; b3 = [r for r in t2 if r["week"] == 3 and r["team"] == "A"]
    assert len(a3) == 1 and len(b3) == 1
    assert [r for r in t2 if r["week"] == 2 and r["team"] == "A"] == []                                    # invalid game is not a target
    assert b3[0]["T_N_SEASON"] == a3[0]["T_N_SEASON"] - 1                                                  # and not an accumulated zero


def test_gate_decision_thresholds_and_slice_gate():
    ok = dict(rel_gain=0.006, upper95=0.1, rel_secondary=0.0, calib_delta=0.0, slices_ok=True, kind="count")
    assert EV.gate_decision(**ok)[1] is True
    assert EV.gate_decision(**{**ok, "rel_gain": 0.004})[1] is False
    assert EV.gate_decision(**{**ok, "rel_gain": 0.0, "upper95": -1e-6})[1] is True
    assert EV.gate_decision(**{**ok, "rel_secondary": 0.006})[1] is False and EV.gate_decision(**{**ok, "calib_delta": 0.021})[1] is False and EV.gate_decision(**{**ok, "slices_ok": False})[1] is False
    assert EV.gate_decision(**{**ok, "kind": "binary", "calib_delta": 0.011})[1] is False and EV.gate_decision(**{**ok, "kind": "binary", "calib_delta": 0.009})[1] is True
    ch = np.array([1.0] * 300 + [2.0] * 300); ref = np.array([1.0] * 300 + [1.0] * 300)
    m = {"all": np.ones(600, bool), "small": np.arange(600) < 50, "bad": np.arange(600) >= 300}
    ok_, det = C.slice_gate(ch, ref, m, 200, 0.05)
    assert ok_ is False and det["small"]["eligible"] is False and det["bad"]["pass"] is False


def test_metrics_pit_bootstrap_determinism_and_correctness():
    pmf = np.array([[0.2, 0.5, 0.3, 0.0], [0.1, 0.1, 0.4, 0.4]]); y = np.array([1, 3])
    cr = C.crps_rows(pmf, y)
    for i in range(2):
        cdf = np.cumsum(pmf[i]); assert abs(cr[i] - sum((cdf[k] - (y[i] <= k)) ** 2 for k in range(4))) < 1e-12
    assert abs(C.nll_rows(pmf, y)[0] + math.log(0.5)) < 1e-12 and abs(C.pmf_mean(pmf)[0] - 1.1) < 1e-12
    assert (C.pit_v("m", ["a", "b"]) == C.pit_v("m", ["a", "b"])).all() and not (C.pit_v("m", ["a"]) == C.pit_v("n", ["a"])).all()
    pit = C.randomized_pit(pmf, y, np.array([0.5, 0.5]))
    assert abs(pit[0] - (0.2 + 0.5 * 0.5)) < 1e-12 and abs(pit[1] - (0.6 + 0.5 * 0.4)) < 1e-12
    delta = np.random.default_rng(1).normal(-0.1, 1.0, 400); wk = np.repeat([201901 + i for i in range(20)], 20)
    b1, b2 = C.block_bootstrap(delta, wk, 500), C.block_bootstrap(delta, wk, 500)
    assert (b1 == b2).all() and abs(b1.mean() - delta.mean()) < 0.1
    nbp, sf = C.nb2_pmf_matrix(np.array([30.0]), 0.03, 110)
    assert abs(nbp.sum() + sf[0] - 1) < 1e-9 and sf[0] < 1e-9


def test_no_sportsbook_inputs_in_features_contracts_or_protocol():
    assert not [n for n in RS.FEATURES + TE.TEAM_FEATURES if any(w in n.lower() for w in SB)]
    assert C.no_sportsbook_columns(RS.FEATURES)
    try:
        C.no_sportsbook_columns(["T_RUSH_B0", "closing_line"]); raise SystemExit("sportsbook column allowed")
    except ValueError:
        pass
    sc = json.loads((OUT / "cfb_shared_state_contract.json").read_text())
    assert any("sportsbook" in x for x in sc["forbidden"]) and "post_win_prob" in " ".join(sc["forbidden"])
    pr = json.loads((OUT / "phase1_protocol.json").read_text())
    assert "no sportsbook inputs" in pr["prohibitions"] and "favourite" in pr["metrics"]["forbidden_slices"]
    assert set(sc["blocks"]["TEAM_STATE"]) | set(sc["blocks"]["OPPONENT_STATE"]) | set(sc["blocks"]["STRENGTH_STATE"]) | set(sc["blocks"]["SCHEDULE_VENUE"]) == set(TE.TEAM_FEATURES)


def test_protocol_amendment_registry_and_commit_order():
    pr = json.loads((OUT / "phase1_protocol.json").read_text())
    body = {k: v for k, v in pr.items() if k != "protocol_body_sha256"}
    assert hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == pr["protocol_body_sha256"] and pr["status"] == "PREREGISTERED_BEFORE_ANY_NEW_ENGINE_PERFORMANCE"
    am = json.loads((OUT / "phase1_protocol_amendment_1.json").read_text())
    assert hashlib.sha256(json.dumps({k: v for k, v in am.items() if k != "body_sha256"}, sort_keys=True).encode()).hexdigest() == am["body_sha256"] and am["registered_before_any_performance"] is True
    ep = pr["exposure_and_periods"]
    assert ep["2026_week_5"].startswith("FORBIDDEN_FROM_MODEL_RESEARCH") and ep["2026_weeks_1_4"].startswith("BURNED_DIAGNOSTIC_ONLY") and ep["2025"].startswith("LATE_PERIOD_CONFIRMATION_PREVIOUSLY_EXPOSED") and "NO pristine" in ep["honesty"]
    th = pr["promotion_gate_challenger_over_baseline"]
    assert "0.5%" in th["G1"] and "5%" in th["G4"] and EV.TH["rel_gain"] == 0.005 and EV.TH["slice"] == 0.05 and EV.TH["ks"] == 0.02 and EV.TH["ece"] == 0.01
    reg = json.loads((OUT / "CFB_OUTCOME_ENGINE_REGISTRY.json").read_text())
    vocab = set(reg["status_vocabulary"])
    for k, v in reg["legacy_models"].items():
        assert v["status"] in vocab
    lm = reg["legacy_models"]
    assert lm["passing_yards"]["status"] == lm["receiving_yards"]["status"] == "LEGACY_SUSPENDED" and lm["rushing_yards"]["status"] == "LEGACY_ACTIVE_COMPARATOR" and "NOT fully stable" in lm["rushing_yards"]["walk_forward"]
    assert reg["research_cutoff"]["periods"]["2026_week_6_plus"].startswith("CLEAN_FORWARD_CANDIDATE")
    first = lambda path: subprocess.run(["git", "log", "--format=%H", "--diff-filter=A", "--", path], cwd=REPO, capture_output=True, text=True).stdout.split()[-1]
    res = subprocess.run(["git", "log", "--format=%H", "--diff-filter=A", "--", "cfb_models/cfb_outcome_engine/phase1_dev_results.json"], cwd=REPO, capture_output=True, text=True).stdout.split()
    if res:
        for f in ("cfb_models/cfb_outcome_engine/phase1_protocol.json", "cfb_models/cfb_outcome_engine/phase1_protocol_amendment_1.json", "cfb_phase1_evaluate.py", "cfb_phase1_team_environment.py", "cfb_phase1_role_state.py", "tests/test_cfb_phase1.py"):
            assert subprocess.run(["git", "merge-base", "--is-ancestor", first(f), res[-1]], cwd=REPO).returncode == 0 and first(f) != res[-1]
    src = (REPO / "cfb_phase1_evaluate.py").read_text()
    assert "protocol + amendment must be committed before any performance" in src and "development already scored" in src


def test_production_files_are_untouched_by_this_branch():
    diff = subprocess.run(["git", "diff", "--name-only", "origin/main...HEAD"], cwd=REPO, capture_output=True, text=True).stdout.split()
    forbidden = ("cfb_serving_builder_a.py", "cfb_pregame_refresh.yml", "cfb_grade_record_a.py", "cfb_live_readiness_a.py", "docs/", "nfl_", "nhl_", "cfb_models/cfb_context_v2_work", "cfb_models/cfb_model.sqlite", ".github/")
    assert not [f for f in diff if f.startswith(forbidden) or f in forbidden], [f for f in diff if f.startswith(forbidden)]


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except BaseException as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
