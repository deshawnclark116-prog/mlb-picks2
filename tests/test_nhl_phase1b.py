"""NHL Phase 1B-A tests (attempt information-value screen). Synthetic data only; no network, no Phase 1A performance recomputation.  python tests/test_nhl_phase1b.py"""
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import nhl_sog_phase1a_data as D  # noqa: E402
import nhl_sog_phase1a_metrics as M  # noqa: E402
import nhl_sog_phase1a_models as MD  # noqa: E402
import nhl_sog_phase1b_attempts as AT  # noqa: E402
import nhl_sog_phase1b_probe as PR  # noqa: E402

OUTD = REPO / "nhl_models" / "nhl_outcome_engine"
T0 = D.epoch("2019-01-20T00:00:00Z")


def iso(sec):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(sec, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def frow(gid, pid, start, toi=1200, sog=2, team=1):
    return {"game_id": gid, "game_start_utc": iso(start), "player_id": pid, "toi_sec": toi, "sog": sog, "team_id": team}


def arow(gid, pid, sog, miss, blk):
    return {"game_id": gid, "player_id": pid, "sog_from_pbp": sog, "missed_attempts": miss, "blocked_attempts": blk, "shot_attempts": sog + miss + blk}


def ev(t, pid, period="REG", key="shootingPlayerId", blocker=None):
    d = {key: pid}
    if blocker:
        d["blockingPlayerId"] = blocker
    return {"typeDescKey": t, "periodDescriptor": {"number": 1, "periodType": period}, "details": d}


# ------------------------------------------------------------------ sampling
def test_deterministic_200_game_hash_sampling_and_label_independence():
    games = {int(f"{s}02{i:04d}"): {"game_start_utc": f"{s}-11-01T00:00:00Z", "sog_total": random.Random(i).randint(0, 90)} for s in (2018, 2019, 2024) for i in range(1, 1300)}
    smp = AT.sample_games(games, seasons=[2018, 2019])
    for s in (2018, 2019):
        pool = [g for g in games if int(str(g)[:4]) == s]
        want = sorted(pool, key=lambda g: (hashlib.sha256(f"nhl-phase1b-a-v1|{g}".encode()).hexdigest(), g))[:200]
        assert smp[s] == want and len(smp[s]) == 200 and len(set(smp[s])) == 200
    assert AT.sample_games(games, seasons=[2018, 2019]) == smp                                                  # deterministic
    other = {g: {"game_start_utc": "x", "sog_total": 1} for g in games}                                       # completely different outcomes/labels, same ids
    assert AT.sample_games(other, seasons=[2018, 2019]) == smp                                                 # independent of anything but the id
    assert AT.sample_games(list(games), seasons=[2018]) == {2018: smp[2018]}                                    # even a bare id list works
    short = {int(f"2020{i:06d}"): {} for i in range(50)}
    assert len(AT.sample_games(short, seasons=[2020])[2020]) == 50                                             # fewer than 200 -> all games
    assert AT.SAMPLE_TAG == "nhl-phase1b-a-v1" and AT.PER_SEASON == 200


def test_2024_2025_excluded_everywhere():
    for s in (2024, 2025):
        try:
            AT.sample_games({int(f"{s}020001"): {}}, seasons=[s]); assert False
        except ValueError:
            pass
        try:
            AT.assert_allowed_seasons([2018, s]); assert False
        except ValueError:
            pass
    assert all(max(tr) <= 2023 and va <= 2023 for _, tr, va in PR.FOLDS) and AT.SAMPLE_SEASONS == [2018, 2019, 2020, 2021, 2022, 2023]
    for f in ("nhl_sog_phase1b_attempts.py", "nhl_sog_phase1b_probe.py"):                                      # no read of Phase 1A confirmation / holdout artifacts
        src = (REPO / f).read_text()
        for bad in ("phase1a_2024_confirmation", "phase1a_2025_holdout", "phase1a_models_2024fit", "phase1a_models_2025fit", "game_crps_2024", "game_crps_2025"):
            assert bad not in src, (f, bad)
    assert "target_seasons=AT.SAMPLE_SEASONS" in (REPO / "nhl_sog_phase1b_probe.py").read_text()


# ------------------------------------------------------------------ history windows
def history_rows(pid=7):
    day = 86400
    rows = []
    for i in range(14):                                                                                         # 14 prior appearances, alternating team 1 / team 2
        rows.append(frow(100 + i, pid, T0 - (20 - i) * day, toi=1200 + 10 * i, sog=i % 4, team=1 + (i % 2)))
    rows.append(frow(500, pid, T0, toi=1300, sog=9))                                                           # the TARGET game
    rows.append(frow(501, pid, T0 + 3 * day, toi=1300, sog=9))                                                 # a future game
    return rows


def test_target_never_in_history_and_only_completed_before_cutoff_games_used():
    rows = history_rows()
    apps, starts = AT.build_appearances(rows)
    w = AT.window_for(apps, starts, 7, T0, 500)
    gids = [a[1] for a in w]
    assert len(w) == 10 and 500 not in gids and 501 not in gids and gids == list(range(104, 114))
    # cutoff boundary: source_start <= target_start - 300 min is usable; one second later is not
    edge = [frow(100, 8, T0 - 300 * 60), frow(101, 8, T0 - 300 * 60 + 1), frow(500, 8, T0)]
    a2, s2 = AT.build_appearances(edge)
    assert [a[1] for a in AT.window_for(a2, s2, 8, T0, 500)] == [100]
    # all-team history is allowed: both team ids appear in the window of the skill window
    teams = {r["team_id"] for r in rows if r["game_id"] in gids}
    assert teams == {1, 2}
    # mutating the target game row cannot change the window
    rows2 = [dict(r, sog=99, toi_sec=1) if r["game_id"] == 500 else r for r in rows]
    a3, s3 = AT.build_appearances(rows2)
    assert AT.window_for(a3, s3, 7, T0, 500) == w
    # a target-game appearance placed inside the window trips the leakage assertion
    try:
        AT.window_for({7: [(T0 - 400 * 60, 500, 1, 1)]}, {7: [T0 - 400 * 60]}, 7, T0, 500); assert False
    except AssertionError:
        pass


def tab_for(rows, pid=7, gid=500):
    return {"game_id": np.array([gid]), "player_id": np.array([pid]), "start": np.array([T0])}


def test_target_row_mutation_leaves_attempt_features_unchanged():
    rows = history_rows()
    apps, starts = AT.build_appearances(rows)
    att = {(100 + i, 7): arow(100 + i, 7, i % 4, i % 3, i % 2) for i in range(14)}
    base = AT.build_attempt_features(tab_for(rows), apps, starts, att)
    att2 = dict(att); att2[(500, 7)] = arow(500, 7, 50, 50, 50); att2[(501, 7)] = arow(501, 7, 50, 50, 50)
    rows2 = [dict(r, sog=77, toi_sec=3) if r["game_id"] in (500, 501) else r for r in rows]
    a2, s2 = AT.build_appearances(rows2)
    mut = AT.build_attempt_features(tab_for(rows2), a2, s2, att2)
    for k in AT.NEW_FEATURES:
        assert np.array_equal(base[k], mut[k], equal_nan=True), k


# ------------------------------------------------------------------ PBP parsing
def test_shootout_excluded_blocked_attributed_to_shooter_and_sog_reconstruction():
    events = [ev("shot-on-goal", 1), ev("goal", 1, key="scoringPlayerId"), ev("missed-shot", 1), ev("blocked-shot", 1, blocker=2), ev("blocked-shot", 3, blocker=1),
              ev("shot-on-goal", 1, period="SO"), ev("goal", 2, period="SO", key="scoringPlayerId"), ev("missed-shot", 3, period="SO"), {"typeDescKey": "hit", "details": {"hittingPlayerId": 1}}]
    c = AT.pbp_counts(events)
    assert c[1] == {"sog": 2, "missed": 1, "blocked": 1, "attempts": 4}                                         # goal counts as SOG; the blocker (player 2) gets nothing for blocked shots
    assert 2 not in c and c[3] == {"sog": 0, "missed": 0, "blocked": 1, "attempts": 1}                         # blocked attempt belongs to the shooter 3 (not blocker 1)
    assert AT.shootout_events_excluded(events) == 3
    pbp = {"plays": events, "rosterSpots": [{"playerId": 9, "positionCode": "G"}]}
    rows, extras = AT.game_player_rows(500, pbp, {1, 3, 4})
    d = {r["player_id"]: r for r in rows}
    assert d[1]["shot_attempts"] == 4 and d[1]["sog_from_pbp"] == 2 and d[4]["shot_attempts"] == 0 and d[3]["blocked_attempts"] == 1
    assert extras["goalie_shooters"] == [] and set(d) == {1, 3, 4}
    for r in rows:
        assert r["shot_attempts"] >= r["sog_from_pbp"] >= 0 and r["shot_attempts"] == r["sog_from_pbp"] + r["missed_attempts"] + r["blocked_attempts"]


def test_pbp_row_order_invariance_and_quality_gate():
    rnd = random.Random(1)
    events = [ev(rnd.choice(["shot-on-goal", "goal", "missed-shot", "blocked-shot"]), rnd.randint(1, 6), period=rnd.choice(["REG", "REG", "OT", "SO"]), key="shootingPlayerId") for _ in range(200)]
    for e in events:
        if e["typeDescKey"] == "goal":
            e["details"] = {"scoringPlayerId": e["details"]["shootingPlayerId"]}
    base = AT.pbp_counts(events)
    for seed in range(5):
        sh = list(events); random.Random(seed).shuffle(sh)
        assert AT.pbp_counts(sh) == base
    cnt = AT.pbp_counts(events)
    official = {p: cnt.get(p, {"sog": 0})["sog"] for p in range(1, 7)}
    raw = json.dumps({"plays": events, "rosterSpots": []}).encode()
    rows, per, q = AT.derive({500: (raw, {"bytes": len(raw)}, False)}, {500: official})
    assert q["mismatches"] == 0 and q["exact_matches"] == 6 and AT.quality_pass(q) and q["order_invariance_failures"] == 0 and q["shootout_events_excluded"] > 0 and len(rows) == 6
    bad = dict(official); bad[1] += 1                                                                            # a genuine attribution mismatch must fail the gate
    _, _, q2 = AT.derive({500: (raw, {"bytes": len(raw)}, False)}, {500: bad})
    assert q2["mismatches"] == 1 and q2["mismatch_examples"][0]["player_id"] == 1 and not AT.quality_pass(q2)
    _, _, q3 = AT.derive({500: (raw, {}, False)}, {500: {1: 0}})                                                 # a non-goalie shooter missing from the frozen table is a mismatch
    assert q3["non_goalie_unmatched_shooters"] and not AT.quality_pass(q3)
    _, _, q4 = AT.derive({500: (None, {"unobtainable": True}, False)}, {500: {1: 0}})
    assert q4["unobtainable_fraction"] == 1.0 and not AT.quality_pass(q4)


# ------------------------------------------------------------------ features
def win(rows_spec):
    """rows_spec: [(game_id, toi, sog_official)] -> window tuples."""
    return [(i, g, toi, sog) for i, (g, toi, sog) in enumerate(rows_spec)]


def test_ratio_of_sums_attempts_per60_and_on_net_rate_and_zero_attempts_missing():
    spec = [(1, 600, 1), (2, 1800, 3), (3, 1200, 0)]
    att = {1: arow(1, 7, 1, 2, 1), 2: arow(2, 7, 3, 3, 3), 3: arow(3, 7, 0, 0, 1)}                              # attempts 4, 9, 1
    f = AT.attempt_features_for_window(win(spec), lambda g: att.get(g))
    assert abs(f["ATTEMPTS_PER60_APP10"] - 3600 * (4 + 9 + 1) / (600 + 1800 + 1200)) < 1e-12                       # ratio of sums
    assert abs(f["ATTEMPTS_PER60_APP10"] - 3600 * np.mean([4 / 600, 9 / 1800, 1 / 1200])) > 1e-6                  # NOT a mean of per-game rates
    assert abs(f["ON_NET_RATE_APP10"] - (1 + 3 + 0) / (4 + 9 + 1)) < 1e-12                                          # ratio of sums
    assert abs(f["ATTEMPTS_MEAN_APP10"] - 14 / 3) < 1e-12 and abs(f["ATTEMPTS_MEAN_APP5"] - 14 / 3) < 1e-12
    big = win([(i, 1200, 1) for i in range(1, 9)]); attb = {i: arow(i, 7, 1, i, 0) for i in range(1, 9)}
    f8 = AT.attempt_features_for_window(big, lambda g: attb.get(g))
    assert abs(f8["ATTEMPTS_MEAN_APP5"] - np.mean([1 + i for i in range(4, 9)])) < 1e-12                           # last 5 only
    z = AT.attempt_features_for_window(win([(1, 1200, 0), (2, 1200, 0)]), lambda g: arow(g, 7, 0, 0, 0))
    assert math.isnan(z["ON_NET_RATE_APP10"]) and z["ATTEMPTS_MEAN_APP10"] == 0.0 and z["ATTEMPTS_PER60_APP10"] == 0.0     # zero attempts: on-net MISSING (not 0)
    e = AT.attempt_features_for_window([], lambda g: None)
    assert all(math.isnan(v) for v in e.values())
    t = AT.attempt_features_for_window(win([(1, 0, 1), (2, 1000, 2)]), lambda g: {1: arow(1, 7, 1, 2, 1), 2: arow(2, 7, 2, 0, 0)}[g])   # TOI = 0 appearance is not usable for per-60 (numerator and denominator alike)
    assert abs(t["ATTEMPTS_PER60_APP10"] - 3600 * 2 / 1000) < 1e-12


def test_no_sportsbook_fields_and_exact_four_features():
    assert AT.NEW_FEATURES == ["ATTEMPTS_MEAN_APP5", "ATTEMPTS_MEAN_APP10", "ATTEMPTS_PER60_APP10", "ON_NET_RATE_APP10"]
    fields = ["game_id", "player_id", "sog_from_pbp", "missed_attempts", "blocked_attempts", "shot_attempts"]
    for name in fields + AT.NEW_FEATURES + PR.A1_FEATURES:
        assert not any(w in name.lower() for w in D.SPORTSBOOK_WORDS), name
    assert set(arow(1, 1, 1, 1, 1)) == set(fields)
    proto = json.loads((OUTD / "phase1b_attempt_signal_protocol.json").read_text())
    assert proto["features"]["allowed"] == AT.NEW_FEATURES and proto["features"]["N_ATTEMPT_APPEARANCES_10"].startswith("NOT used")


# ------------------------------------------------------------------ models
def synth_tab(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t = {k: rng.normal(size=n) for k in PR.A1_FEATURES}
    for b in D.BINARY:
        t[b] = (rng.random(n) < 0.5).astype(float)
    for k in ("ATTEMPTS_PER60_APP10", "ON_NET_RATE_APP10"):
        t[k][rng.random(n) < 0.1] = np.nan
    t["sog"] = rng.poisson(np.exp(0.2 + 0.3 * t["SOG_MEAN_APP10"] + 0.1 * np.nan_to_num(t["ATTEMPTS_MEAN_APP10"]))).astype(int)
    t["game_id"] = np.arange(n) // 10; t["player_id"] = np.arange(n); t["team_id"] = np.arange(n) % 2
    return t


def test_a0_is_exact_frozen_b2_and_a1_differs_only_by_registered_features():
    arch = json.loads((OUTD / "phase1a_selected_architecture.json").read_text())
    assert arch["selected_architecture"] == "B2" and arch["frozen_hyperparameters"]["B1_alpha"] == PR.ALPHA == 0.001
    assert PR.CORE == D.FEATURES == MD.B1_FEATURES and len(PR.CORE) == 33
    assert PR.A1_FEATURES[:33] == PR.CORE and PR.A1_FEATURES[33:] == AT.NEW_FEATURES and len(PR.A1_FEATURES) == 37
    assert set(PR.A1_FEATURES) - set(PR.CORE) == set(AT.NEW_FEATURES)
    tr = synth_tab(800, 1); va = synth_tab(300, 2)
    a = PR.fit_architecture(tr, PR.CORE)
    prep = MD.Preprocessor(MD.B1_FEATURES).fit(tr); X = prep.transform(tr)                                   # the frozen Phase 1A code path, called directly
    m, _ = MD.fit_poisson(X, tr["sog"], 0.001); nb = MD.fit_nb_alpha(tr["sog"], m.predict(X))
    kind, params, mu = PR.predict_architecture(a, va)
    assert kind == "nb2" and np.allclose(mu, m.predict(prep.transform(va))) and abs(params["alpha"] - nb["alpha"]) < 1e-12
    a1 = PR.fit_architecture(tr, PR.A1_FEATURES)
    assert a1["names"] == PR.A1_FEATURES and a1["alpha_poisson"] == 0.001 == a["alpha_poisson"]
    proto = json.loads((OUTD / "phase1b_attempt_signal_protocol.json").read_text())
    assert "alpha=0.001" in proto["models"]["A0"] and "hyperparameter_search" in proto["models"] and proto["models"]["hyperparameter_search"] == "none"


def test_train_only_imputation_for_attempt_features():
    tr = synth_tab(400, 3); va = synth_tab(200, 4)
    va["ATTEMPTS_PER60_APP10"][:] = np.nan; va["ON_NET_RATE_APP10"][:] = 1e6
    prep = MD.Preprocessor(PR.A1_FEATURES).fit(tr)
    med = dict(prep.median)
    X = prep.transform(va)
    assert prep.median == med                                                                                  # validation rows never change training statistics
    j = PR.A1_FEATURES.index("ATTEMPTS_PER60_APP10")
    assert np.allclose(X[:, j], (med["ATTEMPTS_PER60_APP10"] - prep.mean["ATTEMPTS_PER60_APP10"]) / prep.std["ATTEMPTS_PER60_APP10"])   # imputed with the TRAINING median
    miss_cols = [len(PR.A1_FEATURES) + prep.cont.index("ATTEMPTS_PER60_APP10")]
    assert (X[:, miss_cols[0]] == 1).all() and "ATTEMPTS_PER60_APP10" in prep.cont and "ON_NET_RATE_APP10" in prep.cont     # explicit missing indicators, rows kept


# ------------------------------------------------------------------ calibration metrics + gate logic
def test_prospective_attainable_mass_and_randomized_pit_central_coverage():
    pmf = np.array([[0.5, 0.3, 0.15, 0.05], [0.1, 0.2, 0.3, 0.4]])
    y = np.array([1, 3])
    ex = PR.calib_extras("pmf", {"pmf": pmf}, y, "T", np.array([1, 2]), np.array([1, 2]))
    cdf = np.cumsum(pmf, axis=1)
    for c in (0.5, 0.8, 0.9):
        for i in range(2):
            lo = int(np.argmax(cdf[i] >= (1 - c) / 2 - 1e-12)); hi = int(np.argmax(cdf[i] >= (1 + c) / 2 - 1e-12))
            att = cdf[i, hi] - (cdf[i, lo - 1] if lo > 0 else 0.0)
            assert ex["lo"][c][i] == lo and ex["hi"][c][i] == hi and abs(ex["attainable"][c][i] - att) < 1e-12 and att >= c - 1e-9          # an attainable mass can never be below nominal
    assert abs(ex["attainable"][0.8][0] - (0.95 - 0.0)) < 1e-12 and ex["lo"][0.8][0] == 0 and ex["hi"][0.8][0] == 2                         # row 0: F=[.5,.8,.95,1]; lo=0, hi=2 -> F(2)-F(-1)=.95
    rep = PR.calibration_report(ex, y)
    assert abs(rep["80"]["empirical_minus_mean_attainable"] - (rep["80"]["empirical_deterministic_coverage"] - rep["80"]["mean_attainable_model_mass"])) < 1e-12 and rep["80"]["nominal_coverage"] == 0.8
    # randomized PIT central coverage on hand-set PIT values
    fake = {"pit": np.array([0.04, 0.10, 0.5, 0.90, 0.96]), "lo": {c: np.zeros(5, int) for c in (0.5, 0.8, 0.9)}, "hi": {c: np.zeros(5, int) for c in (0.5, 0.8, 0.9)}, "attainable": {c: np.ones(5) for c in (0.5, 0.8, 0.9)}}
    r = PR.calibration_report(fake, np.zeros(5, int))
    assert r["80"]["randomized_pit_central_coverage"] == 3 / 5 and abs(r["80"]["randomized_pit_abs_error"] - 0.2) < 1e-12                       # PIT in [0.1, 0.9]
    assert r["90"]["randomized_pit_central_coverage"] == 3 / 5 and r["50"]["randomized_pit_central_coverage"] == 1 / 5                         # [0.05,0.95]: .10,.5,.90 ; [0.25,0.75]: .5


def test_pass_gate_logic_and_failed_gate_prevents_full_crawl_status():
    ok = dict(rel_improvement=0.006, upper95=-0.0001, fold_deltas=[-1, -1, -1, 1], rel_nll=0.0, slice_ok_fd=True, slice_ok_early=True, ks_delta=0.0, pit80_delta=0.0, pit90_delta=0.0)
    t, s = PR.gate_decision(**ok)
    assert s == "ATTEMPT_SIGNAL_FULL_CRAWL_JUSTIFIED" and all(v["pass"] for v in t.values())
    for key, bad, gate in (("rel_improvement", 0.004, "S1"), ("upper95", 0.0, "S2"), ("fold_deltas", [-1, -1, 1, 1], "S3"), ("rel_nll", 0.0051, "S4"), ("slice_ok_fd", False, "S5"), ("slice_ok_early", False, "S6"),
                           ("ks_delta", 0.0201, "S7"), ("pit80_delta", 0.0201, "S7"), ("pit90_delta", 0.0201, "S7")):
        t, s = PR.gate_decision(**{**ok, key: bad})
        assert not t[gate]["pass"] and s == "ATTEMPT_SIGNAL_NOT_JUSTIFIED", (key, gate)                                   # any failed gate -> NOT_JUSTIFIED, never FULL_CRAWL_JUSTIFIED
    t, s = PR.gate_decision(**{**ok, "rel_improvement": 0.005, "rel_nll": 0.005, "ks_delta": 0.02, "pit80_delta": 0.02, "pit90_delta": 0.02, "fold_deltas": [-1, -1, -1, -1]})
    assert s == "ATTEMPT_SIGNAL_FULL_CRAWL_JUSTIFIED"                                                               # exact thresholds are inclusive where registered
    proto = json.loads((OUTD / "phase1b_attempt_signal_protocol.json").read_text())
    g = proto["pass_gate"]
    assert proto["bootstrap"]["seed"] == M.BOOT_SEED == 20261002 and proto["bootstrap"]["pooled_unit"].startswith("all validation games")
    assert g["pass_status"].startswith("ATTEMPT_SIGNAL_FULL_CRAWL_JUSTIFIED") and g["fail_status"].startswith("ATTEMPT_SIGNAL_NOT_JUSTIFIED")


def test_protocol_registered_before_results_and_untouched_phase1a_artifacts():
    import subprocess
    proto = json.loads((OUTD / "phase1b_attempt_signal_protocol.json").read_text())
    body = {k: v for k, v in proto.items() if k != "protocol_body_sha256"}
    assert hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == proto["protocol_body_sha256"]
    assert proto["status"].startswith("PREREGISTERED_BEFORE") and proto["expected_parent_head"] == "31d4414edd15cd47d581f9bed11daf0ee0657596"
    for f in ("phase1a_2024_confirmation.json", "phase1a_selected_architecture.json", "phase1a_2025_holdout.json", "phase1a_models_2024fit.json", "phase1a_models_2025fit.json"):
        r = subprocess.run(["git", "log", "--format=%H", "--", f"nhl_models/nhl_outcome_engine/{f}"], cwd=REPO, capture_output=True, text=True).stdout.split()
        assert r and "ed877d2" not in r[0] and len(r) >= 1
        diff = subprocess.run(["git", "diff", "--name-only", "31d4414edd15cd47d581f9bed11daf0ee0657596", "HEAD", "--", f"nhl_models/nhl_outcome_engine/{f}"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        assert diff == "", f                                                                                       # byte-identical to the Phase 1B-A parent head


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
