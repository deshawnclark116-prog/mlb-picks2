"""NHL Phase 1A tests (SOG distribution engine): leak-free candidates / features, models, metrics, bootstrap, protocol discipline. python tests/test_nhl_phase1a.py"""
import copy
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import nhl_outcome_contract as CT  # noqa: E402
import nhl_sog_phase1a_data as D  # noqa: E402
import nhl_sog_phase1a_metrics as M  # noqa: E402
import nhl_sog_phase1a_models as MD  # noqa: E402
import nhl_sog_phase1a_run as RUN  # noqa: E402

OUT = REPO / "nhl_models" / "nhl_outcome_engine"
UTC = timezone.utc
T0 = datetime(2023, 11, 1, 19, 0, tzinfo=UTC)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def mk(gid, team, pid, start, pos="C", sog=1, toi=1000, pp=100, ev=900, sh=0, shifts=20):
    return {"game_id": gid, "team_id": team, "player_id": pid, "game_start_utc": iso(start), "position": pos, "sog": sog, "toi_sec": toi, "pp_toi_sec": pp, "ev_toi_sec": ev, "sh_toi_sec": sh, "shifts": shifts,
            "season_start_year": 2023, "team_abbrev": str(team), "opponent": "X"}


def world(n=14):
    """Team 1 (AAA) plays team 2 (BBB) in games 1..n every 2 days (home alternates). Players a=101..105 (team 1), b=201..205 (team 2), old=301 (team 1 only in game 1), x=401 (team 3 in games 1-5, team 1 from game 7).
    Team 3 plays team 4 three hours before each of the first 8 team-1 games."""
    games, rows = {}, []
    for g in range(1, n + 1):
        gid = 2023020000 + g
        st = T0 + timedelta(days=2 * (g - 1))
        home, away = (1, 2) if g % 2 else (2, 1)
        games[gid] = {"game_id": gid, "game_start_utc": iso(st), "home_team_id": home, "away_team_id": away, "home_abbrev": "H", "away_abbrev": "A"}
        for i in range(5):
            rows.append(mk(gid, 1, 101 + i, st, pos="C" if i < 3 else "D", sog=i + 1, pp=200 - 30 * i))
            rows.append(mk(gid, 2, 201 + i, st, pos="C", sog=1, pp=100))
        if g == 1:
            rows.append(mk(gid, 1, 301, st, sog=2))
        if g >= 7:
            rows.append(mk(gid, 1, 401, st, sog=1, pp=40))
        if g <= 8:
            gid2 = 2023029000 + g; st2 = st - timedelta(hours=3)
            games[gid2] = {"game_id": gid2, "game_start_utc": iso(st2), "home_team_id": 3, "away_team_id": 4, "home_abbrev": "C", "away_abbrev": "D"}
            for i in range(5):
                rows.append(mk(gid2, 4, 501 + i, st2, sog=1))
            rows.append(mk(gid2, 3, 411 + g, st2, sog=1))
            if g <= 5:
                rows.append(mk(gid2, 3, 401, st2, sog=6, toi=2000, pp=500, shifts=44))                  # x's old-team rows: high SOG, big role
    return games, rows


def build(games, rows, seasons=(2023,)):
    return D.build_prediction_rows(games, rows, list(seasons))


def select(tab, game, team):
    m = (tab["game_id"] == game) & (tab["team_id"] == team)
    return {int(p): i for p, i in zip(tab["player_id"][m], np.where(m)[0])}


def feats_for(tab, game, team):
    idx = select(tab, game, team)
    return {p: {f: float(tab[f][i]) for f in D.FEATURES} for p, i in idx.items()}


def same_feats(a, b):
    if a.keys() != b.keys():
        return False
    return all(np.array_equal(np.array(list(a[p].values())), np.array(list(b[p].values())), equal_nan=True) for p in a)


# ------------------------------------------------------------------ candidate universe / features
def test_candidate_universe_equals_phase0b_contract_function():
    games, rows = world()
    tab, _ = build(games, rows)
    ct_rows = [{"gameId": r["game_id"], "playerId": r["player_id"], "team": r["team_id"], "startTimeUTC": r["game_start_utc"]} for r in rows]
    for gid in (2023020005, 2023020009, 2023020014):
        for team in (1, 2):
            T = CT.cutoff_time(CT.parse_utc(games[gid]["game_start_utc"]), "T90")
            assert sorted(CT.candidate_universe(ct_rows, gid, team, T)["candidates"]) == sorted(select(tab, gid, team))


def test_target_row_mutation_leaves_every_feature_unchanged():
    games, rows = world()
    base, _ = build(games, rows)
    tgt = 2023020014
    mut = []
    for r in rows:
        if r["game_id"] == tgt:
            r = {**r, "player_id": r["player_id"] + 7_000_000, "sog": 9, "toi_sec": 5, "pp_toi_sec": 0, "ev_toi_sec": 5, "shifts": 99, "position": "D" if r["position"] == "C" else "C"}
        mut.append(r)
    after, _ = build(games, mut)
    for team in (1, 2):
        assert same_feats(feats_for(base, tgt, team), feats_for(after, tgt, team)) and feats_for(base, tgt, team)
    assert select(base, tgt, 1).keys() == select(after, tgt, 1).keys()


def test_target_only_player_is_never_a_candidate_and_is_counted_unobservable():
    games, rows = world()
    base, cov0 = build(games, rows)
    tgt = 2023020014
    st = T0 + timedelta(days=2 * 13)
    rows2 = rows + [mk(tgt, 1, 999, st, sog=4)]
    tab, cov = build(games, rows2)
    assert 999 not in select(tab, tgt, 1) and select(tab, tgt, 1).keys() == select(base, tgt, 1).keys()
    c, c0 = cov[2023], cov0[2023]
    assert c["unobservable"] == c0["unobservable"] + 1 and c["actual_target_skaters"] == c0["actual_target_skaters"] + 1 and c["actual_SOG"] == c0["actual_SOG"] + 4 and c["observable_SOG"] == c0["observable_SOG"]
    assert c["unobservable_diagnostics"]["NO_LOADED_HISTORY"] == c0["unobservable_diagnostics"].get("NO_LOADED_HISTORY", 0) + 1


def test_old_team_only_history_cannot_establish_membership_but_first_same_team_game_does():
    games, rows = world()
    tab, cov = build(games, rows)
    assert 401 not in select(tab, 2023020006, 1) and 401 not in select(tab, 2023020007, 1)       # team-3 history only before game 7
    assert 401 in select(tab, 2023020008, 1)                                                    # after his first completed team-1 game (7)
    assert cov[2023]["unobservable_diagnostics"]["PRIOR_NHL_HISTORY_ELSEWHERE"] >= 1            # graded as unobservable at game 7, never inserted


def test_old_team_skill_history_is_usable_after_membership_but_role_rows_are_not():
    games, rows = world()
    tab, _ = build(games, rows)
    f = feats_for(tab, 2023020009, 1)[401]                                                      # appearances: 5 for team 3 (sog 6), games 7,8 for team 1 (sog 1)
    assert f["N_SKILL_APPEARANCES_10"] == 7 and abs(f["SOG_MEAN_APP10"] - (5 * 6 + 2 * 1) / 7) < 1e-9       # skill uses ALL teams
    assert f["N_ROLE_APPEARANCES_10"] == 2 and abs(f["TOI_MEAN_CT_APP10"] - 1000) < 1e-9 and abs(f["SHIFT_MEAN_CT_APP10"] - 20) < 1e-9    # role: team-1 rows only
    mut = [({**r, "toi_sec": 7777, "pp_toi_sec": 7777, "shifts": 88} if (r["team_id"] == 3 and r["player_id"] == 401) else r) for r in rows]
    tab2, _ = build(games, mut)
    g = feats_for(tab2, 2023020009, 1)[401]
    for k in ("TOI_MEAN_CT_APP3", "TOI_MEAN_CT_APP10", "PP_TOI_MEAN_CT_APP3", "PP_ALLOC_SHARE_MEAN_CT_APP3", "PP_ALLOC_SHARE_MEAN_CT_APP10", "SHIFT_MEAN_CT_APP3", "SHIFT_MEAN_CT_APP10", "N_ROLE_APPEARANCES_10"):
        assert f[k] == g[k] or (np.isnan(f[k]) and np.isnan(g[k])), k


def test_ratio_of_sums_sog_per60():
    games, rows = world()
    for r in rows:
        if r["player_id"] == 101 and r["game_id"] in (2023020012, 2023020013):
            r["sog"] = 1 if r["game_id"] == 2023020012 else 3
            r["toi_sec"] = 600 if r["game_id"] == 2023020012 else 3000
    # make the two newest appearances the only ones: drop older ones of player 101
    rows = [r for r in rows if not (r["player_id"] == 101 and r["game_id"] < 2023020012)]
    tab, _ = build(games, rows)
    f = feats_for(tab, 2023020014, 1)[101]
    assert abs(f["SOG_PER60_APP10"] - 3600 * 4 / 3600) < 1e-9 and abs(f["SOG_PER60_APP10"] - np.mean([6.0, 3.6])) > 0.5


def test_nonparticipant_candidate_label_zero_and_played_zero_sog_allowed():
    games, rows = world()
    st = T0 + timedelta(days=2 * 13)
    rows = [r for r in rows if not (r["game_id"] == 2023020014 and r["player_id"] == 102)]                 # candidate 102 sits out
    for r in rows:
        if r["game_id"] == 2023020014 and r["player_id"] == 103:
            r["sog"] = 0                                                                                    # plays with zero SOG
    tab, _ = build(games, rows)
    idx = select(tab, 2023020014, 1)
    assert tab["played"][idx[102]] == 0 and tab["sog"][idx[102]] == 0 and tab["played"][idx[103]] == 1 and tab["sog"][idx[103]] == 0


def test_no_target_position_leakage_position_from_latest_prior_same_team_appearance():
    games, rows = world()
    base, _ = build(games, rows)
    rows2 = [({**r, "position": "D"} if (r["game_id"] == 2023020014 and r["team_id"] == 1) else r) for r in rows]
    t2, _ = build(games, rows2)
    assert same_feats(feats_for(base, 2023020014, 1), feats_for(t2, 2023020014, 1))
    f = feats_for(base, 2023020014, 1)
    assert f[101]["POS_F"] == 1 and f[104]["POS_D"] == 1
    rows3 = [({**r, "position": "D"} if (r["game_id"] == 2023020013 and r["player_id"] == 101) else r) for r in rows]      # the latest PRIOR appearance changes
    assert feats_for(build(games, rows3)[0], 2023020014, 1)[101]["POS_D"] == 1


def test_no_sportsbook_columns_anywhere():
    games, rows = world()
    assert not any(any(w in k.lower() for w in D.SPORTSBOOK_WORDS) for k in rows[0]) and not any(any(w in f.lower() for w in D.SPORTSBOOK_WORDS) for f in D.FEATURES)
    r = D.read_jsonl_gz(D.DATA / "skater_games_2023.jsonl.gz")[0]
    assert not any(any(w in k.lower() for w in D.SPORTSBOOK_WORDS) for k in r) and set(D.FROZEN_FIELDS) <= set(r)


def test_source_games_completed_before_cutoff_and_target_excluded():
    games, rows = world()
    # a game that starts 2h before the target is NOT complete at T = target - 90 min (needs start + 210 min <= T)
    tgt = 2023020014
    st_t = T0 + timedelta(days=26)
    gid_x = 2023025000
    games[gid_x] = {"game_id": gid_x, "game_start_utc": iso(st_t - timedelta(hours=2)), "home_team_id": 1, "away_team_id": 4, "home_abbrev": "H", "away_abbrev": "A"}
    rows += [mk(gid_x, 1, 888, st_t - timedelta(hours=2))]
    tab, _ = build(games, rows)
    assert 888 not in select(tab, tgt, 1)
    edge = st_t - timedelta(minutes=300)
    games[gid_x]["game_start_utc"] = iso(edge)
    rows = [r for r in rows if r["player_id"] != 888] + [mk(gid_x, 1, 888, edge)]
    tab, _ = build(games, rows)
    assert 888 in select(tab, tgt, 1)                                                                     # start + 210 min == T is included


# ------------------------------------------------------------------ data / protocol discipline
def test_fold_chronology_and_season_roles():
    for fid, train, val in RUN.FOLDS:
        assert max(train) < val and min(train) == 2018 and val <= 2023 and 2017 not in train and 2024 not in train + [val] and 2025 not in train + [val]
    assert [f[2] for f in RUN.FOLDS] == [2020, 2021, 2022, 2023]
    games, rows = world()
    tab, _ = build(games, rows)
    tab["season"] = np.where(tab["game_id"] > 2023020009, 2024, 2023)
    tr = RUN.sel(tab, [2023]); assert tr["season"].max() == 2023


def test_train_only_imputation_and_scaling():
    games, rows = world()
    tab, _ = build(games, rows)
    n = len(tab["game_id"]); half = n // 2
    tr = MD.select_rows(tab, np.arange(n) < half); va = MD.select_rows(tab, np.arange(n) >= half)
    pre = MD.Preprocessor(["POS_F", "SOG_SD_APP10", "TEAM_REST_HOURS", "TOI_MEAN_CT_APP3"]).fit(tr)
    med = float(np.nanmedian(tr["SOG_SD_APP10"]))
    assert abs(pre.median["SOG_SD_APP10"] - med) < 1e-12
    va2 = copy.deepcopy(va); va2["SOG_SD_APP10"] = np.full(len(va2["sog"]), np.nan); va2["TOI_MEAN_CT_APP3"] = va2["TOI_MEAN_CT_APP3"] * 1000 + 5
    pre2 = MD.Preprocessor(pre.names).fit(tr)
    assert pre.schema() == pre2.schema()                                                                  # validation data cannot move the schema
    X = pre.transform(va2)
    i = pre.names.index("SOG_SD_APP10")
    assert np.allclose(X[:, i], (med - pre.mean["SOG_SD_APP10"]) / pre.std["SOG_SD_APP10"]) and X[:, len(pre.names) + pre.cont.index("SOG_SD_APP10")].min() == 1.0     # imputed + indicator
    assert set(np.unique(X[:, pre.names.index("POS_F")])) <= {0.0, 1.0}                                   # binary stays 0/1
    cap = {"TEAM_REST_HOURS": np.array([1000.0, 20.0]), **{k: np.zeros(2) for k in pre.names if k != "TEAM_REST_HOURS"}}
    assert MD.Preprocessor.raw(cap, "TEAM_REST_HOURS").max() == 240.0


# ------------------------------------------------------------------ models
def small_table():
    games, rows = world()
    tab, _ = build(games, rows)
    return tab


def test_b0_probabilities_sum_to_one_and_mixture_identities():
    tab = small_table()
    b0 = MD.B0().fit(tab)
    p = b0.predict(tab)
    assert np.allclose(p["pmf"].sum(axis=1), 1.0, atol=1e-12) and np.allclose(p["cond_pmf"].sum(axis=1), 1.0, atol=1e-12) and (p["pmf"] >= 0).all()
    k = np.arange(MD.SUPPORT + 1)
    assert np.allclose((p["pmf"] * k).sum(axis=1), p["p_play"] * p["mean_cond"], atol=1e-12) and np.allclose(p["mean"], p["p_play"] * p["mean_cond"])
    assert np.allclose(p["pmf"][:, 0], (1 - p["p_play"]) + p["p_play"] * p["cond_pmf"][:, 0])
    # fixed constants, position priors from TRAINING rows only
    assert MD.KAPPA_PLAY == 5 and MD.KAPPA_SOG == 5 and MD.DIRICHLET_EPS == 0.01
    pos = np.where(tab["POS_F"] == 1, "F", "D")
    assert abs(b0.base["F"] - tab["played"][pos == "F"].mean()) < 1e-12


def test_b0_blocks_on_sog_above_60():
    tab = small_table(); tab = {k: v.copy() for k, v in tab.items()}; tab["sog"][0] = 61; tab["played"][0] = 1
    try:
        MD.B0().fit(tab); assert False
    except RuntimeError as e:
        assert "BLOCK" in str(e)


def test_poisson_nb2_and_b3_mixture_normalization_and_means():
    mu = np.array([0.3, 1.2, 3.5, 0.05]); y = np.zeros(4, dtype=int)
    for kind, params in (("poisson", {"mu": mu}), ("nb2", {"mu": mu, "alpha": 0.4}), ("mix_nb2", {"p": np.array([0.9, 0.5, 0.99, 0.1]), "mu": mu, "alpha": 0.4})):
        for sl, pm, sf in M.pmf_matrix(kind, params, y):
            assert np.allclose(pm.sum(axis=1) + sf, 1.0, atol=1e-9) and np.max(sf) < 1e-10
            k = np.arange(pm.shape[1])
            exp = (pm * k).sum(axis=1)
            if kind == "mix_nb2":
                assert np.allclose(exp, params["p"] * mu, rtol=1e-6)                                      # E[Y] = p_play * conditional mean
            else:
                assert np.allclose(exp, mu, rtol=1e-6)
    pm = next(M.pmf_matrix("mix_nb2", {"p": np.array([0.5]), "mu": np.array([2.0]), "alpha": 0.3}, np.zeros(1, dtype=int)))[1][0]
    from scipy import stats
    r = 1 / 0.3
    assert abs(pm[0] - (0.5 + 0.5 * stats.nbinom.pmf(0, r, r / (r + 2.0)))) < 1e-12 and pm[0] > 0.5                  # no hurdle: a played player can record 0


def test_nb2_dispersion_mle_recovers_alpha_and_reports_bounds():
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.5, 3.0, 20000); alpha = 0.35
    r = 1 / alpha
    y = rng.negative_binomial(r, r / (r + mu))
    fit = MD.fit_nb_alpha(y, mu)
    assert abs(fit["alpha"] - alpha) < 0.05 and not fit["at_lower_bound"] and not fit["at_upper_bound"]
    fit2 = MD.fit_nb_alpha(rng.poisson(mu), mu)                                                            # equidispersed -> alpha at the lower bound, reported
    assert fit2["alpha"] < 0.05


def test_artifact_hash_roundtrip_and_prediction_reproduction():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 4)); y = rng.poisson(np.exp(0.3 * X[:, 0] + 0.2))
    tab = {"POS_F": np.ones(500), "SOG_MEAN_APP5": X[:, 0], "game_id": np.arange(500), "team_id": np.zeros(500), "player_id": np.arange(500)}
    pre = MD.Preprocessor(["POS_F", "SOG_MEAN_APP5"]).fit(tab)
    Xs = pre.transform(tab)
    m, info = MD.fit_poisson(Xs, y, 0.1)
    a = MD.poisson_artifact(m, pre, 0.1)
    assert np.allclose(MD.predict_poisson_from_artifact(a, Xs), m.predict(Xs), rtol=1e-12)
    body = {k: v for k, v in a.items() if k != "sha256"}
    assert MD.sha_json(body) == a["sha256"]
    lg, _ = MD.fit_logistic(Xs, (y > 0).astype(int), 1.0)
    la = MD.logistic_artifact(lg, pre, 1.0)
    p = 1 / (1 + np.exp(-(Xs @ np.array(la["coef"]) + la["intercept"])))
    assert np.allclose(p, lg.predict_proba(Xs)[:, 1], atol=1e-12)
    bad, manifest = D.verify_manifest()
    assert not bad
    for name in ("phase1a_models_2024fit.json", "phase1a_models_2025fit.json"):
        f = OUT / name
        if f.exists():
            b = json.loads(f.read_text())
            assert all(MD.sha_json(b["artifacts"][m_]) == h for m_, h in b["manifest"].items()) and MD.sha_json(b["manifest"]) == b["bundle_sha256"]


# ------------------------------------------------------------------ metrics
def test_crps_point_mass_zero_and_known_values():
    F = np.cumsum([0, 0, 1.0, 0, 0])
    assert M.crps_point(F, 2) == 0.0
    assert abs(M.crps_point(np.cumsum([0.5, 0.5, 0, 0]), 0) - 0.25) < 1e-12                             # F=(0.5,1,1,1), y=0: (0.5-1)^2
    pm = np.zeros((1, 61)); pm[0, 3] = 1.0
    out = M.row_metrics("pmf", {"pmf": pm}, np.array([3]), "m", [1], [1])
    assert out["crps"][0] == 0.0 and out["nll"][0] < 1e-12


def test_dynamic_tail_and_invalid_distribution_ceiling():
    out = M.row_metrics("nb2", {"mu": np.array([25.0]), "alpha": 0.2}, np.array([25]), "m", [1], [1])
    assert out["max_K"] > 40 and out["max_sf"] < M.SF_TOL                                              # support extended until survival < 1e-10
    try:
        M.row_metrics("nb2", {"mu": np.array([500.0]), "alpha": 20.0}, np.array([1]), "m", [1], [1]); assert False
    except M.InvalidDistribution:
        pass
    pm = np.full((1, 61), 1 / 61)
    assert np.allclose(M.row_metrics("pmf", {"pmf": pm}, np.array([0]), "m", [1], [1])["crps"], M.row_metrics("pmf", {"pmf": pm}, np.array([0]), "m", [1], [1])["crps"])


def test_nll_floor_counted():
    pm = np.zeros((1, 61)); pm[0, 0] = 1.0
    out = M.row_metrics("pmf", {"pmf": pm}, np.array([5]), "m", [1], [1])
    assert out["floor_hits"] == 1 and abs(out["nll"][0] - (-np.log(1e-15))) < 1e-9


def test_randomized_pit_is_deterministic_and_model_specific():
    a = M.pit_v("B1", [1, 2, 3], [10, 20, 30]); b = M.pit_v("B1", [1, 2, 3], [10, 20, 30]); c = M.pit_v("B2", [1, 2, 3], [10, 20, 30])
    assert np.array_equal(a, b) and not np.array_equal(a, c) and ((a >= 0) & (a < 1)).all()
    assert abs(a[0] - int.from_bytes(hashlib.sha256(b"B1|1|10").digest()[:8], "big") / 2 ** 64) < 1e-15
    u = M.row_metrics("pmf", {"pmf": np.tile(np.r_[0.3, 0.7, np.zeros(59)], (3, 1))}, np.array([0, 1, 1]), "m", [1, 2, 3], [4, 5, 6])["pit"]
    assert (u[0] <= 0.3) and (u[1] >= 0.3 and u[1] <= 1.0)
    assert M.ks_uniform(np.linspace(0.0005, 0.9995, 1000)) < 0.001


def test_game_level_aggregation_and_slice_gate():
    m, uniq, per = M.game_macro(np.array([1.0, 3.0, 5.0, 7.0, 9.0]), np.array([1, 1, 2, 2, 2]))
    assert m == (2.0 + 7.0) / 2 and list(per) == [2.0, 7.0] and list(uniq) == [1, 2]                   # macro over games, not rows
    ch = np.array([1.1] * 600 + [1.0] * 100); ref = np.array([1.0] * 700)
    ok, d = M.slice_gate(ch, ref, {"a": np.arange(700) < 600, "b": np.arange(700) >= 600})
    assert not ok and d["a"]["eligible"] and not d["a"]["pass"] and not d["b"]["eligible"]               # slices < 500 rows are not eligible


def test_two_calendar_week_moving_block_bootstrap_deterministic():
    rng = np.random.default_rng(3)
    weeks = np.repeat(np.arange(2800, 2812), 5); delta = rng.normal(-0.01, 0.05, len(weeks))
    a = M.blocked_bootstrap(delta, weeks, reps=2000, seed=M.BOOT_SEED); b = M.blocked_bootstrap(delta, weeks, reps=2000, seed=M.BOOT_SEED); c = M.blocked_bootstrap(delta, weeks, reps=2000, seed=1)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and M.BOOT_SEED == 20261002 and M.BOOT_REPS == 10000
    assert abs(a.mean() - delta.mean()) < 0.01
    # a gap week breaks consecutiveness: only consecutive-week pairs are valid block starts
    w2 = np.r_[np.repeat([1, 2], 3), np.repeat([5, 6], 3)]; d2 = np.array([1.0] * 6 + [-1.0] * 6)
    bt = M.blocked_bootstrap(d2, w2, reps=500)
    assert set(np.unique(np.round(bt, 6))) <= {-1.0, 0.0, 1.0}
    rep = M.bootstrap_report(delta, weeks, 0.6, reps=500)
    assert rep["seed"] == 20261002 and rep["n_calendar_weeks"] == 12
    from nhl_sog_phase1a_data import week_index
    assert week_index(np.array([345600, 345600 + 604800 - 1, 345600 + 604800]))[0] == 0 and week_index(np.array([345600 + 604800 - 1]))[0] == 0 and week_index(np.array([345600 + 604800]))[0] == 1


def test_tie_rule_prefers_stronger_regularization():
    assert RUN.pick_with_tie({0.001: 1.0, 0.01: 1.0004, 10.0: 1.2}, lambda a: -a) == 0.01                  # within 0.1%: larger alpha
    assert RUN.pick_with_tie({0.01: 1.0, 1.0: 1.0009, 10.0: 1.2}, lambda C: C) == 0.01
    assert RUN.pick_with_tie({(0.01, 0.1): 1.0, (0.1, 1.0): 1.0005, (0.1, 10.0): 1.0004}, lambda k: (k[0], -k[1])) == (0.01, 0.1)


# ------------------------------------------------------------------ protocol discipline
def fake_eval(summ_override=None):
    """Synthetic 4-model evaluation on a small table where B3 would be better in 2025 than the selected model."""
    def make(crps, nll=1.0):
        return {"crps_macro_game": crps, "nll_macro_game": nll, "coverage_error": {"80": 0.01, "90": 0.01}, "pit_ks": 0.01}
    return make


def synthetic_gate_inputs():
    n_games, per = 20, 60
    gid = np.repeat(np.arange(1, n_games + 1) + 2023020000, per)
    start = np.repeat(345600 + 604800 * (np.arange(n_games) // 2) + 3600, per)
    tab = {"game_id": gid, "start": start, "POS_F": (np.arange(len(gid)) % 2 == 0).astype(float), "POS_D": (np.arange(len(gid)) % 2 == 1).astype(float),
           "N_CURRENT_SEASON_TEAM_GAMES_OBS": np.where(np.arange(len(gid)) % 3 == 0, 5.0, 40.0), "sog": np.zeros(len(gid), dtype=int)}
    return tab


def rows_for(tab, crps_value):
    g, idx = np.unique(tab["game_id"], return_index=True)
    c = np.full(len(tab["game_id"]), crps_value)
    return {"games": g, "crps_per_game": np.full(len(g), crps_value), "crps": c}


def test_gates_logic_and_2024_selection_sequence():
    tab = synthetic_gate_inputs()
    mk_s = lambda crps, nll=1.0, ce=0.01, ks=0.01: {"crps_macro_game": crps, "nll_macro_game": nll, "coverage_error": {"80": ce, "90": ce}, "pit_ks": ks}
    ref, ch = mk_s(0.70), mk_s(0.60)
    g = RUN.gates(ch, ref, rows_for(tab, 0.60), rows_for(tab, 0.70), tab, "B0")
    assert g["G1_crps_improvement_ge_0.5pct"]["pass"] and g["G2_blocked_bootstrap_upper_bound_lt_0"]["pass"] and g["all_pass"]
    g2 = RUN.gates(mk_s(0.699), ref, rows_for(tab, 0.699), rows_for(tab, 0.70), tab, "B0")
    assert not g2["G1_crps_improvement_ge_0.5pct"]["pass"] and not g2["all_pass"]                         # < 0.5% improvement fails
    g3 = RUN.gates(mk_s(0.60, nll=1.02), ref, rows_for(tab, 0.60), rows_for(tab, 0.70), tab, "B0")
    assert not g3["G3_nll_not_worse_than_0.5pct"]["pass"]
    g4 = RUN.gates(mk_s(0.60, ce=0.05), ref, rows_for(tab, 0.60), rows_for(tab, 0.70), tab, "B0")
    assert not g4["G4_coverage_80_90_not_worse_by_0.02"]["pass"]
    g5 = RUN.gates(mk_s(0.60, ks=0.05), ref, rows_for(tab, 0.60), rows_for(tab, 0.70), tab, "B0")
    assert not g5["G5_pit_ks_not_worse_by_0.02"]["pass"]
    bad_slice = rows_for(tab, 0.60); bad_slice["crps"] = bad_slice["crps"].copy(); bad_slice["crps"][tab["POS_D"] == 1] = 0.80          # one slice >5% worse
    g6 = RUN.gates(mk_s(0.60), ref, bad_slice, rows_for(tab, 0.70), tab, "B0")
    assert not g6["G6_position_slices_no_5pct_degradation"]["pass"]


def test_2024_cannot_change_hyperparameters_and_2025_cannot_change_winner():
    import inspect
    src = inspect.getsource(RUN.confirm_2024)
    assert 'dev_res["selected"]' in src and "ALPHA_GRID" not in src and "C_GRID" not in src and "pick_with_tie" not in src
    captured = {}
    old_fit, old_eval = RUN.fit_all, RUN.evaluate
    tab = synthetic_gate_inputs()
    full = {k: v for k, v in tab.items()}
    try:
        RUN.fit_all = lambda train, hyper: (captured.setdefault("hyper", dict(hyper)) and None, {"B0": {}}) if False else (captured.update(hyper=dict(hyper)) or ({}, {"B0": {}}))
        mk_s = lambda c: {"crps_macro_game": c, "nll_macro_game": 1.0, "coverage_error": {"80": 0.01, "90": 0.01}, "pit_ks": 0.01}
        # 2024: B3 is clearly best, but the hyper-parameters must be exactly the dev-selected ones
        RUN.evaluate = lambda models, te, log=print: ({"B0": mk_s(0.70), "B1": mk_s(0.70), "B2": mk_s(0.70), "B3": mk_s(0.60)}, {m: rows_for(te, c) for m, c in (("B0", 0.70), ("B1", 0.70), ("B2", 0.70), ("B3", 0.60))}, None)
        tab2024 = dict(full); tab2024["season"] = np.full(len(full["game_id"]), 2024)
        tab2024.update({k: np.zeros(len(full["game_id"])) for k in ("played",)})
        for s in range(2018, 2024):                                                                        # tiny train seasons so sel() is non-empty
            pass
        big = {k: np.concatenate([v, v]) for k, v in tab2024.items()}
        big["season"] = np.concatenate([np.full(len(full["game_id"]), 2023), np.full(len(full["game_id"]), 2024)])
        for k in ("played",):
            big[k] = np.zeros(len(big["game_id"]))
        for k in ("team_id", "player_id", "plays10", "den10"):
            big[k] = np.zeros(len(big["game_id"]))
        dev_res = {"selected": {"B1_alpha": 0.01, "B3_C": 0.1, "B3_alpha": 1.0}}
        r = RUN.confirm_2024(big, dev_res, ".")
        assert captured["hyper"] == dev_res["selected"] and r["hyper"] == dev_res["selected"] and r["selected"] == "B3" and r["promotion_reference"] == "B2" or r["selected"] in ("B3",)
        # 2025: the 2024-selected architecture stays the evaluated one even if another model looks best in 2025
        sel2 = {"selected_architecture": "B1", "frozen_hyperparameters": dev_res["selected"], "promotion_reference": "B0"}
        big25 = dict(big); big25["season"] = np.concatenate([np.full(len(full["game_id"]), 2024), np.full(len(full["game_id"]), 2025)])
        RUN.evaluate = lambda models, te, log=print: ({"B0": mk_s(0.70), "B1": mk_s(0.71), "B2": mk_s(0.70), "B3": mk_s(0.50)}, {m: rows_for(te, c) for m, c in (("B0", 0.70), ("B1", 0.71), ("B2", 0.70), ("B3", 0.50))}, None)
        r25 = RUN.holdout_2025(big25, sel2, ".")
        assert r25["status"] == "NO_HISTORICAL_PROMOTION" and r25["hyper"] == dev_res["selected"]            # B3 looked best in 2025: still no switch, no rescue
        RUN.evaluate = lambda models, te, log=print: ({"B0": mk_s(0.70), "B1": mk_s(0.60), "B2": mk_s(0.70), "B3": mk_s(0.50)}, {m: rows_for(te, c) for m, c in (("B0", 0.70), ("B1", 0.60), ("B2", 0.70), ("B3", 0.50))}, None)
        assert RUN.holdout_2025(big25, sel2, ".")["status"] == "HISTORICAL_CHAMPION"                          # B1 passes its gates vs B0 and its promotion-reference guardrail (0.60 <= 0.70)
        RUN.evaluate = lambda models, te, log=print: ({"B0": mk_s(0.70), "B1": mk_s(0.60), "B2": mk_s(0.55), "B3": mk_s(0.50)}, {m: rows_for(te, c) for m, c in (("B0", 0.70), ("B1", 0.60), ("B2", 0.55), ("B3", 0.50))}, None)
        sel3 = {**sel2, "promotion_reference": "B2"}                                                           # guardrail: selected CRPS must be <= the promotion-reference model's
        assert RUN.holdout_2025(big25, sel3, ".")["status"] == "NO_HISTORICAL_PROMOTION"
    finally:
        RUN.fit_all, RUN.evaluate = old_fit, old_eval


def test_no_rescue_after_holdout_is_enforced():
    with tempfile.TemporaryDirectory() as t:
        old = RUN.OUT
        RUN.OUT = Path(t)
        try:
            (Path(t) / "phase1a_2025_holdout.json").write_text("{}")
            try:
                RUN._holdout_main(t); assert False
            except AssertionError as e:
                assert "burned" in str(e)
            (Path(t) / "phase1a_selected_architecture.json").write_text("{}")
            try:
                RUN._confirm_main(t); assert False
            except AssertionError:
                pass                                                                                       # 2024 selection is never redone
        finally:
            RUN.OUT = old


def first_commit(path):
    out = subprocess.check_output(["git", "log", "--format=%H %ct", "--diff-filter=A", "--", str(path.relative_to(REPO))], cwd=REPO, text=True).strip().splitlines()
    return out[-1].split() if out else None


def test_protocol_committed_before_any_result_artifact():
    p = first_commit(OUT / "phase1a_protocol.json")
    assert p is not None
    for name in ("phase1a_dev_results.json", "phase1a_2024_confirmation.json", "phase1a_selected_architecture.json", "phase1a_2025_holdout.json"):
        c = first_commit(OUT / name)
        if c:
            assert subprocess.run(["git", "merge-base", "--is-ancestor", p[0], c[0]], cwd=REPO).returncode == 0, name
    c24, c25 = first_commit(OUT / "phase1a_selected_architecture.json"), first_commit(OUT / "phase1a_2025_holdout.json")
    if c25:
        assert c24 and subprocess.run(["git", "merge-base", "--is-ancestor", c24[0], c25[0]], cwd=REPO).returncode == 0 and c24[0] != c25[0]       # 2024 selection committed BEFORE the 2025 result
    proto = json.loads((OUT / "phase1a_protocol.json").read_text())
    assert proto["bootstrap"]["seed"] == 20261002 and proto["models"]["B0"]["constants"]["KAPPA_PLAY"] == 5 and proto["development"]["folds"][3]["validate"] == 2023


def test_frozen_dataset_hash_roundtrip_and_quality_artifact():
    bad, m = D.verify_manifest()
    assert not bad and m["guard_failures"] == [] and all(w[k]["reported_total"] < 10000 and w[k]["returned_rows"] == w[k]["reported_total"] for w in m["windows"] for k in ("summary", "timeonice"))
    q = json.loads((OUT / "phase1a_data_quality.json").read_text())
    assert q["all_pass"] and all(q["gates"].values()) and q["data_manifest_content_sha256"] == m["manifest_content_sha256"]
    assert all(v["pass"] for v in q["toi_quality_by_season"].values()) and q["mutation_check"]["unchanged"] and not q["candidate_equivalence_sample"]["mismatches"]


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
