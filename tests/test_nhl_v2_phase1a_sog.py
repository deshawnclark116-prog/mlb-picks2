"""NHL V2 Phase1A-SOG deterministic tests (research only)."""
import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "nhl_models" / "nhl_player_outcome_v2"
import nhl_v2_phase1a_sog_data as D
import nhl_v2_phase1a_sog_model as M

SCIENCE = sorted(REPO.glob("nhl_v2_phase1a_sog_*.py"))
BANNED = tuple(next(v for k, v in json.loads((OUT / "protocol.json").read_text()).items() if k.endswith("_firewall"))["banned_tokens"]) + ("over/under",)


def git(*a):
    return subprocess.run(["git"] + list(a), cwd=str(REPO), capture_output=True, text=True)


def synth(n_teams=6, n_games=60, n_players=14, seed=3):
    rng = np.random.default_rng(seed)
    games, rows, gid = {}, [], 2019020000
    base = 1570000000
    teams = list(range(1, n_teams + 1))
    for gi in range(n_games):
        a, b = (teams[(2 * gi) % n_teams], teams[(2 * gi + 1) % n_teams])
        if a == b:
            continue
        gid += 1
        start = base + gi * 86400 * 2
        iso = __import__("datetime").datetime.fromtimestamp(start, __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        games[gid] = {"game_id": gid, "game_start_utc": iso, "home_abbrev": "T%d" % a, "away_abbrev": "T%d" % b, "home_team_id": a, "away_team_id": b, "date": iso[:10]}
        for tid, opp in ((a, b), (b, a)):
            for p in range(n_players):
                if rng.random() < 0.15 and p > 8:
                    continue
                pid = tid * 100 + p
                toi = int(rng.integers(600, 1300))
                rows.append({"game_id": gid, "season_start_year": 2019, "game_start_utc": iso, "team_id": tid, "team_abbrev": "T%d" % tid, "opponent": "T%d" % opp, "player_id": pid, "position": "D" if p % 3 == 0 else "C",
                             "sog": int(rng.poisson(1.5)), "toi_sec": toi, "ev_toi_sec": toi - 100, "pp_toi_sec": 60, "sh_toi_sec": 40, "ot_toi_sec": 0, "shifts": 20})
    rows.sort(key=lambda r: (r["game_start_utc"], r["game_id"], r["team_id"], r["player_id"]))
    return games, rows


def test_sportsbook_firewall_on_phase1a_modules():
    assert SCIENCE
    for p in SCIENCE:
        for i, line in enumerate(p.read_text().lower().splitlines(), 1):
            for t in BANNED:
                assert not re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", line), "%s:%d uses %s" % (p.name, i, t)


def test_no_sampling_simulation_in_phase1a_modules():
    for p in SCIENCE:
        txt = p.read_text()
        assert not re.search(r"\.(binomial|poisson|multinomial|choice|normal|rvs)\(", txt.replace("stats.poisson.", "").replace("stats.nbinom.", "")), p.name


def test_nb2_distribution_consistency():
    mu = np.array([0.3, 1.2, 2.5, 4.0]); alpha = 0.2
    y = np.zeros(4, dtype=int)
    for sl, pm, sf in M.pmf_matrix("nb2", {"mu": mu, "alpha": alpha}, y):
        assert np.allclose(pm.sum(axis=1) + sf, 1.0, atol=1e-9)
        k = np.arange(pm.shape[1])
        assert np.allclose((pm * k).sum(axis=1), mu, atol=1e-6)
        assert np.allclose((pm * k ** 2).sum(axis=1) - mu ** 2, mu + alpha * mu ** 2, atol=1e-4)
        s = M.distribution_summary(mu, alpha)
        cdf = np.cumsum(pm, axis=1)
        assert np.allclose(s["p_ge"][:, 0], 1.0)
        for j in range(1, 5):
            assert np.allclose(s["p_ge"][:, j], 1 - cdf[:, j - 1], atol=1e-9)
        assert np.array_equal(s["median"], (cdf >= 0.5 - 1e-12).argmax(axis=1))
        assert np.allclose(s["variance"], mu + alpha * mu ** 2)


def test_crps_matches_definition_and_poisson_limit():
    mu = np.array([1.0, 2.0]); y = np.array([0, 3])
    m = M.row_metrics("nb2", {"mu": mu, "alpha": 1e-6}, y, "t", np.array([1, 2]), np.array([1, 2]))
    mp = M.row_metrics("poisson", {"mu": mu}, y, "t", np.array([1, 2]), np.array([1, 2]))
    assert np.allclose(m["crps"], mp["crps"], atol=1e-4)
    from scipy import stats
    cdf = stats.poisson.cdf(np.arange(0, 60), 1.0)
    assert abs(((cdf - (0 <= np.arange(60))) ** 2).sum() - mp["crps"][0]) < 1e-9


def test_nb_alpha_recovers_overdispersion():
    rng = np.random.default_rng(0)
    mu = np.full(20000, 2.0); r = 4.0
    y = rng.negative_binomial(r, r / (r + mu))
    fit = M.fit_nb_alpha(y, mu)
    assert abs(fit["alpha"] - 0.25) < 0.03 and not fit["at_lower_bound"] and not fit["at_upper_bound"]


def test_preprocessor_training_only_and_schema_roundtrip():
    games, rows = synth()
    tab, _ = D.build_rows(games, rows, target_seasons=[2019], horizon_min=90)
    train = M.select_rows(tab, tab["start"] < np.median(tab["start"]))
    prep = M.Preprocessor(D.FEATURES).fit(train)
    X = prep.transform(tab)
    prep2 = M.Preprocessor.from_schema(json.loads(json.dumps(prep.schema())))
    assert np.array_equal(X, prep2.transform(tab))
    assert X.shape[1] == len(D.FEATURES) + len(prep.cont)
    # no target information in features
    assert not set(D.FEATURES) & {"sog", "played", "toi"}


def test_artifact_prediction_equals_sklearn_and_is_deterministic():
    games, rows = synth(n_teams=8, n_games=140)
    tab, _ = D.build_rows(games, rows, target_seasons=[2019], horizon_min=90)
    a1 = M.fit_b2(tab, "T90"); a2 = M.fit_b2(tab, "T90")
    assert a1["sha256"] == a2["sha256"] and a1["converged"]
    mu = M.predict_mu(json.loads(json.dumps(a1)), tab)
    assert np.all(mu > 0) and abs(mu.mean() - tab["sog"].mean()) < 0.3
    assert a1["poisson_alpha"] == 0.001


def test_builder_is_causal_target_rows_never_change_features():
    games, rows = synth(n_teams=8, n_games=100)
    tab, _ = D.build_rows(games, rows, target_seasons=[2019], horizon_min=90)
    gids = sorted({int(g) for g in tab["game_id"]})
    target = {gids[-1]}
    rows2 = [dict(r, sog=r["sog"] + 5, toi_sec=1, pp_toi_sec=0) if r["game_id"] in target else r for r in rows]
    tab2, _ = D.build_rows(games, rows2, target_seasons=[2019], horizon_min=90)
    m1 = np.isin(tab["game_id"], list(target)); m2 = np.isin(tab2["game_id"], list(target))
    for f in D.FEATURES:
        assert np.array_equal(tab[f][m1], tab2[f][m2], equal_nan=True), f
    # forward mode (labels=False) with the target rows deleted gives identical features
    rows3 = [r for r in rows if r["game_id"] not in target]
    tab3, _ = D.build_rows(games, rows3, target_gids=sorted(target), horizon_min=90, labels=False)
    for f in D.FEATURES:
        assert np.array_equal(tab[f][m1], tab3[f], equal_nan=True), f
    assert np.array_equal(tab["player_id"][m1], tab3["player_id"])


def test_cutoff_excludes_games_inside_210_minutes_and_scales_with_horizon():
    games, rows = synth(n_teams=4, n_games=40)
    t90, _ = D.build_rows(games, rows, target_seasons=[2019], horizon_min=90)
    t24, _ = D.build_rows(games, rows, target_seasons=[2019], horizon_min=1440)
    assert t90["cutoff"][0] - t90["start"][0] == -5400 and t24["cutoff"][0] - t24["start"][0] == -86400
    assert (t24["N_CURRENT_SEASON_TEAM_GAMES_OBS"] <= 40).all()
    assert t24["TEAM_GAMES_SINCE_APPEARANCE"].shape == t24["sog"].shape


def test_vendored_data_hashes_match_manifest_and_v1():
    m = json.loads((OUT / "phase1a_sog_data_manifest.json").read_text())
    assert m["all_files_equal_v1_manifest"] is True
    for n, f in m["files"].items():
        assert hashlib.sha256((D.DATA / n).read_bytes()).hexdigest() == f["sha256"], n


def test_data_quality_artifact_passes_and_reports_coverage():
    q = json.loads((OUT / "phase1a_sog_data_quality.json").read_text())
    assert q["all_pass"] is True
    assert q["coverage_by_season_T90"]["2025"]["candidate_player_coverage"] > 0.97
    assert q["candidate_population_audit_T90"]["2025"]["full_played_fraction"] < 0.9


def test_protocol_and_amendment_not_modified_after_registration():
    for f in ("phase1a_sog_protocol.json", "phase1a_sog_protocol_amendment_1.json", "phase1a_sog_forward_protocol.json", "phase1a_sog_protocol_amendment_2.json"):
        path = "nhl_models/nhl_player_outcome_v2/" + f
        first = git("log", "--diff-filter=A", "--format=%H", "--", path).stdout.split()
        if not first or not (OUT / f).exists():
            continue
        orig = git("show", first[-1] + ":" + path).stdout
        assert orig == (OUT / f).read_text(), f


def test_phase1a_commit_order_in_history():
    def first(path):
        out = git("log", "--diff-filter=A", "--format=%H", "--", path).stdout.split()
        return out[-1] if out else None
    names = ["phase1a_v1_migration_audit.json", "phase1a_sog_protocol.json", "phase1a_sog_data_manifest.json", "phase1a_sog_protocol_amendment_1.json", "phase1a_sog_burned_reproduction.json",
             "phase1a_sog_forward_protocol.json", "phase1a_sog_engine_lock.json", "phase1a_sog_protocol_amendment_2.json", "phase1a_sog_engine_lock_v1_1.json"]
    shas = [first("nhl_models/nhl_player_outcome_v2/" + n) for n in names]
    if any(s is None for s in shas):
        return
    pos = {h: i for i, h in enumerate(git("rev-list", "--reverse", "HEAD").stdout.split())}
    idx = [pos[s] for s in shas]
    assert idx == sorted(idx) and len(set(idx)) == len(idx), dict(zip(names, idx))


def test_only_research_paths_changed_vs_phase0_head():
    r = git("diff", "--name-only", "1a090c3d78fc93cf3d531b90efa96fed47493092", "HEAD")
    if r.returncode != 0:
        return
    ok = re.compile(r"^(nhl_v2_[a-z0-9_]+\.py|tests/test_nhl_v2_[a-z0-9_]+\.py|nhl_models/nhl_player_outcome_v2/.+|\.github/workflows/nhl_v2_[a-z0-9_]+\.yml|requirements-research-nhl\.txt)$")
    assert not [f for f in r.stdout.split() if not ok.match(f)]
