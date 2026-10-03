"""
Phase 1 hardening / Phase 1B invariant tests (research, shadow). Standalone and pytest-compatible.

  python -u tests/test_nfl_phase1b_invariants.py

Phase 1B unit invariants (synthetic): valid probabilities, bins partition the yard grid and support negative yards, two-process
(ordinary + explosive) reconstruction, catch/air/YAC and completion/air/YAC reconstruction of yards, tail integration, CRN determinism,
inactive -> zero, Tilt recovery, Phase 1A samples consumed without recomputation, no sportsbook / participation source in Phase 1B code.
Data invariants (need --data-dir): perturbing week-w tallies leaves week-w features unchanged; fits only in allowed periods.

Hardening: protocol/architecture consistency, historical-proxy wording, forward snapshot immutability / hash /
retrieval<=forecast time / append-only manifest / T24-vs-T90 independence.
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_phase1_snapshots as SN  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
UTC = timezone.utc


def test_protocol_availability_model_matches_selected_architecture():
    proto = json.loads((REPO / "nfl_models/nfl_player_outcome_phase1_protocol.json").read_text())
    sel = json.loads((REPO / "nfl_models/nfl_player_outcome_phase1a/selected_architecture.json").read_text())
    av = sel["availability"]
    assert all("xgb" in av[h][s] for h in ("T24", "T90") for s in ("offense", "defense"))
    model = proto["availability_handling"]["model"].lower()
    assert model.split(",")[1].strip().startswith("gradient boosting")


def test_historical_injury_wording_is_proxy():
    for f in ("nfl_phase1_data.py", "nfl_models/nfl_player_outcome_phase1a/README.md"):
        t = (REPO / f).read_text()
        assert "final-weekly-report proxy" in t
    proto = json.loads((REPO / "nfl_models/nfl_player_outcome_phase1_protocol.json").read_text())
    assert {"E", "F"} <= {a["id"] for a in proto["amendments_v1_1"]}


def _fetcher(b):
    return lambda: b


def test_snapshot_hash_immutable_and_never_overwritten():
    with tempfile.TemporaryDirectory() as d:
        ko = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
        now = ko - timedelta(hours=30)
        rec = SN.take_snapshot("injuries", _fetcher(b"a,b\n1,2\n"), ko, "T24", root=d, now=now)
        assert rec["sha256"] == SN.sha256_hex(b"a,b\n1,2\n")
        assert SN.load_snapshot(rec, d) == b"a,b\n1,2\n"
        p = Path(d) / rec["path"]
        try:
            open(p, "wb").write(b"x"); ok_write = False
        except PermissionError:
            ok_write = True
        assert ok_write or os.geteuid() == 0
        try:
            SN.take_snapshot("injuries", _fetcher(b"a,b\n1,2\n"), ko, "T24", root=d, now=now); assert False
        except SN.SnapshotError:
            pass
        os.chmod(p, 0o644); p.write_bytes(b"tampered")
        try:
            SN.load_snapshot(rec, d); assert False
        except SN.SnapshotError:
            pass
        assert any("hash mismatch" in x for x in SN.verify_all(d))


def test_retrieval_after_forecast_time_rejected_and_horizons_independent():
    with tempfile.TemporaryDirectory() as d:
        ko = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
        try:
            SN.take_snapshot("injuries", _fetcher(b"x"), ko, "T24", root=d, now=ko - timedelta(hours=23)); assert False
        except SN.SnapshotError:
            pass
        r24 = SN.take_snapshot("injuries", _fetcher(b"v24"), ko, "T24", root=d, now=ko - timedelta(hours=25))
        # a T90 forecast must not silently reuse the T24 snapshot: it needs its own download
        try:
            SN.usable_snapshot("injuries", "T90", ko, d); assert False
        except SN.SnapshotError:
            pass
        r90 = SN.take_snapshot("injuries", _fetcher(b"v90"), ko, "T90", root=d, now=ko - timedelta(minutes=100))
        assert r24["path"] != r90["path"] and r24["sha256"] != r90["sha256"]
        b24, pv = SN.forecast_inputs(["injuries"], ko, "T24", d)
        b90, _ = SN.forecast_inputs(["injuries"], ko, "T90", d)
        assert b24["injuries"] == b"v24" and b90["injuries"] == b"v90"
        assert SN.parse_iso(pv["injuries"]["retrieval_ts"]) <= SN.parse_iso(pv["injuries"]["forecast_ts"])
        assert SN.verify_all(d) == []
        lines = SN.read_manifest(d)
        assert [l["horizon"] for l in lines] == ["T24", "T90"]


def test_forecast_ignores_late_snapshot():
    with tempfile.TemporaryDirectory() as d:
        ko = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
        SN.take_snapshot("injuries", _fetcher(b"early"), ko, "T90", root=d, now=ko - timedelta(minutes=120))
        # a later manifest line retrieved after the forecast time (written manually) must never be selected
        rec = dict(SN.read_manifest(d)[0]); rec["retrieval_ts"] = SN.iso(ko - timedelta(minutes=10)); rec["path"] = "injuries/late"
        (Path(d) / "manifest.jsonl").open("a").write(json.dumps(rec) + "\n")
        b, _ = SN.forecast_inputs(["injuries"], ko, "T90", d)
        assert b["injuries"] == b"early"


import numpy as np  # noqa: E402
import nfl_phase1_efficiency as F  # noqa: E402
import nfl_phase1_receiving_efficiency as RC  # noqa: E402
import nfl_phase1_rushing_efficiency as RU  # noqa: E402
import nfl_phase1b_data as B  # noqa: E402
import nfl_phase1b_evaluate as EV  # noqa: E402


def test_bins_partition_grid_and_support_negative_yards():
    for M in (B.M_R, B.M_A, B.M_Y):
        assert (M.sum(1) == 1).all()
    assert RU.BIN_VALS[0] < 0 and (RU.BIN_VALS < 0).sum() >= 4          # lump (-10..-4) and singles -3, -2, -1
    assert B.gclip(-25) == 0 and B.gclip(150) == B.G - 1


def test_hier_pmf_and_tilt_valid_probabilities_and_negative_mass():
    rng = np.random.default_rng(0)
    n, K = 50, B.K_R
    cl = rng.poisson(3, (1, K)).astype(float) + 1
    p = F.hier_pmf(rng.poisson(1, (n, K)).astype(float), np.tile(cl, (n, 1)) * 10, np.tile(cl, (n, 1)) * 100, 20.0, 50.0)
    assert np.allclose(p.sum(1), 1) and (p > 0).all()
    assert (p[:, RU.BIN_VALS < 0].sum(1) > 0).all()
    X = rng.normal(size=(n, 3)); C_ = np.array([rng.multinomial(10, q) for q in p])
    m = F.Tilt(RU.PHI_SINGLE, l2=10.0).fit(C_, np.log(p), X)
    P = m.predict(np.log(p), X)
    assert np.allclose(P.sum(1), 1) and (P > 0).all() and (P[:, RU.BIN_VALS < 0].sum(1) > 0).all()


def test_tilt_recovers_known_effect():
    rng = np.random.default_rng(0)
    base = np.array([.1, .3, .3, .15, .1, .05]); phi = np.array([[0], [0], [0], [0], [1], [2]], float)
    X = rng.normal(size=(3000, 1))
    P = np.exp(np.log(base)[None] + 0.5 * X * phi[:, 0][None]); P /= P.sum(1, keepdims=True)
    Cn = np.array([rng.multinomial(20, q) for q in P])
    m = F.Tilt(phi, l2=1.0).fit(Cn, np.log(np.tile(base, (3000, 1))), X)
    assert abs(m.W[1, 0] / 1.0 - 0.5 * X.std()) < 0.05


def test_two_process_reconstructs_rush_pmf_and_mean():
    P0 = np.random.default_rng(1).dirichlet(np.ones(B.K_R), 5)
    body, tail = RU.BIN_VALS < 20, RU.BIN_VALS >= 20
    h = P0[:, tail].sum(1)
    Pb = P0[:, body] / P0[:, body].sum(1, keepdims=True); Pt = P0[:, tail] / P0[:, tail].sum(1, keepdims=True)
    P = np.zeros_like(P0); P[:, body] = (1 - h)[:, None] * Pb; P[:, tail] = h[:, None] * Pt
    assert np.allclose(P, P0)
    m = (1 - h) * (Pb @ RU.BIN_VALS[body]) + h * (Pt @ RU.BIN_VALS[tail])
    assert np.allclose(m, P0 @ RU.BIN_VALS)


def test_chain_reconstructs_receiving_yards_and_zero_when_inactive():
    rng = np.random.default_rng(3)
    Pa = rng.dirichlet(np.ones(B.K_A)); Py = rng.dirichlet(np.ones(B.K_Y), 4)
    ns = np.array([5, 0, 12])
    y1, c1 = EV.chain(Pa, np.ones(4), Py, ns, np.random.default_rng(9))
    y0, c0 = EV.chain(Pa, np.zeros(4), Py, ns, np.random.default_rng(9))
    assert (c1 == ns).all() and (c0 == 0).all() and (y0 == 0).all() and y1[1] == 0 and c1[1] == 0   # zero targets -> zero everything
    # yards == air + yac on every catch, reconstructed independently from the same uniforms
    r = np.random.default_rng(9); tot = int(ns.sum())
    u1, u2, u3 = r.random(tot), r.random(tot), r.random(tot)
    a = EV.draw_bins(Pa, u1); bk = RC.AIR_BUCKET[a]
    yb = np.zeros(tot, int)
    for b in range(4):
        m = bk == b
        yb[m] = EV.draw_bins(Py[b], u3[m])
    per = RC.AIR_V[a] + RC.YAC_V[yb]
    seg = np.repeat(np.arange(len(ns)), ns)
    assert np.allclose(np.bincount(seg, weights=per, minlength=len(ns)), y1)


def test_chain_group_probs_sum_to_one():
    rng = np.random.default_rng(4)
    Pa = rng.dirichlet(np.ones(B.K_A), 6); Py = rng.dirichlet(np.ones(B.K_Y), (6, 4)); c = rng.uniform(0.3, 0.9, (6, 4))
    g = RC.chain_group_probs(Pa, c, Py)
    assert np.allclose(g.sum(1), 1) and (g >= -1e-12).all()


def test_ordinal_crps_and_sample_crps_integrate():
    vals = np.arange(5.0)
    P = np.array([[0, 0, 1.0, 0, 0]]); Cn = np.array([[0, 0, 0, 1.0, 0]])
    assert abs(F.ordinal_crps_records(P, Cn, vals)[0] - 1.0) < 1e-9                    # point mass at 2 vs outcome 3 -> |2-3|
    S = np.random.default_rng(0).normal(size=(3, 4000))
    assert np.allclose(F.crps_samples(S, np.zeros(3)), 0.2337, atol=0.02)               # N(0,1) at 0: (sqrt(2)-1)/sqrt(pi)


def test_crn_determinism_of_simulation_helpers():
    p = np.random.default_rng(1).dirichlet(np.ones(B.K_R)); ns = np.array([3, 0, 7, 1])
    a = EV.sum_rush(p, RU.BIN_VALS, ns, EV.rng_for(("k", 1), "rush")); b = EV.sum_rush(p, RU.BIN_VALS, ns, EV.rng_for(("k", 1), "rush"))
    assert np.array_equal(a, b) and a[1] == 0


def test_phase1a_samples_consumed_not_recomputed_and_no_forbidden_sources():
    src = (REPO / "nfl_phase1b_evaluate.py").read_text()
    assert "run_cfg(" not in src and "sample_alloc(" not in src and "tune_alpha(" not in src
    for f in ("nfl_phase1b_data.py", "nfl_phase1_efficiency.py", "nfl_phase1_event_models.py", "nfl_phase1_rushing_efficiency.py", "nfl_phase1_receiving_efficiency.py",
              "nfl_phase1_passing_efficiency.py", "nfl_phase1_defense_events.py", "nfl_phase1b_evaluate.py"):
        t = (REPO / f).read_text().lower()
        code = "\n".join(l for l in t.splitlines() if not l.strip().startswith(("#", '"', "'")))
        for bad in ("fanduel", "moneyline", "spread_line", "total_line", "over_odds", "under_odds", "offense_players", "defense_players", "participation_"):
            assert bad not in code, (f, bad)


def test_extras_records_are_inactive_zero_count():
    rows = [{"s": 2025, "w": 3, "key": ("a",), "pg": "RB", "n": 0.0, "cnt": {"y": np.zeros(B.K_R)}, "base": {"pl": np.zeros((3, B.K_R))}, "fam": {"team": [1.0]}, "extra": True},
            {"s": 2025, "w": 3, "key": ("b",), "pg": "RB", "n": 5.0, "cnt": {"y": np.ones(B.K_R)}, "base": {"pl": np.zeros((3, B.K_R))}, "fam": {"team": [1.0]}}]
    R = B.Records(rows)
    assert list(R.active) == [False, True]
    tr, va, dv = F.masks(R)
    assert not dv[0] and dv[1]


def run_data_tests(data_dir, report_path=None):
    """Perturbation / audit tests on real data (about a minute)."""
    import nfl_phase1_data as P1
    D = P1.Data(data_dir)
    T = B.PlayTallies(data_dir, [2024, 2025])
    class StubPD:
        def def_profile(self, s, w, team):
            return {}
    inj = B.build_inj_index(D)
    R1 = B.build_records(D, T, StubPD(), inj)["rush"]
    # pick a rusher in 2025 week 8 and perturb that week's tallies (player history, team context and opponent context)
    k = next(k for k in T.rush if k[0] == 2025 and k[1] == 8)
    team, opp = T.tm[k]
    T.rush[k][B.gclip(45)] += 6
    T.rush_ev[k][0] += 3; T.rush_ev[k][1] += 3
    for key in ((2025, 8, team), ("D", 2025, 8, opp)):
        T.rush_team[key][0] += 6; T.rush_team[key][1] += 270; T.rush_team[key][3] += 6
    R2 = B.build_records(D, T, StubPD(), inj)["rush"]
    i1 = {k_: i for i, k_ in enumerate(R1.key)}; i2 = {k_: i for i, k_ in enumerate(R2.key)}
    week8 = [k_ for k_ in R1.key if k_[0] == 2025 and k_[1] == 8]
    same = all(np.array_equal(R1.base["pl"][i1[k_]], R2.base["pl"][i2[k_]]) and np.allclose(R1.fam["opp"][i1[k_]], R2.fam["opp"][i2[k_]], equal_nan=True)
               and np.allclose(R1.fam["team"][i1[k_]], R2.fam["team"][i2[k_]], equal_nan=True) for k_ in week8)
    assert same, "week-8 features changed when week-8 outcomes were perturbed (leak)"
    later = [k_ for k_ in R1.key if k_[0] == 2025 and k_[1] == 9 and k_[2] == k[2]]
    assert not later or not np.array_equal(R1.base["pl"][i1[later[0]]], R2.base["pl"][i2[later[0]]]), "week-9 history should reflect week-8 outcomes"
    if report_path and Path(report_path).exists():
        rep = json.loads(Path(report_path).read_text())
        bad = [a for a in rep["fit_audit"] if a["label"].startswith("phase1b") and any(w.endswith(":dev") for w in a["windows"])]
        assert not bad, bad
    print("PASS data invariants (perturbation, fit audit)")


if __name__ == "__main__":
    if "--data-dir" in sys.argv:
        run_data_tests(sys.argv[sys.argv.index("--data-dir") + 1], sys.argv[sys.argv.index("--report") + 1] if "--report" in sys.argv else None)
        sys.exit(0)
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                f(); print("PASS", n)
            except Exception as e:  # noqa
                fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
