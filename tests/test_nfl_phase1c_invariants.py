"""
Phase 1C invariant tests (research, shadow). Standalone and pytest-compatible.

  python -u tests/test_nfl_phase1c_invariants.py

Synthetic-game invariants for the joint simulator (event coherence, reconciliation, red-zone nesting, inactive => zero, determinism),
append-only store semantics (idempotency, conflict = hard error, partial write, corrupt batch), and source scans.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_phase1_store as ST  # noqa: E402
import nfl_phase1b_data as B  # noqa: E402
import nfl_phase1c_sim as SM  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONSTS = {"no_target_rate_of_nonsack_attempts": 0.04, "scramble_share_of_qb_carries": 0.5, "league_sack_rate_per_dropback": 0.066, "half_sack_probability": 0.11,
          "tackle_credits_per_eligible_play": 1.26, "rz_latent_gamma_shape": 5.0, "gl_share_of_rz_rushes": 0.3, "gl_share_of_rz_targets": 0.2,
          "rz_share_of_rushes": 0.17, "rz_share_of_targets": 0.13, "scramble_per_dropback": 0.04}


def synthetic(seed=0):
    r = np.random.default_rng(seed)
    n_c, n_t, n_q, n_d = 3, 4, 2, 6
    ids_c, ids_t, ids_q, ids_d = ["r1", "r2", "q1"], ["w1", "w2", "w3", "r1"], ["q1", "q2"], [f"d{i}" for i in range(n_d)]
    g = {"key": (2025, 3, "AAA"), "types": {
        "carry": {"ids": ids_c, "P1": np.array([.5, .25, .1]), "pact24": np.array([.95, .9, .9]), "pact90": np.array([1., 1., 0.]), "mu": 25.0, "k": 40.0},
        "target": {"ids": ids_t, "P1": np.array([.3, .25, .2, .1]), "pact24": np.array([.9, .9, .8, .95]), "pact90": np.array([1., 1., 1., 1.]), "mu": 33.0, "k": 60.0},
        "qb_att": {"ids": ids_q, "P1": np.array([.8, .1]), "pact24": np.array([.97, .5]), "pact90": np.array([1., 1.]), "mu": 37.0, "k": 60.0},
        "rz_carry": {"ids": ids_c, "P1": np.array([.6, .2, .05]), "mu": 4.0, "k": 20.0}, "rz_target": {"ids": ids_t, "P1": np.array([.3, .25, .2, .1]), "mu": 4.0, "k": 20.0},
        "def_snap": {"ids": ids_d, "P1": np.full(n_d, 0.8), "pact24": np.full(n_d, .9), "pact90": np.full(n_d, .9), "mu": 60.0, "k": 50.0, "share_sd": 0.1}},
        "_meta": {"carry": {"other": 0.08, "alpha": 20.0}, "target": {"other": 0.03, "alpha": 40.0}, "qb_att": {"other": 0.15, "alpha": 5.0}}}
    dirich = lambda K, n=None: r.dirichlet(np.ones(K) * 2, size=n)
    n_r, n_e, n_p = len(ids_c), len(ids_t), len(ids_q)
    eff = {"rush": dirich(B.K_R, n_r), "rush_td": (np.full(n_r, .04), np.full(n_r, .005)), "air": dirich(B.K_A, n_e), "catch": r.uniform(.4, .8, (n_e, 4)),
           "yac": dirich(B.K_Y, (n_e, 4)), "rec_td": (np.full(n_e, .03), np.full(n_e, .004)), "qb_sack": np.full(n_p, .066), "qb_int": np.full(n_p, .02),
           "qb_comp": r.uniform(.5, .8, (n_p, 4)), "qb_comp_league": r.uniform(.5, .8, (n_p, 4)),
           "def_rate": {t: np.full(n_d, v) for t, v in (("tackles", .09), ("sacks", .004), ("interceptions", .001))}}
    idx = {"rush": {(2025, 3, i): k for k, i in enumerate(ids_c)}, "rec": {(2025, 3, i): k for k, i in enumerate(ids_t)}, "pass": {(2025, 3, i): k for k, i in enumerate(ids_q)}}
    defaults = {"rush": eff["rush"][0], "rush_td": (.04, .005), "air": eff["air"][0], "catch": eff["catch"][0], "yac": eff["yac"][0], "rec_td": (.03, .004), "qb_sack": .066, "qb_int": .02}
    return g, eff, idx, defaults


def run(seed=1, horizon="T24", N=400, trace=False):
    g, eff, idx, defaults = synthetic()
    C = SM.Const(CONSTS)
    ev = SM.EffView(eff, idx, 2025, 3, g["types"]["carry"]["ids"], g["types"]["target"]["ids"], g["types"]["qb_att"]["ids"], defaults)
    rng = SM.rng_key(seed, "off")
    off = SM.simulate_offense(g, ev, C, N, rng, horizon, trace=trace)
    de = SM.simulate_defense(g, off, {t: eff["def_rate"][t] for t in eff["def_rate"]}, C, N, SM.rng_key(seed, "def"), horizon)
    return g, off, de


def test_qb_passing_yards_and_tds_equal_receiver_events():
    g, off, de = run()
    assert np.array_equal(off["qb"]["yds"].sum(1), off["rc"]["yds"].sum(1))
    assert np.array_equal(off["qb"]["td"].sum(1), off["rc"]["td"].sum(1))
    assert np.array_equal(off["qb"]["cmp"].sum(1), off["rc"]["rec"].sum(1))


def test_interception_sack_and_completion_are_mutually_exclusive():
    g, off, de = run()
    att, cmp_, it = off["qb"]["att"], off["qb"]["cmp"], off["qb"]["int"]
    assert (cmp_ + it <= att).all()                                   # a completion is never an interception; both are attempts
    # dropbacks = attempts + sacks + scrambles for every QB (sacks and scrambles are never attempts)
    d_q = off["dropbacks_q"]
    lhs = att[:, :-1] + off["sacks_q"] + off["scr_q"]
    assert ((lhs == d_q) | (off["adj_q"] > 0)).all()
    assert (off["rc"]["rec"].sum(1) <= off["Tt"]).all()               # receptions only happen on targets
    assert np.array_equal(att.sum(1), off["A"])


def test_opportunity_reconciliation_and_nonnegative():
    g, off, de = run()
    assert np.array_equal(off["rush_att"].sum(1) + off["rush_att_out"], off["Rn"])
    assert np.array_equal(off["tgt_named"].sum(1) + off["tgt_out"] + off["NT"], off["A"])
    assert np.array_equal(off["rc"]["tgt"].sum(1), off["Tt"])
    assert (off["rush_att"] >= 0).all() and (off["tgt_named"] >= 0).all() and (off["qb"]["att"] >= 0).all()
    assert np.array_equal(off["dropbacks_q"].sum(1) + off["dropbacks_out"], off["Dn"])


def test_red_zone_nested_inside_totals():
    g, off, de = run()
    assert (off["rz_rush"] <= off["rush_att"]).all() and (off["rz_rush"].sum(1) <= off["Rn"]).all()
    assert (off["rz_tgt"] <= off["tgt_named"]).all() and (off["rz_tgt"].sum(1) <= off["Tt"]).all()
    assert (off["rz_tgt_out"] <= off["tgt_out"]).all()


def test_inactive_players_have_zero_opportunities_and_stats():
    g, off, de = run(horizon="T90")                                   # pact90 = 0 for the third rusher (a QB); QB carries must be zero there
    assert (off["rush_att"][:, 2] == 0).all() and (off["rush_yds"][:, 2] == 0).all() and (off["rush_td"][:, 2] == 0).all()
    g2, off2, _ = run(horizon="T24")
    inactive = ~off2["act_c"][:, 0]
    assert (off2["rush_att"][inactive, 0] == 0).all()
    inactive_q = ~off2["act_q"][:, 1]
    assert (off2["qb"]["att"][inactive_q, 1] == 0).all() and (off2["dropbacks_q"][inactive_q, 1] == 0).all()


def test_defence_events_reconcile_with_offensive_events():
    g, off, de = run()
    assert np.allclose(de["sacks"].sum(1), off["sacks_total"])         # half-sacks are 0.5 + 0.5, totals reconcile
    assert np.array_equal(de["interceptions"].sum(1), off["tot_int"])
    assert (de["tackles"] >= 0).all() and (de["tackles"].sum(1) == de["credits"]).all()
    assert (de["credits"] >= (off["Rn"] - off["rush_td_total"]) + (off["tot_cmp"] - off["tot_ptd"]) + off["sacks_total"]).all()


def test_same_seed_same_result_and_different_seed_differs():
    _, a, da = run(seed=5); _, b, db = run(seed=5); _, c, _ = run(seed=6)
    assert np.array_equal(a["rush_yds"], b["rush_yds"]) and np.array_equal(da["tackles"], db["tackles"])
    assert not np.array_equal(a["rush_yds"], c["rush_yds"])


def test_touchdown_yardage_bounded_by_field_position_at_event_level():
    g, off, de = run(N=4000, trace=True)
    r = off["_rush_events"]
    assert (r["yards"][r["td"] & r["rz"]] >= 1).all() and (r["yards"][r["td"] & r["rz"]] <= 20).all()
    assert (r["yards"][r["td"] & ~r["rz"]] >= 21).all()                   # a TD from outside the red zone needs at least 21 yards
    assert (r["yards"][~r["td"] & r["rz"]] <= 19).all()                   # a red-zone carry that is not a TD stays short of the goal line
    p = off["_pass_events"]
    assert not (p["int"] & p["comp"]).any() and not (p["comp"] & ~p["targeted"]).any() and not (p["td"] & ~p["comp"]).any()
    assert (p["yards"][p["td"] & p["rz"]] >= 1).all() and (p["yards"][p["td"] & p["rz"]] <= 20).all() and (p["yards"][p["td"] & ~p["rz"]] >= 21).all()
    assert (p["yards"][~p["comp"]] == 0).all()                            # no yards without a completion


def test_goal_line_nested_and_probabilities_valid_and_quantiles_monotonic():
    import nfl_phase1c_metrics as MT
    g, off, de = run()
    assert (off["gl_rush"] <= off["rz_rush"]).all() and (off["gl_tgt"] <= off["rz_tgt"]).all()
    S = off["rc"]["yds"][:, :4].T.astype(float)
    sm = MT.summary_row(S)
    qs = np.column_stack([sm[f"p{int(round(p * 100)):02d}"] for p in MT.QS])
    assert (np.diff(qs, axis=1) >= -1e-9).all()
    p_ev = (off["rc"]["td"][:, :4] >= 1).mean(0)
    assert ((p_ev >= 0) & (p_ev <= 1)).all()


def test_engine_modules_do_not_import_sportsbook_bearing_v2_v3():
    for f in ("nfl_phase1c_sim.py", "nfl_phase1c_fit.py", "nfl_phase1c_script.py", "nfl_phase1_forecast.py", "nfl_phase1_score.py", "nfl_phase1_store.py", "nfl_phase1_board.py",
              "nfl_phase1c_evaluate.py", "nfl_phase1b_data.py", "nfl_phase1_efficiency.py"):
        t = (REPO / f).read_text()
        assert "nfl_yardage_v2" not in t and "nfl_yardage_v3" not in t, f          # only nfl_phase1c_v2comp (a comparator) may touch the v2 recipe


def test_no_forbidden_sources_in_phase1c_code():
    for f in ("nfl_phase1c_sim.py", "nfl_phase1c_fit.py", "nfl_phase1c_script.py", "nfl_phase1c_constants.py", "nfl_phase1c_evaluate.py", "nfl_phase1_store.py"):
        t = (REPO / f).read_text().lower()
        code = "\n".join(l for l in t.splitlines() if not l.strip().startswith(("#", '"', "'")))
        for bad in ("fanduel", "moneyline", "spread_line", "total_line", "over_odds", "under_odds", "offense_players", "defense_players"):
            assert bad not in code, (f, bad)


# ------------------------------------------------------------------ store
def rec(i, v=1.0):
    return {"id": ST.make_id("m", "g", i), "player": i, "mean": v}


def test_store_idempotent_noop_and_conflict_hard_error():
    with tempfile.TemporaryDirectory() as d:
        s = ST.Store(d, "forecasts")
        r = s.append_batch("g1", {"generated_at": "t1"}, [rec("a"), rec("b")])
        assert r == {"written": 2, "verified_duplicates": 0}
        r2 = s.append_batch("g1", {"generated_at": "LATER"}, [rec("a"), rec("b")])           # generated_at differs but records identical
        assert r2 == {"written": 0, "verified_duplicates": 2}
        try:
            s.append_batch("g1", {}, [rec("a", 2.0)]); assert False
        except ST.HardError:
            pass
        assert len(s.batch_files()) == 1


def test_store_partial_write_and_corruption_detected():
    with tempfile.TemporaryDirectory() as d:
        s = ST.Store(d, "forecasts")
        s.append_batch("g1", {}, [rec("a")])
        tmp = s.dir / "batches" / "g2.123.tmp"
        tmp.write_bytes(b'{"_batch":"g2"}\n{"id":"x"')                                         # crashed writer
        assert s.recover() == [str(tmp)] and set(s.index()) == {rec("a")["id"]}                 # ignored, never loaded
        f = s.batch_files()[0]
        os.chmod(f, 0o644); raw = f.read_bytes(); f.write_bytes(raw[:-40])                       # truncated finalized batch
        try:
            s.index(); assert False
        except ST.StoreCorrupt:
            pass


def test_store_crash_halfway_rerun_completes_without_duplicates():
    with tempfile.TemporaryDirectory() as d:
        s = ST.Store(d, "forecasts")
        s.append_batch("game1", {}, [rec("a"), rec("b")])                                        # process crashed after game 1
        for name, recs in (("game1", [rec("a"), rec("b")]), ("game2", [rec("c")])):              # rerun the whole job
            s.append_batch(name, {}, recs)
        assert sorted(r["player"] for r in s.all_records()) == ["a", "b", "c"]


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                f(); print("PASS", n)
            except Exception as e:  # noqa
                import traceback; traceback.print_exc()
                fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
