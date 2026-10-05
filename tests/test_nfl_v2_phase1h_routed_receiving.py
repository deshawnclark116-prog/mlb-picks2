"""Deterministic temporal, routing, gate and validation-discipline tests."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_v2_phase1h_routed_receiving as H


def row(i=0, season=2024, week=2):
    return {"season": season, "week": week, "game_id": f"{season}_{week:02d}_{i//3}",
            "player_id": f"p{i}", "team": "A", "opponent": "B", "position": "WR",
            "targets": 5., "receptions": 3., "receiving_yards": 36.,
            "actual_completed_air": 21., "actual_yac": 15.,
            "component_labels_reconciled": True, "predicted_targets": 6.,
            "baseline_catch_probability": .6, "baseline_air_component": 7.,
            "baseline_yac_component": 5.,
            "features": {c: (i % 5 - 2)*.1 for f in H.ROUTES.values() for cs in f.values() if isinstance(cs,list) for c in cs},
            "phase1f_ypt": 7.5, "history_ypt": 7.8,
            "phase1g_catch": .62, "phase1g_air": 8., "phase1g_yac": 5., "phase1g_ypt": 8.06}


def ev(week=1, team="A", pid="stable", date="2024-09-08", **kw):
    return {"key": (2024,week), "game_id": f"2024_{week:02d}_{team}", "date": date,
            "player_id": pid, "receptions": 1., "targets": 1., "rec_yards": 15.,
            "completed_air": 10., "yac": 5., "air_yards": 10.,
            "pressure": 0., "blitz": 0., "scheme": "ZONE_COVERAGE", **kw}


def test_mechanical_identity_and_receipt_reconstruction():
    r = row(); q = H.project(r)
    assert q["ypt"] == pytest.approx(.6*(7+5))
    assert q["final_direct_receiving_yard_projection"] == pytest.approx(6*q["ypt"])
    assert all(q[k] == 0 for k in ("matchup_catch_adjustment", "matchup_air_adjustment", "matchup_yac_adjustment"))


@pytest.mark.parametrize("family,unchanged", [("depth_explosive",("catch","yac")),
    ("pressure_blitz",("air","yac")), ("man_zone",("yac",)), ("yac_matchup",("catch","air"))])
def test_feature_families_cannot_wander_between_components(family, unchanged):
    train = [row(i) for i in range(30)]
    for r in train:
        r["receptions"] += r["features"].get("scheme_catch_edge",0)
        r["actual_completed_air"] += 3*r["features"]["player_adot_minus_position"]
        r["actual_yac"] += 2*r["features"]["def_position_yac_delta"]
    spec = H.fit(train, family, 10)
    q, base = H.project(row(4),spec), H.project(row(4))
    for c in unchanged:
        assert q[c] == base[c]
    assert set(spec["models"][family]) <= {"catch","air","yac"}
    assert all("yards" not in m.get("target","") for m in spec["models"][family].values())


def test_same_week_future_and_uncompleted_date_excluded():
    m = object.__new__(H.Mechanics)
    vals = [ev(1), ev(2,date="2024-09-10"),ev(2,date="2024-09-15"),ev(3,date="2024-09-22")]
    idx = {"stable": vals}
    m.keys = {(id(idx),"stable"): [e["key"] for e in vals]}
    assert m.prior(idx,"stable",(2024,2),"2024-09-12") == vals[:1]
    assert m.prior(idx,"stable",(2024,3),"2024-09-12") == vals[:2]
    assert all(e["date"] < "2024-09-12" for e in m.prior(idx,"stable",(2024,3),"2024-09-12"))


def test_skill_identity_survives_transfer_and_game_window_is_not_week_window():
    m = object.__new__(H.Mechanics)
    vals = [ev(1,team="OLD"),ev(3,team="NEW",date="2024-09-22")]
    idx = {"stable": vals}; m.keys = {(id(idx),"stable"): [e["key"] for e in vals]}
    assert len(m.prior(idx,"stable",(2024,5),"2024-10-01",games=2)) == 2


def test_missing_labels_never_become_zero_and_rates_use_correct_denominators():
    p = H.profile([ev(),ev(receptions=0.,completed_air=0.,yac=0.,rec_yards=0.),ev(air_yards=None,yac=None,completed_air=None)])
    assert p["catch"] == pytest.approx(2/3)
    assert p["yac"] == 5
    assert p["air"] == 10
    assert p["adot"] == 10
    assert H.boolean("") is None
    assert H.boolean("FALSE") == 0


def test_negative_completed_air_preserved():
    assert H.profile([ev(completed_air=-3.,rec_yards=2.)])["air"] == -3.


def test_no_target_labels_or_sportsbook_inputs_in_projection():
    r = row(4); q = H.project(r,H.fit([row(i) for i in range(30)],"depth_explosive",100))
    other = copy.deepcopy(r)
    other.update(receiving_yards=10000., targets=500., receptions=400., actual_yac=6000.,
                 actual_completed_air=4000., sportsbook_line=10., odds=-200., spread_line=5.)
    assert H.project(other,H.fit([row(i) for i in range(30)],"depth_explosive",100)) == q


def test_fitting_2025_or_2026_forbidden():
    for year in (2025,2026):
        with pytest.raises(ValueError,match="only on 2024"):
            H.fit([row(season=year)],"man_zone",10)


def test_repeated_fitting_evaluation_and_block_bootstrap_deterministic():
    rs = [row(i,week=1+i//6) for i in range(48)]
    a,b = H.fit(rs,"yac_matchup",100),H.fit(rs,"yac_matchup",100)
    assert a == b
    pa,pb = H.predictions(rs,a),H.predictions(rs)
    assert H.losses(rs,pa) == H.losses(rs,H.predictions(rs,b))
    assert H.bootstrap(rs,pa,pb) == H.bootstrap(rs,pa,pb)
    assert H.bootstrap(rs,pb,pb)["ci95"] == [0.,0.]


def test_block_bootstrap_keeps_game_rows_and_is_not_row_wise():
    rs = [row(i,week=1+i//6) for i in range(48)]
    pa,pb = H.predictions(rs),H.predictions(rs,comparator="history")
    # Duplicate the whole paired panel; row-wise resampling would narrow CI.
    assert H.bootstrap(rs,pa,pb) == H.bootstrap(rs+rs,pa+pa,pb+pb)


def test_no_cancellation_or_component_failure_can_pass():
    base = {"oracle_target_rec_yds_mae": 20.,"yards_per_target_mae":4.,"catch_rate_mae":.2}
    bad = {**base,"oracle_target_rec_yds_mae":19.,"catch_rate_mae":.21}
    assert not H.gate(bad,base,["catch_rate_mae"])["passes"]
    bad = {**base,"oracle_target_rec_yds_mae":20.1,"full_yards_mae":10.}
    assert not H.gate(bad,base,[])["passes"]
    bad = {**base,"oracle_target_rec_yds_mae":19.,"yards_per_target_mae":4.1}
    assert not H.gate(bad,base,[])["passes"]


def test_efficiency_zero_targets_excluded_but_full_zero_rows_preserved():
    r = row(); zero = {**r,"targets":0.,"receptions":0.,"receiving_yards":0.}
    rs = [r,zero]; ps = H.predictions(rs)
    assert H.losses(rs,ps)["n"] == 1
    assert H.full_metrics(rs,ps)["n"] == 2


def test_component_unreconciled_labels_not_scored_or_fit_as_fabricated():
    r = {**row(),"component_labels_reconciled":False,"actual_completed_air":None,"actual_yac":None}
    m = H.losses([r],H.predictions([r]))
    assert m["n"] == 1
    assert "completed_air_per_target_mae" not in m


def test_frozen_development_selection_ignores_validation_labels(tmp_path):
    rows = [row(i,week=1+i//6) for i in range(108)]
    audit = tmp_path/"audit.json";audit.write_text("{}")
    original = H.develop(rows+[row(season=2025)],{},audit)
    changed = H.develop(rows+[{**row(season=2025),"receiving_yards":1000000.}],{},audit)
    assert original == changed
    assert original["selected"] is None


def test_confirm_does_not_fit_or_rescue_and_skips_full_on_failure(monkeypatch,tmp_path):
    audit = tmp_path/"audit.json"; audit.write_text("{}")
    lock = H.develop([row(i,week=1+i//6) for i in range(108)],{},audit)
    monkeypatch.setattr(H,"fit",lambda *args: pytest.fail("Confirmation cannot fit"))
    monkeypatch.setattr(H,"full_metrics",lambda *args: pytest.fail("Failed efficiency cannot score full yards"))
    out, receipts = H.confirm([row(season=2025),row(season=2026)],lock)
    assert out["verdict"] == "REJECTED_EFFICIENCY_REPLACEMENT"
    assert out["full_projection_evaluation"]["status"] == "NOT_RUN_EFFICIENCY_GATE_FAILED"
    assert receipts


def test_protocol_changed_after_freeze_refused(tmp_path,monkeypatch):
    path = tmp_path/"protocol.json"; path.write_text("changed")
    monkeypatch.setattr(H,"PROTOCOL",path)
    with pytest.raises(ValueError,match="Protocol changed"):
        H.confirm([], {"protocol_sha256":"old"})


def test_true_pressure_is_not_sack_proxy_and_scheme_splits_are_components():
    assert H.ROUTES["pressure_blitz"]["catch"] == ["def_pressure_minus_league","def_blitz_minus_league","pressure_exposure_x_player_catch_split"]
    assert "scheme_ypt_edge" not in json.dumps(H.ROUTES)
    assert set(H.ROUTES["man_zone"]) == {"catch","air","mechanism"}


def test_future_period_prohibition_and_protected_scope():
    protocol = json.loads(H.PROTOCOL.read_text())
    assert protocol["periods"]["future"].startswith("2026 W5+")
    assert protocol["no_rescue"] and not protocol["sportsbook_inputs"] and not protocol["monte_carlo"]
    import subprocess
    changed = subprocess.check_output(["git","diff","--name-only",protocol["base_head"]],cwd=H.ROOT,text=True).splitlines()
    allowed = {"nfl_v2_phase1h_sources.py","nfl_v2_phase1h_routed_receiving.py",
               "tests/test_nfl_v2_phase1h_routed_receiving.py",".github/workflows/nfl_v2_phase1h_routed_receiving.yml",
               "nfl_models/nfl_player_outcome_v2/research_registry.json"}
    assert all(f in allowed or f.startswith("nfl_models/nfl_player_outcome_v2/phase1h_") for f in changed)
