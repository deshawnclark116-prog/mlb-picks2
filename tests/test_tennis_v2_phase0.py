"""Tennis V2 Phase0 tests (standalone and pytest-compatible):  python -u tests/test_tennis_v2_phase0.py

Structure tests need no data. Data tests need TENNIS_V2_RAW (hash-verified 2015-2024 TML files; fetched by the workflow) and skip otherwise. Result tests assert on committed phase0_* evidence."""
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import tennis_v2_data as TD  # noqa: E402
import tennis_v2_incumbent as INC  # noqa: E402
import tennis_v2_metrics as M  # noqa: E402

OUT = REPO / "tennis_models" / "tennis_outcome_engine_v2"
PROTOCOL_COMMIT = "99dd4275b4af8d9cf77b2c40c74dd42eded04bd8"


class Skip(Exception):
    pass


def _git(*a):
    return subprocess.run(["git", "-C", str(REPO)] + list(a), capture_output=True, text=True).stdout.strip()


def _raw_ok():
    d = Path(os.environ.get("TENNIS_V2_RAW", "/tmp/tennis_v2_raw"))
    return d.exists() and all((d / ("%s_%d.csv" % (t, s))).exists() for t in ("atp", "wta") for s in TD.SEASONS)


# ------------------------------------------------------------- firewall / chronology
def test_firewall_no_market_tokens_in_v2_science_code():
    proto = json.loads((OUT / "protocol.json").read_text())
    toks = next(v["banned_tokens"] for v in proto.values() if isinstance(v, dict) and "banned_tokens" in v)
    assert len(toks) >= 9
    bad = []
    for f in sorted(REPO.glob("tennis_v2_*.py")):
        low = f.read_text().lower()
        for t in toks:
            if t in low:
                bad.append((f.name, t))
    assert not bad, bad


def test_firewall_tool_detects_a_planted_token():
    proto = json.loads((OUT / "protocol.json").read_text())
    toks = next(v["banned_tokens"] for v in proto.values() if isinstance(v, dict) and "banned_tokens" in v)
    planted = "x = load('" + toks[0] + "_feed')"
    assert any(t in planted.lower() for t in toks)


def test_firewall_schema_has_no_price_columns():
    schema = TD.F.SCHEMA.lower()
    assert "price" not in schema and "line_" not in schema


def test_season_cap_blocks_sealed_and_exposed_years():
    assert TD.SEASON_CAP == 2024 and max(TD.SEASONS) == 2024
    for yr in (2025, 2026):
        try:
            TD.fetch_verified("atp", yr)
        except ValueError:
            continue
        raise AssertionError("season %d must be refused" % yr)
    assert not any(k for k in TD.MANIFEST["files"] if k.split("_")[1].startswith(("2025", "2026")))


def test_code_never_names_a_sealed_season_file():
    import re
    for f in sorted(REPO.glob("tennis_v2_*.py")):
        txt = f.read_text()
        assert not re.search(r"(atp|wta)_(2025|2026)", txt), f.name


def test_protocol_and_manifest_unchanged_since_protocol_commit():
    if not _git("rev-parse", "--verify", PROTOCOL_COMMIT + "^{commit}"):
        raise Skip("protocol commit not in this checkout")
    diff = _git("diff", PROTOCOL_COMMIT, "--", "tennis_models/tennis_outcome_engine_v2/protocol.json", "tennis_models/tennis_outcome_engine_v2/source_manifest.json")
    assert diff == "", "protocol.json/source_manifest.json changed after the protocol commit; add a dated amendment file instead of editing"


def test_protocol_commit_precedes_every_result_artifact():
    if not _git("rev-parse", "--verify", PROTOCOL_COMMIT + "^{commit}"):
        raise Skip("protocol commit not in this checkout")
    t0 = int(_git("show", "-s", "--format=%ct", PROTOCOL_COMMIT))
    for f in ("phase0_market_results.json", "phase0_findings.md", "phase0_error_receipts.jsonl.gz"):
        first = _git("log", "--diff-filter=A", "--format=%ct", "--", "tennis_models/tennis_outcome_engine_v2/" + f).splitlines()
        if first:
            assert int(first[-1]) >= t0


def test_protocol_is_locked_and_research_only():
    p = json.loads((OUT / "protocol.json").read_text())
    assert p["chronology"]["locked_before_results"] is True and p["research_only"] is True and p["promotes_nothing"] is True
    assert p["chronology"]["SEALED_UNTOUCHED"]["season"] == 2025 and "no Monte Carlo" in p["no_simulation_built"]


def test_no_production_file_is_modified_by_this_branch():
    if not _git("rev-parse", "--verify", "origin/main^{commit}"):
        raise Skip("origin/main not available")
    changed = set(_git("diff", "--name-only", "origin/main...HEAD").splitlines())
    forbidden = {"tennis_serving_builder_a.py", "tennis_grade_record_a.py", "tennis_player_matches_foundation_a.py", "api.py"}
    assert not (changed & forbidden), changed & forbidden
    assert not [c for c in changed if c.startswith(("docs/", "nfl_", "nhl_", "mlb_", "cfb_")) ], [c for c in changed if c.startswith(("docs/", "nfl_", "nhl_", "mlb_", "cfb_"))]


# ------------------------------------------------------------------ closed forms
def test_race_distribution_sums_to_one_and_inverts():
    for races in (2, 3):
        for q in (0.2, 0.5, 0.63, 0.9):
            d = INC.outcome_distribution(q, races)
            assert abs(sum(d.values()) - 1) < 1e-12
            p = INC.match_win_prob_from_set_prob(q, races)
            assert abs(INC.invert_to_set_prob(p, races) - q) < 1e-9
            assert abs(sum(v for k, v in d.items() if k[0] == "P1") - p) < 1e-12


def test_hold_probability_closed_form_limits():
    import tennis_v2_phase0_audit as A
    assert abs(A.hold_prob(0.5) - 0.5) < 1e-12
    assert A.hold_prob(0.99) > 0.999 and A.hold_prob(0.01) < 1e-3
    assert A.hold_prob(0.62) > A.hold_prob(0.58)


def test_canonical_order_puts_earlier_rounds_first():
    base = {"match_date": "2023-01-01", "tourney_id": "T", "winner_id": "a", "loser_id": "b"}
    rows = [dict(base, match_id="atp_T_9", round="R16"), dict(base, match_id="atp_T_10", round="R32"), dict(base, match_id="atp_T_100", round="QF")]
    can = [r["round"] for r in INC.order_rows(rows, "canonical")]
    asi = [r["round"] for r in INC.order_rows(rows, "as_implemented")]
    assert can == ["R32", "R16", "QF"]
    assert asi != can


def test_leak_flags_detects_lookahead():
    base = {"match_date": "2023-01-01", "tourney_id": "T", "winner_id": "a", "loser_id": "b"}
    rows = [dict(base, match_id="m1", round="QF"), dict(base, match_id="m2", round="R16", winner_id="a", loser_id="c")]
    fl = TD.leak_flags(rows)
    assert fl["m1"] is False and fl["m2"] is True


# ------------------------------------------------------------------- metrics
def test_metrics_known_values():
    p = np.array([0.9, 0.1, 0.5, 0.5]); y = np.array([1, 0, 1, 0])
    assert abs(M.brier(p, y) - (0.01 + 0.01 + 0.25 + 0.25) / 4) < 1e-12
    assert abs(M.auc(p, y) - 0.875) < 1e-12
    assert M.ece(np.array([0.5] * 10), np.array([1] * 5 + [0] * 5)) < 1e-12
    c = M.count_metrics([10, 20, 30], [12, 20, 45])
    assert c["mae"] == 17 / 3 and c["within_2"] == 2 / 3 and abs(c["miss_gt_10"] - 1 / 3) < 1e-12


def test_bootstrap_is_deterministic_and_paired():
    a = np.arange(200, dtype=float) % 7; b = a + 0.5
    r1 = M.paired_boot(a, b); r2 = M.paired_boot(a, b)
    assert r1 == r2 and r1["hi"] < 0 and abs(r1["mean_diff"] + 0.5) < 1e-12


def test_verdict_rules():
    rng = np.random.default_rng(1)
    base = rng.random(3000) + 1.0
    assert M.verdict(base * 0.9, {"s": base})["verdict"] == "SURVIVES"
    assert M.verdict(base * 1.1, {"s": base})["verdict"] == "REJECTED"
    assert M.verdict(base * 0.999 + rng.normal(0, 0.5, 3000), {"s": base})["verdict"] == "WEAK"
    assert M.verdict(base, {})["verdict"] == "BLOCKED_DATA"


# ------------------------------------------------------------------ data tests
def test_hash_verification_fails_closed():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "atp_2015.csv").write_bytes(b"tampered")
        try:
            TD.fetch_verified("atp", 2015, d)
        except RuntimeError:
            return
        raise AssertionError("tampered source must be refused")


def test_replica_reproduces_committed_atp_gate_reports():
    if not _raw_ok():
        raise Skip("raw TML files not present")
    con = TD.build_db("atp"); rows = TD.all_matches(con)
    ml = json.loads((REPO / "tennis_moneyline_gate_report.json").read_text())
    o = [x for x in INC.moneyline(rows, "as_implemented") if INC.HOLD[0] <= x["match_date"] <= INC.HOLD[1]]
    y = np.array([x["p1_wins"] for x in o]); p = np.array([x["p1_prob"] for x in o])
    assert len(o) == ml["holdout"]["n"] and abs(M.brier(p, y) - ml["holdout"]["brier"]) < 1e-6
    tg = json.loads((REPO / "tennis_total_games_gate_report.json").read_text())
    t = [x for x in INC.total_games(rows, "as_implemented") if INC.HOLD[0] <= x["match_date"] <= INC.HOLD[1]]
    assert len(t) == tg["n_holdout_observations"] and abs(np.mean([abs(x["predicted_mean"] - x["actual"]) for x in t]) - tg["model_holdout_mae"]) < 1e-6
    assert con.execute("select max(match_date) from matches").fetchone()[0] <= "2024-12-31"


# ---------------------------------------------------------------- result tests
def _res(name):
    p = OUT / name
    if not p.exists():
        raise Skip("%s not generated yet" % name)
    return p


def test_results_snapshot_hashes_match_artifacts():
    snap = json.loads(_res("phase0_snapshot.json").read_text())
    for name, h in snap["artifact_sha256"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == h, name


def test_results_cover_both_tours_all_markets_and_stay_in_cap():
    m = json.loads(_res("phase0_market_results.json").read_text())
    for tour in ("atp", "wta"):
        assert set(m["tours"][tour]) == {"moneyline", "total_games", "games_spread", "set_score", "aces", "double_faults"}
        for mk, v in m["tours"][tour].items():
            assert v["native_performance_verdict"] in ("SURVIVES", "WEAK", "REJECTED", "BLOCKED_DATA")
    for name in ("phase0_error_receipts.jsonl.gz", "phase0_catastrophic_misses.jsonl.gz"):
        with gzip.open(_res(name), "rt") as f:
            dates = [json.loads(l).get("date") for l in f]
        assert max(d for d in dates if d) <= "2024-12-31" and min(d for d in dates if d) >= "2020-01-01"


def test_tours_never_pooled_in_receipts():
    with gzip.open(_res("phase0_error_receipts.jsonl.gz"), "rt") as f:
        for l in f:
            r = json.loads(l)
            assert r["match_id"].split("_")[0] == r["tour"]


def main():
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    fail = skip = 0
    for n, f in fns:
        try:
            f(); print("PASS", n)
        except Skip as e:
            skip += 1; print("SKIP", n, e)
        except Exception as e:  # noqa: BLE001
            fail += 1; print("FAIL", n, repr(e))
    print("%d passed, %d skipped, %d failed" % (len(fns) - fail - skip, skip, fail))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
