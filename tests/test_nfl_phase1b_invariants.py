"""
Phase 1 hardening / Phase 1B invariant tests (research, shadow). Standalone and pytest-compatible.

  python -u tests/test_nfl_phase1b_invariants.py

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


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                f(); print("PASS", n)
            except Exception as e:  # noqa
                fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
