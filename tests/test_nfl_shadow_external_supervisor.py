"""External supervisor decision tests. No tokens, no network, no live forecast."""
import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_shadow_external_supervisor as S

NOW = S.utc("2026-10-10T13:00:00Z")
KICK = S.utc("2026-10-11T13:30:00Z")
GAME = "2026_05_PHI_JAX"


def key(hz):
    from nfl_phase1d_schedule import forecast_cutoff, iso
    return GAME + "|" + hz + "|" + iso(forecast_cutoff(KICK, hz))


def test_independent_worker_prewarms_before_four_minute_capture():
    result = S.decisions({GAME: {"kick": KICK}}, {}, set(), NOW)
    assert result["dispatch_shadow"]
    assert key("T24") in result["shadow_keys"]
    assert not result["incidents"]


def test_active_collector_prevents_duplicate_dispatch():
    result = S.decisions({GAME: {"kick": KICK}}, {}, set(), NOW, shadow_active=True)
    assert not result["dispatch_shadow"]


def test_done_but_absent_from_page_triggers_publisher():
    led = {key("T24"): {"state": "DONE"}}
    result = S.decisions({GAME: {"kick": KICK}}, led, set(), NOW)
    assert result["dispatch_publisher"]


def test_public_row_ends_publisher_retry():
    led = {key("T24"): {"state": "DONE"}}
    result = S.decisions({GAME: {"kick": KICK}}, led, {(GAME, "T24")}, NOW)
    assert not result["dispatch_publisher"]


def test_missing_dispatch_key_after_cutoff_generates_actionable_alert():
    late = S.utc("2026-10-10T13:39:00Z")
    result = S.decisions({GAME: {"kick": KICK}}, {}, set(), late)
    assert any(x["kind"] == "MISSED_OR_UNFINISHED" and x["key"] == key("T24")
               and x["state"] == "MISSING_DISPATCH_KEY" for x in result["incidents"])


def test_publisher_not_allowed_to_erase_missed_forecast():
    late = S.utc("2026-10-10T13:39:00Z")
    led = {key("T24"): {"state": "MISSED_REAL_CUTOFF"}}
    result = S.decisions({GAME: {"kick": KICK}}, led, set(), late)
    assert not result["dispatch_publisher"]
    assert result["incidents"][0]["state"] == "MISSED_REAL_CUTOFF"


def test_unsupported_raw_public_schema_fails_closed():
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    with TemporaryDirectory() as td:
        class FakeGithub:
            def __init__(self):
                self.calls = []
            def content(self, path, branch):
                if path == "dispatch_ledger.jsonl":
                    return b""
                return b'{"row_fields":[],"forecasts":{"T24":[],"T90":[]}}'
            def runs(self, name):
                return []
        def fake_fetch():
            raise RuntimeError("provider down")
        try:
            S.one_tick(FakeGithub(), Path(td) / "state.json", NOW, fake_fetch)
        except RuntimeError as e:
            assert "provider down" in str(e)
        else:
            raise AssertionError("independent worker must fail on provider outage")


def test_worker_dedupe_window_survives_restart(tmp_path):
    s = tmp_path / "worker.json"
    now = NOW
    S.save_log(s, {"dispatch": {"shadow:" + key("T24"): now.isoformat()}, "alerts": {}})
    doc = S.load_log(s)
    assert S.recent(doc["dispatch"]["shadow:" + key("T24")], now + timedelta(minutes=3), 12)
    assert not S.recent(doc["dispatch"]["shadow:" + key("T24")], now + timedelta(minutes=15), 12)
