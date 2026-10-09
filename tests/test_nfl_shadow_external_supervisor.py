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
    led = {key("T24"): {"state": "DONE", "n_records": 1}}
    result = S.decisions({GAME: {"kick": KICK}}, led, {}, NOW)
    assert result["dispatch_publisher"]


def test_public_row_ends_publisher_retry():
    led = {key("T24"): {"state": "DONE", "n_records": 1}}
    result = S.decisions({GAME: {"kick": KICK}}, led, {(GAME, "T24"): frozenset({"forecast-id"})}, NOW)
    assert not result["dispatch_publisher"]


def test_partial_public_rows_do_not_count_as_delivery():
    led = {key("T24"): {"state": "DONE", "n_records": 3}}
    public = {(GAME, "T24"): frozenset({"only-one-id"})}
    early = S.decisions({GAME: {"kick": KICK}}, led, public, NOW)
    assert early["dispatch_publisher"]
    assert not early["incidents"]
    late = S.decisions({GAME: {"kick": KICK}}, led, public,
                       S.utc("2026-10-10T13:39:00Z"))
    assert late["incidents"][0]["kind"] == "DONE_NOT_FULLY_PUBLISHED"
    assert late["incidents"][0]["expected_rows"] == 3
    assert late["incidents"][0]["published_rows"] == 1


def test_missing_expected_count_does_not_claim_delivery():
    led = {key("T24"): {"state": "DONE"}}
    res = S.decisions({GAME: {"kick": KICK}}, led,
                      {(GAME, "T24"): frozenset({"row"})}, NOW)
    assert res["dispatch_publisher"]


def test_read_public_fails_on_duplicate_receipt_ids():
    import pytest
    values = {
        "row_fields": ["game_id", "id"],
        "forecasts": {"T24": [[GAME, "a"], [GAME, "a"]], "T90": []},
    }
    with pytest.raises(RuntimeError, match="PUBLICATION_DUPLICATE_ID"):
        S.read_public(json.dumps(values).encode())


def test_read_public_returns_verified_id_groups():
    values = {
        "row_fields": ["game_id", "id"],
        "forecasts": {"T24": [[GAME, "a"], [GAME, "b"]], "T90": []},
    }
    assert S.read_public(json.dumps(values).encode()) == {
        (GAME, "T24"): frozenset({"a", "b"})
    }


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


def test_early_warning_when_dispatch_stopped_before_cutoff():
    now = S.utc("2026-10-10T13:23:00Z")  # seven minutes before T24
    result = S.decisions({GAME: {"kick": KICK}}, {}, {}, now,
                         shadow_active=False)
    assert key("T24") in result["shadow_keys"]
    assert result["dispatch_shadow"]
    assert any(x["kind"] == "AT_RISK_NO_ACTIVE_COLLECTOR"
               and x["key"] == key("T24") for x in result["incidents"])
    active = S.decisions({GAME: {"kick": KICK}}, {}, {}, now,
                         shadow_active=True)
    assert not any(x["kind"] == "AT_RISK_NO_ACTIVE_COLLECTOR"
                   for x in active["incidents"])


def test_supervisor_source_failure_alert_is_durable_and_rate_limited(tmp_path):
    class MockGH:
        def __init__(self):
            self.alerts = []
        def alert(self, msg):
            self.alerts.append(msg)
    gh = MockGH()
    st = tmp_path / "alerts.json"
    err = RuntimeError("schedule provider unavailable")
    assert S.report_poll_failure(gh, st, NOW, err)
    assert len(gh.alerts) == 1
    assert "SUPERVISOR" in gh.alerts[0]
    assert not S.report_poll_failure(gh, st, NOW + timedelta(minutes=3), err)
    assert len(gh.alerts) == 1
    assert S.report_poll_failure(gh, st, NOW + timedelta(minutes=31), err)
    assert len(gh.alerts) == 2


def test_queued_collector_near_cutoff_is_not_falsely_healthy():
    now = S.utc("2026-10-10T13:23:00Z")
    result = S.decisions({GAME: {"kick": KICK}}, {}, {}, now,
                         shadow_active=True, shadow_running=False)
    assert not result["dispatch_shadow"]  # already queued; do not flood Actions
    assert any(i["kind"] == "AT_RISK_COLLECTOR_QUEUED"
               for i in result["incidents"])


def test_retry_dispatch_near_cutoff_after_recent_unstarted_request(tmp_path, monkeypatch):
    import nfl_phase1d_schedule as SCH
    import nfl_shadow_schedule as SOURCES
    now = S.utc("2026-10-10T13:23:00Z")
    monkeypatch.setattr(SCH, "parse_schedule", lambda raw: {GAME: {"kick": KICK}})
    monkeypatch.setattr(SOURCES, "sanitize", lambda raw, now=None: (b"schema-safe", "hash"))
    state_file = tmp_path / "supervisor.json"
    S.save_log(state_file, {
        "dispatch": {"shadow:" + key("T24"): (now - timedelta(minutes=5)).isoformat()},
        "alerts": {},
    })

    class FakeGithub:
        def __init__(self):
            self.calls = []
            self.alerts = []
        def content(self, path, branch):
            if path == "dispatch_ledger.jsonl":
                return b""
            return json.dumps({"row_fields": ["game_id", "id"],
                               "forecasts": {"T24": [], "T90": []}}).encode()
        def runs(self, name):
            return []
        def dispatch(self, name, inputs=None):
            self.calls.append((name, inputs))
        def alert(self, msg):
            self.alerts.append(msg)

    gh = FakeGithub()
    fetch = lambda: (b"synthetic-schedule-data", {})
    result = S.one_tick(gh, state_file, now=now, fetch_schedule=fetch)
    assert result["shadow_dispatched"]
    assert gh.calls == [("nfl_phase1e_shadow.yml",
                         {"mode": "run", "duration_min": "90"})]
    # One-minute retry is suppressed; a two-minute retry is allowed if no
    # worker ever runs. GitHub Actions concurrency and CAS still arbitrate.
    S.one_tick(gh, state_file, now=now + timedelta(minutes=1),
               fetch_schedule=fetch)
    assert len(gh.calls) == 1
    S.one_tick(gh, state_file, now=now + timedelta(minutes=3),
               fetch_schedule=fetch)
    assert len(gh.calls) == 2


def test_render_supervisor_blueprint_is_isolated_and_secretless():
    import yaml
    path = Path(__file__).resolve().parents[1] / "docs/operations/nfl_supervisor.render.yaml"
    cfg = yaml.safe_load(path.read_text())
    assert len(cfg["services"]) == 1
    svc = cfg["services"][0]
    assert svc["type"] == "worker"
    assert svc["branch"] == "main"
    assert svc["numInstances"] == 1
    assert svc["disk"]["mountPath"] == "/var/data"
    assert "nfl_shadow_external_supervisor.py" in svc["startCommand"]
    token = [e for e in svc["envVars"] if e["key"] == "NFL_SUPERVISOR_GITHUB_TOKEN"]
    assert len(token) == 1 and token[0].get("sync") is False
    assert "value" not in token[0]


def test_external_worker_exits_after_sustained_poll_failures(tmp_path, monkeypatch):
    """A dead source cannot leave a green-looking background worker forever."""
    import pytest
    class FakeGH:
        def __init__(self, token):
            self.token = token
        def alert(self, message):
            pass
    def raise_failure(*args, **kwargs):
        raise RuntimeError("synthetic source outage")

    monkeypatch.setattr(S, "Github", FakeGH)
    monkeypatch.setattr(S, "one_tick", raise_failure)
    monkeypatch.setattr(S.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("NFL_SUPERVISOR_GITHUB_TOKEN", "fake-nonlive-token")
    monkeypatch.setattr(sys, "argv", ["supervisor", "--interval-sec", "30",
                                    "--state-file", str(tmp_path / "monitor.json")])
    with pytest.raises(SystemExit, match="SUPERVISOR_UNHEALTHY_AFTER_5_FAILED_POLLS"):
        S.main()


def test_worker_dedupe_window_survives_restart(tmp_path):
    s = tmp_path / "worker.json"
    now = NOW
    S.save_log(s, {"dispatch": {"shadow:" + key("T24"): now.isoformat()}, "alerts": {}})
    doc = S.load_log(s)
    assert S.recent(doc["dispatch"]["shadow:" + key("T24")], now + timedelta(minutes=3), 12)
    assert not S.recent(doc["dispatch"]["shadow:" + key("T24")], now + timedelta(minutes=15), 12)
