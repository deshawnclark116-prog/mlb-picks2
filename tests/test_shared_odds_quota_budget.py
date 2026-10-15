"""Adversarial shared odds-credit controls: no fabricated market lines, no replay spend."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from shared_odds_api_quota_guard import the_odds_api_quota_locked, source_status

UTC = timezone.utc
WHEN = datetime(2026,10,15,2,0,tzinfo=UTC)


def status(path, *, provider="the_odds_api",
           state="PROVIDER_OUT_OF_USAGE_CREDITS_DO_NOT_AUTOMATICALLY_RETRY",
           error="PROVIDER_HTTP_STATUS_401__OUT_OF_USAGE_CREDITS",
           at="2026-10-15T01:59:00Z"):
    path.write_text(json.dumps({"provider":provider,"status":state,
                                "provider_access_blocked":error,
                                "captured_at_utc":at}),encoding="utf-8")


def test_live_provider_exhaustion_shared_across_sports(tmp_path):
    path=tmp_path/"status.json"
    status(path)
    assert the_odds_api_quota_locked(status_file=path,now=WHEN)
    assert source_status(status_file=path,now=WHEN).startswith("QUOTA_EXHAUSTED")


def test_lock_requires_exact_provider_and_out_of_credits_not_network_guess(tmp_path):
    p=tmp_path/"status.json"
    status(p,provider="odds-api.io")
    assert not the_odds_api_quota_locked(status_file=p,now=WHEN)
    status(p,state="NO_ADMISSIBLE_REAL_FANDUEL_MAINLINES")
    assert not the_odds_api_quota_locked(status_file=p,now=WHEN)
    status(p,error="PROVIDER_HTTP_STATUS_401__INVALID_KEY")
    assert not the_odds_api_quota_locked(status_file=p,now=WHEN)


def test_missing_corrupt_and_future_status_cannot_poison_present(tmp_path):
    p=tmp_path/"missing"
    assert not the_odds_api_quota_locked(status_file=p,now=WHEN)
    p.write_text("{wrong",encoding="utf-8")
    assert not the_odds_api_quota_locked(status_file=p,now=WHEN)
    status(p,at="2026-10-16T04:00:00Z")
    assert not the_odds_api_quota_locked(status_file=p,now=WHEN)


def test_expiration_only_on_new_calendar_month_not_next_scheduled_run(tmp_path):
    p=tmp_path/"status.json"
    status(p)
    assert the_odds_api_quota_locked(
        status_file=p,now=datetime(2026,10,31,23,59,tzinfo=UTC))
    assert not the_odds_api_quota_locked(
        status_file=p,now=datetime(2026,11,1,0,0,tzinfo=UTC))


def test_tennis_results_cached_across_fresh_process_equivalent(tmp_path,monkeypatch):
    import tennis_serving_builder_a as t
    p=tmp_path/"tennis_odds_cache.json"
    monkeypatch.setattr(t,"TENNIS_ODDS_CACHE_PATH",p)
    monkeypatch.setattr(t,"the_odds_api_quota_locked",lambda:False)
    calls=[]
    class Resp:
        def __init__(self,data):
            self.data=data
        def raise_for_status(self):
            return None
        def json(self):
            return self.data
    def mock_get(url,*,params,timeout):
        calls.append(url)
        if url.endswith("/sports/"):
            return Resp([{"key":"tennis_atp_test","active":True}])
        return Resp([{"home_team":"Test A","away_team":"Test B",
                      "bookmakers":[{"key":"fanduel","markets":[{
                          "key":"totals","outcomes":[
                          {"name":"Over","point":21.5,"price":-110},
                          {"name":"Under","point":21.5,"price":-110}]}]}]}])
    monkeypatch.setattr(t.requests,"get",mock_get)
    first,err=t.fetch_totals_odds("dummy-in-test")
    assert len(first)==1 and err is None
    assert len(calls)==2
    assert p.exists()
    second,cached=t.fetch_totals_odds("dummy-in-test")
    assert second==first
    assert "cache" in cached
    assert len(calls)==2


def test_tennis_shared_quota_lock_prevents_even_sports_discovery(tmp_path,monkeypatch):
    import tennis_serving_builder_a as t
    monkeypatch.setattr(t,"TENNIS_ODDS_CACHE_PATH",tmp_path/"not_existing")
    monkeypatch.setattr(t,"the_odds_api_quota_locked",lambda:True)
    def forbidden(*args,**kwargs):
        raise AssertionError("No paid or event-list calls during shared quota exhaustion")
    monkeypatch.setattr(t.requests,"get",forbidden)
    rows,reason=t.fetch_totals_odds("demo")
    assert rows==[] and "quota exhausted" in reason


def test_tennis_empty_market_cache_remains_valid_and_prevents_repeat_spend(tmp_path,monkeypatch):
    import tennis_serving_builder_a as t
    monkeypatch.setattr(t,"TENNIS_ODDS_CACHE_PATH",tmp_path/"empty.json")
    monkeypatch.setattr(t,"the_odds_api_quota_locked",lambda:False)
    calls=[]
    class Resp:
        def raise_for_status(self): pass
        def json(self): return []
    def request(*args,**kw):
        calls.append(1)
        return Resp()
    monkeypatch.setattr(t.requests,"get",request)
    assert t.fetch_totals_odds("test")[0]==[]
    assert t.fetch_totals_odds("test")[0]==[]
    assert len(calls)==1


def test_all_three_sports_import_single_shared_guard():
    repo=Path(__file__).resolve().parents[1]
    for name in ("api.py","nfl_serving_builder_a.py","tennis_serving_builder_a.py"):
        source=(repo/name).read_text()
        assert "from shared_odds_api_quota_guard import the_odds_api_quota_locked" in source
        assert "if the_odds_api_quota_locked():" in source
    workflow=(repo/".github/workflows/nhl_v2_fanduel_live_source.yml").read_text()
    assert "github.event_name == 'workflow_dispatch'" in workflow
