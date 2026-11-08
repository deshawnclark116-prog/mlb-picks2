"""Regression: transient nflverse 504 cannot strand all Sunday NFL projections."""
from __future__ import annotations

import io
import urllib.error
from unittest.mock import patch
import pytest
import nfl_player_games_foundation_a as f

def http(code):
    return urllib.error.HTTPError(
        "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv",
        code, "temporary outage",{},None)

def valid_csv():
    header="game_id,season,week,gameday,gametime,home_team,away_team,home_score,away_score\n"
    line="2026_09_CIN_ATL,2026,9,2026-11-08,09:30,ATL,CIN,,\n"
    return header + line*22000

def test_504_falls_back_only_to_original_nfldata_csv(monkeypatch):
    calls=[]
    def fake(source,timeout=30):
        calls.append(source)
        if source=="schedules":raise http(504)
        if source==f.SCHEDULES_ORIGINAL_CSV:return valid_csv()
        raise AssertionError(source)
    monkeypatch.setattr(f,"fetch_text",fake)
    result=f.fetch_schedules_resilient("schedules")
    assert result.startswith("game_id")
    assert calls==["schedules",f.SCHEDULES_ORIGINAL_CSV]
    records=f.load_schedules(result,{2026})
    assert records[0]["game_id"]=="2026_09_CIN_ATL"
    assert records[0]["kickoff_utc"]=="2026-11-08T14:30:00+00:00"

def test_401_or_wrong_source_does_not_rewrite_schedule(monkeypatch):
    calls=[]
    def fetch(source,timeout=30):
        calls.append(source)
        raise http(401)
    monkeypatch.setattr(f,"fetch_text",fetch)
    with pytest.raises(urllib.error.HTTPError) as exc:
        f.fetch_schedules_resilient("schedules")
    assert exc.value.code==401 and calls==["schedules"]
    with pytest.raises(urllib.error.HTTPError):
        f.fetch_schedules_resilient("https://custom.invalid/my-games.csv")
    assert f.SCHEDULES_ORIGINAL_CSV not in calls

def test_missing_or_truncated_original_mirror_is_rejected(monkeypatch):
    monkeypatch.setattr(f,"fetch_text",lambda source,timeout=30:
        (_ for _ in ()).throw(http(504)) if source=="schedules" else
        "2026_09_CIN_ATL,2026,9\n")
    with pytest.raises(ValueError,match="MIRROR_SCHEMA_INVALID"):
        f.fetch_schedules_resilient("schedules")
    monkeypatch.setattr(f,"fetch_text",lambda source,timeout=30:
        (_ for _ in ()).throw(http(503)) if source=="schedules" else
        "game_id,season,week,gameday,home_team,away_team\n")
    with pytest.raises(ValueError,match="UNEXPECTEDLY_SMALL"):
        f.fetch_schedules_resilient("schedules")

def test_retry_only_transient_gateway_failures(monkeypatch):
    sleep=[]
    monkeypatch.setattr(f.time,"sleep",lambda n:sleep.append(n))
    class Resp:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return b"verified current upstream"
    count=[0]
    def urlopen(req,timeout=60):
        count[0]+=1
        if count[0]<3:raise http(504)
        return Resp()
    monkeypatch.setattr(f.urllib.request,"urlopen",urlopen)
    assert f._http_get_bytes("https://github.com/example/releases") == b"verified current upstream"
    assert count[0]==3 and sleep==[1,3]

def test_auth_and_real_404_are_not_retried(monkeypatch):
    sleep=[]
    monkeypatch.setattr(f.time,"sleep",lambda n:sleep.append(n))
    seen=[]
    def bad(req,timeout=60):
        seen.append(req)
        raise http(404)
    monkeypatch.setattr(f.urllib.request,"urlopen",bad)
    with pytest.raises(urllib.error.HTTPError) as err:
        f._http_get_bytes("https://github.com/nflverse/nflverse-data/missing")
    assert err.value.code==404 and len(seen)==1 and sleep==[]
