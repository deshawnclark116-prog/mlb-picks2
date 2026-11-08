"""Fail closed on malformed yardage, preserve authentic priced numeric ranges."""
from datetime import datetime,timedelta,timezone

import pytest

import nfl_serving_builder_a as serving
from nfl_game_day_contract import validate

def test_real_book_line_never_erases_original_simulated_yardage_quantiles(monkeypatch):
    prior={
      "side":"OVER","side_prob":.726,
      "mean":67.3,"median":64.2,
      "p10":18.4,"p90":122.1,
    }
    monkeypatch.setattr(serving.nfl_sim,"simulate",lambda *args,**kwargs:prior)
    odds={
      "line":49.5,"over_price":-110,"under_price":-110,"book":"FanDuel"
    }
    p=serving.make_real_odds_pick(
      "rushing_yards","Test Runner","00-9999999","DAL","IND",
      2026,9,[17,19,16],[7.,2.,14.],odds,6
    )
    assert p is not None
    assert p["line"]==49.5
    assert p["pick"]=="OVER 49.5"
    assert p["model_prob"]==.726
    assert p["projected_mean"]==67.3
    assert p["projected_median"]==64.2
    assert p["projected_low"]==18.4
    assert p["projected_high"]==122.1
    assert p["odds"]==-110
    assert p["book"]=="FanDuel"
    assert 0<=p["projected_low"]<=p["projected_median"]<=p["projected_high"]
    clock=datetime.now(timezone.utc)
    kickoff=clock+timedelta(hours=5)
    p["kickoff_utc"]=kickoff.isoformat()
    d={
      "season":2026,"week":9,"generated_at_utc":clock.isoformat(),
      "scheduled_games":[{"home_team":"IND","away_team":"DAL",
                          "kickoff_utc":kickoff.isoformat()}],
      "picks":[p],
    }
    receipt=validate(d,now=clock+timedelta(seconds=1))
    assert receipt["status"]=="VALID_PREGAME_PUBLICATION"
    assert receipt["numeric"]==1
    # This is the exact incomplete shape that prevented Sunday publication.
    before={**p}
    for k in ("projected_low","projected_mean","projected_high"):
      before.pop(k)
    d["picks"]=[before]
    with pytest.raises(ValueError,match="INVALID_NUMERICAL_PROJECTION"):
      validate(d,now=clock+timedelta(seconds=1))

def test_no_book_line_already_keeps_same_real_moments(monkeypatch):
    fake={"side":"OVER","side_prob":.8,"median":55.0,"mean":59.0,
          "p10":15.0,"p90":101.0}
    monkeypatch.setattr(serving.nfl_sim,"simulate",lambda *args,**kwargs:fake)
    p=serving.make_real_odds_pick("receiving_yards","Test Receiver",
       "00-9999998","ATL","CIN",2026,9,[8,9],[11.0,13.0],None,5)
    assert p["model_prob"] if "model_prob" in p else True
    assert p["projected_median"]==55
    assert p["projected_mean"]==59
    assert p["projected_low"]==15
    assert p["projected_high"]==101
    assert "odds" not in p
    assert "line" not in p
