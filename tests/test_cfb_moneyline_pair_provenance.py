import json
from pathlib import Path
import pytest
import cfb_moneyline_pair_provenance as p
import cfb_moneyline_firstseen_diagnostic as d


def test_context_platt_then_pair_normalization_alters_stated_probability_without_recalibration_proof():
    home=.86;away=.29
    served=home/(home+away)
    desc=p.describe_pair(home,away,served)
    assert desc["pre_pair_sum"]==1.15
    assert desc["pair_normalization_shift"]<0
    assert abs(desc["published_probability"]-served)<1e-6
    assert desc["model_changed"] is False


def test_invalid_or_non_normalized_context_pair_is_not_a_fake_receipt():
    with pytest.raises(ValueError,match="does not match"):
        p.describe_pair(.86,.29,.86)
    with pytest.raises(ValueError,match="no normalizable"):
        p.describe_pair(0,0,0)
    with pytest.raises(ValueError,match="invalid"):
        p.describe_pair(float("nan"),.6,.4)


def sample(logged="2026-10-04T08:27:57+00:00",raw=.4):
    source={"season":2026,"week":6,"game_id":"401856712",
       "market":"moneyline","team":"Alabama Crimson Tide",
       "model_source":"context_v2","raw_prob":raw,
       "model_prob":.7593,"logged_at":logged,
       "kickoff_utc":"2026-10-10T23:30:00Z"}
    result={"season":2026,"week":6,"game_id":"401856712",
       "market":"moneyline","team":"Alabama Crimson Tide",
       "generation":"in_season_context_v2","model_prob":.7593,
       "evaluation_probability_source":"FIRST_PREGAME_LEDGER_ENTRY",
       "logged_at":"2026-10-04T08:27:57+00:00",
       "result":"miss"}
    return source,result


def test_read_only_moneyline_diagnostic_preserves_original_side_and_prob():
    l,r=sample()
    frozen=json.dumps([l,r],sort_keys=True)
    doc=d.analyze({"results":[r]},[l])
    assert json.dumps([l,r],sort_keys=True)==frozen
    assert doc["n_2026_context_original_first_pregame"]==1
    assert doc["all"]["actual_win_rate"]==0
    assert doc["all"]["served_mean_confidence"]==.7593
    assert doc["all"]["raw_probability_below_half_for_served_side"]==1
    assert doc["status"]=="OBSERVATIONAL_ALREADY_CONSULTED_NO_PROMOTION"


def test_do_not_use_after_kickoff_or_wrong_first_pregame_version():
    late,r=sample(logged="2026-10-10T23:31:00Z")
    assert d.analyze({"results":[r]},[late])["n_2026_context_original_first_pregame"]==0
    wrong,r=sample()
    r["team"]="Georgia Bulldogs"
    assert d.analyze({"results":[r]},[wrong])["n_2026_context_original_first_pregame"]==0


def test_production_builder_attaches_provenance_only_after_original_probabilities(monkeypatch):
    text=Path("cfb_serving_builder_a.py").read_text()
    assert "pre_pair_probabilities = np.array(cal, dtype=float, copy=True)" in text
    assert 'pick["probability_provenance"] = describe_pair(' in text
    assert 'pick["model_prob"]' not in text # picks assign initial literals only, never reassign


def test_real_first_pregame_2026_already_consulted_diagnostic():
    ledger=Path("docs/cfb_picks_log.jsonl")
    record=Path("docs/cfb_record.json")
    if not ledger.exists() or not record.exists():pytest.skip("source files absent")
    originals=[json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    evaluated=json.loads(record.read_text())
    audit=d.analyze(evaluated,originals)
    assert audit["status"]=="OBSERVATIONAL_ALREADY_CONSULTED_NO_PROMOTION"
    assert audit["n_2026_context_original_first_pregame"]>=20
    assert 0<=audit["all"]["served_mean_confidence"]<=1
    assert audit["all"]["raw_probability_below_half_for_served_side"]>=1
    assert audit["paired_game_bootstrap95_for_served_minus_raw_brier"] is not None
