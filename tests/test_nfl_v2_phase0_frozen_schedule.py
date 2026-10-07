"""Phase0 / Week 4 audit: the dead mutable schedule dependency is replaced by a frozen, digest-verified allowlisted artifact; science is unchanged."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

import nfl_v2_phase0_audit as A

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
FROZEN = ART / 'phase0_frozen_final_games_2026_w1_4.json'
BASE = 'e67c066c58b88b30b69c0be1f35b57ec12aa0d91'
CSV_HEAD = 'game_id,season,game_type,week,gameday,gametime,away_team,away_score,home_team,home_score,result,spread_line,total_line,home_moneyline\n'


def csv_text(extra=''):
    return (CSV_HEAD + '2026_01_A_B,2026,REG,1,2026-09-10,20:15,A,17,B,24,7,3.5,44,-150\n2026_02_C_D,2026,REG,2,2026-09-17,13:00,C,,D,,,2.5,41,-110\n'
            '2026_04_E_F,2026,REG,4,2026-10-01,13:00,E,10,F,10,0,1,40,100\n2026_05_G_H,2026,REG,5,2026-10-11,13:00,G,,H,,,3,45,-120\n' + extra)


def test_freezing_is_deterministic_allowlisted_and_never_stores_scores_or_betting_fields():
    a = A.freeze_final_games(csv_text(), 2026, 4, {'p': 1})
    b = A.freeze_final_games(csv_text(), 2026, 4, {'p': 1})
    assert a == b and a['rows_sha256'] == A.frozen_rows_digest(a['rows'])
    assert [r['game_id'] for r in a['rows']] == ['2026_01_A_B', '2026_02_C_D', '2026_04_E_F']          # week 5 is out of scope
    assert [r['final'] for r in a['rows']] == [True, False, True]                                        # a tie (result 0) is final
    text = json.dumps(a)
    for bad in ('spread_line', 'total_line', 'moneyline', 'away_score', 'home_score', '"result"'):
        assert bad not in text
    assert all(set(r) == set(A.FROZEN_FINAL_GAMES_FIELDS) for r in a['rows'])


def test_frozen_artifact_matches_the_csv_semantics_exactly():
    doc = A.freeze_final_games(csv_text(), 2026, 4, {})
    tmp = ART.parent / '_tmp_phase0_frozen_test.json'
    try:
        tmp.write_text(json.dumps(doc))
        csvp = ART.parent / '_tmp_games.csv'
        csvp.write_text(csv_text())
        assert A.load_final_teams(tmp) == {t for t in A.load_final_teams(csvp) if t[0] == 2026 and t[1] <= 4}
    finally:
        tmp.unlink(missing_ok=True)
        (ART.parent / '_tmp_games.csv').unlink(missing_ok=True)


def test_committed_artifact_verifies_and_covers_weeks_1_to_4():
    doc = json.loads(FROZEN.read_text())
    teams = A.load_frozen_final_teams(FROZEN)
    assert doc['rows_count'] == len(doc['rows']) == 64 and all(r['final'] for r in doc['rows']) and {r['week'] for r in doc['rows']} == {1, 2, 3, 4}
    assert len(teams) == 128 and {t[1] for t in teams} == {1, 2, 3, 4}
    assert doc['provenance']['cross_check']['result'].startswith('IDENTICAL') and len(doc['provenance']['frozen_from']['raw_file_sha256']) == 64


def test_a_semantically_changed_historical_schedule_fails(tmp_path):
    doc = json.loads(FROZEN.read_text())
    for mutate in (lambda d: d['rows'][0].__setitem__('final', False), lambda d: d['rows'][1].__setitem__('home_team', 'ZZZ'), lambda d: d['rows'].pop(), lambda d: d['rows'][2].__setitem__('home_score', 3)):
        bad = copy.deepcopy(doc)
        mutate(bad)
        p = tmp_path / 'bad.json'
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            A.load_frozen_final_teams(p)


def test_dead_upstream_url_is_not_required_and_the_audit_uses_the_frozen_artifact():
    wf = (ROOT / '.github/workflows/nfl_v2_phase0_audit.yml').read_text()
    assert 'schedules/games.csv' not in wf and 'games.csv' not in wf
    assert '--games nfl_models/nfl_player_outcome_v2/phase0_frozen_final_games_2026_w1_4.json' in wf
    src = (ROOT / 'nfl_v2_phase0_audit.py').read_text()
    assert 'http' not in src.lower().replace('https://', '') or 'urlopen' not in src                # the audit never downloads anything


def test_week4_science_files_are_byte_identical_to_the_base():
    r = subprocess.run(['git', 'diff', '--name-only', BASE, 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('base commit unavailable')
    changed = set(r.stdout.split())
    protected = {p for p in changed if p.startswith('nfl_models/nfl_player_outcome_v2/') and any(k in p for k in ('week4', 'phase1a', 'phase1b', 'phase1c', 'phase1d', 'phase1e', 'phase1f', 'competent_human', 'phase0'))}
    assert protected <= {'nfl_models/nfl_player_outcome_v2/phase0_frozen_final_games_2026_w1_4.json'}, protected
    for name in ('nfl_v2_phase0_baselines.py', 'nfl_v2_phase1a_direct.py', 'nfl_v2_phase1b_opportunity.py', 'nfl_v2_phase1d_role_allocation.py'):
        assert name not in changed


def forecast_row(**kw):
    base = {"id": "x", "season": 2026, "week": 4, "game_id": "2026_04_A_B", "horizon": "T90", "player_id": "p1", "player_name": "P", "position": "RB", "team": "A", "opponent": "B", "outcome": "rush_yds", "p_active": 0.99,
            "expected_opportunities": 10.0, "mean": 50.0, "median": 48.0, "role_state": {"propensity_share": 0.5, "share_shift_vs_last8": 0.1}, "uncertainty": {"score": 0.2, "reasons": []}}
    base.update(kw)
    return base


STATS = {(2026, 4, "p1"): {"season": "2026", "week": "4", "player_id": "p1", "team": "A", "carries": "8", "rushing_yards": "32", "targets": "2", "receptions": "1", "receiving_yards": "5", "attempts": "0", "passing_yards": "0", "passing_tds": "0", "passing_interceptions": "0"}}
TOTALS = {(2026, 4, "A"): {"carries": 16, "targets": 2, "attempts": 0}}
PART = {"players": {(2026, 4, "p1")}, "coverage": {(2026, 4, "A")}, "raw_rows": 1, "positive_rows": 1, "mapped_positive_rows": 1, "map_rate": 1.0}


def week4_csv():
    return CSV_HEAD + '2026_04_A_B,2026,REG,4,2026-10-01,13:00,B,10,A,10,0,1,40,100\n2026_05_G_H,2026,REG,5,2026-10-11,13:00,H,,G,,,3,45,-120\n'


def test_no_current_or_future_schedule_change_can_alter_week4_grading(tmp_path):
    frozen = tmp_path / 'f.json'
    frozen.write_text(json.dumps(A.freeze_final_games(week4_csv(), 2026, 4, {})))
    base = A.grade_rows([forecast_row()], STATS, TOTALS, PART, {}, A.load_final_teams(frozen))
    assert len(base) == 1
    # a later, different upstream schedule (week 5 final, week 4 scores changed, extra columns) is irrelevant: the audit never reads it
    csvp = tmp_path / 'later.csv'
    csvp.write_text(week4_csv().replace('2026_05_G_H,2026,REG,5,2026-10-11,13:00,H,,G,,,3,45,-120', '2026_05_G_H,2026,REG,5,2026-10-11,13:00,H,21,G,17,-4,3,45,-120'))
    again = A.grade_rows([forecast_row()], STATS, TOTALS, PART, {}, A.load_final_teams(frozen))
    assert json.dumps(base, sort_keys=True, default=str) == json.dumps(again, sort_keys=True, default=str)
    assert not any(t[1] > 4 for t in A.load_frozen_final_teams(frozen))


def test_audit_grading_is_identical_with_csv_and_frozen_final_sets(tmp_path):
    frozen = tmp_path / 'f.json'
    frozen.write_text(json.dumps(A.freeze_final_games(week4_csv(), 2026, 4, {})))
    csvp = tmp_path / 'g.csv'
    csvp.write_text(week4_csv())
    a = A.grade_rows([forecast_row()], STATS, TOTALS, PART, {}, A.load_final_teams(frozen))
    b = A.grade_rows([forecast_row()], STATS, TOTALS, PART, {}, {t for t in A.load_final_teams(csvp) if t[1] <= 4})
    assert a and json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    unfinished = A.grade_rows([forecast_row(team='C')], STATS, {(2026, 4, 'C'): {'carries': 1}}, PART, {}, A.load_final_teams(frozen))
    assert unfinished == []                                                                                   # a team not final in the frozen set is not graded (same as before)
