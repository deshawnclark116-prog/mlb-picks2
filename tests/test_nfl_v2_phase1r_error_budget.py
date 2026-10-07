"""Phase1R-VOI deterministic checks: exact Shapley, actual values only in the diagnostic branch, identical cohorts, frozen inputs, registry history."""
import ast
import gzip
import json
from pathlib import Path
import subprocess
import tempfile

import pytest

import nfl_v2_phase1r_error_budget as E

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
BASE = E.BASE_HEAD


def row(pred, actual, fn=lambda v: v['T'] * v['E'], y=None):
    names = tuple(pred)
    r = E.Row(('x',), 0.0, names, pred, actual, fn)
    r.y = fn({n: (actual[n] if actual[n] is not None else pred[n]) for n in names}) if y is None else y
    r._cache.clear()
    return r


# ---------------------------------------------------------------- Shapley and oracle mechanics
def test_two_component_shapley_matches_hand_calculation():
    r = row({'T': 2.0, 'E': 3.0}, {'T': 4.0, 'E': 3.0})            # y = 12, normal prediction 6
    assert r.err(()) == 6 and r.err(('T',)) == 0 and r.err(('E',)) == 6 and r.err(('T', 'E')) == 0
    phi = r.shapley()
    assert phi['T'] == pytest.approx(6.0) and phi['E'] == pytest.approx(0.0)       # the correct component gets everything, a right component gets zero


def test_shapley_is_efficient_symmetric_and_handles_cancelling_errors():
    r = row({'T': 2.0, 'C': 0.5, 'E': 10.0}, {'T': 3.0, 'C': 0.8, 'E': 7.0}, fn=lambda v: v['T'] * v['C'] * v['E'])
    phi = r.shapley()
    assert sum(phi.values()) == pytest.approx(r.err(()) - r.err(('T', 'C', 'E')))
    s = row({'T': 2.0, 'E': 2.0}, {'T': 3.0, 'E': 3.0})
    assert s.shapley()['T'] == pytest.approx(s.shapley()['E'])                       # symmetric components share equally
    c = row({'T': 2.0, 'E': 3.0}, {'T': 3.0, 'E': 2.0})                              # errors cancel: 6 -> 6
    assert c.err(()) == 0 and sum(c.shapley().values()) == pytest.approx(0.0)


def test_actual_values_enter_only_through_the_subset_and_undefined_actuals_keep_the_prediction():
    r = row({'T': 2.0, 'E': 3.0}, {'T': 4.0, 'E': None})
    assert r.value(()) == 6.0 and r.value(('T',)) == 12.0 and r.value(('E',)) == 6.0           # E has no realized value: prediction kept
    assert r.value(('T', 'E')) == 12.0
    assert r.pred == {'T': 2.0, 'E': 3.0}                                                       # predictions never mutated


def test_budget_reports_non_additivity_and_exact_identity():
    rows = [row({'T': 2.0, 'E': 3.0}, {'T': 4.0, 'E': 4.0}), row({'T': 3.0, 'E': 3.0}, {'T': 2.0, 'E': 2.0})]
    b, _ = E.budget(rows)
    assert b['shapley_sum_minus_addressable'] == pytest.approx(0.0, abs=1e-12)
    assert b['non_additivity']['sum_of_single_oracle_gains'] != pytest.approx(b['normal_mae'])


def test_dominant_cause_classification_is_deterministic():
    names = ('T', 'C', 'A', 'Y')
    assert E.dominant({'T': 60, 'C': 20, 'A': 15, 'Y': 5}, 100.0, names) == 'OPPORTUNITY_UNSPLIT'
    assert E.dominant({'T': 30, 'C': 40, 'A': 20, 'Y': 10}, 100.0, names) == 'MULTIPLE_COMPONENTS'
    assert E.dominant({'T': 10, 'C': 10, 'A': 55, 'Y': 25}, 100.0, names) == 'COMPLETED_AIR'
    assert E.dominant({'T': 0, 'C': 0, 'A': 0, 'Y': 0}, 0.0, names) == 'UNRESOLVED'
    assert E.dominant({'T': 50, 'C': 50, 'A': 0, 'Y': 0}, 100.0, names) == 'OPPORTUNITY_UNSPLIT'   # exact tie: the lexicographically larger component name wins, stable


def test_tier_rules():
    th = E.protocol()['tiering']
    base = {'timing': 'conditional', 'directly_measures': True, 'represented_by_failed': 'NONE', 'validation_path': 'HISTORICAL_AND_LIVE_PLAUSIBLE', 'measurement_confidence': 'MEDIUM_HIGH', 'id': 'x', 'quantification': 'ROW_LEVEL'}
    assert E.tier_for(base, 0.30, th) == 'TIER_1_ACQUIRE_FIRST'
    assert E.tier_for(base, 0.12, th) == 'TIER_2_HIGH_VALUE_IF_ACCESSIBLE'
    assert E.tier_for({**base, 'represented_by_failed': 'PARTIAL_PROXY_FAILED'}, 0.9, th) == 'TIER_2_HIGH_VALUE_IF_ACCESSIBLE'      # partial proxy failure caps at Tier 2
    assert E.tier_for({**base, 'measurement_confidence': 'LOW'}, 0.9, th) == 'TIER_2_HIGH_VALUE_IF_ACCESSIBLE'
    assert E.tier_for({**base, 'validation_path': 'NONE_IDENTIFIED'}, 0.9, th) == 'TIER_3_SECONDARY'
    assert E.tier_for({**base, 'represented_by_failed': 'FAILED_SAME_SIGNAL'}, 0.9, th) == 'TIER_4_LOW_EXPECTED_VALUE'
    assert E.tier_for({**base, 'timing': 'no'}, 0.9, th) == 'DO_NOT_PURSUE' and E.tier_for({**base, 'directly_measures': False}, 0.9, th) == 'DO_NOT_PURSUE'


# ---------------------------------------------------------------- real frozen cohorts
@pytest.fixture(scope='module')
def built():
    return E.run()


def test_every_cohort_is_identical_across_substitutions_and_all_oracle_equals_actual(built):
    heads = built[0]
    for name, (rows, _k, _c, _d) in heads.items():
        assert len({r.key for r in rows}) == len(rows) and len({r.names for r in rows}) == 1, name
        assert max(abs(r.value(r.names) - r.y) for r in rows) < 1e-3, name              # fully reconciled chain (receipts store 6-decimal shares)
    a = [r.key for r in heads['receiving_yards_decomposed'][0]]
    assert a == [r.key for r in heads['receiving_yards_incumbent_ypt'][0]]              # same rows for the incumbent comparison


def test_cohort_counts_and_exclusions_are_reported_and_consistent(built):
    cohorts = built[3]
    c = cohorts['receiving_yards_decomposed']
    assert c['source_population_n'] == c['common_cohort_n'] + sum(c['exclusions'].values())
    assert c['excluded_rows_normal_mae'] is not None                                      # exclusions are not hiding difficult rows
    r = cohorts['rushing_yards_2024_in_sample']
    assert r['source_population_n'] == r['common_cohort_n'] + sum(r['exclusions'].values()) and 'NOT_IN_PHASE1D' in next(iter(r['exclusions']))


def test_shapley_sums_to_normal_error_on_every_head_and_nothing_is_called_predictive_lift(built):
    for name, b in built[1].items():
        assert abs(b['shapley_sum_minus_addressable']) < 1e-9, name
        assert b['all_oracle_mae'] < 1e-3
    for name, s in built[2].items():
        assert 'not predictive lift' in s['caveat']
    voi = json.loads((ART / 'phase1r_information_value_ranking.json').read_text()) if (ART / 'phase1r_information_value_ranking.json').exists() else None
    if voi:
        assert voi['status'] == 'UPPER_BOUNDS_NOT_EXPECTED_IMPROVEMENT' and all('UPPER BOUND' in x['warning'] for x in voi['families'])


def test_catastrophic_labels_are_supported_and_unsupported_causes_are_never_inferred(built):
    catas = built[4]
    assert catas and {x['dominant_cause'] for x in catas} <= set(E.LABELS)
    assert not {x['dominant_cause'] for x in catas} & {'AVAILABILITY', 'GAME_SCRIPT', 'QB_STATE'}
    assert all(x['absolute_error'] > min(E.THRESHOLDS[{'receiving_yards_decomposed': 'receiving_yards', 'receiving_yards_incumbent_ypt': 'receiving_yards', 'receptions': 'receptions', 'targets_2024_in_sample': 'targets', 'carries_2024_in_sample': 'carries', 'rushing_yards_2024_in_sample': 'rushing_yards'}[x['head']]]) for x in catas)
    assert any(x['dominant_cause'] == 'UNRESOLVED' for x in catas)                          # rows without a component split are shown, not hidden


def test_rushing_tail_partition_is_an_exact_identity(built):
    heads = built[0]
    tail, supported = E.tail_diagnostic(heads['rushing_yards_2024_in_sample'][0])
    assert supported > 400
    for v in list(tail.values())[:50]:
        assert v['within_tier_residual_yards'] + v['tail_mix_surprise_yards'] + v['tail_model_minus_incumbent_yards'] == pytest.approx(v['efficiency_error_yards'], abs=1e-6)


def test_week4_table_parses_and_attribution_is_exact(built):
    w4 = E.week4_receipts()
    assert len(w4) == 10 and {x['player'] for x in w4} >= {'Kenneth Walker', 'Carnell Tate', 'Courtland Sutton'}
    for x in w4:
        assert x['shapley_opportunity'] + x['shapley_efficiency'] == pytest.approx(x['absolute_error'], abs=1e-9)
        assert x['q1_team_opportunity_wrong'].startswith('UNKNOWN') and x['q4_explosive_tail_dominant'].startswith('UNKNOWN')


def test_qb_decomposition_is_blocked_not_bypassed():
    res = json.loads((ART / 'phase1r_error_budget_results.json').read_text())
    assert res['qb']['decomposition'] == 'BLOCKED_STARTER_STATE_DATA' and res['qb']['aggregate_diagnostic']['status'].startswith('POSTGAME_DIAGNOSTIC_ONLY')
    assert not any('qb' in k for k in res['heads'])


# ---------------------------------------------------------------- scope, inputs, reproducibility
def test_no_models_network_sportsbook_or_week5_access():
    tree = ast.parse((ROOT / 'nfl_v2_phase1r_error_budget.py').read_text())
    mods = {n.module if isinstance(n, ast.ImportFrom) else a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in (n.names if isinstance(n, ast.Import) else [None])}
    assert {m for m in mods if m} <= {'__future__', 'argparse', 'collections', 'gzip', 'hashlib', 'itertools', 'json', 'math', 'pathlib', 're'}, mods
    text = (ROOT / 'nfl_v2_phase1r_error_budget.py').read_text().lower()
    for bad in ('numpy', 'sklearn', 'scipy', 'spread_line', 'moneyline', 'odds', 'sportsbook_inputs_used": true', 'urllib', 'requests', 'curl', 'http', 'week 5', 'week5', 'games.csv', 'monte_carlo(', 'fit('):
        assert bad not in text, bad
    p = E.protocol()
    assert not any(('2026' in f and 'wk' not in f) for f in p['inputs'])
    assert all(f.endswith(('.json', '.jsonl.gz', '.md')) for f in p['inputs'])


def test_changed_frozen_input_is_rejected(monkeypatch):
    monkeypatch.setattr(E, 'sha', lambda path: 'deadbeef')
    with pytest.raises(ValueError, match='changed or missing'):
        E.verify_inputs()


def test_prior_frozen_artifacts_are_byte_identical_to_the_base():
    r = subprocess.run(['git', 'diff', '--name-status', BASE, 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('base commit unavailable')
    allowed_modified = {'nfl_models/nfl_player_outcome_v2/research_registry.json', 'tests/test_nfl_v2_phase1l_qb_opportunity.py', 'tests/test_nfl_v2_phase1m_qb_state.py', 'tests/test_nfl_v2_phase1n_team_opportunity.py'}
    for line in r.stdout.splitlines():
        status, _, path = line.partition('\t')
        if status == 'A':
            assert 'phase1r' in path or 'phase2a' in path or 'nfl_v2_qb_state_ingestion' in path, path
        else:
            assert status == 'M' and path in allowed_modified, (status, path)
    for name in E.protocol()['inputs']:
        assert subprocess.run(['git', 'diff', '--quiet', BASE, 'HEAD', '--', f'nfl_models/nfl_player_outcome_v2/{name}'], cwd=ROOT).returncode == 0, name


def test_registry_history_is_preserved_and_only_high_level_state_changed():
    base = subprocess.run(['git', 'show', f'{BASE}:nfl_models/nfl_player_outcome_v2/research_registry.json'], cwd=ROOT, capture_output=True, text=True)
    if base.returncode != 0:
        pytest.skip('base commit unavailable')
    old, new = json.loads(base.stdout), json.loads((ART / 'research_registry.json').read_text())
    changed = {'status', 'next_milestone'}
    for k, v in old.items():
        if k not in changed:
            assert new[k] == v, k
    assert set(new) - set(old) <= {'current_state', 'phase1r_error_budget', 'phase2a_forward'}
    assert new['open_research_tracks'] == old['open_research_tracks'] and new['frozen_or_blocked'] == old['frozen_or_blocked']      # legacy lists are pinned by the Phase1I/K archival gates and stay byte-identical
    assert 'authoritative current state' in new['current_state']['supersession_note']
    assert new['current_state']['phase1o_family_B_snap_change'].startswith('RESOLVED')


def test_outputs_reproduce_byte_identically():
    with tempfile.TemporaryDirectory() as d:
        E.assemble(d)
        for p in sorted(Path(d).iterdir()):
            assert p.read_bytes() == (ART / p.name).read_bytes(), p.name


def test_snapshot_hashes_match_committed_artifacts():
    snap = json.loads((ART / 'phase1r_snapshot.json').read_text())
    for name, h in snap['artifacts_sha256'].items():
        assert E.sha(ART / name) == h, name
    assert snap['code_sha256'] == E.code_sha256() and snap['no_new_modeling'] is True
