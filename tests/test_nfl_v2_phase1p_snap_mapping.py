"""Phase1P-DATA deterministic checks: exact identity rules, no names/outcomes, frozen bytes, gate decisions."""
import gzip
import io
import json
from pathlib import Path

import pytest

import nfl_v2_phase1p_identity_sources as I
import nfl_v2_phase1p_snap_mapping as M

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'


def test_extract_is_deterministic_identity_only_and_drops_blank_ids():
    text = ('gsis_id,display_name,pfr_id,position,height,status\n'
            '00-2,Bob B,BBBB01,WR,70,ACT\n00-1,Al A,AAAA01,TE,72,RET\n00-3,No Pfr,,QB,75,ACT\n,No Gsis,CCCC01,RB,68,ACT\n00-4,NA Pfr,NA,RB,68,ACT\n')
    a, b = I.extract_bytes(text), I.extract_bytes(text)
    assert a == b
    rows = gzip.decompress(a).decode().splitlines()
    assert rows == ['gsis_id,pfr_id,display_name,position', '00-1,AAAA01,Al A,TE', '00-2,BBBB01,Bob B,WR']


def test_resolution_hierarchy_is_exact():
    canon = {'A': ('g1',), 'B': ('g2',), 'C': ('g3', 'g4'), 'D': ('g5',)}
    roster = {'A': ('g1',), 'R': ('g9',), 'D': ('gX',), 'B2': ('g7', 'g8')}
    blank = {'Z'}
    assert M.resolve('A', canon, roster, blank)[:3] == ('g1', 'AGREE_CANONICAL_ROSTER', None)
    assert M.resolve('B', canon, roster, blank)[:3] == ('g2', 'CANONICAL_ONLY', None)
    assert M.resolve('R', canon, roster, blank)[:3] == ('g9', 'ROSTER_ONLY', None)
    assert M.resolve('D', canon, roster, blank)[2] == 'SOURCE_CONFLICT'
    assert M.resolve('C', canon, roster, blank)[2] == 'AMBIGUOUS_PFR_TO_GSIS'
    assert M.resolve('B2', canon, roster, blank)[2] == 'AMBIGUOUS_PFR_TO_GSIS'
    assert M.resolve('', canon, roster, blank)[2] == 'PFR_ID_MISSING'
    assert M.resolve('Z', canon, roster, blank)[2] == 'ROSTER_CROSSWALK_MISSING'
    assert M.resolve('Q', canon, roster, blank)[2] == 'NO_CANONICAL_ID_RECORD'


def row(pfr, name='X', team='AAA', week=1, pos='WR'):
    return {'season': '2024', 'week': str(week), 'game_type': 'REG', 'team': team, 'player': name, 'pfr_player_id': pfr, 'position': pos, 'offense_pct': '0.5'}


def test_names_never_change_a_mapping():
    canon = {'A': ('g1',)}
    one = M.classify([row('A', 'Alpha'), row('Q', 'Alpha')], canon, {}, set())
    two = M.classify([row('A', 'Zed'), row('Q', 'Zed')], canon, {}, set())
    assert [(x['gsis'], x['reason']) for x in one] == [(x['gsis'], x['reason']) for x in two] == [('g1', None), (None, 'NO_CANONICAL_ID_RECORD')]


def test_many_to_one_in_one_team_week_is_a_conflict_but_across_weeks_is_not():
    canon = {'A': ('g1',), 'B': ('g1',)}
    same = M.classify([row('A'), row('B')], canon, {}, set())
    assert [x['reason'] for x in same] == ['SOURCE_CONFLICT', 'SOURCE_CONFLICT'] and not any(x['gsis'] for x in same)
    apart = M.classify([row('A', week=1), row('B', week=2)], canon, {}, set())
    assert all(x['gsis'] == 'g1' for x in apart)


def test_gate_decisions_and_threshold_do_not_move():
    p = json.loads((ART / I.PROTOCOL).read_text())
    assert M.THRESHOLD == 0.95 == p['acceptance']['threshold'] and p['acceptance']['threshold_moves'] is False
    assert M.decide(0.95, 0.0, 0) == ('PHASE1Q_SNAP_CAN_OPEN', True)
    assert M.decide(0.9499, 0.0, 0) == ('PHASE1Q_SNAP_REMAINS_BLOCKED_ID_MAPPING', True)
    assert M.decide(0.99, 0.006, 0) == ('PHASE1Q_SNAP_BLOCKED_SOURCE_INTEGRITY', False)
    assert M.decide(0.99, 0.0, 1)[0] == 'PHASE1Q_SNAP_BLOCKED_SOURCE_INTEGRITY'
    assert set(M.DECISIONS) == set(p['acceptance']['decisions'])


def test_only_identity_columns_are_read_and_no_fuzzy_matcher_exists():
    outcome = {'targets', 'carries', 'receiving_yards', 'rushing_yards', 'receptions', 'offense_snaps', 'defense_snaps', 'st_snaps'}
    for cols in (I.SNAP_COLUMNS, I.ROSTER_COLUMNS, I.EXTRACT_COLUMNS):
        assert not outcome & set(cols)
    import ast
    for f in M.CODE_FILES:
        tree = ast.parse((ROOT / f).read_text())
        mods = {n.module if isinstance(n, ast.ImportFrom) else a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in (n.names if isinstance(n, ast.Import) else [None])}
        assert not {m for m in mods if m and any(w in m.lower() for w in ('difflib', 'fuzz', 'leven', 'jellyfish', 'rapidfuzz', 'unidecode'))}, f
        calls = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not calls & {'lower', 'upper', 'casefold', 'ratio', 'get_close_matches'}, f       # no name normalisation
    import inspect
    assert list(inspect.signature(M.resolve).parameters) == ['pfr', 'canon', 'roster', 'roster_blank']   # no name argument
    text = ''.join((ROOT / f).read_text() for f in M.CODE_FILES)
    for bad in ('spread_line', 'games.csv', 'stats_player_week', '_2025', '_2026'):
        assert bad not in text, bad
    assert I.YEARS == (2023, 2024)


def test_changed_identity_extract_is_rejected(tmp_path, monkeypatch):
    bad = tmp_path / I.EXTRACT
    bad.write_bytes(b'tampered')
    monkeypatch.setattr(I, 'ART', tmp_path)
    monkeypatch.setattr(I, 'protocol', lambda: json.loads((ART / I.PROTOCOL).read_text()))
    with pytest.raises(ValueError, match='changed or missing'):
        I.verify_extract()


def test_committed_extract_matches_its_pin_and_has_unique_pairs():
    I.verify_extract()
    canon, g2p, _ = I.canonical_map()
    assert all(len(v) == 1 for v in canon.values()) and all(len(v) == 1 for v in g2p.values())
    assert I.protocol()['identity_source']['upstream_mutable'] is True


def test_committed_results_are_consistent_with_the_rules():
    res = json.loads((ART / 'phase1p_snap_mapping_results.json').read_text())
    audit = json.loads((ART / 'phase1p_identity_source_audit.json').read_text())
    c = res['combined_primary']
    assert c['mapped'] + c['unmapped'] == c['rows'] and c['mapping_share'] == pytest.approx(c['mapped'] / c['rows'], abs=1e-6)
    assert res['decision'] == M.decide(c['mapping_share'], res['source_conflict_rate'], c['reason_counts']['OTHER'])[0]
    assert audit['outcome_columns_read'] == [] and audit['fuzzy_or_name_matching_used'] is False and audit['models_fitted'] is False and audit['later_seasons_opened'] is False
    assert res['old_mapping']['share_including_qb'] < M.THRESHOLD                       # reproduces the Phase1O blocker
    rows = [json.loads(l) for l in gzip.open(ART / 'phase1p_unmapped_snap_rows.jsonl.gz', 'rt')]
    assert len(rows) == sum(v['combined']['unmapped'] for v in res['by_position'].values())
    allowed = {'season', 'week', 'team', 'pfr_id', 'player', 'position', 'offense_pct', 'primary_position', 'candidate_gsis', 'reason'}
    assert all(set(r) == allowed and r['reason'] in M.REASONS for r in rows)
    assert sum(1 for r in rows if r['primary_position']) == c['unmapped']


def test_committed_snapshot_hashes_match_artifacts():
    snap = json.loads((ART / 'phase1p_identity_snapshot.json').read_text())
    for name, h in snap['artifacts_sha256'].items():
        assert I.sha(ART / name) == h, name
    assert snap['code_sha256'] == M_code_sha()
    assert snap['decision'] == json.loads((ART / 'phase1p_snap_mapping_results.json').read_text())['decision']


def M_code_sha():
    import hashlib
    h = hashlib.sha256()
    for f in M.CODE_FILES:
        h.update(f.encode())
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()
