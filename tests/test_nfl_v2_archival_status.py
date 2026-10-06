"""Archival-integrity gate for the frozen Phase1H/I/K workflows: honest, strict, and unable to run science."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import types

import pytest
import yaml

import nfl_v2_archival_status as A

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
WORKFLOWS = {'h': 'nfl_v2_phase1h_routed_receiving.yml', 'i': 'nfl_v2_phase1i_target_depth.yml', 'k': 'nfl_v2_phase1k_rushing_efficiency.yml'}


def static(key):
    return json.loads((ART / A.PHASES[key]['status_file']).read_text())


def fake_fetch_factory(static_doc, drift_2026=True, break_stable=None, stable_missing=None, missing_2026=False, urls=None):
    """Serve the exact pinned bytes (as content whose sha we control) without a network."""
    pins = {p['url']: p for p in static_doc['available_pins'] + static_doc['unavailable_pins']}
    blobs = {}

    def fetch(url, dest, timeout=0):
        if urls is not None:
            urls.append(url)
        pin = pins[url]
        if pin['local_name'] == stable_missing:
            return types.SimpleNamespace(returncode=22, stderr='404')
        if pin['year'] == '2026' and missing_2026:
            return types.SimpleNamespace(returncode=22, stderr='404')
        content = (pin['expected_sha256'] + ':' + ('REVISED' if (pin['year'] == '2026' and drift_2026) or pin['local_name'] == break_stable else 'FROZEN')).encode()
        Path(dest).write_bytes(content)
        return types.SimpleNamespace(returncode=0, stderr='')
    return fetch


@pytest.fixture
def fake_sha(monkeypatch):
    """sha_file on the fake bytes returns the pin digest when the content says FROZEN, otherwise a different digest."""
    real = A.sha_file

    def sha(path):
        b = Path(path).read_bytes()
        if b.endswith(b':FROZEN') and b'.' not in b[:5]:
            return b.split(b':')[0].decode()
        if b.endswith(b':REVISED'):
            return 'f' * 64
        return real(path)
    monkeypatch.setattr(A, 'sha_file', sha)


# ---------------------------------------------------------------- committed static records
@pytest.mark.parametrize('key', sorted(A.PHASES))
def test_static_archival_record_is_precise(key):
    d = static(key)
    assert d['status'] == 'EXPECTED_FROZEN_SOURCE_UNAVAILABLE' and d['scientific_status'] == 'FROZEN_VALID_RESULT_REPRODUCTION_CURRENTLY_BLOCKED'
    assert d['replacement_data_used'] is False and d['model_rerun'] is False and d['selection_rerun'] is False and d['scientific_artifacts_changed'] is False
    assert d['frozen_verdict'] == 'REJECTED_EFFICIENCY_REPLACEMENT' and d['frozen_source_run'] and len(d['frozen_source_head']) == 40
    assert 'REPRODUCTION_BLOCKED_EXACT_2026_BYTES_UNAVAILABLE' in d['wording'] and 'SCIENTIFIC_RESULT_FROZEN' in d['wording'] and 'ARCHIVAL_INTEGRITY_VERIFIED' in d['wording']
    assert 'REPRODUCTION_PASSED' not in json.dumps(d['wording']) and d['not_claimed'].startswith('REPRODUCTION_PASSED')
    assert {p['year'] for p in d['available_pins']} == {'2023', '2024', '2025'} and {p['year'] for p in d['unavailable_pins']} == {'2026'}
    assert d['expected_frozen_digest'] == {p['local_name']: p['expected_sha256'] for p in d['unavailable_pins']}
    results = json.loads((ART / A.PHASES[key]['results']).read_text())
    assert d['frozen_verdict'] == results['verdict']


@pytest.mark.parametrize('key', sorted(A.PHASES))
def test_frozen_scientific_artifacts_are_byte_identical_to_the_frozen_head(key):
    cfg, d = A.PHASES[key], static(key)
    assert A.hashed_files(cfg) == d['artifacts_sha256']                       # nothing changed since the archival record
    head = d['frozen_source_head']
    probe = subprocess.run(['git', 'cat-file', '-e', head], cwd=ROOT, capture_output=True)
    if probe.returncode != 0:
        pytest.skip('frozen head not in this clone')
    for rel, digest in d['artifacts_sha256'].items():
        if rel.startswith('tests/'):
            continue                                                            # a test file may legitimately widen later guards; science does not
        blob = subprocess.run(['git', 'show', f'{head}:{rel}'], cwd=ROOT, capture_output=True)
        assert blob.returncode == 0, rel
        import hashlib
        assert hashlib.sha256(blob.stdout).hexdigest() == digest, rel          # findings, snapshots, results, receipts, protocol, code: byte-identical to the frozen run's head


# ---------------------------------------------------------------- the gate
def test_known_2026_drift_passes_only_with_the_exact_condition_and_leaves_no_bytes(tmp_path, fake_sha):
    d = static('k')
    urls = []
    code, rt = A.run_gate('k', tmp_path / 'data', tmp_path / 'rt.json', fetch=fake_fetch_factory(d, urls=urls))
    assert code == 0 and rt['status'] == 'EXPECTED_FROZEN_SOURCE_UNAVAILABLE' and rt['replacement_data_used'] is False and rt['model_rerun'] is False and rt['selection_rerun'] is False
    assert rt['scientific_artifacts_changed'] is False and rt['observed_2026_bytes_deleted'] is True
    assert {p['reason'] for p in rt['unavailable_pins']} == {'DIGEST_MISMATCH'} and len(rt['available_pins_verified']) == len(d['available_pins'])
    assert not (tmp_path / 'data' / '_observed_2026_scratch').exists()
    assert not any(p.name.endswith(('pbp_2026.csv.gz', 'stats_player_week_2026.csv')) for p in (tmp_path / 'data').rglob('*'))     # revised 2026 bytes are never kept
    assert set(urls) == {p['url'] for p in d['available_pins'] + d['unavailable_pins']} and not any('api.github.com' in u or 'latest' in u for u in urls)
    assert json.loads((tmp_path / 'rt.json').read_text())['wording'] == d['wording']


def test_all_pins_available_means_the_original_reproduction_may_run(tmp_path, fake_sha):
    code, rt = A.run_gate('k', tmp_path / 'data', tmp_path / 'rt.json', fetch=fake_fetch_factory(static('k'), drift_2026=False))
    assert code == 0 and rt['status'] == 'ALL_AVAILABLE' and rt['unavailable_pins'] == [] and rt['wording'] == ['ARCHIVAL_INTEGRITY_VERIFIED']


def test_2026_not_served_is_the_same_expected_condition(tmp_path, fake_sha):
    code, rt = A.run_gate('k', tmp_path / 'data', tmp_path / 'rt.json', fetch=fake_fetch_factory(static('k'), missing_2026=True))
    assert code == 0 and rt['status'] == 'EXPECTED_FROZEN_SOURCE_UNAVAILABLE' and {p['reason'] for p in rt['unavailable_pins']} == {'NOT_SERVED'}


def test_a_changed_older_stable_pin_still_fails(tmp_path, fake_sha):
    d = static('k')
    code, rt = A.run_gate('k', tmp_path / 'data', tmp_path / 'rt.json', fetch=fake_fetch_factory(d, break_stable=d['available_pins'][0]['local_name']))
    assert code == 1 and rt['status'] == 'STABLE_PIN_MISMATCH' and not (tmp_path / 'data' / d['available_pins'][0]['local_name']).exists()
    assert rt['wording'] == [] and rt['scientific_status'] == 'NOT_VERIFIED'


def test_an_unobtainable_stable_pin_fails(tmp_path, fake_sha):
    d = static('k')
    code, rt = A.run_gate('k', tmp_path / 'data', tmp_path / 'rt.json', fetch=fake_fetch_factory(d, stable_missing=d['available_pins'][1]['local_name']))
    assert code == 1 and rt['status'] == 'STABLE_PIN_UNOBTAINABLE'


def test_a_changed_or_missing_scientific_artifact_fails_before_any_download(tmp_path, fake_sha):
    d = static('k')
    bad = deepcopy(d)
    first = sorted(bad['artifacts_sha256'])[0]
    bad['artifacts_sha256'][first] = '0' * 64                                   # an artifact whose bytes differ from the record
    p = tmp_path / 's.json'
    p.write_text(json.dumps(bad))
    called = []
    code, rt = A.run_gate('k', tmp_path / 'data', tmp_path / 'rt.json', fetch=lambda *a, **k: called.append(a) or types.SimpleNamespace(returncode=0, stderr=''), status_path=p)
    assert code == 1 and rt['status'] == 'FROZEN_ARTIFACT_CHANGED_OR_MISSING' and rt['scientific_artifacts_changed'] is True and called == []
    missing = deepcopy(d)
    missing['artifacts_sha256']['nfl_models/nfl_player_outcome_v2/phase1k_does_not_exist.json'] = '1' * 64
    p.write_text(json.dumps(missing))
    code, rt = A.run_gate('k', tmp_path / 'data2', tmp_path / 'rt2.json', fetch=fake_fetch_factory(d), status_path=p)
    assert code == 1 and rt['status'] == 'FROZEN_ARTIFACT_CHANGED_OR_MISSING'


@pytest.mark.parametrize('key', sorted(A.PHASES))
def test_gate_module_cannot_execute_model_code_and_workflows_gate_every_science_step(key):
    tree = ast.parse((ROOT / 'nfl_v2_archival_status.py').read_text())
    imported = {n.names[0].name.split('.')[0] if isinstance(n, ast.Import) else n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))}
    assert imported <= {'__future__', 'argparse', 'hashlib', 'json', 'os', 'pathlib', 'shutil', 'subprocess'}          # stdlib only: no numpy, no phase code
    assert 'import nfl_v2_phase' not in (ROOT / 'nfl_v2_archival_status.py').read_text() and 'numpy' not in (ROOT / 'nfl_v2_archival_status.py').read_text()
    wf = yaml.safe_load((ROOT / '.github/workflows' / WORKFLOWS[key]).read_text())
    steps = wf['jobs']['research']['steps']
    gate = next(s for s in steps if s.get('id') == 'archival')
    assert f'nfl_v2_archival_status.py --phase {key}' in gate['run'] and 'continue-on-error' not in gate and '|| true' not in gate['run']
    idx = steps.index(gate)
    for step in steps[idx + 1:]:
        text = step.get('run', '') + json.dumps(step.get('with', {}))
        science = 'nfl_v2_phase1' in text and 'phase1' in step.get('run', '') or 'cmp nfl_models' in step.get('run', '')
        if science:
            assert "steps.archival.outputs.status == 'ALL_AVAILABLE'" in step['if'], step['name']
    arch_comment = next(s for s in steps if s.get('name', '').startswith('Archival status on existing PR'))
    assert "steps.archival.outputs.status == 'EXPECTED_FROZEN_SOURCE_UNAVAILABLE'" in arch_comment['if']
    assert 'Scientific verdict' in arch_comment['with']['script'] and 'No replacement data were used' in arch_comment['with']['script']
    assert 'REPRODUCTION_PASSED' not in (ROOT / '.github/workflows' / WORKFLOWS[key]).read_text()


def test_scientific_artifacts_are_not_modified_by_this_change():
    r = subprocess.run(['git', 'diff', '--name-status', '35e3be75f8cdf8a546c2dcd938bb8453b0d80897', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('base commit unavailable')
    for line in r.stdout.splitlines():
        status, _, path = line.partition('\t')
        if path.startswith('nfl_models/nfl_player_outcome_v2/phase1[hik]_'[:36]) and any(path.split('/')[-1].startswith(f'phase1{k}_') for k in 'hik'):
            assert status == 'A' and path.endswith('_archival_status.json'), (status, path)      # only the new archival records; no frozen science file changes
        assert not (path.startswith('nfl_v2_phase1') and path.endswith('.py') and path.split('_')[2][:6] in ('phase1h', 'phase1i', 'phase1k')), path
