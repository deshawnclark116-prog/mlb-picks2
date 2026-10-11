#!/usr/bin/env python3
"""Archival-integrity gate for the frozen Phase1H / Phase1I / Phase1K research workflows.

Stdlib only. It imports NO model, fitting or phase code, so the archival success path cannot execute science.

What it does, in order:
  1. verifies every frozen scientific artifact and code file against the committed archival status file (any change fails);
  2. downloads and verifies every still-obtainable frozen pin (2023-2025); any mismatch or missing file FAILS;
  3. observes the frozen 2026 pins in a scratch directory, compares digests, then DELETES the observed bytes so nothing can be
     scored on them;
  4. emits ALL_AVAILABLE (every pin matches: the original reproduction steps may run) or EXPECTED_FROZEN_SOURCE_UNAVAILABLE
     (the only mismatches are the known 2026 pins), writing a machine-readable runtime status artifact.
It never substitutes newer 2026 files, reruns a model or selection, or claims a reproduction it did not perform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
COVERAGE = ART / 'phase1h_source_coverage.json'
ALL_AVAILABLE = 'ALL_AVAILABLE'
UNAVAILABLE = 'EXPECTED_FROZEN_SOURCE_UNAVAILABLE'
STABLE_YEARS = ('2023', '2024', '2025')
EXPECTED_UNAVAILABLE_YEARS = ('2026',)

PHASES = {
    'h': {'phase': 'PHASE1H-R routed receiving', 'prefix': 'phase1h_', 'kinds': None, 'workflow': 'nfl_v2_phase1h_routed_receiving.yml',
          'code': ('nfl_v2_phase1h_sources.py', 'nfl_v2_phase1h_routed_receiving.py', 'tests/test_nfl_v2_phase1h_routed_receiving.py'),
          'results': 'phase1h_routed_receiving_results.json', 'snapshot': 'phase1h_routed_receiving_snapshot.json',
          'frozen_source_run': 37362164432, 'frozen_source_head': '761f060a1ed5eaab235fb1056fd59e6168c48372', 'freeze_commit': '63d44d453f726991be6d3e8fd04ec86d28c8d636',
          'comment_marker': 'nfl-phase1h-r', 'status_file': 'phase1h_archival_status.json', 'extra_hashed': ()},
    'i': {'phase': 'PHASE1I-A target depth / catchability', 'prefix': 'phase1i_', 'kinds': None, 'workflow': 'nfl_v2_phase1i_target_depth.yml',
          'code': ('nfl_v2_phase1i_sources.py', 'nfl_v2_phase1i_target_depth.py', 'tests/test_nfl_v2_phase1i_target_depth.py'),
          'results': 'phase1i_target_depth_results.json', 'snapshot': 'phase1i_target_depth_snapshot.json',
          'frozen_source_run': 37383777787, 'frozen_source_head': '411d6f8ca12f24e1f192ab2d7768589cf77d1bc0', 'freeze_commit': '411d6f8ca12f24e1f192ab2d7768589cf77d1bc0',
          'comment_marker': 'nfl-phase1i-a', 'status_file': 'phase1i_archival_status.json', 'extra_hashed': ('phase1h_source_coverage.json',)},
    'k': {'phase': 'PHASE1K-R PBP-only rushing efficiency', 'prefix': 'phase1k_', 'kinds': ('pbp', 'stats'), 'workflow': 'nfl_v2_phase1k_rushing_efficiency.yml',
          'code': ('nfl_v2_phase1k_sources.py', 'nfl_v2_phase1k_rushing_efficiency.py', 'tests/test_nfl_v2_phase1k_rushing_efficiency.py'),
          'results': 'phase1k_rushing_results.json', 'snapshot': 'phase1k_rushing_snapshot.json',
          'frozen_source_run': 37396653808, 'frozen_source_head': '28c96ef98cb0835f732d822dc8d1a5adaaa942f3', 'freeze_commit': '28c96ef98cb0835f732d822dc8d1a5adaaa942f3',
          'comment_marker': 'nfl-phase1k-r', 'status_file': 'phase1k_archival_status.json', 'extra_hashed': ('phase1h_source_coverage.json',)},
}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(value):
    return json.dumps(value, indent=2, sort_keys=True) + '\n'


def frozen_pins(cfg):
    """[{year, kind, local_name, url, expected_sha256}] from the frozen Phase1H coverage manifest (pins only; no sha = not published)."""
    seasons = json.loads(COVERAGE.read_text())['seasons']
    pins = []
    for year in sorted(seasons):
        for kind in sorted(seasons[year]):
            if cfg['kinds'] and kind not in cfg['kinds']:
                continue
            m = seasons[year][kind]
            if not m.get('sha256'):
                continue
            pins.append({'year': year, 'kind': kind, 'local_name': m['local_name'], 'url': m['url'], 'expected_sha256': m['sha256']})
    return pins


def hashed_files(cfg):
    """Frozen scientific artifacts, code and shared inputs whose bytes must never change."""
    files = sorted(p.name for p in ART.glob(cfg['prefix'] + '*') if p.name != cfg['status_file'])
    out = {'nfl_models/nfl_player_outcome_v2/' + n: sha_file(ART / n) for n in files}
    for n in cfg['extra_hashed']:
        out['nfl_models/nfl_player_outcome_v2/' + n] = sha_file(ART / n)
    for rel in cfg['code']:
        out[rel] = sha_file(ROOT / rel)
    return dict(sorted(out.items()))


def build_static(phase_key):
    cfg = PHASES[phase_key]
    pins = frozen_pins(cfg)
    results = json.loads((ART / cfg['results']).read_text())
    available = [p for p in pins if p['year'] in STABLE_YEARS]
    unavailable = [p for p in pins if p['year'] in EXPECTED_UNAVAILABLE_YEARS]
    return {
        'schema': 'nfl-v2-archival-status-v1', 'phase': cfg['phase'], 'workflow': cfg['workflow'],
        'frozen_source_run': cfg['frozen_source_run'], 'frozen_source_head': cfg['frozen_source_head'], 'scientific_freeze_commit': cfg['freeze_commit'],
        'frozen_verdict': results['verdict'],
        'status': UNAVAILABLE, 'scientific_status': 'FROZEN_VALID_RESULT_REPRODUCTION_CURRENTLY_BLOCKED',
        'status_note': 'Registered classification observed on 2026-10-06: the exact pinned 2026 upstream bytes are no longer served (nflverse rewrites the 2026 release assets as the season progresses). The runtime gate re-observes this on every run and fails if anything else changed.',
        'wording': ['SCIENTIFIC_RESULT_FROZEN', 'REPRODUCTION_BLOCKED_EXACT_2026_BYTES_UNAVAILABLE', 'ARCHIVAL_INTEGRITY_VERIFIED'],
        'not_claimed': 'REPRODUCTION_PASSED (the old experiment is not rerun while its exact 2026 bytes are unavailable)',
        'available_pins': available, 'unavailable_pins': unavailable,
        'expected_frozen_digest': {p['local_name']: p['expected_sha256'] for p in unavailable},
        'artifacts_sha256': hashed_files(cfg),
        'replacement_data_used': False, 'model_rerun': False, 'selection_rerun': False, 'scientific_artifacts_changed': False,
    }


def curl(url, dest, timeout=280):
    return subprocess.run(['curl', '-fsSL', '--retry', '5', '--retry-all-errors', '--retry-delay', '5', '--max-time', str(timeout), url, '-o', str(dest)], capture_output=True, text=True)


def run_gate(phase_key, data_dir, runtime_out, fetch=curl, status_path=None, attempts=4, sleep=time.sleep):
    """Returns (exit_code, runtime_document). exit_code 0 only for ALL_AVAILABLE or the exact known 2026 condition."""
    cfg = PHASES[phase_key]
    static = json.loads(Path(status_path or ART / cfg['status_file']).read_text())
    runtime = {k: static[k] for k in ('schema', 'phase', 'frozen_source_run', 'frozen_source_head', 'frozen_verdict', 'expected_frozen_digest')}
    runtime.update({'replacement_data_used': False, 'model_rerun': False, 'selection_rerun': False, 'scientific_artifacts_changed': False,
                    'available_pins_verified': [], 'unavailable_pins': [], 'current_upstream_digest_if_observed': {}, 'observed_2026_bytes_deleted': True})

    def finish(code, status, reason=None):
        runtime['status'], runtime['exit_code'] = status, code
        if reason:
            runtime['failure'] = reason
        runtime['scientific_status'] = static['scientific_status'] if status in (ALL_AVAILABLE, UNAVAILABLE) else 'NOT_VERIFIED'
        runtime['wording'] = static['wording'] if status == UNAVAILABLE else (['ARCHIVAL_INTEGRITY_VERIFIED'] if status == ALL_AVAILABLE else [])
        Path(runtime_out).parent.mkdir(parents=True, exist_ok=True)
        Path(runtime_out).write_text(dump(runtime))
        return code, runtime

    # 1. frozen scientific artifacts and code must be byte-identical to the committed archival record
    current = hashed_files(cfg)
    if current != static['artifacts_sha256']:
        diff = sorted(set(current) ^ set(static['artifacts_sha256']) | {k for k in current if k in static['artifacts_sha256'] and current[k] != static['artifacts_sha256'][k]})
        runtime['scientific_artifacts_changed'] = True
        return finish(1, 'FROZEN_ARTIFACT_CHANGED_OR_MISSING', diff)
    # 2. every still-obtainable frozen pin must verify
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for pin in static['available_pins']:
        dest = data_dir / pin['local_name']
        if not (dest.exists() and sha_file(dest) == pin['expected_sha256']):
            # transient transport failures (many CI jobs download the same releases at once) are retried; a persistent failure or any digest mismatch still FAILS
            for attempt in range(1, attempts + 1):
                r = fetch(pin['url'], dest)
                if r.returncode == 0 and dest.exists():
                    break
                sleep(15 * attempt)
            else:
                return finish(1, 'STABLE_PIN_UNOBTAINABLE', {'pin': pin['local_name'], 'attempts': attempts, 'stderr': (getattr(r, 'stderr', '') or '')[-200:]})
        got = sha_file(dest)
        if got != pin['expected_sha256']:
            dest.unlink()
            return finish(1, 'STABLE_PIN_MISMATCH', {'pin': pin['local_name'], 'expected': pin['expected_sha256'], 'observed': got})
        runtime['available_pins_verified'].append({'local_name': pin['local_name'], 'sha256': got})
    # 3. observe the frozen 2026 pins in a scratch directory; the bytes never reach the science data directory
    scratch = data_dir / '_observed_2026_scratch'
    scratch.mkdir(exist_ok=True)
    mismatched = []
    try:
        for pin in static['unavailable_pins']:
            dest = scratch / pin['local_name']
            r = fetch(pin['url'], dest)
            observed = sha_file(dest) if r.returncode == 0 and dest.exists() else None
            runtime['current_upstream_digest_if_observed'][pin['local_name']] = observed
            if observed != pin['expected_sha256']:
                mismatched.append(pin['local_name'])
                runtime['unavailable_pins'].append({'local_name': pin['local_name'], 'expected_sha256': pin['expected_sha256'], 'observed_sha256': observed,
                                                    'reason': 'NOT_SERVED' if observed is None else 'DIGEST_MISMATCH'})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if not mismatched:
        return finish(0, ALL_AVAILABLE)
    expected = {p['local_name'] for p in static['unavailable_pins']}
    if not set(mismatched) <= expected:
        return finish(1, 'UNEXPECTED_MISMATCH', sorted(set(mismatched) - expected))
    return finish(0, UNAVAILABLE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=sorted(PHASES))
    ap.add_argument('--data-dir')
    ap.add_argument('--runtime-out')
    ap.add_argument('--write-static', action='store_true', help='regenerate the committed static archival record (maintainer use; never run by CI)')
    ap.add_argument('--github-output', default=os.environ.get('GITHUB_OUTPUT'))
    args = ap.parse_args()
    if args.write_static:
        for key in sorted(PHASES):
            (ART / PHASES[key]['status_file']).write_text(dump(build_static(key)))
        return
    code, runtime = run_gate(args.phase, args.data_dir, args.runtime_out)
    print(json.dumps({k: runtime[k] for k in ('status', 'frozen_verdict') if k in runtime}))
    if args.github_output:
        with open(args.github_output, 'a') as f:
            f.write('status=%s\n' % runtime['status'])
    raise SystemExit(code)


if __name__ == '__main__':
    main()
