"""Research source audit only. Reuses frozen bytes; never fetches or fits a model."""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import nhl_v2_phase1a_sog_forward as F

OUT = F.OUT


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def nested_values(value, key):
    if isinstance(value, dict):
        if key in value:
            yield value[key]
        for v in value.values():
            yield from nested_values(v, key)
    elif isinstance(value, list):
        for v in value:
            yield from nested_values(v, key)


def audit():
    F.verify_lock()
    ledger = F.Ledger()
    count = ledger.verify()
    forecasts = [r for r in ledger.rows() if r['record_type'] == 'FORECAST']
    decisions = {}
    mismatch = Counter()
    for r in forecasts:
        decisions.setdefault((r['game_id'], r['forecast_horizon'], r['scheduled_start']), r)
        # Frozen summary helper shifted survival indices. Preserve original fields,
        # disclose separately; repaired metrics compute directly from NB2, not P1..P5.
        from scipy.stats import nbinom
        shape = 1 / r['dispersion']
        for k in range(1, 6):
            correct = nbinom.sf(k - 1, shape, shape / (shape + r['expected_sog']))
            if abs(r['P' + str(k)] - correct) > 1e-8:
                mismatch['P' + str(k)] += 1
    horizons = {}
    blobs_checked = set()
    for h in F.HORIZONS:
        rs = [r for (_, hh, _), r in decisions.items() if hh == h]
        complete = timed = roster = scratches = 0
        for r in rs:
            sources = r['raw_source_hashes']['game_payloads']
            if len(sources) != 3:
                continue
            complete += 1
            eligible = all(s.get('retrieval_completed_utc') and F.parse_iso(s['retrieval_completed_utc']) <= F.parse_iso(r['cutoff_at']) for s in sources)
            timed += eligible
            found_roster = found_scratch = False
            for s in sources:
                raw = gzip.decompress((F.BLOBS / (s['sha256'] + '.gz')).read_bytes())
                if hashlib.sha256(raw).hexdigest() != s['sha256']:
                    raise ValueError('source blob hash mismatch')
                blobs_checked.add(s['sha256'])
                data = json.loads(raw)
                found_roster |= any(bool(x) for x in nested_values(data, 'rosterSpots'))
                found_scratch |= any(bool(x) for x in nested_values(data, 'scratches'))
            roster += eligible and found_roster
            scratches += eligible and found_scratch
        horizons[h] = {'decisions': len(rs), 'complete_endpoint_captures': complete,
                       'complete_pre_cutoff_captures': timed, 'nonempty_roster_observations': roster,
                       'nonempty_scratch_observations': scratches, 'confirmed_dressed_captures': 0,
                       'complete_truth_comparisons': 0, 'dressed_recall': None,
                       'availability_certified': False, 'status': 'UNQUALIFIED_NO_DRESSED_EVIDENCE'}
    coverage = {}
    for season in range(2017, 2026):
        path = OUT / 'phase1a_data' / f'skater_games_{season}.jsonl.gz'
        rows = [json.loads(l) for l in gzip.open(path, 'rt')]
        fields = {}
        for key in ('player_id', 'team_id', 'toi_sec', 'ev_toi_sec', 'pp_toi_sec', 'sh_toi_sec', 'sog', 'shot_attempts', 'ev_attempts', 'pp_attempts', 'pk_attempts', 'completed_at', 'line', 'pp_unit', 'zone_starts'):
            n = sum(r.get(key) is not None for r in rows)
            fields[key] = {'present': n, 'missing': len(rows) - n, 'usable': n == len(rows)}
        coverage[str(season)] = {'rows': len(rows), 'games': len({r['game_id'] for r in rows}),
                                'file_sha256': F.sha_file(path), 'fields': fields,
                                'TOI_reconciles': all(r['toi_sec'] == r['ev_toi_sec'] + r['pp_toi_sec'] + r['sh_toi_sec'] for r in rows)}
    return {'evidence': 'EXISTING_FROZEN_CORPUS_NOT_NEW_COLLECTION', 'ledger_rows': count,
            'ledger_sha256': F.sha_file(F.LEDGER), 'game_source_blobs_verified': len(blobs_checked),
            'horizons': horizons, 'historical_coverage': coverage,
            'B2_frozen_survival_field_mismatches': dict(mismatch),
            'B2_policy': 'Do not change frozen forecasts or engine. Stored P1..P5 are shifted (P1=1); use NB2(mu,alpha) for repaired evaluations and disclose. New engine tests correct P>=k semantics.',
            'publication_times_in_existing_game_payloads': 'NOT_PROVEN; retrieval proves observation only',
            'new_collection': 'DISABLED_PENDING_AUTHORIZED_ACCESS',
            'attempt_component': 'BLOCKED_NO_AUTHORIZED_ATTEMPT_CORPUS',
            'professional_comparator': 'NO_INDEPENDENT_PROSPECTIVE_FORECASTS_AVAILABLE'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(OUT / 'phase1b_source_audit.json'))
    args = parser.parse_args()
    write_json(args.output, audit())
