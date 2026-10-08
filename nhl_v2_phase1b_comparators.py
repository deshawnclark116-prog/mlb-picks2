"""Prospective independent analyst intake; never invents missing professional forecasts."""
import math
from nhl_v2_phase1b_snapshots import timestamp
from nhl_v2_phase1a_sog_forward import HORIZON_MIN, canon, sha_text, Ledger


def accept_independent_forecast(record, decision, now, ledger_path):
    required = ('author', 'method_version', 'publication_timestamp', 'retrieval_timestamp',
                'evidence_hashes', 'expected_sog', 'pmf', 'rights_basis', 'independent_of_engine')
    if not all(record.get(k) is not None for k in required) or not record['author'] or not record['rights_basis'] or record['independent_of_engine'] is not True:
        raise ValueError('independent authorized evidence required')
    if record.get('football_evidence_only') is not True or not record['evidence_hashes']:
        raise ValueError('independent football evidence required')
    for h in record['evidence_hashes']:
        if len(h) != 64 or any(x not in '0123456789abcdef' for x in h):
            raise ValueError('invalid evidence hash')
    if any(record[k] != decision[k] for k in ('game_id', 'player_id', 'team_id', 'horizon', 'scheduled_start')):
        raise ValueError('different analyst decision')
    if record['horizon'] not in HORIZON_MIN:
        raise ValueError('unsupported horizon')
    if any(timestamp(record[t]) > timestamp(decision['cutoff']) for t in ('publication_timestamp', 'retrieval_timestamp')) or timestamp(now) > timestamp(decision['cutoff']):
        raise ValueError('late professional forecast cannot be backfilled')
    pm = record['pmf']
    if not pm or any(not math.isfinite(p) or p < 0 for p in pm) or abs(sum(pm)-1) > 1e-10:
        raise ValueError('invalid analyst PMF')
    expected = sum(i*p for i,p in enumerate(pm))
    if abs(expected-record['expected_sog']) > 1e-8:
        raise ValueError('mean does not match PMF')
    store = Ledger(ledger_path)
    store.verify()
    key = tuple(record[k] for k in ('game_id','player_id','team_id','horizon','scheduled_start','author','method_version'))
    if any(tuple(r[k] for k in ('game_id','player_id','team_id','horizon','scheduled_start','author','method_version')) == key for r in store.rows()):
        raise ValueError('professional decision already frozen')
    return store.append(dict(record, evidence='INDEPENDENT_PROSPECTIVE_NOT_SCRIPTED', record_hash=sha_text(canon(record))))
