"""Separate fail-closed Phase1B ledger gate. No automatic scheduler or network calls.

Only a genuinely ready, pushed lock and independently qualified components may
produce new predictions. The current partial lock is explicitly NOT ready.
"""
from pathlib import Path
from datetime import timedelta

from nhl_v2_phase1a_sog_forward import Ledger, canon, sha_text, sha_file
from nhl_v2_phase1b_snapshots import timestamp


def verify_lock(lock, repo):
    for path, digest in lock['files'].items():
        if sha_file(Path(repo) / path) != digest:
            raise ValueError('Phase1B lock mismatch: ' + path)
    return True


def append_forecast(lock, repo, receipt, now, sources, ledger_path, lock_commit_pushed=False, ci_green=False):
    verify_lock(lock, repo)
    if lock.get('status') != 'READY_FOR_FORWARD' or not lock_commit_pushed or not ci_green:
        raise ValueError('Phase1B blocked: ready pushed lock and green CI required')
    horizon = receipt['horizon']
    cutoff = timestamp(receipt['cutoff'])
    if cutoff < timestamp(lock['eligible_from_cutoff_utc']) or not cutoff-timedelta(seconds=900) <= timestamp(now) <= cutoff:
        raise ValueError('outside valid prospective cutoff window')
    from nhl_v2_phase1a_sog_forward import HORIZON_MIN
    if timestamp(receipt['scheduled_start'])-timedelta(minutes=HORIZON_MIN[horizon]) != cutoff:
        raise ValueError('invalid horizon cutoff')
    cert = lock['availability_certification'][horizon]
    if not cert['qualified_candidate_source'] or not lock['components_validated']:
        raise ValueError('uncertified source/components')
    if not sources or any(not s['timing_eligible'] or s['game_id'] != receipt['game_id'] or s['horizon'] != horizon or s['scheduled_start'] != receipt['scheduled_start'] or timestamp(s['retrieved_at']) > cutoff for s in sources):
        raise ValueError('ineligible forecast evidence')
    ledger = Ledger(ledger_path)
    ledger.verify()
    rows = ledger.rows()
    key = (receipt['game_id'], receipt['player_id'], horizon, receipt['scheduled_start'])
    for r in rows:
        if (r['game_id'], r['player_id'], r['horizon'], r['scheduled_start']) == key:
            raise ValueError('duplicate or cross-team forecast decision')
    rec = dict(receipt, record_type='FORECAST', engine_version=lock['engine_version'],
               lock_hash=sha_text(canon(lock)), generated_at=now,
               source_hashes=[s['record_sha256'] for s in sources])
    return ledger.append(rec)
