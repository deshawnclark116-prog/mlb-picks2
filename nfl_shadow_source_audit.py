"""Real-clock operational source probe. No model fitting, no forecast or horizon capture."""
import json
from datetime import datetime, timezone
from pathlib import Path
import nfl_phase1d_cas as CAS
import nfl_phase1d_schedule as SCH
import nfl_phase1e_live as LIVE
import nfl_phase1e_ops as OPS
import nfl_shadow_ownership as OWN


def audit(root):
    OWN.install(); root = Path(root)
    got, sources, missing, raw_schedule = LIVE.fetch_sources()
    retrieved = datetime.now(timezone.utc)
    store = CAS.BlobStore(root / 'cas')
    hashes = {name: store.put(b[0] if isinstance(b, tuple) else b)['sha256'] for name, b in got.items()}
    raw = CAS.BlobStore(root / 'cas_v2_raw_schedule').put(raw_schedule)
    schedule = SCH.parse_schedule(got['games.csv'][0])
    season = max(g['season'] for g in schedule.values())
    lag = LIVE.provider_lag_problems(got, schedule, retrieved, season)
    row = {'retrieval_ts': CAS.iso(retrieved), 'kind': 'OPERATIONAL_SOURCE_AUDIT_NOT_FORECAST',
           'sources': sources, 'stored_hashes': hashes, 'raw_schedule_sha256': raw['sha256'],
           'optional_unavailable': missing, 'provider_lag_missing': lag,
           'code_identity': OPS.code_identity(), 'prefit_verified': {}, 'no_forecast_generated': True}
    for s, w in OPS.prefit_window_weeks(schedule, retrieved)[0]:
        pf = OPS.prefit_row(root, s, w)
        try:
            if pf is None:
                raise CAS.CASError('prefit_missing')
            OPS.load_verified_prefit(root, pf)
            row['prefit_verified'][f'{s}-{w}'] = {'ok': True, 'bundle_sha256': pf['artifact_bundle_sha256']}
        except CAS.CASError as e:
            row['prefit_verified'][f'{s}-{w}'] = {'ok': False, 'error': str(e)}
    row['OK'] = not lag and all(x['ok'] for x in row['prefit_verified'].values())
    CAS.Ledger(root, 'source_audits.jsonl').append_many([row])
    print(json.dumps(row, sort_keys=True))
    if not row['OK']:
        raise CAS.CASError('DATA_UNAVAILABLE: source audit or prefit verification failed')
    return row


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True)
    audit(ap.parse_args().root)
