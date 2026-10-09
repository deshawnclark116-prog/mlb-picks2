"""Optional real frozen-engine plumbing test on real source bytes, with SYNTHETIC TIME.

No forecasts leave pytest's temporary directory. This is NOT real horizon evidence.
Set NFL_SHADOW_INTEGRATION_STATE to a verified source-audit checkout to run it.
"""
import json
import os
import shutil
import sys
from datetime import timedelta
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.skipif(not os.environ.get('NFL_SHADOW_INTEGRATION_STATE'), reason='requires real verified source-audit fixture')
def test_frozen_engine_both_horizons_durable_publisher(tmp_path, monkeypatch):
    import nfl_shadow_publish_evidence as E
    import nfl_shadow_durability as DUR
    import nfl_phase1d_cas as CAS
    import nfl_phase1d_schedule as SCH
    import nfl_phase1d_live as LV
    import nfl_phase1e_live as LIVE
    import nfl_phase1e_scheduler as DS
    import nfl_phase1_publisher as PUB

    source = Path(os.environ['NFL_SHADOW_INTEGRATION_STATE'])
    audit = json.loads((source / 'source_audits.jsonl').read_text().splitlines()[-1])
    assert audit['OK'] and audit['no_forecast_generated']
    # The comparator raw schedule has its own namespace; never use it as model input.
    from nfl_shadow_durability import restore
    restore(source)
    raw = CAS.BlobStore(source / 'cas_v2_raw_schedule').get(audit['raw_schedule_sha256'])
    got = {name: E.blob(source, sha) for name, sha in audit['stored_hashes'].items()}
    schedule = SCH.parse_schedule(got['games.csv'])
    now = LIVE.utcnow()
    gid, game = min(((gid, g) for gid, g in schedule.items() if SCH.forecast_cutoff(g['kick'], 'T24') > now), key=lambda x:x[1]['kick'])
    root = tmp_path / 'SYNTHETIC_TIME_REAL_ENGINE'; root.mkdir()
    shutil.copytree(source / 'artifacts', root / 'artifacts')
    shutil.copyfile(source / 'prefit_ledger.jsonl', root / 'prefit_ledger.jsonl')
    clock = [SCH.forecast_cutoff(game['kick'], 'T24') - timedelta(minutes=4)]
    monkeypatch.setattr(LIVE, 'utcnow', lambda:clock[0])
    original_prepare = LV.prepare
    # Frozen preparation depends on source content/target week, not horizon;
    # cache only in this test to avoid repeating the identical preparation.
    def cached_prepare(*a, **kw):
        kw.update(cache_dir=root/'tmp', cache_key='synthetic-identical-source')
        (root/'tmp').mkdir(exist_ok=True)
        return original_prepare(*a, **kw)
    monkeypatch.setattr(LV,'prepare',cached_prepare)
    runner = LIVE.LiveRunner(root, 100000, fetcher=lambda:(got,audit['sources'],{},raw), log=lambda m:print(m,flush=True))
    runner.checkpoint_snapshot = lambda:DUR.seal(root)
    d = DS.Dispatcher(root, 100000, runner=runner, horizons=('T24',), fetch_schedule=lambda:(raw,'fixture'), clock=lambda:clock[0], log=lambda m:print(m,flush=True))
    first = d.tick(); assert first['done'] >= 1, first
    # Only the selected kickoff group is in the near-horizon window. Other games
    # may be planned but cannot be forced through selection.
    clock[0] = SCH.forecast_cutoff(game['kick'],'T90') - timedelta(minutes=4)
    d.horizons = ('T90',)
    second = d.tick(); assert second['done'] >= 1, second
    d.heartbeat(); DUR.seal(root)
    shutil.rmtree(root/'cas'/'blobs')
    doc=PUB.build(root)
    counts={h:sum(r[0]==gid for r in doc['forecasts'][h]) for h in PUB.HORIZONS}
    assert all(counts.values()), doc['status']['verification_errors']
    active_index = doc['row_fields'].index('p_active')
    meaningful = {h:sum(r[0] == gid and r[active_index] >= .75 for r in doc['forecasts'][h]) for h in PUB.HORIZONS}
    assert all(meaningful.values())
    assert not doc['status']['verification_errors']
    report={'label':'SYNTHETIC TIME — REAL FROZEN ENGINE PLUMBING ONLY, NOT LIVE EVIDENCE',
            'game_id':gid,'rows':counts,'meaningful_rows_p_active_ge_75':meaningful,'forecast_ids':{h:[r[-1] for r in doc['forecasts'][h] if r[0]==gid] for h in PUB.HORIZONS},
            'source_audit_retrieval':audit['retrieval_ts'],'n_draws':100000,'model_versions':doc['status']['model_versions'],
            'cache_deleted_publication_verified':True,'state_path':str(root)}
    out=os.environ.get('NFL_SHADOW_INTEGRATION_REPORT')
    if out: Path(out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='forecast_ids'}))
