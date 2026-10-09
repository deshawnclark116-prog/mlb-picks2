"""Operational fault injection. Synthetic clocks/stubs live exclusively in tests."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_phase1d_cas as CAS
import nfl_phase1e_scheduler as DS
import nfl_phase1_store_lock as LK
import nfl_shadow_schedule as SCHEDULE
import nfl_shadow_durability as DUR
import nfl_shadow_ownership as OWN
import nfl_shadow_watchdog as DOG
import nfl_phase1_publisher as PUB
from test_nfl_phase1e import Clock, KICK, make, sched_bytes, FakeRunner
from test_nfl_new_engine_frontend import make_state, rec, G1, C24, C90


def test_host_mismatch_is_not_contention(tmp_path):
    (tmp_path / 'host_binding.json').write_text(json.dumps({'host_id': 'foreign-runner'}))
    d, _ = make(tmp_path, Clock(KICK - timedelta(hours=30)))
    with pytest.raises(CAS.CASError, match='HOST_BINDING_MISMATCH'):
        d.tick()
    assert not (tmp_path / 'dispatch_ledger.jsonl').exists()


def test_hundred_skipped_ticks_cannot_succeed(tmp_path):
    d, _ = make(tmp_path, Clock(KICK - timedelta(hours=30)))
    # Regression for the historical 100-skipped-tick green Actions run.
    with patch.object(d, 'tick', return_value={'skipped': 'held'}), patch.object(d, 'heartbeat'):
        with pytest.raises(CAS.CASError, match='LOCK_CONTENTION'):
            d.run(duration_min=100, every_sec=60, do_prefit=False)


def test_schedule_404_is_failure_and_status_not_ready(tmp_path):
    d, _ = make(tmp_path, Clock(KICK - timedelta(hours=30)))
    d.fetch_schedule = lambda: (_ for _ in ()).throw(RuntimeError('SCHEDULE_FETCH_404: provider'))
    with pytest.raises(RuntimeError, match='SCHEDULE_FETCH_404'):
        d.tick()
    s = json.loads((tmp_path / 'status.json').read_text())
    assert not s['readiness']['READY']
    assert 'SCHEDULE_FETCH_404' in s['last_operational_event']['error']
    assert s['last_successful_schedule_capture_utc'] is None


def test_schedule_allowlist_no_scores_or_starters_or_markets():
    raw = sched_bytes().decode().replace('total_line\n', 'total_line,home_score,home_qb_id\n').replace('44.5\n', '44.5,99,future-qb\n').encode()
    out, sha, tr = SCHEDULE.sanitize(raw, KICK - timedelta(hours=30))
    assert sha == hashlib.sha256(raw).hexdigest()
    assert all(x not in out for x in (b'home_score', b'qb_id', b'spread_line', b'total_line', b'future-qb'))
    assert b'COMPLETED' not in out and 'metadata' in tr


def test_stale_planned_closes_before_kickoff_and_never_backfills(tmp_path):
    c = Clock(KICK - timedelta(hours=30)); d, _ = make(tmp_path, c); d.tick()
    c.t = KICK - timedelta(hours=23); out = d.tick()
    assert out['expired'] == 1
    assert next(r for r in d.states().values() if r['horizon'] == 'T24')['state'] == 'MISSED_REAL_CUTOFF'
    assert not d.runner.calls


def test_watchdog_invocation_is_not_capture_and_overdue_keys_wake(tmp_path):
    c = Clock(KICK - timedelta(hours=30)); d, _ = make(tmp_path, c); d.tick()
    (tmp_path / 'status.json').write_text(json.dumps({'last_invocation_utc': CAS.iso(c())}))
    out = DOG.decide(tmp_path, KICK + timedelta(hours=1))
    assert out['action'] == 'dispatch' and out['stale']
    assert DOG.decide(tmp_path, KICK + timedelta(hours=1), active_runs=1)['action'] == 'skip'


def test_durable_blobs_survive_cache_loss_and_corruption_fails(tmp_path):
    store = CAS.BlobStore(tmp_path / 'cas'); b = b'original source bytes' * 500
    sha = store.put(b)['sha256']; assert DUR.seal(tmp_path) == 1
    shutil.rmtree(tmp_path / 'cas' / 'blobs')
    assert DUR.restore(tmp_path) == 1 and store.get(sha) == b
    store.path(sha).unlink(); z = next((tmp_path / 'durable_blobs').rglob('*.gz')); z.write_bytes(b'bad')
    with pytest.raises(RuntimeError, match='chunk hash'):
        DUR.restore(tmp_path)


def test_checkpoints_fail_loudly(tmp_path):
    d, _ = make(tmp_path, Clock(KICK - timedelta(hours=30)))
    with pytest.raises(CAS.CASError, match='NOT saved'):
        d.checkpoint_if_changed(None, 'exit 1')


def test_synthetic_t24_t90_done_durable_and_nonempty_publication(tmp_path):
    # Real scheduler + CAS + immutable forecast-store + strict publisher. Forecast
    # values are synthetic, NOT operational/live scientific evidence.
    from test_nfl_phase1e import adv
    c = Clock(KICK - timedelta(hours=24, minutes=4))
    root = tmp_path / 'state'; root.mkdir()
    class Runner(FakeRunner):
        def group_name(self, s, w, h, k):
            return f'LIVE_{s}_{w}_{h}_{CAS.iso(k)}'
        def run_group(self, s, w, h, k, gids, run_id):
            import nfl_phase1_store as ST
            cut = k - (timedelta(hours=24) if h == 'T24' else timedelta(minutes=90))
            sr = CAS.take_snapshot_set(CAS.BlobStore(root / 'cas'), CAS.Ledger(root / 'cas'),
                {'games.csv': ('synthetic:test', lambda: SCHEDULE.sanitize(sched_bytes(k, gids[0]), c())[0])},
                h, k, cut, c(), self.group_name(s,w,h,k), time_travel=False)
            self.sets[self.group_name(s,w,h,k)] = sr
            CAS.Ledger(root, 'prefit_ledger.jsonl').append_many([{'season':s,'week':w,
                'artifact_bundle_sha256':'test-only', 'created_at':CAS.iso(cut-timedelta(hours=1)),
                'source_identity':{'fit_retrieval_ts':CAS.iso(cut-timedelta(hours=2))}}])
            r = rec(gids[0], h, CAS.iso(cut), 'synthetic eligible RB', 'rush_yds', kickoff=CAS.iso(k),
                input_snapshots={'snapshot_set_id':sr['set_id'], 'retrieval_ts':sr['retrieval_ts'],
                                 'artifact_bundle_sha256':'test-only', 'time_travel':False})
            ST.Store(root,'forecasts').append_batch(h, {'generated_at':CAS.iso(c()),'game_id':gids[0]}, [r])
            return sr, 'test-only', [{'game_id':gids[0], 'status':'FORECAST_SUCCESS', 'n_records':1,'written':1}]
    d = DS.Dispatcher(root, 1000, runner=Runner(root), fetch_schedule=lambda:(sched_bytes(), 'test'), clock=c, sleep=adv(c), log=lambda x:None)
    assert d.tick()['done'] == 1
    c.t = KICK - timedelta(minutes=94); assert d.tick()['done'] == 1
    assert {r['state'] for r in d.states().values()} == {'DONE'}
    d.heartbeat(); DUR.seal(root); shutil.rmtree(root / 'cas' / 'blobs')
    out = tmp_path / 'new-engine.json'; PUB.publish(root, out)
    doc = json.loads(out.read_text())
    assert doc['status']['publication_state'] == 'PUBLISHED'
    assert len(doc['forecasts']['T24']) == len(doc['forecasts']['T90']) == 1
    assert not doc['status']['verification_errors']
    # Simulate restart without Actions cache; hashes and receipts still verify.
    DUR.restore(root); assert d.tick()['done'] == 0


def test_publisher_rejects_late_or_missing_proof_and_reports_empty(tmp_path):
    r = rec(G1,'T90',C90,'alpha','rush_yds')
    st = make_state(tmp_path, [(G1,'T90',C90,['DONE'],[r],{})])
    shutil.rmtree(st / 'cas' / 'blobs')
    d = PUB.build(st)
    assert not d['forecasts']['T90'] and d['status']['publication_state'] == 'PUBLISH_EMPTY'
    assert 'OPERATIONAL_FAILURE' in d['empty_message'] and d['status']['verification_errors']


def git_repo(tmp):
    bare = tmp / 'remote.git'; subprocess.run(['git','init','--bare',str(bare)],check=True,capture_output=True)
    root=tmp/'state'; subprocess.run(['git','clone',str(bare),str(root)],check=True,capture_output=True)
    OWN.git(root,'checkout','-b',OWN.BRANCH); OWN.git(root,'config','user.name','test'); OWN.git(root,'config','user.email','test@example.com')
    (root/'.gitignore').write_text('*.lock\ncas/blobs/\n')
    (root/'host_binding.json').write_text(json.dumps({'host_id':'old-host'}))
    (root/'historical.jsonl').write_text('{"frozen":true}\n')
    OWN.git(root,'add','.'); OWN.git(root,'commit','-m','original'); OWN.git(root,'push','origin',OWN.BRANCH)
    return root


def test_rotating_actions_hosts_preserve_bindings_and_git_cas(tmp_path, monkeypatch):
    root=git_repo(tmp_path); before=(root/'host_binding.json').read_bytes()
    for k,v in {'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':OWN.REPO,'GITHUB_WORKFLOW_REF':OWN.REPO+'/'+OWN.WORKFLOW+'@refs/heads/main',
                'NFL_SHADOW_CONCURRENCY':'nfl-phase1e-shadow','GITHUB_RUN_ID':'1','GITHUB_RUN_ATTEMPT':'1'}.items(): monkeypatch.setenv(k,v)
    original=LK.bind_host
    try:
        # Install operational binding adapter in child, each using different host.
        code="import nfl_shadow_ownership as O; O.install(); import nfl_phase1_store_lock as L; L.bind_host(__import__('sys').argv[1])"
        env_path = str(Path(__file__).resolve().parents[1]); monkeypatch.setenv('PYTHONPATH', env_path)
        for host in ['runner-A','runner-B']:
            monkeypatch.setenv('NFL_STORE_HOST_ID',host)
            assert OWN.run(root,[sys.executable,'-c',code,str(root)]) == 0
        assert (root/'host_binding.json').read_bytes()==before
        transitions=[json.loads(x) for x in (root/'ownership_transitions.jsonl').read_text().splitlines()]
        assert len(transitions)==2
        assert all(r['legacy_binding_probe']['status']=='HOST_BINDING_MISMATCH' for r in transitions)
        assert all('old-host' in r['legacy_binding_probe']['exact_error'] for r in transitions)
        # No audited context: original physical binding still refuses.
        monkeypatch.delenv('NFL_SHADOW_SESSION',raising=False)
        with pytest.raises(LK.LockError): LK.bind_host(root)
    finally:
        LK.bind_host=original


def test_concurrent_remote_advance_and_ledger_rewrite_fail_closed(tmp_path):
    root=git_repo(tmp_path); base=OWN.git(root,'rev-parse','HEAD')
    (root/'historical.jsonl').write_text('{"changed":true}\n')
    with pytest.raises(RuntimeError,match='rewritten historical'):
        OWN.immutable_check(root,base)
    (root/'historical.jsonl').write_text('{"frozen":true}\n{"append":true}\n'); OWN.immutable_check(root,base)
    fake=tmp_path/'session.json'; fake.write_text(json.dumps({'root':str(root),'base':'stale'}))
    with patch.object(OWN,'proof',return_value=(fake,{'root':str(root),'base':'stale'})):
        with pytest.raises(RuntimeError,match='remote advanced'):
            OWN.checkpoint(root)


def test_frozen_predictive_code_hashes_unchanged():
    import nfl_phase1_forecast as FC
    repo=Path(__file__).resolve().parents[1]
    for f in FC.CODE_FILES:
        original=subprocess.check_output(['git','show',f'origin/main:{f}'],cwd=repo)
        assert hashlib.sha256(original).hexdigest()==FC.file_hashes()[f], f
    import nfl_phase1e_ops as OPS
    assert OPS.code_identity()=='b77dc75b8587a359ae481b1c24cbaee0633915b20a79f63028a8582241bb079d'


def test_actual_workflow_and_checkpoint_cli_arguments_parse():
    a, child = OWN.parse_cli(['run', '--root', 'state', '--', 'python', 'nfl_phase1e_scheduler.py', 'run', '--root', 'state'])
    assert a.root == 'state' and a.cmd == 'run' and child[-2:] == ['--root', 'state']
    a, child = OWN.parse_cli(['checkpoint', '--root', '/absolute/state'])
    assert a.cmd == 'checkpoint' and a.root == '/absolute/state' and not child


def test_actual_concurrent_push_race_loses_lease_without_rewriting(tmp_path):
    root = git_repo(tmp_path); base = OWN.git(root, 'rev-parse', 'HEAD')
    other = tmp_path/'other'
    subprocess.run(['git','clone','-b',OWN.BRANCH,str(tmp_path/'remote.git'),str(other)],check=True,capture_output=True)
    OWN.git(other,'config','user.name','other'); OWN.git(other,'config','user.email','other@example.com')
    (other/'winner.txt').write_text('concurrent winner')
    OWN.git(other,'add','.'); OWN.git(other,'commit','-m','winner')
    winner = OWN.git(other,'rev-parse','HEAD')
    (root/'loser.txt').write_text('must remain local for diagnosis')
    session=tmp_path/'session.json'
    proof={'root':str(root),'base':base,'context':{'GITHUB_RUN_ID':'test'}}
    real_run=subprocess.run
    def race(args, *a, **kw):
        if len(args)>3 and args[0]=='git' and 'push' in args and any('--force-with-lease=' in x for x in args):
            OWN.git(other,'push','origin',OWN.BRANCH)
        return real_run(args,*a,**kw)
    with patch.object(OWN,'proof',return_value=(session,proof)), patch('subprocess.run',side_effect=race):
        with pytest.raises(RuntimeError,match='NOT saved'):
            OWN.checkpoint(root)
    assert OWN.remote_head(root)==winner
    assert OWN.git(root,'rev-parse','HEAD') != winner
    assert (root/'historical.jsonl').read_text()=='{"frozen":true}\n'


def test_transient_data_failure_remains_retryable_before_cutoff(tmp_path):
    class UnavailableRunner(FakeRunner):
        def run_group(self, *a):
            raise CAS.CASError('DATA_UNAVAILABLE: transient source error')
    c=Clock(KICK-timedelta(hours=24,minutes=4))
    d,_=make(tmp_path,c,runner=UnavailableRunner(tmp_path))
    out=d.tick()
    r=next(r for r in d.states().values() if r['horizon']=='T24')
    assert r['state']=='STARTED' and out['done']==0
    d.heartbeat()
    assert json.loads((tmp_path/'status.json').read_text())['last_operational_event']['status']=='FAILED'
    c.t=KICK-timedelta(hours=24)+timedelta(seconds=1)
    d.tick()
    assert d.states()[r['key']]['state']=='MISSED_REAL_CUTOFF'


def test_publisher_empty_failure_is_red_after_diagnostic_is_saved(tmp_path):
    st=make_state(tmp_path,[(G1,'T90',C90,['MISSED_REAL_CUTOFF'],[],{})])
    out=tmp_path/'diagnostic.json'
    result=subprocess.run([sys.executable,str(Path(PUB.__file__)), '--state',str(st),'--out',str(out),'--fail-on-operational-empty'],capture_output=True,text=True)
    assert result.returncode != 0 and 'PUBLISH_EMPTY' in result.stderr
    assert out.exists() and json.loads(out.read_text())['status']['publication_state']=='PUBLISH_EMPTY'
