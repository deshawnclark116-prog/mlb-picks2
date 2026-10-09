"""Actions logical ownership + host-local session flock + optimistic Git compare-and-swap.

Only the serialized shadow workflow may opt into this ownership protocol. Historical
physical host bindings are evidence and are never rewritten. Ordinary local stores
continue to enforce the original host binding.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = 'deshawnclark116-prog/mlb-picks2'
BRANCH = 'nfl-shadow-state'
WORKFLOW = '.github/workflows/nfl_phase1e_shadow.yml'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def remote_head(root):
    out = git(root, 'ls-remote', 'origin', f'refs/heads/{BRANCH}')
    if not out:
        raise RuntimeError('DURABLE_STATE_CONFLICT: missing state branch')
    return out.split()[0]


def context():
    if (os.environ.get('GITHUB_ACTIONS') != 'true' or os.environ.get('GITHUB_REPOSITORY') != REPO or
        not os.environ.get('GITHUB_WORKFLOW_REF', '').startswith(REPO + '/' + WORKFLOW + '@') or
        os.environ.get('NFL_SHADOW_CONCURRENCY') != 'nfl-phase1e-shadow'):
        raise RuntimeError('HOST_BINDING_MISMATCH: audited Actions writer context required')
    return {k: os.environ[k] for k in ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_WORKFLOW_REF')}


def proof():
    p = Path(os.environ['NFL_SHADOW_SESSION'])
    r = json.loads(p.read_text())
    if r['context'] != context() or r['host'] != physical_host():
        raise RuntimeError('HOST_BINDING_MISMATCH: ownership session context changed')
    try:
        os.kill(r['pid'], 0)
    except OSError as e:
        raise RuntimeError('HOST_BINDING_MISMATCH: ownership session is no longer alive') from e
    return p, r


def physical_host():
    import nfl_phase1_store_lock as LK
    return LK.host_id()


def install():
    """Opt-in operational adapter; frozen lock/model source files and code hashes stay unchanged."""
    if not os.environ.get('NFL_SHADOW_SESSION'):
        return
    import nfl_phase1_store_lock as LK
    original = getattr(LK.bind_host, '_physical_binding', LK.bind_host)

    def audited_bind(root, rebind=False):
        _, session = proof()
        path = Path(root).resolve()
        base = Path(session['root'])
        if path != base and base not in path.parents:
            return original(root, rebind)
        if rebind:
            raise LK.LockError('HOST_BINDING_MISMATCH: physical rebinding forbidden in Actions')
        # Local flock still happens in unchanged StoreLock.acquire. Session flock spans
        # capture, forecast, heartbeat AND Git checkpoints, not only individual appends.
        return {'host_id': session['host'], 'ownership': 'actions-concurrency+git-cas',
                'run_id': session['context']['GITHUB_RUN_ID']}

    audited_bind._physical_binding = original
    LK.bind_host = audited_bind


def immutable_check(root, base):
    """Existing scientific state cannot be edited; ledgers may only grow by complete lines."""
    for rel in git(root, 'ls-tree', '-r', '--name-only', base).splitlines():
        if rel in ('status.json', 'readiness.json'):
            continue
        before = subprocess.check_output(['git', '-C', str(root), 'show', f'{base}:{rel}'])
        p = Path(root) / rel
        if not p.exists():
            raise RuntimeError(f'DURABLE_STATE_CONFLICT: deleted {rel}')
        after = p.read_bytes()
        ok = after.startswith(before) and (after == before or after.endswith(b'\n')) if rel.endswith('.jsonl') and '/batches/' not in rel else after == before
        if not ok:
            raise RuntimeError(f'DURABLE_STATE_CONFLICT: rewritten historical {rel}')


def checkpoint(root):
    from nfl_shadow_durability import seal
    p, s = proof(); root = Path(root).resolve()
    if str(root) != s['root']:
        raise RuntimeError('DURABLE_STATE_CONFLICT: wrong checkpoint root')
    expected = s['base']
    if remote_head(root) != expected:
        raise RuntimeError('DURABLE_STATE_CONFLICT: remote advanced; refusing rebase of immutable ledgers')
    seal(root); immutable_check(root, expected)
    git(root, 'config', 'user.name', 'nfl-shadow-bot')
    git(root, 'config', 'user.email', 'nfl-shadow-bot@users.noreply.github.com')
    git(root, 'add', '-A')
    if subprocess.run(['git', '-C', str(root), 'diff', '--cached', '--quiet']).returncode == 0:
        return expected
    git(root, 'commit', '-m', f"shadow durable checkpoint run {s['context']['GITHUB_RUN_ID']}")
    head = git(root, 'rev-parse', 'HEAD')
    git(root, 'merge-base', '--is-ancestor', expected, head)
    for i in range(4):
        r = subprocess.run(['git', '-C', str(root), 'push',
                            f'--force-with-lease=refs/heads/{BRANCH}:{expected}', 'origin', f'HEAD:refs/heads/{BRANCH}'])
        if r.returncode == 0:
            s['base'] = head; p.write_text(json.dumps(s)); return head
        if remote_head(root) != expected:
            break
        time.sleep(i + 1)
    raise RuntimeError('DURABLE_STATE_CONFLICT: checkpoint NOT saved; no rebase or success claim')


def _run(root, command):
    from nfl_shadow_durability import restore
    ctx = context(); root = Path(root).resolve()
    if git(root, 'branch', '--show-current') != BRANCH:
        raise RuntimeError('DURABLE_STATE_CONFLICT: wrong state branch')
    base = git(root, 'rev-parse', 'HEAD')
    if remote_head(root) != base:
        raise RuntimeError('DURABLE_STATE_CONFLICT: stale state checkout')
    with open(root / 'writer-session.lock', 'a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError('LOCK_CONTENTION: another local writer session') from e
        bindings = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.rglob('host_binding.json')}
        entry = {'at': datetime.now(timezone.utc).isoformat(), 'context': ctx, 'physical_host': physical_host(),
                 'base': base, 'preserved_bindings': bindings, 'protocol': 'actions-concurrency+local-session-flock+git-cas-v1'}
        with open(root / 'ownership_transitions.jsonl', 'a') as f:
            f.write(json.dumps(entry, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
        with tempfile.TemporaryDirectory(prefix='nfl-shadow-session-') as tmp:
            p = Path(tmp) / 'session.json'
            p.write_text(json.dumps({'root': str(root), 'base': base, 'context': ctx, 'host': physical_host(), 'pid': os.getpid()}))
            os.environ['NFL_SHADOW_SESSION'] = str(p)
            restore(root)
            checkpoint(root) # ownership transition durable before child can mutate state
            try:
                return subprocess.run(command).returncode
            finally:
                checkpoint(root) # failed captures/diagnostics are durable, too


def run(root, command):
    previous = os.environ.get('NFL_SHADOW_SESSION')
    try:
        return _run(root, command)
    finally:
        if previous is None:
            os.environ.pop('NFL_SHADOW_SESSION', None)
        else:
            os.environ['NFL_SHADOW_SESSION'] = previous


def parse_cli(argv):
    # Split the child command explicitly: argparse REMAINDER would consume
    # --root after the positional mode, making the actual Actions entrypoint fail.
    split = argv.index('--') if '--' in argv else len(argv)
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['run', 'checkpoint'])
    ap.add_argument('--root', required=True)
    a = ap.parse_args(argv[:split])
    return a, argv[split + 1:]


if __name__ == '__main__':
    a, command = parse_cli(sys.argv[1:])
    if a.cmd == 'checkpoint':
        print(checkpoint(a.root))
    else:
        if not command:
            raise SystemExit('writer session requires a child command')
        sys.exit(run(a.root, command))
