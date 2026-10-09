"""Read-only verification of live snapshot provenance for presentation publishing."""
import gzip
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path


def t(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00'))


def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def blob(root, sha):
    p = root / 'cas' / 'blobs' / sha[:2] / sha
    if p.exists():
        b = p.read_bytes()
    else:
        d = root / 'durable_blobs' / 'cas' / sha
        m = json.loads((d / 'manifest.json').read_text())
        parts = []
        for c in m['chunks']:
            if Path(c['name']).name != c['name']:
                raise ValueError('invalid chunk path')
            z = (d / c['name']).read_bytes()
            if hashlib.sha256(z).hexdigest() != c['sha256']:
                raise ValueError('durable chunk hash mismatch')
            parts.append(gzip.decompress(z))
        b = b''.join(parts)
        if len(b) != m['bytes']:
            raise ValueError('durable blob size mismatch')
    if hashlib.sha256(b).hexdigest() != sha:
        raise ValueError('source blob hash mismatch')
    return b


class Evidence:
    def __init__(self, state):
        self.root = Path(state)
        self.manifest = rows(self.root / 'cas' / 'manifest.jsonl')
        self.sets = {r['set_id']: r for r in self.manifest if r.get('type') == 'set'}
        self.prefits = rows(self.root / 'prefit_ledger.jsonl')
        self.cache = {}
        self.problems = set()

    def valid(self, r, led, header):
        try:
            s = r['input_snapshots']; sid = s['snapshot_set_id']
            key = (sid, s.get('artifact_bundle_sha256'))
            cutoff, kick = t(r['cutoff']), t(r['kickoff'])
            if cutoff != kick - (timedelta(hours=24) if r['horizon'] == 'T24' else timedelta(minutes=90)):
                raise ValueError('horizon does not match kickoff')
            if header.get('generated_at') is None or t(header['generated_at']) >= kick:
                raise ValueError('missing or post-kickoff generation timestamp')
            if led.get('snapshot_set_id') != sid or led.get('retrieval_ts') != s['retrieval_ts']:
                raise ValueError('dispatch/snapshot identity mismatch')
            sr = self.sets[sid]
            if sr['cutoff'] != r['cutoff'] or sr['kickoff'] != r['kickoff'] or sr['horizon'] != r['horizon']:
                raise ValueError('snapshot horizon/cutoff mismatch')
            if sr.get('time_travel') or s.get('time_travel') or sr['retrieval_ts'] != s['retrieval_ts']:
                raise ValueError('non-live snapshot')
            if not cutoff - timedelta(minutes=5) <= t(sr['retrieval_ts']) <= cutoff:
                raise ValueError('snapshot outside legitimate pre-cutoff capture window')
            if key not in self.cache:
                if not sr['files'] or 'games.csv' not in sr['files']:
                    raise ValueError('missing schedule provenance')
                files = {x['logical_name']: x for x in self.manifest if x.get('type') == 'file' and x.get('set_id') == sid}
                for name, sha in sr['files'].items():
                    if name not in files or files[name]['sha256'] != sha or files[name]['retrieval_ts'] != sr['retrieval_ts']:
                        raise ValueError('source manifest mismatch')
                    blob(self.root, sha)
                pf = [p for p in self.prefits if p['artifact_bundle_sha256'] == key[1] and
                      (p['season'], p['week']) == (r['season'], r['week']) and t(p['created_at']) <= cutoff and
                      t(p['source_identity']['fit_retrieval_ts']) <= cutoff]
                if not pf:
                    raise ValueError('missing pre-cutoff artifact receipt')
                self.cache[key] = True
            return True
        except (ValueError, KeyError, OSError, TypeError) as e:
            self.problems.add(str(e))
            return False
