"""Cross-platform evidence verification, never prediction or model selection.

Frozen raw artifacts remain untouched. Full receipts must retain their own raw
hashes and match every original field within 1e-10 absolute roundoff. Aggregate
floats permit only 1e-10 absolute roundoff. The observed legacy plot-bin exchange
has a narrowly recorded exception; populations, labels and decisions stay exact.
"""
import argparse
import gzip
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path

from nhl_v2_phase1a_sog_forward import canon, sha_file


def semantic(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('nonfinite receipt')
        return {'__typed_float_12g__': format(value, '.12g')}
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items()}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def receipt_fingerprint(path):
    content = hashlib.sha256()
    typed = hashlib.sha256()
    count = 0
    with gzip.open(path, 'rb') as file:
        for line in file:
            content.update(line)
            row = json.loads(line)
            recorded = row.pop('receipt_sha256')
            if hashlib.sha256(canon(row).encode()).hexdigest() != recorded:
                raise ValueError('receipt self hash mismatch')
            typed.update((canon(semantic(row))+'\n').encode())
            count += 1
    return {'rows': count, 'semantic_sha256': typed.hexdigest(),
            'content_sha256': content.hexdigest(), 'archive_sha256': sha_file(path)}


def compare(expected, actual, path='', exceptions=None):
    exceptions = exceptions or {}
    if type(expected) is not type(actual):
        raise ValueError('type mismatch: '+path)
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError('keys mismatch: '+path)
        for key in expected:
            compare(expected[key], actual[key], path+'/'+key, exceptions)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError('length mismatch: '+path)
        for i, (a,b) in enumerate(zip(expected,actual)):
            compare(a,b,path+'/'+str(i),exceptions)
    elif isinstance(expected, float):
        limit = exceptions.get(path, 1e-10)
        if not math.isfinite(actual) or abs(expected-actual) > limit:
            raise ValueError('numeric mismatch: '+path)
    elif expected != actual:
        raise ValueError('exact mismatch: '+path)


def verify_receipts(reference,actual):
    content=hashlib.sha256();count=0
    with gzip.open(reference,'rb') as a, gzip.open(actual,'rb') as b:
        for i,(left,right) in enumerate(zip_longest(a,b)):
            if left is None or right is None:
                raise ValueError('receipt population length mismatch')
            content.update(right);count+=1
            expected,current=json.loads(left),json.loads(right)
            for row in (expected,current):
                recorded=row.pop('receipt_sha256')
                if hashlib.sha256(canon(row).encode()).hexdigest()!=recorded:
                    raise ValueError('receipt self hash mismatch')
            compare(expected,current,'receipt/'+str(i))
    return {'rows':count,'content_sha256':content.hexdigest(),'archive_sha256':sha_file(actual)}


def verify(base, output):
    base, output = Path(base), Path(output)
    contract = json.loads((base/'phase1b_reproduction_contract.json').read_text())
    expected_lock = json.loads((base/'phase1b_development_lock.json').read_text())
    actual_lock = json.loads((output/'phase1b_development_lock.json').read_text())
    expected_results = json.loads((base/'phase1b_results.json').read_text())
    actual_results = json.loads((output/'phase1b_results.json').read_text())
    for period, specification in contract['receipts'].items():
        reference=base/'phase1b_frozen_receipts'/specification['file']
        if sha_file(reference)!=specification['archive_sha256']:
            raise ValueError('frozen reference mismatch')
        fingerprint=verify_receipts(reference,output/specification['file'])
        if fingerprint['rows']!=specification['rows']:
            raise ValueError('receipt population mismatch')
        a = actual_results['development'] if period == 'development' else actual_results['diagnostics'][period]
        e = expected_results['development'] if period == 'development' else expected_results['diagnostics'][period]
        if a['receipt_sha256'] != fingerprint['content_sha256'] or a['receipt_file_sha256'] != fingerprint['archive_sha256']:
            raise ValueError('generated archive provenance mismatch')
        if e['receipt_sha256'] != specification['content_sha256'] or e['receipt_file_sha256'] != specification['archive_sha256']:
            raise ValueError('frozen archive provenance mismatch')
        if period == 'development':
            if actual_lock['development_receipts'] != fingerprint['archive_sha256'] or expected_lock['development_receipts'] != specification['archive_sha256']:
                raise ValueError('development receipt lock mismatch')
        # Raw digests differ across platforms; never waive the full typed fingerprint.
        for key in ('receipt_sha256','receipt_file_sha256'):
            a[key] = e[key]
    actual_lock['development_receipts'] = expected_lock['development_receipts']
    compare(expected_lock,actual_lock)
    # The two old plotting bins may exchange at most two successes. Their total
    # must remain exact, and all underlying labels were checked exactly above.
    exceptions = contract['legacy_plot_absolute_limits']
    for path in exceptions:
        parts=path.split('/')
        e,a=expected_results,actual_results
        for part in parts[1:-2]:
            e=e[part]; a=a[part]
        expected_positive = sum(round(r['n']*r['observed']) for r in e)
        actual_positive = sum(round(r['n']*r['observed']) for r in a)
        if expected_positive != actual_positive:
            raise ValueError('legacy plot event total mismatch')
    compare(expected_results,actual_results,'phase1b_results.json',exceptions)
    compare(json.loads((base/'phase1b_retrospective_uncertainty.json').read_text()),
            json.loads((output/'phase1b_retrospective_uncertainty.json').read_text()))
    return {'status':'VERIFIED_EVERY_RECEIPT_FIELD_AND_FROZEN_DECISIONS',
            'frozen_artifacts_rewritten':False, 'float_absolute_tolerance':1e-10,
            'legacy_plot_exception_paths':list(exceptions)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=Path('nhl_models/nhl_player_outcome_v2'))
    parser.add_argument('--output',type=Path,default=Path('research_out/nhl_phase1b'))
    args=parser.parse_args()
    print(json.dumps(verify(args.base,args.output),sort_keys=True))
