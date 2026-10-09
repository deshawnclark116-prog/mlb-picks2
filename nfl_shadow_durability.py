"""Durable, hash-verified CAS backup on the state branch; Actions cache is only an accelerator."""
import gzip
import hashlib
import json
import os
from pathlib import Path

CHUNK_BYTES = 16 * 1024 * 1024
STORES = ('cas', 'cas_v2_raw_schedule')


def digest(b):
    return hashlib.sha256(b).hexdigest()


def atomic_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with open(tmp, 'xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        if path.read_bytes() != data:
            raise RuntimeError(f'DURABLE_STATE_CONFLICT: {path}')
    finally:
        tmp.unlink()


def seal(root):
    root = Path(root)
    count = 0
    for store in STORES:
        for p in sorted((root / store / 'blobs').glob('*/*')):
            if not p.is_file() or p.name.startswith('.'):
                continue
            dest = root / 'durable_blobs' / store / p.name
            manifest = dest / 'manifest.json'
            if manifest.exists():
                continue
            b = p.read_bytes()
            if digest(b) != p.name:
                raise RuntimeError(f'DATA_UNAVAILABLE: corrupt CAS blob {p.name}')
            chunks = []
            for i, start in enumerate(range(0, len(b), CHUNK_BYTES)):
                z = gzip.compress(b[start:start + CHUNK_BYTES], compresslevel=6, mtime=0)
                name = f'{i:05d}.gz'; atomic_new(dest / name, z)
                chunks.append({'name': name, 'sha256': digest(z)})
            atomic_new(manifest, json.dumps({'sha256': p.name, 'bytes': len(b), 'chunks': chunks}, sort_keys=True).encode())
            count += 1
    return count


def restore(root):
    root = Path(root)
    count = 0
    for store in STORES:
        for m in sorted((root / 'durable_blobs' / store).glob('*/manifest.json')):
            info = json.loads(m.read_text())
            if info['sha256'] != m.parent.name:
                raise RuntimeError('DATA_UNAVAILABLE: durable manifest identity mismatch')
            target = root / store / 'blobs' / info['sha256'][:2] / info['sha256']
            if target.exists():
                if digest(target.read_bytes()) != info['sha256']:
                    raise RuntimeError('DATA_UNAVAILABLE: corrupt cached blob; refusing overwrite')
                continue
            parts = []
            for c in info['chunks']:
                if Path(c['name']).name != c['name']:
                    raise RuntimeError('invalid durable chunk name')
                z = (m.parent / c['name']).read_bytes()
                if digest(z) != c['sha256']:
                    raise RuntimeError('DATA_UNAVAILABLE: durable chunk hash mismatch')
                parts.append(gzip.decompress(z))
            b = b''.join(parts)
            if len(b) != info['bytes'] or digest(b) != info['sha256']:
                raise RuntimeError('DATA_UNAVAILABLE: durable blob hash mismatch')
            atomic_new(target, b); target.chmod(0o444); count += 1
    return count
