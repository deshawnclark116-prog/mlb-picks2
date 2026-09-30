"""
NFL_PHASE1_STORE_LOCK  (Phase 1D)  -- inter-process write lock for the append-only stores

Guarantee provided: on ONE host, at most one process holds the lock of a given store at a time.

  * mechanism: kernel advisory lock (fcntl.flock) on a lock file. The kernel drops the lock when the holder dies (kill -9, crash, OOM), so a dead
    writer can never leave the store locked: there is no stale-lock state to clean up by hand. The owner record written INTO the lock file (pid, host,
    start time) is only a diagnostic; when a new holder finds an owner record whose pid is dead it logs `recovered_stale_owner` and replaces it.
  * timeout: acquisition polls without blocking forever; on timeout it raises LockTimeout naming the current owner. Nothing is written without the lock.
  * NOT distributed: flock is host-local (and unreliable on network filesystems). The store is therefore BOUND to one host (host_binding.json, written by
    the first writer). A writer on another host is refused with LockError before any write. Multi-host / multi-container operation is unsupported
    until a distributed lock (or a transactional ledger service) replaces this module; that constraint is enforced, not assumed. Re-binding a store to a
    new host is an explicit, logged act (bind_host(..., rebind=True)).
"""
import fcntl
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path


class LockError(RuntimeError):
    pass


class LockTimeout(LockError):
    pass


def host_id():
    """Stable identifier of the machine that may write. NFL_STORE_HOST_ID overrides (persistent volume moved between containers on purpose)."""
    env = os.environ.get("NFL_STORE_HOST_ID")
    if env:
        return env
    try:
        mid = Path("/etc/machine-id").read_text().strip()
        if mid:
            return "machine:" + mid
    except OSError:
        pass
    return "host:" + socket.gethostname()


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def bind_host(root, rebind=False):
    """Bind the store directory to this host on first use; refuse foreign hosts afterwards."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p = root / "host_binding.json"
    me = host_id()
    if not p.exists():
        tmp = root / f".host_binding.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"host_id": me, "bound_at": datetime.now(timezone.utc).isoformat(), "note": "flock is host-local; other hosts are refused"}, fh)
            fh.flush(); os.fsync(fh.fileno())
        try:
            os.link(tmp, p)
        except FileExistsError:
            pass
        os.unlink(tmp)
    cur = json.loads(p.read_text())
    if cur["host_id"] != me:
        if not rebind:
            raise LockError(f"store {root} is bound to host {cur['host_id']!r}; this host is {me!r}. flock is host-local, distributed writers are unsupported "
                            f"(explicit rebind required)")
        cur = {"host_id": me, "bound_at": datetime.now(timezone.utc).isoformat(), "rebound_from": cur["host_id"]}
        p.write_text(json.dumps(cur))
    return cur


class StoreLock:
    """Re-entrant within one process instance (a nested `with` on the same object is a no-op); exclusive across processes on this host."""

    def __init__(self, path, timeout=60.0, poll=0.05, bind_dir=None):
        self.path = Path(path)
        self.timeout = timeout
        self.poll = poll
        self.bind_dir = Path(bind_dir) if bind_dir else self.path.parent
        self._fh = None
        self._depth = 0
        self.last_recovery = None

    def acquire(self):
        if self._depth:
            self._depth += 1
            return self
        bind_host(self.bind_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        t0 = time.time()
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t0 >= self.timeout:
                    owner = self._read_owner(fh)
                    fh.close()
                    raise LockTimeout(f"could not acquire {self.path} within {self.timeout}s; owner record: {owner}")
                time.sleep(self.poll)
        owner = self._read_owner(fh)
        self.last_recovery = None
        if owner and owner.get("pid") and not pid_alive(owner["pid"]) and owner.get("host") == host_id():
            self.last_recovery = {"event": "recovered_stale_owner", "dead_pid": owner["pid"], "was_acquired_at": owner.get("acquired_at")}
        fh.seek(0); fh.truncate()
        fh.write(json.dumps({"pid": os.getpid(), "host": host_id(), "acquired_at": datetime.now(timezone.utc).isoformat()}))
        fh.flush(); os.fsync(fh.fileno())
        self._fh, self._depth = fh, 1
        return self

    @staticmethod
    def _read_owner(fh):
        try:
            fh.seek(0)
            t = fh.read().strip()
            return json.loads(t) if t else None
        except Exception:
            return None

    def release(self):
        if not self._depth:
            return
        self._depth -= 1
        if self._depth:
            return
        try:
            self._fh.seek(0); self._fh.truncate(); self._fh.flush()
            fcntl.flock(self._fh, fcntl.LOCK_UN)
        finally:
            self._fh.close(); self._fh = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False
