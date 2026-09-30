"""
NFL_PHASE1D_CAS  (Phase 1D, shadow research)  -- complete input provenance: content-addressed immutable blob store

Every source file that can influence a Phase 1 forecast is stored as an immutable blob addressed by its sha256:

    blobs/<sha[:2]>/<sha>          read-only, never overwritten, hash re-verified on every read
    manifest.jsonl                 append-only ledger; one row per (snapshot set, logical file) plus one "set" row per snapshot set

If two retrievals return identical bytes the blob is stored once and referenced twice. A manifest row records: logical file name, provider
identifier / URL, retrieval timestamp, information cutoff (forecast time) and horizon, sha256, byte count, blob path, parser/schema version,
and (when the stored bytes are a documented transform of what the provider served) the raw sha256 plus the transform name.

`materialize` builds a directory of read-only symlinks (logical name -> blob) after re-verifying every hash; the unchanged Phase 1A/1B loaders
read that directory, so a forecast can only see bytes that were snapshotted before its cutoff.

Time-travel (burned-week operational validation): `TimeTravelSource` produces the bytes a provider WOULD have served at a past cutoff, from the
local full files, by dropping every row whose game completed after the cutoff (kickoff + 24h, assumption A1), every injury / roster row of a
later week, and every depth-chart snapshot later than the cutoff. That is a simulation of retrieval; the manifest labels such sets `time_travel`.

Nothing here reads or stores sportsbook columns: the schedule is stored with the market columns removed (documented transform).
"""
import csv
import gzip
import hashlib
import io
import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nfl_phase1_data as P1
import nfl_phase1_store_lock as LK

UTC = timezone.utc
PARSER_VERSION = "phase1-parsers-2026.1"
SEASONS = P1.SEASONS
SCHEDULE_STRIP = tuple(P1.MARKET_COLUMNS) + ("old_game_id",)   # sportsbook columns are never stored; old_game_id is unused


class CASError(RuntimeError):
    pass


class Unavailable(CASError):
    """An OPTIONAL source could not be retrieved (e.g. play participation of the live season is not published); recorded in the set, never invented."""


def sha256_hex(b):
    return hashlib.sha256(b).hexdigest()


def iso(ts):
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def canon_json(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=str).encode()


# ------------------------------------------------------------------ logical files that Phase 1 reads
SEASON_FILES = {   # logical-name template -> (nflverse release tag, provider file name); layout as used by the project's own downloaders
    "injuries_{s}.csv": ("injuries", "injuries_{s}.csv"),
    "roster_weekly_{s}.csv": ("weekly_rosters", "roster_weekly_{s}.csv"),
    "stats_player_week_{s}.csv": ("stats_player", "stats_player_week_{s}.csv"),
    "snap_counts_{s}.csv": ("snap_counts", "snap_counts_{s}.csv"),
    "pbp_{s}.csv.gz": ("pbp", "play_by_play_{s}.csv.gz"),
    "participation_{s}.csv": ("pbp_participation", "pbp_participation_{s}.csv"),
    "ftn_{s}.csv": ("ftn_charting", "ftn_charting_{s}.csv"),
    "depth_charts_{s}.csv": ("depth_charts", "depth_charts_{s}.csv"),
}
STATIC_FILES = {"games.csv": ("schedules", "games.csv"), "players.csv": ("players", "players.csv")}
RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"


def provider_url(name, base=None):
    """The provider URL of a logical file (nflverse release layout)."""
    base = base or RELEASE
    if name in STATIC_FILES:
        rel, fn = STATIC_FILES[name]
        return f"{base}/{rel}/{fn}"
    for tpl, (rel, fn) in SEASON_FILES.items():
        stem = tpl.split("{s}")
        if name.startswith(stem[0]) and name.endswith(stem[1]):
            s = name[len(stem[0]): len(name) - len(stem[1])]
            if s.isdigit():
                return f"{base}/{rel}/{fn.format(s=s)}"
    raise CASError(f"no provider URL for {name}")
NO_FILE = {("participation_{s}.csv", 2026), ("depth_charts_{s}.csv", 2022), ("depth_charts_{s}.csv", 2023), ("depth_charts_{s}.csv", 2024)}   # not published / not consumed (A5)


def logical_files(seasons=SEASONS, data_dir=None):
    """All logical files a forecast reads (existing in data_dir when given)."""
    out = []
    for name in STATIC_FILES:
        out.append(name)
    for s in seasons:
        for tpl in SEASON_FILES:
            if (tpl, s) in NO_FILE:
                continue
            f = tpl.format(s=s)
            if data_dir is not None and not (Path(data_dir) / f).exists():
                continue
            out.append(f)
    return out


def provider_id(name):
    if name in STATIC_FILES:
        return f"nflverse:{STATIC_FILES[name][0]}:{STATIC_FILES[name][1]}"
    for tpl, (rel, fn) in SEASON_FILES.items():
        stem = tpl.split("{s}")
        if name.startswith(stem[0]) and name.endswith(stem[1]):
            s = name[len(stem[0]): len(name) - len(stem[1])] if stem[1] else name[len(stem[0]):]
            if s.isdigit():
                return f"nflverse:{rel}:{fn.format(s=s)}"
    return f"unknown:{name}"


# ------------------------------------------------------------------ blob store
class BlobStore:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)

    def path(self, sha):
        return self.root / "blobs" / sha[:2] / sha

    def put(self, data):
        """Store bytes; returns {'sha256','bytes','blob','deduplicated'}. An existing blob is re-verified, never rewritten."""
        if not isinstance(data, (bytes, bytearray)) or not len(data):
            raise CASError("empty or non-bytes payload")
        sha = sha256_hex(data)
        p = self.path(sha)
        if p.exists():
            if sha256_hex(p.read_bytes()) != sha:
                raise CASError(f"existing blob {sha} is corrupt on disk (refusing to overwrite)")
            return {"sha256": sha, "bytes": len(data), "blob": str(p.relative_to(self.root)), "deduplicated": True}
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / f".{sha}.{os.getpid()}.tmp"
        with open(tmp, "xb") as fh:
            fh.write(data); fh.flush(); os.fsync(fh.fileno())
        os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            os.link(tmp, p)                       # atomic and fails if it exists: no overwrite, ever
        except FileExistsError:
            pass
        finally:
            os.unlink(tmp)
        return {"sha256": sha, "bytes": len(data), "blob": str(p.relative_to(self.root)), "deduplicated": False}

    def get(self, sha):
        p = self.path(sha)
        if not p.exists():
            raise CASError(f"missing blob {sha}")
        b = p.read_bytes()
        if sha256_hex(b) != sha:
            raise CASError(f"blob {sha} failed hash verification (modified on disk)")
        return b

    def verify_all(self):
        problems = []
        for p in sorted((self.root / "blobs").rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                if sha256_hex(p.read_bytes()) != p.name:
                    problems.append(f"hash mismatch {p}")
                if os.access(p, os.W_OK) and os.geteuid() != 0:
                    problems.append(f"blob is writable {p}")
        return problems

    def blob_count(self):
        return sum(1 for p in (self.root / "blobs").rglob("*") if p.is_file() and not p.name.startswith("."))


# ------------------------------------------------------------------ ledger
class Ledger:
    """Append-only manifest (jsonl). Appends take the store's inter-process lock; reads never do."""

    def __init__(self, root, name="manifest.jsonl", lock_timeout=60.0):
        self.root = Path(root)
        self.path = self.root / name
        self.lock = LK.StoreLock(self.root / (name + ".lock"), timeout=lock_timeout)

    def read(self):
        if not self.path.exists():
            return []
        rows = []
        for i, line in enumerate(self.path.read_bytes().split(b"\n")):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise CASError(f"{self.path}: unparsable ledger line {i}: {e}")
        return rows

    def append_many(self, recs):
        with self.lock:
            with open(self.path, "ab") as fh:
                fh.write(b"".join(canon_json(r) + b"\n" for r in recs)); fh.flush(); os.fsync(fh.fileno())


# ------------------------------------------------------------------ snapshot sets
def take_snapshot_set(store, ledger, sources, horizon, kickoff, cutoff, retrieval_ts, group, note=None, time_travel=False):
    """sources: {logical_name: (provider_id, fetch() -> bytes | (bytes, raw_sha256, transform))}.
    Raises CASError when the retrieval happened after the cutoff. Returns the set record (also appended to the ledger)."""
    if retrieval_ts > cutoff:
        raise CASError(f"retrieval {iso(retrieval_ts)} is after the information cutoff {iso(cutoff)}")
    files, rows, unavailable = {}, [], {}
    for name in sorted(sources):
        prov, fetch = sources[name]
        try:
            got = fetch()
        except Unavailable as e:
            unavailable[name] = str(e)
            continue
        raw_sha, transform = None, None
        if isinstance(got, tuple):
            got, raw_sha, transform = got
        info = store.put(got)
        files[name] = info["sha256"]
        rows.append({"type": "file", "logical_name": name, "source": prov, "retrieval_ts": iso(retrieval_ts), "cutoff": iso(cutoff), "horizon": horizon,
                     "sha256": info["sha256"], "bytes": info["bytes"], "blob": info["blob"], "deduplicated": info["deduplicated"], "parser_version": PARSER_VERSION,
                     "raw_sha256": raw_sha, "transform": transform, "time_travel": bool(time_travel)})
    content_id = sha256_hex(canon_json(sorted(files.items())))[:32]
    set_id = sha256_hex(canon_json([content_id, horizon, iso(cutoff), group]))[:32]
    for r in rows:
        r["set_id"] = set_id
    setrec = {"type": "set", "set_id": set_id, "content_id": content_id, "horizon": horizon, "kickoff": iso(kickoff), "cutoff": iso(cutoff), "group": group,
              "retrieval_ts": iso(retrieval_ts), "files": files, "unavailable_optional_sources": unavailable, "parser_version": PARSER_VERSION, "time_travel": bool(time_travel), "note": note}
    ledger.append_many(rows + [setrec])
    return setrec


def get_set(ledger, set_id):
    for r in ledger.read():
        if r.get("type") == "set" and r["set_id"] == set_id:
            return r
    raise CASError(f"unknown snapshot set {set_id}")


def verify_set(store, ledger, setrec, required=None):
    """Every file of the set exists as a blob with the recorded hash, the manifest rows agree with the set record, retrieval <= cutoff,
    and (optionally) every required logical file is present."""
    problems = []
    if parse_iso(setrec["retrieval_ts"]) > parse_iso(setrec["cutoff"]):
        problems.append("retrieval after cutoff")
    rows = {r["logical_name"]: r for r in ledger.read() if r.get("type") == "file" and r.get("set_id") == setrec["set_id"]}
    for name, sha in setrec["files"].items():
        if name not in rows or rows[name]["sha256"] != sha:
            problems.append(f"manifest row missing/different for {name}")
        try:
            store.get(sha)
        except CASError as e:
            problems.append(str(e))
    for name in (required or []):
        if name not in setrec["files"]:
            problems.append(f"required source missing: {name}")
    return problems


def materialize(store, setrec, dest):
    """Directory of symlinks logical name -> verified blob. Loaders read this directory unchanged."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, sha in setrec["files"].items():
        store.get(sha)                                            # mandatory hash verification on read
        link = dest / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(store.path(sha).resolve(), link)
    return dest


# ------------------------------------------------------------------ time-travel source (burned weeks only)
def _csv_bytes(header, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _gz(b):
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", mtime=0, compresslevel=6) as g:
        g.write(b)
    return bio.getvalue()


BLANK_UNTIL_COMPLETE = ("result", "total", "home_score", "away_score", "overtime", "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name", "referee")


def sanitize_schedule(raw, done=None):
    """Schedule snapshot bytes: sportsbook / unused columns removed; with `done` (time travel) the outcome fields of games not completed by the cutoff are blanked.
    Returns (bytes, raw sha256, transform name)."""
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    keep = [c for c in rows[0].keys() if c not in SCHEDULE_STRIP]
    out = []
    for r in rows:
        r2 = {c: r[c] for c in keep}
        if done is not None and r["game_id"] not in done:
            for c in BLANK_UNTIL_COMPLETE:
                if c in r2:
                    r2[c] = ""
        out.append(r2)
    return _csv_bytes(keep, out), sha256_hex(raw), ("strip_market_columns+blank_incomplete_outcomes" if done is not None else "strip_market_columns")


class TimeTravelSource:
    """What the providers would have served at `cutoff`, derived from the full local files. Deterministic bytes; unchanged files are returned untouched
    (so identical content dedupes against every other set)."""

    def __init__(self, data_dir, seasons=SEASONS, contaminate=None):
        """contaminate=(season, week): TEST ONLY. Keep every stats / play / snap / charting row of that week's games in the snapshot, i.e. hand the loader a snapshot that
        contains the target game's outcome, to prove the loader is as-of correct even when it is (wrongly) given the future."""
        self.dir = Path(data_dir)
        self.seasons = seasons
        self.contaminate = contaminate
        rows = list(csv.DictReader(open(self.dir / "games.csv", newline="", encoding="utf-8")))
        self.games_header = list(rows[0].keys())
        self.games = rows
        self.kick = {}                    # game_id -> kickoff utc
        self.key_kick = {}                # (season, week, team) -> kickoff utc
        for r in rows:
            if r["game_type"] not in ("REG", "POST") or not r["season"].isdigit() or int(r["season"]) not in seasons:
                continue
            k = P1.kickoff_utc(r["gameday"], r["gametime"])
            self.kick[r["game_id"]] = k
            for t in (r["home_team"], r["away_team"]):
                self.key_kick[(int(r["season"]), int(r["week"]), t)] = k
        self._cache = {}

    def completed_ids(self, cutoff):
        done = {g for g, k in self.kick.items() if k + timedelta(hours=24) <= cutoff}
        if self.contaminate:
            cs, cw = self.contaminate
            done |= {g for g in self.kick if g.startswith(f"{cs}_{cw:02d}_")}
        return frozenset(done)

    def _memo(self, name, key, fn):
        k = (name, key)
        if k not in self._cache:
            self._cache[k] = fn()
        return self._cache[k]

    def bytes_for(self, name, cutoff, current, horizon):
        """current = (season, week) of the target game(s). Returns (bytes, raw_sha256|None, transform|None)."""
        cs, cw = current
        done = self.completed_ids(cutoff)
        if name == "games.csv":
            return self._memo(name, done, lambda: self._games(done))
        if name == "players.csv":
            return (self.dir / name).read_bytes(), None, None
        s = int(name.split("_")[-1].split(".")[0])
        if name.startswith("injuries_"):
            return self._memo(name, ("week", cs, cw), lambda: self._by_week(name, s, cs, cw))
        if name.startswith("roster_weekly_"):
            lim = cw if horizon == "T90" else cw - 1          # game-day roster status (A3) is usable at T-90m only (also under contamination: only game OUTCOME rows are contaminated)
            return self._memo(name, ("week", cs, lim), lambda: self._by_week(name, s, cs, lim))
        if name.startswith("depth_charts_"):
            return self._memo(name, ("dt", cutoff), lambda: self._depth(name, cutoff))
        if name.startswith("pbp_"):
            return self._memo(name, ("gid", done), lambda: self._by_game(name, s, done, "game_id", gz=True))
        if name.startswith("participation_") or name.startswith("ftn_"):
            return self._memo(name, ("gid", done), lambda: self._by_game(name, s, done, "nflverse_game_id"))
        if name.startswith("stats_player_week_") or name.startswith("snap_counts_"):
            return self._memo(name, ("gid", done), lambda: self._by_game(name, s, done, "game_id"))
        raise CASError(f"time-travel: unknown source {name}")

    def _games(self, done):
        return sanitize_schedule((self.dir / "games.csv").read_bytes(), done)

    def _by_week(self, name, s, cs, cw):
        raw = (self.dir / name).read_bytes()
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        header = list(rows[0].keys()) if rows else []
        if s > cs:
            keep = []
        elif s < cs:
            keep = rows
        else:
            keep = [r for r in rows if r.get("week") in (None, "") or int(r["week"]) <= cw]
        if len(keep) == len(rows):
            return raw, None, None
        return _csv_bytes(header, keep), sha256_hex(raw), f"drop_rows_after_week_{cw}"

    def _depth(self, name, cutoff):
        raw = (self.dir / name).read_bytes()
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        header = list(rows[0].keys())
        lim = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
        keep = [r for r in rows if r.get("dt") and r["dt"] <= lim]
        if len(keep) == len(rows):
            return raw, None, None
        return _csv_bytes(header, keep), sha256_hex(raw), "drop_snapshots_after_cutoff"

    def _by_game(self, name, s, done, col, gz=False):
        p = self.dir / name
        raw = p.read_bytes()
        text = gzip.decompress(raw).decode("utf-8") if gz else raw.decode("utf-8")
        rdr = csv.DictReader(io.StringIO(text, newline=""))
        header = rdr.fieldnames
        ids_this_season = {g for g in self.kick if g.startswith(f"{s}_")}
        kept, dropped = [], 0
        for r in rdr:
            g = r[col]
            if g in ids_this_season and g not in done:
                dropped += 1
                continue
            kept.append(r)
        if not dropped:
            return raw, None, None
        b = _csv_bytes(header, kept)
        return (_gz(b) if gz else b), sha256_hex(raw), "drop_rows_of_games_not_completed_by_cutoff"


def time_travel_sources(tt, cutoff, current, horizon, seasons=SEASONS, data_dir=None, skip=()):
    """{logical_name: (provider_id, fetch)} for a time-travel snapshot set."""
    out = {}
    for name in logical_files(seasons, data_dir):
        if name in skip:
            continue
        out[name] = (provider_id(name) + "#time_travel", (lambda n=name: tt.bytes_for(n, cutoff, current, horizon)))
    return out


# ------------------------------------------------------------------ live retrieval (forward system)
OPTIONAL_LIVE = ("participation_", "ftn_", "depth_charts_")      # not published (yet) for every season / time; their absence is recorded, never imputed


def live_sources(seasons=SEASONS, base=None, timeout=120, headers=None):
    """{logical_name: (provider URL, fetch)} for the forward system: every source is downloaded at retrieval time and stored by hash. `base` overrides the nflverse
    release root (tests point it at a local mirror)."""
    import urllib.error
    import urllib.request

    def make(name, url):
        def fetch():
            req = urllib.request.Request(url, headers={"User-Agent": "nfl-phase1-snapshots", **(headers or {})})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    raw = r.read()
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                if name.startswith(OPTIONAL_LIVE):
                    raise Unavailable(f"{url}: {e}")
                raise CASError(f"required source {name} not retrievable: {url}: {e}")
            if not raw:
                raise CASError(f"required source {name} returned an empty payload")
            return sanitize_schedule(raw) if name == "games.csv" else raw
        return fetch
    out = {}
    for name in logical_files(seasons):
        url = provider_url(name, base)
        out[name] = (url, make(name, url))
    return out
