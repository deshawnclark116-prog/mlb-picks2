"""
NFL_PHASE1E_LIVE (Phase 1E, shadow) -- real-provider snapshot retrieval + live forecast execution on top of the frozen Phase 1D Runner.

`LiveRunner` differs from the time-travel Runner only in WHERE the snapshot comes from: every source is downloaded from the real provider at execution time,
the retrieval timestamp is the real clock (taken AFTER the last byte arrived), and a set is refused if that timestamp is after the real cutoff.
The forecast path after the snapshot (materialize -> LiveLoader -> serialized walk-forward bundle -> Phase 1C simulator -> calibration -> records) is the
unchanged `Runner.run_context`.

Provider-lag policy (frozen in live_source_registry.json): the latest bytes obtainable before the cutoff are used and their age is recorded; a later revision is never
labelled with an earlier horizon; a lagging provider (completed games missing from the data) -> SAFE_EXPLICIT_FAILURE('provider_lag'); nothing is invented.
"""
import csv
import hashlib
import io
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import nfl_phase1_data as P1
import nfl_phase1_store as ST
import nfl_phase1d_cas as CAS
import nfl_phase1d_runner as RN
import nfl_phase1d_schedule as SCH

UTC = timezone.utc


def utcnow():
    return datetime.now(UTC)


class ProviderError(CAS.CASError):
    pass


def http_get(url, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-phase1-snapshots"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Last-Modified"), r.geturl()


def fetch_sources(seasons=P1.SEASONS, log=print):
    """Download every logical source from the real provider. Returns ({name: bytes_or_transform_tuple}, {name: audit}, unavailable)."""
    got, audit, unavailable = {}, {}, {}
    for name in CAS.logical_files(seasons):
        url = CAS.provider_url(name)
        try:
            raw, lm, final = http_get(url)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            if name.startswith(CAS.OPTIONAL_LIVE):
                unavailable[name] = f"{url}: {e}"
                continue
            raise ProviderError(f"required source {name} not retrievable: {url}: {e}")
        if not raw:
            raise ProviderError(f"required source {name} returned an empty payload")
        got[name] = CAS.sanitize_schedule(raw) if name == "games.csv" else raw
        audit[name] = {"url": url, "raw_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "last_modified": lm, "provider_final_url": final}
    return got, audit, unavailable


def provider_lag_problems(got, schedule_games, cutoff, season):
    """Operational validity (not a model input): every game of the live season completed by the cutoff (kickoff + 24h <= cutoff) must be present in the
    provider's player-stats file. Returns the list of missing (week, team) cells."""
    stats = got.get(f"stats_player_week_{season}.csv")
    if stats is None:
        return ["stats_player file missing"]
    have = {(r["week"], r["team"]) for r in csv.DictReader(io.StringIO(stats.decode("utf-8")))}
    missing = []
    for g in schedule_games.values():
        if g["season"] == season and g["kick"] + timedelta(hours=24) <= cutoff:
            for tm in (g["home"], g["away"]):
                if (str(g["week"]), tm) not in have:
                    missing.append(f"{season} wk{g['week']} {tm}")
    return missing


class LateGeneration(CAS.CASError):
    pass


class GuardedStore:
    """Forecast store that refuses to append at or after kickoff (a forecast generated after the game started is not a pregame forecast)."""

    def __init__(self, inner, kicks):
        self.inner, self.kicks = inner, kicks

    def append_batch(self, name, header, records):
        gid = header["game_id"]
        if utcnow() >= self.kicks[gid]:
            raise LateGeneration(f"forecast for {gid} generated after kickoff {SCH.iso(self.kicks[gid])}")
        return self.inner.append_batch(name, header, records)


class LiveRunner(RN.Runner):
    def __init__(self, root, n_draws, cas_root=None, log=print, fetcher=fetch_sources):
        super().__init__(root, None, n_draws, log=log, horizons=("T24", "T90"), cas_root=cas_root)
        self.fetcher = fetcher
        self.lag = CAS.Ledger(self.root, name="provider_lag.jsonl")

    def group_name(self, season, week, hz, kick):
        return f"LIVE_{season}_{week:02d}_{hz}_{SCH.iso(kick)}"

    def existing_set(self, group):
        prev = [r for r in self.ledger.read() if r.get("type") == "set" and r["group"] == group]
        return prev[-1] if prev else None

    def snapshot_live(self, season, week, hz, kick, expected_game_ids):
        """Retrieve (or reuse, after a restart) the real snapshot set of one (kickoff group, horizon). Raises CAS.CASError when it cannot be a valid pre-cutoff snapshot."""
        cutoff = SCH.forecast_cutoff(kick, hz)
        group = self.group_name(season, week, hz, kick)
        prev = self.existing_set(group)
        if prev:
            return prev, "reused_after_restart"
        if utcnow() >= cutoff:
            raise CAS.CASError("MISSED_REAL_CUTOFF: the cutoff passed before retrieval started")
        got, audit, unavailable = self.fetcher()
        retrieval = utcnow()                                           # AFTER the last byte arrived
        if retrieval > cutoff:
            raise CAS.CASError(f"MISSED_REAL_CUTOFF: retrieval completed {SCH.iso(retrieval)} after the cutoff {SCH.iso(cutoff)}")
        games_bytes = got["games.csv"][0] if isinstance(got["games.csv"], tuple) else got["games.csv"]
        sched = SCH.parse_schedule(games_bytes, seasons={season})
        for gid in expected_game_ids:
            g = sched.get(gid)
            if g is None:
                raise CAS.CASError(f"game {gid} absent from the retrieved schedule")
            if SCH.iso(g["kick"]) != SCH.iso(kick):
                raise CAS.CASError(f"kickoff of {gid} changed in the retrieved schedule ({SCH.iso(g['kick'])} vs planned {SCH.iso(kick)}); the dispatcher re-plans")
        miss = provider_lag_problems(got, sched, cutoff, season)
        self.lag.append_many([{"group": group, "horizon": hz, "retrieval_ts": SCH.iso(retrieval), "cutoff": SCH.iso(cutoff), "sources": audit,
                               "provider_lag_missing": miss, "unavailable_optional": unavailable}])
        if miss:
            raise CAS.CASError("provider_lag: completed games missing from the provider's data: " + "; ".join(miss[:8]))
        srcs = {n: (CAS.provider_id(n), (lambda b=b: b)) for n, b in got.items()}
        def _unavail(msg):
            def f():
                raise CAS.Unavailable(msg)
            return f
        for n, msg in unavailable.items():
            srcs[n] = (CAS.provider_id(n), _unavail(msg))
        note = "REAL PROVIDER RETRIEVAL; shadow; not clean-forward evidence"
        rec = CAS.take_snapshot_set(self.store, self.ledger, srcs, hz, kick, cutoff, retrieval, group, note=note, time_travel=False)
        return rec, "retrieved_now"

    def run_group(self, season, week, hz, kick, game_ids, run_id):
        """Snapshot + forecast one kickoff group. Returns (set_record|None, [status logs])."""
        rec, how = self.snapshot_live(season, week, hz, kick, game_ids)
        fstore = GuardedStore(ST.Store(self.root, "forecasts"), {g: kick for g in game_ids})
        bstore = ST.Store(self.root, "baselines")
        logs = self.run_context(hz, [(kick, rec, sorted(game_ids))], season, week, run_id, fstore, bstore, only_games=set(game_ids))
        return rec, how, logs

