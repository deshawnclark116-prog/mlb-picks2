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
from pathlib import Path
from datetime import datetime, timedelta, timezone

import nfl_phase1_data as P1
import nfl_phase1_store as ST
import nfl_phase1d_cas as CAS
import nfl_phase1d_runner as RN
import nfl_phase1d_schedule as SCH
import nfl_shadow_schedule as SCHEDULE

UTC = timezone.utc
MAX_RETRIEVAL_LEAD = timedelta(minutes=5)        # a COMPLETED retrieval earlier than this before the cutoff is REJECTED (never silently labelled T24/T90)
CAPTURE_START_LEAD = timedelta(seconds=240)      # pre-registered: the dispatcher may wake earlier but starts the capture only this long before the cutoff (measured full fetch ~25 s)


def utcnow():
    return datetime.now(UTC)


class ProviderError(CAS.CASError):
    pass


def http_get(url, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-phase1-snapshots", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not r.geturl().startswith("https://"):
            raise ProviderError("DATA_UNAVAILABLE: non-HTTPS provider redirect")
        return r.read(), r.headers.get("Last-Modified"), r.geturl()


def fetch_sources(seasons=P1.SEASONS, log=print):
    """Download every logical source from the real provider. Returns ({name: bytes_or_transform_tuple}, {name: audit}, unavailable)."""
    got, audit, unavailable = {}, {}, {}
    raw_schedule = None
    for name in CAS.logical_files(seasons):
        url = SCHEDULE.provider_url(name)
        try:
            raw, lm, final = http_get(url)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            if name.startswith(CAS.OPTIONAL_LIVE):
                unavailable[name] = f"{url}: {e}"
                continue
            raise ProviderError(f"{'SCHEDULE_FETCH_404' if name == 'games.csv' and getattr(e, 'code', None) == 404 else 'DATA_UNAVAILABLE'}: required source {name} not retrievable: {url}: {e}")
        if not raw:
            raise ProviderError(f"required source {name} returned an empty payload")
        if name == "games.csv":
            raw_schedule = raw                                           # kept ONLY for the v2 comparator's own (separate) store; Phase 1 stores the sanitized bytes
        got[name] = SCHEDULE.sanitize(raw) if name == "games.csv" else raw
        audit[name] = {"url": url, "raw_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "last_modified": lm, "provider_final_url": final}
    return got, audit, unavailable, raw_schedule


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


def joint_ready(phase1_ok, v2_status):
    """A game-horizon is FULLY ready only if Phase 1 is ready AND the frozen v2 comparator has its shared schedule provenance and was logged."""
    return bool(phase1_ok) and v2_status == "V2_LOGGED"


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
        self.v2raw = CAS.BlobStore(self.root / "cas_v2_raw_schedule")
        self.v2frozen = None

    def group_name(self, season, week, hz, kick):
        return f"LIVE_{season}_{week:02d}_{hz}_{SCH.iso(kick)}"

    def weekly_artifacts(self, D, U, target_sw, set_content_id):
        """LIVE path: load the immutable PREFIT artifact (hash + code-identity verified, and existing BEFORE this forecast's cutoff); never fit at a cutoff."""
        import nfl_phase1e_ops as OPS
        cutoff = getattr(self, "current_cutoff", None)
        if cutoff is None:
            raise CAS.CASError("PREFIT_NOT_READY_AT_CUTOFF: no cutoff context for the prefit check")
        row = OPS.prefit_gate(self.root, target_sw[0], target_sw[1], cutoff, utcnow())
        art = OPS.load_verified_prefit(self.root, row)
        self.prefit_used = row
        return art

    def existing_set(self, group):
        prev = [r for r in self.ledger.read() if r.get("type") == "set" and r["group"] == group]
        return prev[-1] if prev else None

    def snapshot_live(self, season, week, hz, kick, expected_game_ids):
        """Retrieve (or reuse, after a restart) the real snapshot set of one (kickoff group, horizon). Raises CAS.CASError when it cannot be a valid pre-cutoff snapshot."""
        cutoff = SCH.forecast_cutoff(kick, hz)
        group = self.group_name(season, week, hz, kick)
        prev = self.existing_set(group)
        if prev:
            bad = CAS.verify_set(self.store, self.ledger, prev)
            if bad:
                raise CAS.CASError("stored_snapshot_invalid: " + "; ".join(bad[:4]))
            return prev, "reused_after_restart"
        if utcnow() >= cutoff:
            raise CAS.CASError("MISSED_REAL_CUTOFF: the cutoff passed before retrieval started")
        got, audit, unavailable, raw_sched = self.fetcher()
        retrieval = utcnow()                                           # AFTER the last byte arrived
        if retrieval > cutoff:
            raise CAS.CASError(f"MISSED_REAL_CUTOFF: retrieval completed {SCH.iso(retrieval)} after the cutoff {SCH.iso(cutoff)}")
        if cutoff - retrieval > MAX_RETRIEVAL_LEAD:
            raise CAS.CASError(f"early_snapshot_rejected: retrieved {SCH.iso(retrieval)}, {int((cutoff - retrieval).total_seconds() / 60)} min before the {hz} cutoff {SCH.iso(cutoff)} (max lead {int(MAX_RETRIEVAL_LEAD.total_seconds() / 60)} min); not labelled {hz}")
        games_bytes = got["games.csv"][0] if isinstance(got["games.csv"], tuple) else got["games.csv"]
        sched = SCH.parse_schedule(games_bytes, seasons={season})
        for gid in expected_game_ids:
            g = sched.get(gid)
            if g is None:
                raise CAS.CASError(f"game {gid} absent from the retrieved schedule")
            if SCH.iso(g["kick"]) != SCH.iso(kick):
                raise CAS.CASError(f"kickoff of {gid} changed in the retrieved schedule ({SCH.iso(g['kick'])} vs planned {SCH.iso(kick)}); the dispatcher re-plans")
        miss = provider_lag_problems(got, sched, cutoff, season)
        raw_info = self.v2raw.put(raw_sched)                           # sportsbook columns live only in this separate store, read only by the v2 comparator logger
        self.lag.append_many([{"group": group, "raw_schedule_sha256": raw_info["sha256"], "horizon": hz, "retrieval_ts": SCH.iso(retrieval), "cutoff": SCH.iso(cutoff), "sources": audit,
                               "provider_lag_missing": miss, "unavailable_optional": unavailable}])
        if miss:
            raise CAS.CASError("provider_lag: completed games missing from the provider's data: " + "; ".join(miss[:8]))
        srcs = {n: (SCHEDULE.provider_id(n), (lambda b=b: b)) for n, b in got.items()}
        def _unavail(msg):
            def f():
                raise CAS.Unavailable(msg)
            return f
        for n, msg in unavailable.items():
            srcs[n] = (CAS.provider_id(n), _unavail(msg))
        note = "REAL PROVIDER RETRIEVAL for a live horizon. Temporal evidence: clean-forward when retrieved within 5 min before the real cutoff (see live_events.jsonl). Operational N only (R11 blocker): not freeze evidence, not production-promotion evidence"
        rec = CAS.take_snapshot_set(self.store, self.ledger, srcs, hz, kick, cutoff, retrieval, group, note=note, time_travel=False)
        return rec, "retrieved_now"

    def run_group(self, season, week, hz, kick, game_ids, run_id):
        """Snapshot + forecast one kickoff group. Returns (set_record|None, [status logs])."""
        import nfl_phase1e_ops as OPS
        self.current_cutoff = SCH.forecast_cutoff(kick, hz)
        OPS.prefit_gate(self.root, season, week, self.current_cutoff, utcnow())              # a valid pre-cutoff prefit must exist BEFORE any live capture / forecast work
        rec, how = self.snapshot_live(season, week, hz, kick, game_ids)
        if getattr(self, "checkpoint_snapshot", None):
            self.checkpoint_snapshot() # persist exact sources BEFORE expensive generation
        v2 = self.log_v2(rec, season, week, hz, kick, game_ids, run_id)
        fstore = GuardedStore(ST.Store(self.root, "forecasts"), {g: kick for g in game_ids})
        bstore = ST.Store(self.root, "baselines")
        logs = self.run_context(hz, [(kick, rec, sorted(game_ids))], season, week, run_id, fstore, bstore, only_games=set(game_ids))
        import nfl_phase1e_ops as OPS
        for l in logs:
            l.setdefault("season", season); l.setdefault("week", week)
            l["v2"] = v2.get(l["game_id"])
            l["joint_ready"] = joint_ready(l.get("status") == "FORECAST_SUCCESS", (l["v2"] or {}).get("status"))
            OPS.record_live_event(self, rec, how, hz, kick, l)
        return rec, how, logs

    def log_v2(self, rec, season, week, hz, kick, game_ids, run_id):
        """v2 frozen-comparator pre-kickoff append-only log for the same snapshot. A v2 failure never blocks Phase 1; it is recorded (HardError conflicts propagate)."""
        import nfl_phase1e_v2 as V2
        group = self.group_name(season, week, hz, kick)
        row = [r for r in self.lag.read() if r["group"] == group and r.get("raw_schedule_sha256")]
        if not row:
            return {g: {"status": "V2_FAILED", "reason": "no provider-lag row (raw schedule hash) for the group"} for g in game_ids}
        raw = self.v2raw.get(row[-1]["raw_schedule_sha256"])
        d = CAS.materialize(self.store, rec, self.root / "tmp" / f"v2_{rec['set_id'][:16]}")
        link = Path(d) / "games.csv"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.write_bytes(raw)
        try:
            self.v2frozen = self.v2frozen or V2.Frozen()
            cut = SCH.iso(SCH.forecast_cutoff(kick, hz))
            prov = {"retrieval_ts": rec["retrieval_ts"], "input_hashes": {"snapshot_set_id": rec["set_id"], "content_id": rec["content_id"], "raw_schedule_sha256": row[-1]["raw_schedule_sha256"],
                                                                         **{k: v for k, v in rec["files"].items() if k.startswith(("stats_player_week_", "snap_counts_"))}}}
            out = V2.log_group(self.root, self.v2frozen, str(d), season, week, game_ids, hz, {g: SCH.iso(kick) for g in game_ids}, {g: cut for g in game_ids}, prov, utcnow, run_id)
            return {g: {"status": "V2_LOGGED", **v} for g, v in out.items()}
        except ST.HardError:
            raise
        except Exception as e:                                          # noqa
            return {g: {"status": "V2_FAILED", "reason": f"{type(e).__name__}: {e}"[:300]} for g in game_ids}

