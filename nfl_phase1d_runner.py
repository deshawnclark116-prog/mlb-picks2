"""
NFL_PHASE1D_RUNNER  (Phase 1D, shadow research)  -- the shadow forecast runner on top of the content-addressed snapshot store

One (season, week) is run as:

  1. for every kickoff group and horizon: retrieve (time-travel: reconstruct) EVERY source into the CAS, schedule first; the kickoff and the cutoff come
     from the schedule snapshot; the retrieval must be <= the cutoff (otherwise SAFE_EXPLICIT_FAILURE);
  2. get-or-fit the WEEKLY artifacts (walk-forward refit from completed weeks before the target week; every refit has a new bundle hash and a ledger row);
  3. group snapshot sets by content id; for each content id materialize the verified blobs, run the LiveLoader (`nfl_phase1d_live.prepare`) and simulate
     every target game;
  4. write append-only, idempotent forecast batches (locked, atomic) and one status row per game-horizon (FORECAST_SUCCESS | SAFE_EXPLICIT_FAILURE + reason).

Nothing reads a burned-week stage output; nothing here can see the target game's outcome (the snapshot contains no row of any game that had not completed by
the cutoff).
"""
import argparse
import hashlib
import io
import json
import shutil
import time
import zlib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import nfl_phase1_forecast as FC
import nfl_phase1_snapshots as SN
import nfl_phase1_store as ST
import nfl_phase1_store_lock as LK
import nfl_phase1c_sim as SM
import nfl_phase1d_baselines as BL
import nfl_phase1d_cas as CAS
import nfl_phase1d_live as LV
import nfl_phase1d_p1a as P
import nfl_phase1d_schedule as SCH

REPO = Path(__file__).resolve().parent
P1C = REPO / "nfl_models" / "nfl_player_outcome_phase1c"
UTC = timezone.utc
from nfl_phase1_data import SEASONS as P1_SEASONS  # noqa: E402
STATUS_OK, STATUS_FAIL = "FORECAST_SUCCESS", "SAFE_EXPLICIT_FAILURE"
RETRIEVAL_MARGIN = timedelta(hours=1)          # simulated retrieval time of a time-travel snapshot = cutoff - 1h


def load_config():
    """Frozen component architecture, constants, calibration (development-selected; not modified by Phase 1D)."""
    comp = json.load(open(P1C / "selected_components.json"))["components"]
    cfg = {}
    for name, e in comp.items():
        cfg[name] = {"level": e["level"], "structure": e.get("structure", "single")}
    consts = json.load(open(P1C / "constants.json"))
    cal = json.load(open(P1C / "calibration_depth.json"))["final_maps"]
    return cfg, consts, cal


class Runner:
    def __init__(self, root, data_dir, n_draws=25000, use_cache=False, log=print, horizons=("T24", "T90"), cas_root=None, contaminate=None, group_suffix=""):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.n_draws = n_draws
        self.log = log
        self.horizons = horizons
        cas = Path(cas_root) if cas_root else self.root / "cas"        # several runners (weeks) may share one content-addressed store (locked; host-local)
        self.store = CAS.BlobStore(cas)
        self.ledger = CAS.Ledger(cas)
        self.sched = SCH.ScheduleLedger(self.root)
        self.status = CAS.Ledger(self.root, name="runs.jsonl")
        self.wf_ledger = CAS.Ledger(self.root, name="weekly_fits.jsonl")
        self.fstore = None
        self.cfg, self.consts, self.cal = load_config()
        self.p1b_hyper = LV.load_p1b_hyper()
        self.frozen = P.load_frozen()
        self.tt = CAS.TimeTravelSource(data_dir, contaminate=contaminate) if data_dir else None
        self.group_suffix = group_suffix
        self.use_cache = use_cache

    # ------------------------------------------------------------ snapshots
    def plan_week(self, season, week):
        """{game_id: (home, away, kick)} from the (time-travel) schedule source for this regular-season week."""
        out = {}
        for r in self.tt.games:
            if r["season"] == str(season) and r["week"] == str(week) and r["game_type"] == "REG" and r.get("gameday"):
                out[r["game_id"]] = (r["home_team"], r["away_team"], self.tt.kick[r["game_id"]])
        return out

    def snapshot_sets(self, season, week, horizons, games=None):
        """Take one snapshot set per (kickoff group, horizon). Returns [(horizon, kick, setrec, [game_ids])] in wall-clock order."""
        plan = self.plan_week(season, week)
        if games is not None:
            plan = {g: v for g, v in plan.items() if g in games}
        by_kick = defaultdict(list)
        for gid, (h, a, k) in plan.items():
            by_kick[k].append(gid)
        items = sorted(((hz, k) for hz in horizons for k in by_kick), key=lambda x: SCH.forecast_cutoff(x[1], x[0]))
        out = []
        for hz, k in items:
            cutoff = SCH.forecast_cutoff(k, hz)
            retrieval = cutoff - RETRIEVAL_MARGIN
            group = f"{season}_{week:02d}_{hz}_{SCH.iso(k)}{self.group_suffix}"
            srcs = CAS.time_travel_sources(self.tt, cutoff, (season, week), hz, data_dir=self.data_dir)
            rec = self.take_set_idempotent(srcs, hz, k, cutoff, retrieval, group)
            out.append((hz, k, rec, sorted(by_kick[k])))
        return out

    def take_set_idempotent(self, srcs, hz, kick, cutoff, retrieval, group):
        """Same group + same bytes -> the existing set (no new ledger rows); same group + different bytes -> HARD ERROR (never silently replace a snapshot)."""
        rows = self.ledger.read()
        prev = [r for r in rows if r.get("type") == "set" and r["group"] == group]
        rec = CAS.take_snapshot_set(self.store, _NullLedger(), srcs, hz, kick, cutoff, retrieval, group, note="TIME-TRAVEL SIMULATED RETRIEVAL (burned week)", time_travel=True) if prev \
            else None
        if prev:
            if rec["content_id"] != prev[-1]["content_id"]:
                raise ST.HardError(f"snapshot group {group} re-retrieved with DIFFERENT bytes (content {rec['content_id']} vs {prev[-1]['content_id']}); refusing to replace")
            return prev[-1]
        return CAS.take_snapshot_set(self.store, self.ledger, srcs, hz, kick, cutoff, retrieval, group, note="TIME-TRAVEL SIMULATED RETRIEVAL (burned week)", time_travel=True)

    # ------------------------------------------------------------ weekly artifacts (walk-forward refit)
    def weekly_artifacts(self, D, U, target_sw, set_content_id):
        s, w = target_sw
        lock = LK.StoreLock(self.root / f"weekly_fit_{s}_{w:02d}.lock", timeout=3600, bind_dir=self.root)
        with lock:
            rows = [r for r in self.wf_ledger.read() if r["season"] == s and r["week"] == w]
            if rows:
                art = P.Artifacts.load(self.root / "artifacts" / rows[-1]["artifact_bundle_sha256"])
                return art
            completed = {k for k in D.stat}
            win = P.walk_forward_windows(completed, (s, w))
            t0 = time.time()
            art = P.fit_artifacts(U, win, self.frozen, code=FC.file_hashes(), audit_week=(s, w),
                                  extra_meta={"target_week": [s, w], "fit_set_content_id": set_content_id, "algorithm": "walk-forward weekly refit v1 (nfl_phase1d_p1a.fit_artifacts)"})
            man = art.save(self.root / "artifacts" / art.manifest()["bundle_sha256"])
            self.wf_ledger.append_many([{"season": s, "week": w, "artifact_bundle_sha256": man["bundle_sha256"], "training_cutoff_last_week": win.detail["last_training_week"],
                                         "n_train_weeks": len(win.detail["train_weeks"]), "n_valid_weeks": len(win.detail["valid_weeks"]), "fit_set_content_id": set_content_id,
                                         "code_sha": FC.git_sha(), "code_hashes": FC.file_hashes(), "fit_seconds": round(time.time() - t0),
                                         "p1b_hyper_sha256": hashlib.sha256(json.dumps(self.p1b_hyper, sort_keys=True, default=float).encode()).hexdigest(),
                                         "phase1b_fit_end": LV.week_fit_end(s, w)}])
            return art

    # ------------------------------------------------------------ one context (= one distinct snapshot content)
    def run_context(self, hz, entries, season, week, run_id, fstore, bstore=None, only_games=None):
        """entries: [(kick, setrec, [game_ids])] sharing one content id."""
        setrec0 = entries[0][1]
        asof = CAS.materialize(self.store, setrec0, self.root / "tmp" / f"{setrec0['content_id'][:16]}")
        sched = SCH.parse_schedule((Path(asof) / "games.csv").read_bytes(), seasons={season})
        logs, targets, wanted = [], [], []
        for kick, setrec, gids in entries:
            for gid in gids:
                if only_games is not None and gid not in only_games:
                    continue
                g = sched.get(gid)
                base = {"run_id": run_id, "game_id": gid, "horizon": hz, "season": season, "week": week, "set_id": setrec["set_id"], "content_id": setrec["content_id"]}
                if g is None:
                    logs.append({**base, "status": STATUS_FAIL, "reason": "game absent from the schedule snapshot"}); self.sched.observe(gid, None, setrec["set_id"], SCH.parse_iso(setrec["retrieval_ts"]), "absent_from_schedule")
                    continue
                cutoff = SCH.forecast_cutoff(g["kick"], hz)
                row, changed = self.sched.observe(gid, g["kick"], setrec["set_id"], SCH.parse_iso(setrec["retrieval_ts"]))
                if SCH.parse_iso(setrec["retrieval_ts"]) > cutoff:
                    logs.append({**base, "status": STATUS_FAIL, "reason": "late_retrieval: snapshot retrieved after the cutoff derived from the schedule snapshot"}); continue
                if SCH.iso(g["kick"]) != SCH.iso(kick):
                    logs.append({**base, "status": STATUS_FAIL, "reason": f"kickoff in the schedule snapshot ({SCH.iso(g['kick'])}) differs from the planned group kickoff ({SCH.iso(kick)})"}); continue
                wanted.append((gid, g, cutoff, setrec, row))
                targets += [(season, week, g["home"]), (season, week, g["away"])]
        if not wanted:
            self.status.append_many(logs)
            return logs
        wf = {"art": None}

        def art_provider(D, U, target_sw):
            wf["art"] = self.weekly_artifacts(D, U, target_sw, setrec0["content_id"])
            return wf["art"]
        prep = LV.prepare(str(asof), targets, art_provider, self.cfg, self.consts, self.cal, self.n_draws, self.p1b_hyper, (season, week), log=self.log)
        inj = FC._parse_csv(self.store.get(setrec0["files"][f"injuries_{season}.csv"]))
        ros_name = f"roster_weekly_{season}.csv"
        ros = FC._parse_csv(self.store.get(setrec0["files"][ros_name])) if ros_name in setrec0["files"] else []
        for gid, g, cutoff, setrec, srow in wanted:
            base = {"run_id": run_id, "game_id": gid, "horizon": hz, "season": season, "week": week, "set_id": setrec["set_id"], "content_id": setrec["content_id"],
                    "kickoff": SCH.iso(g["kick"]), "cutoff": SCH.iso(cutoff), "schedule_revision": srow["revision"], "model_version": prep.bundle.model_version,
                    "artifact_bundle_sha256": prep.art_manifest["bundle_sha256"]}
            try:
                gs, rule_log = {}, []
                missing = None
                for tm in (g["home"], g["away"]):
                    gi = prep.pack["games"].get((season, week, tm))
                    if gi is None:
                        missing = tm; break
                    gs[tm] = FC.apply_snapshot_rules(gi, hz, inj, ros, season, week, tm, rule_log)
                if missing:
                    logs.append({**base, "status": STATUS_FAIL, "reason": f"no Phase 1A inputs for {missing}"}); continue
                view = {"games": {(season, week, tm): gs[tm] for tm in gs}, "meta": prep.pack["meta"]}
                seed = zlib.crc32(repr((prep.bundle.model_version, gid, hz)).encode())
                b = prep.bundle
                res, gsx = SM.run_game(view, g["home"], g["away"], season, week, b.eff, b.idx, b.defaults, b.C, b.n_draws, seed, hz, None, b.qb_adjust)
                prov = {"snapshot_set_id": setrec["set_id"], "content_id": setrec["content_id"], "retrieval_ts": setrec["retrieval_ts"], "cutoff": SCH.iso(cutoff),
                        "artifact_bundle_sha256": prep.art_manifest["bundle_sha256"], "schedule_revision": srow["revision"], "time_travel": bool(setrec.get("time_travel"))}
                recs = FC.build_records(b, prep.D, prep.names, season, week, gsx, res, hz, g["kick"], cutoff, gid, prov, rule_log)
                hdr = {"generated_at": datetime.now(UTC).isoformat(), "run_id": run_id, "code_sha": FC.git_sha(), "horizon": hz, "game_id": gid}
                r = fstore.append_batch(f"{season}_wk{week:02d}_{hz}_{gid}", hdr, recs)
                if bstore is not None:
                    def status_lookup(tm, tname, j, _gs=gs):
                        return float(_gs[tm]["types"][tname]["pact_lookup"][j])       # already 0 for T-90m game-day inactives (apply_snapshot_rules)
                    seasons_ = [s_ for s_ in P1_SEASONS if f"stats_player_week_{s_}.csv" in setrec0["files"]]
                    stats_blobs = [self.store.get(setrec0["files"][f"stats_player_week_{s_}.csv"]) for s_ in seasons_]
                    pos_map = {gid_: ((prep.D.roster.get((season, week, gid_)) or prep.D.players.get(gid_) or {}).get("pos")) for tm in gs for t in gs[tm]["types"].values() for gid_ in t["ids"]}
                    brecs = BL.build_records({tm: gs[tm] for tm in gs}, status_lookup, stats_blobs, season, week, gid, [g["home"], g["away"]], hz, SCH.iso(g["kick"]), SCH.iso(cutoff),
                                             pos_map, {"snapshot_set_id": setrec["set_id"], "content_id": setrec["content_id"]})
                    bstore.append_batch(f"{season}_wk{week:02d}_{hz}_{gid}", {**hdr, "baseline_version": BL.BASELINE_VERSION}, brecs)
                logs.append({**base, "status": STATUS_OK, "n_records": len(recs), "snapshot_rules_applied": len(rule_log), **r})
            except ST.HardError:
                raise
            except (LK.LockError, CAS.CASError) as e:
                logs.append({**base, "status": STATUS_FAIL, "reason": f"{type(e).__name__}: {e}"})
        self.status.append_many(logs)
        return logs

    # ------------------------------------------------------------ one week
    def run_week(self, season, week, run_id, kind="forecasts", horizons=None, games=None, crash_after=None):
        horizons = horizons or self.horizons
        fstore = ST.Store(self.root, kind)
        bstore = ST.Store(self.root, kind.replace("forecasts", "baselines"))
        sets = self.snapshot_sets(season, week, horizons, games)
        by_hz = defaultdict(lambda: defaultdict(list))
        for hz, k, rec, gids in sets:
            by_hz[hz][rec["content_id"]].append((k, rec, gids))
        all_logs, n = [], 0
        for hz in horizons:
            for cid, entries in by_hz[hz].items():
                if crash_after is not None and n >= crash_after:
                    raise RuntimeError("simulated crash mid-run")
                all_logs += self.run_context(hz, entries, season, week, run_id, fstore, bstore, only_games=games)
                n += 1
        return all_logs


class _NullLedger:
    """Used only to compute a set's content id for an already-recorded group without writing anything."""

    def append_many(self, recs):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--data-dir", default="/tmp/nflcsv")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--run-id", default="run-A")
    a = ap.parse_args()
    r = Runner(a.root, a.data_dir, a.n, log=lambda m: print(m, flush=True))
    logs = r.run_week(a.season, a.week, a.run_id)
    print(json.dumps({"ok": sum(l["status"] == STATUS_OK for l in logs), "failed": [l for l in logs if l["status"] != STATUS_OK]}, default=str)[:2000])


if __name__ == "__main__":
    main()
