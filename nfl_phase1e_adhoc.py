"""
NFL_PHASE1E_ADHOC -- one explicitly separate AD_HOC_PREGAME run of the Phase 1 engine + frozen v2 comparator for ONE game, from current real-provider bytes.

NOT a T24 / T90 forecast, NOT freeze evidence, R11 remains BLOCKER (N=100,000 is an operational choice). Refuses if kickoff has occurred. The engine is the unchanged Phase 1
live path run with its T90 information rules (game-day inactives, roster) -- the information cutoff is the REAL retrieval timestamp, which is written to every record
(cutoff = retrieval), and horizon is rewritten to AD_HOC_PREGAME with a new deterministic id so it can never be confused with or collide with a T24/T90 forecast.
Outputs are append-only stores under --root; docs/nfl_predictions.json is never touched.
"""
import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import nfl_phase1_store as ST
import nfl_phase1d_cas as CAS
import nfl_phase1d_schedule as SCH
import nfl_phase1e_live as LVE
import nfl_phase1e_v2 as V2

LABEL = "AD_HOC_PREGAME"
NOTE = "AD_HOC_PREGAME: operational shadow forecast from current real-provider bytes; N=100000 is operational only; NOT T24/T90; NOT freeze evidence; R11 remains BLOCKER"


class RelabelStore:
    """Wraps a Store: every record gets horizon=AD_HOC_PREGAME, a new deterministic id, and the label; batch names are prefixed."""

    def __init__(self, inner, kick, retrieval_iso):
        self.inner, self.kick, self.ret = inner, kick, retrieval_iso

    def append_batch(self, name, header, records):
        if LVE.utcnow() >= self.kick:
            raise LVE.LateGeneration("kickoff has occurred")
        out = []
        for r in records:
            r = dict(r)
            r["source_id"] = r["id"]
            r["id"] = ST.make_id("adhoc", r["id"])
            r["horizon"] = LABEL
            r["adhoc_label"] = NOTE
            r["information_cutoff_is_retrieval_ts"] = self.ret
            out.append(r)
        return self.inner.append_batch("ADHOC_" + name, {**header, "horizon": LABEL, "adhoc_label": NOTE}, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed-root", help="root holding a previously prepared weekly artifact (same code, completed weeks < target week)")
    a = ap.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    if a.seed_root:
        import shutil
        for n in ("artifacts",):
            if not (root / n).exists():
                shutil.copytree(Path(a.seed_root) / n, root / n)
        if not (root / "weekly_fits.jsonl").exists():
            shutil.copy(Path(a.seed_root) / "weekly_fits.jsonl", root / "weekly_fits.jsonl")
    r = LVE.LiveRunner(root, a.n, cas_root=root / "cas", log=lambda m: print(m, flush=True))
    got, audit, unavailable, raw_sched = LVE.fetch_sources()
    retrieval = LVE.utcnow()                                            # after the last byte arrived
    games_bytes = got["games.csv"][0]
    sched = SCH.parse_schedule(games_bytes)
    g = sched[a.game]
    kick = g["kick"]
    if retrieval >= kick:
        print(f"STOP: kickoff {SCH.iso(kick)} has occurred (retrieval {SCH.iso(retrieval)}); no pregame forecast generated")
        sys.exit(3)
    season, week = g["season"], g["week"]
    miss = LVE.provider_lag_problems(got, sched, retrieval, season)
    group = f"ADHOC_{season}_{week:02d}_{SCH.iso(kick)}_{SCH.iso(retrieval)}"
    raw_info = r.v2raw.put(raw_sched)
    srcs = {n: (CAS.provider_id(n), (lambda b=b: b)) for n, b in got.items()}
    for n, msg in unavailable.items():
        srcs[n] = (CAS.provider_id(n), (lambda m=msg: (_ for _ in ()).throw(CAS.Unavailable(m))))
    rec = CAS.take_snapshot_set(r.store, r.ledger, srcs, LABEL, kick, retrieval, retrieval, group, note=NOTE, time_travel=False)
    r.lag.append_many([{"group": group, "raw_schedule_sha256": raw_info["sha256"], "retrieval_ts": SCH.iso(retrieval), "sources": audit, "provider_lag_missing": miss, "unavailable_optional": unavailable}])
    print("snapshot", rec["set_id"], rec["content_id"], SCH.iso(retrieval), "lag_missing", miss, flush=True)
    # information cutoff = the real retrieval time (the engine's T90 information rules; nothing is labelled T90)
    orig = SCH.forecast_cutoff
    SCH.forecast_cutoff = lambda k, hz: retrieval if hz == "T90" else orig(k, hz)
    fst = RelabelStore(ST.Store(root, "forecasts"), kick, SCH.iso(retrieval))
    bst = RelabelStore(ST.Store(root, "baselines"), kick, SCH.iso(retrieval))
    logs = r.run_context("T90", [(kick, rec, [a.game])], season, week, "adhoc-" + SCH.iso(retrieval), fst, bst, only_games={a.game})
    SCH.forecast_cutoff = orig
    print(json.dumps(logs, default=str), flush=True)
    # frozen v2 comparator from the SAME retrieval (raw schedule with market inputs from the separate store)
    d = CAS.materialize(r.store, rec, root / "tmp" / f"v2_{rec['set_id'][:16]}")
    lk = Path(d) / "games.csv"
    if lk.is_symlink() or lk.exists():
        lk.unlink()
    lk.write_bytes(r.v2raw.get(raw_info["sha256"]))
    fr = V2.Frozen()
    prov = {"retrieval_ts": SCH.iso(retrieval), "input_hashes": {"snapshot_set_id": rec["set_id"], "content_id": rec["content_id"], "raw_schedule_sha256": raw_info["sha256"],
                                                                 **{k: v for k, v in rec["files"].items() if k.startswith(("stats_player_week_", "snap_counts_"))}}}
    v2 = V2.log_group(root, fr, str(d), season, week, [a.game], LABEL, {a.game: SCH.iso(kick)}, {a.game: SCH.iso(retrieval)}, prov, LVE.utcnow, "adhoc")
    print(json.dumps({"v2": v2, "retrieval_ts": SCH.iso(retrieval), "set_id": rec["set_id"], "content_id": rec["content_id"]}), flush=True)


if __name__ == "__main__":
    main()
