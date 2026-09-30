"""
NFL_PHASE1C_DRYRUN  (Phase 1C, shadow research)

Complete shadow dry run on a BURNED historical week that is treated as if it were the future, plus the chaos suite.

  python -u nfl_phase1c_dryrun.py --scratch DIR --adj FILES... --season 2025 --week 10 --out DIR [--chaos]

Nothing here touches production, the clean-forward window, or any sportsbook source.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import pickle
import shutil
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

import nfl_phase1_data as P1
import nfl_phase1_forecast as FC
import nfl_phase1_score as SCORE
import nfl_phase1_snapshots as SN
import nfl_phase1_store as ST
import nfl_phase1c_evaluate as EV
import nfl_phase1c_fit as FT
import nfl_phase1c_sim as SM

UTC = SN.UTC


def make_bundle(scratch, adj_files, n_draws, calibration=None, fit_end=202418, variant="adjudicated"):
    import nfl_phase1c_adjudicate as AJ
    rec = pickle.load(open(Path(scratch) / "records_full.pkl", "rb"))
    Rs, drecs = AJ.mask_warmup(rec["Rs"], rec["drecs"])
    cfg = EV.load_config(adj_files)
    eff = FT.fit_all(Rs, drecs, cfg, variant, fit_end)
    d = SM.defaults_from(eff, Rs)
    d["def_rate"] = {t: float(np.median(eff["def_rate"][t][eff["def_rate"][t] > 0])) for t in eff["def_rate"]}
    idx = {"rush": {k: i for i, k in enumerate(Rs["rush"].key)}, "rec": {k: i for i, k in enumerate(Rs["rec"].key)},
           "pass": {k: i for i, k in enumerate(Rs["pass"].key)}, "def": {(r["s"], r["w"], r["gid"]): i for i, r in enumerate(drecs)}}
    consts = json.load(open(EV.OUT / "constants.json"))
    return FC.Bundle(cfg, consts, calibration or {}, n_draws, eff, d, idx, extra={"fit_end": fit_end, "variant": variant})


def truncated_csv(path, season, week, cutoff=None, dt_col=None):
    """Bytes of a source file as it could have looked at the time: rows with week <= target week (and dt <= cutoff for timestamped depth charts)."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    keep = []
    for r in rows:
        if dt_col:
            if r.get(dt_col) and r[dt_col] <= cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"):
                keep.append(r)
        elif r.get("week") in (None, "") or int(r["week"]) <= week:
            keep.append(r)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader(); w.writerows(keep)
    return buf.getvalue().encode()


def snapshot_game(root, data_dir, kickoff, season, week, horizon, retrieved_offset=timedelta(hours=1), skip=()):
    cutoff = SN.forecast_time(kickoff, horizon)
    now = cutoff - retrieved_offset
    d = Path(data_dir)
    files = {"injuries": (d / f"injuries_{season}.csv", None), "weekly_rosters": (d / f"roster_weekly_{season}.csv", None),
             "depth_charts": (d / f"depth_charts_{season}.csv", "dt")}
    recs = {}
    for src, (path, dt) in files.items():
        if src in skip:
            continue
        b = truncated_csv(path, season, week, cutoff, dt)
        recs[src] = SN.take_snapshot(src, (lambda b=b: b), kickoff, horizon, root=root, now=now)
    return recs


def names_map(D, pack):
    out = {}
    for (s, w, tm), g in pack["games"].items():
        for t in g["types"].values():
            for gid in t["ids"]:
                r = D.roster.get((s, w, gid))
                out[(s, w, gid)] = (r or {}).get("name", "")
    return out


def week_games(pack, D, s, w):
    seen, out = set(), []
    for (ss, ww, tm) in sorted(pack["games"]):
        if (ss, ww) != (s, w):
            continue
        opp = D.game[(ss, ww, tm)]["opp"]
        k = tuple(sorted((tm, opp)))
        if k in seen or (ss, ww, opp) not in pack["games"]:
            continue
        seen.add(k); out.append((tm, opp))
    return out


def run_week(bundle, loader, D, names, store, s, w, games, horizons, snap_root, run_id, crash_after=None):
    logs, n = [], 0
    for hz in horizons:
        for (A, B) in games:
            if crash_after is not None and n >= crash_after:
                raise RuntimeError("simulated crash mid-run")
            logs.append(FC.run_game(bundle, loader, D, names, store, s, w, A, B, hz, snap_root, run_id))
            n += 1
    return logs


def record_hashes(store):
    return {r["id"]: ST.sha(ST.canon({k: v for k, v in r.items() if not k.startswith("_")})) for r in FC.read_forecasts(store)}


def dry_run(a):
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scratch = Path(a.scratch)
    bundle = make_bundle(scratch, a.adj, a.n)
    D = P1.Data(a.data_dir)
    pack = pickle.load(open(scratch / "p1a_inputs.pkl", "rb"))
    loader = FC.BurnedWeekLoader(pack)
    names = names_map(D, pack)
    s, w = a.season, a.week
    games = week_games(pack, D, s, w)
    work = Path(a.work); shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    snap_root = work / "snapshots"
    res = {"label": "PRE-FREEZE DRY RUN on a BURNED week treated as future (not a holdout, not clean-forward)", "season": s, "week": w, "n_games": len(games),
           "model_version": bundle.model_version, "n_draws": a.n}
    # 1. input snapshots (T24 and T90 independently), one set per distinct kickoff
    kicks = sorted({D.game[(s, w, A)]["kick"] for A, _ in games})
    for k in kicks:
        for hz in ("T24", "T90"):
            snapshot_game(snap_root, a.data_dir, k, s, w, hz)
    res["snapshot_audit"] = {"problems": SN.verify_all(snap_root), "n_snapshots": len(SN.read_manifest(snap_root)),
                             "retrieved_before_cutoff": all(SN.parse_iso(r["retrieval_ts"]) <= SN.parse_iso(r["forecast_ts"]) for r in SN.read_manifest(snap_root))}
    # 2. T24 + T90 forecasts
    fstore = ST.Store(work, "forecasts")
    logs1 = run_week(bundle, loader, D, names, fstore, s, w, games, ("T24", "T90"), snap_root, "run-A")
    ref_hash = record_hashes(fstore)
    res["run_A"] = {"games_ok": sum(l["status"] == "ok" for l in logs1), "skipped": [l for l in logs1 if l["status"] != "ok"], "records": len(ref_hash), "seconds": round(time.time() - t0)}
    # 3. restart / idempotency: rerun everything into the same store
    logs2 = run_week(bundle, loader, D, names, fstore, s, w, games, ("T24", "T90"), snap_root, "run-B")
    res["idempotent_rerun"] = {"new_records_written": sum(l.get("written", 0) for l in logs2), "verified_duplicates": sum(l.get("verified_duplicates", 0) for l in logs2),
                               "store_unchanged": record_hashes(fstore) == ref_hash}
    # 4. crash halfway then rerun into a fresh store must converge to the same bytes
    cstore = ST.Store(work, "forecasts_crash")
    crashed = False
    try:
        run_week(bundle, loader, D, names, cstore, s, w, games, ("T24", "T90"), snap_root, "run-C", crash_after=max(1, len(games) // 2))
    except RuntimeError:
        crashed = True
    partial = len(record_hashes(cstore))
    run_week(bundle, loader, D, names, cstore, s, w, games, ("T24", "T90"), snap_root, "run-D")
    res["crash_restart"] = {"crashed": crashed, "records_after_crash": partial, "records_after_restart": len(record_hashes(cstore)), "identical_to_reference": record_hashes(cstore) == ref_hash,
                            "tmp_files_left": cstore.recover()}
    # 5. deterministic reproduction from scratch (fresh store, fresh run)
    rstore = ST.Store(work, "forecasts_repro")
    run_week(bundle, loader, D, names, rstore, s, w, games, ("T24", "T90"), snap_root, "run-E")
    res["reproduction"] = {"identical_bytes": record_hashes(rstore) == ref_hash, "n": len(ref_hash)}
    # 6. grading against the official file
    sstore = ST.Store(work, "scores")
    stats_b = (Path(a.data_dir) / f"stats_player_week_{s}.csv").read_bytes(); games_b = (Path(a.data_dir) / "games.csv").read_bytes()
    g1 = SCORE.grade(fstore, sstore, stats_b, games_b)
    g2 = SCORE.grade(fstore, sstore, stats_b, games_b)
    res["grading"] = {"first": g1, "rerun": g2, "table": SCORE.summarize(sstore)}
    # 7. forecasts unaffected by grading; hashes re-verified
    res["hash_verification"] = {"forecasts_unchanged_after_grading": record_hashes(fstore) == ref_hash, "snapshots_verify_all": SN.verify_all(snap_root) == [],
                                "store_index_ok": bool(fstore.index())}
    json.dump(res, open(out / "dry_run_report.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("grading",)}, indent=1, default=float)[:3000])
    return res


def chaos(a):
    """Failure injection. Every case must fail SAFELY: no invented data, no overwritten forecast, a named reason."""
    scratch = Path(a.scratch)
    bundle = make_bundle(scratch, a.adj, 200)
    D = P1.Data(a.data_dir)
    pack = pickle.load(open(scratch / "p1a_inputs.pkl", "rb"))
    loader = FC.BurnedWeekLoader(pack)
    names = names_map(D, pack)
    s, w = a.season, a.week
    A, B = week_games(pack, D, s, w)[0]
    kick = D.game[(s, w, A)]["kick"]
    R = {}
    base = Path(a.work) / "chaos"; shutil.rmtree(base, ignore_errors=True); base.mkdir(parents=True)

    def fresh(name, skip=(), horizons=("T24", "T90")):
        root = base / name / "snap"
        for hz in horizons:
            snapshot_game(root, a.data_dir, kick, s, w, hz, skip=skip)
        return root, ST.Store(base / name, "forecasts")

    def one(store, root, hz, ld=loader, bd=bundle):
        return FC.run_game(bd, ld, D, names, store, s, w, A, B, hz, root, "chaos")
    # 1. missing injury snapshot
    root, st = fresh("missing_injuries", skip=("injuries",))
    r = one(st, root, "T24"); R["missing_injury_snapshot"] = {"status": r["status"], "reason": r.get("reason"), "records": len(st.all_records()), "pass": r["status"] == "skipped" and not st.all_records()}
    # 2. corrupt snapshot bytes
    root, st = fresh("corrupt")
    m = SN.read_manifest(root)[0]; p = root / m["path"]; os.chmod(p, 0o644); p.write_bytes(p.read_bytes() + b"tampered")
    r = one(st, root, "T24" if m["horizon"] == "T24" else "T90"); R["corrupt_snapshot_hash"] = {"status": r["status"], "reason": r.get("reason"), "records": len(st.all_records()), "pass": r["status"] == "skipped" and "hash" in r.get("reason", "")}
    # 3. snapshot retrieved after the cutoff
    root2 = base / "late" / "snap"
    try:
        SN.take_snapshot("injuries", lambda: b"x", kick, "T24", root=root2, now=SN.forecast_time(kick, "T24") + timedelta(minutes=1)); late_blocked = False
    except SN.SnapshotError:
        late_blocked = True
    R["late_snapshot_refused"] = {"pass": late_blocked}
    # 4. duplicate forecast id: identical bytes = verified no-op ; same id with different bytes = hard error
    root, st = fresh("dup")
    r1 = one(st, root, "T24"); r2 = one(st, root, "T24")
    recs = FC.read_forecasts(st)
    tampered = {k: v for k, v in recs[0].items() if not k.startswith("_")}; tampered["mean"] = tampered["mean"] + 1.0
    try:
        st.append_batch("evil", {}, [tampered]); hard = False
    except ST.HardError:
        hard = True
    R["duplicate_forecast_id"] = {"rerun_written": r2.get("written"), "rerun_duplicates": r2.get("verified_duplicates"), "conflict_is_hard_error": hard, "pass": r2.get("written") == 0 and hard}
    # 5. changed model bytes => new model_version => new ids; old records intact
    before = record_hashes_safe(st)
    bd2 = FC.Bundle(bundle.config, bundle.constants, {"rush_yds": {"method": "scale", "tau": 1.05}}, bundle.n_draws, bundle.eff, bundle.defaults, bundle.idx, extra={"different": 1})
    r3 = one(st, root, "T24", bd=bd2)
    after = record_hashes_safe(st)
    R["changed_model_bytes"] = {"new_model_version": bd2.model_version != bundle.model_version, "old_records_intact": all(after.get(k) == v for k, v in before.items()), "new_records_written": r3.get("written"),
                                "pass": bd2.model_version != bundle.model_version and all(after.get(k) == v for k, v in before.items()) and r3.get("written", 0) > 0}
    # 6. partially written forecast file
    root, st = fresh("partial")
    one(st, root, "T24")
    tmpf = st.dir / "batches" / "crashed.999.tmp"; tmpf.write_bytes(b'{"_batch":"x"}\n{"id":"deadbeef"')
    ok_ignored = tmpf.name in [Path(x).name for x in st.recover()] and "deadbeef" not in st.index()
    f = st.batch_files()[0]; os.chmod(f, 0o644); raw = f.read_bytes(); f.write_bytes(raw[:-60])
    try:
        one(st, root, "T24"); detected = False
    except ST.StoreCorrupt:
        detected = True
    R["partially_written_file"] = {"tmp_ignored": ok_ignored, "truncated_final_batch_detected_hard": detected, "pass": ok_ignored and detected}
    # 7. unavailable route data: the simulator does not consume route data; a pack without route / rz types must run and REPORT the fallback
    g = pack["games"][(s, w, A)]
    pk2 = {"games": {**pack["games"], (s, w, A): {**g, "types": {k: v for k, v in g["types"].items() if k not in ("rz_carry", "rz_target")}}}, "meta": pack["meta"]}
    root, st = fresh("route")
    r = one(st, root, "T24", ld=FC.BurnedWeekLoader(pk2))
    R["unavailable_route_or_rz_inputs"] = {"status": r["status"], "note": "route data is not a Phase 1C input; missing red-zone inputs fall back to the league red-zone share (constants.json), never to invented player data", "pass": r["status"] == "ok"}
    # 8. player absent from expected roster (no efficiency record): league default, counted in the log, never silent
    idx2 = {**bundle.idx, "rush": {k: v for k, v in bundle.idx["rush"].items() if k[2] != g["types"]["carry"]["ids"][0]}}
    bd3 = FC.Bundle(bundle.config, bundle.constants, {}, bundle.n_draws, bundle.eff, bundle.defaults, idx2, extra={"chaos": "absent_player"})
    root, st = fresh("absent")
    r = one(st, root, "T24", bd=bd3)
    R["player_absent_from_expected_roster"] = {"status": r["status"], "records": r.get("n_records"), "pass": r["status"] == "ok"}
    # 9/10. QB ruled out (T24 injuries snapshot lists the QB as Out although the model thinks he plays) and last-minute inactive at T90
    qb = g["types"]["qb_att"]["ids"][int(np.argmax(g["types"]["qb_att"]["P1"]))]
    root = base / "qb_out" / "snap"
    for hz in ("T24", "T90"):
        cutoff = SN.forecast_time(kick, hz)
        inj = truncated_csv(Path(a.data_dir) / f"injuries_{s}.csv", s, w)
        rows = list(csv.DictReader(io.StringIO(inj.decode())))
        for r_ in rows:
            if r_["gsis_id"] == qb and int(r_["week"]) == w:
                r_["report_status"] = "Out"
        if not any(r_["gsis_id"] == qb and int(r_["week"]) == w for r_ in rows):
            rows.append({**rows[0], "gsis_id": qb, "team": A, "week": str(w), "season": str(s), "report_status": "Out"})
        buf = io.StringIO(); wr = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n"); wr.writeheader(); wr.writerows(rows)
        SN.take_snapshot("injuries", (lambda b=buf.getvalue().encode(): b), kick, hz, root=root, now=cutoff - timedelta(hours=1))
        SN.take_snapshot("weekly_rosters", (lambda: truncated_csv(Path(a.data_dir) / f"roster_weekly_{s}.csv", s, w)), kick, hz, root=root, now=cutoff - timedelta(hours=1))
        SN.take_snapshot("depth_charts", (lambda c=cutoff: truncated_csv(Path(a.data_dir) / f"depth_charts_{s}.csv", s, w, c, "dt")), kick, hz, root=root, now=cutoff - timedelta(hours=1))
    st = ST.Store(base / "qb_out", "forecasts")
    r = one(st, root, "T24")
    recs = [x for x in FC.read_forecasts(st) if x["player_id"] == qb and x["outcome"] == "pass_yds"]
    R["qb_ruled_out"] = {"status": r["status"], "qb": qb, "overrides": r.get("snapshot_rules_applied"), "qb_p_active": recs[0]["p_active"] if recs else None,
                         "qb_mean_pass_yds": recs[0]["mean"] if recs else None, "pass": bool(recs) and recs[0]["p_active"] <= 0.02 + 1e-9}
    starter = g["types"]["target"]["ids"][int(np.argmax(g["types"]["target"]["P1"]))]
    root, st = fresh("inactive")
    ros = list(csv.DictReader(io.StringIO(truncated_csv(Path(a.data_dir) / f"roster_weekly_{s}.csv", s, w).decode())))
    for r_ in ros:
        if r_["gsis_id"] == starter and int(r_["week"]) == w:
            r_["status"] = "INA"
    buf = io.StringIO(); wr = csv.DictWriter(buf, fieldnames=list(ros[0].keys()), lineterminator="\n"); wr.writeheader(); wr.writerows(ros)
    root_i = base / "inactive2" / "snap"
    for hz in ("T24", "T90"):
        cutoff = SN.forecast_time(kick, hz)
        SN.take_snapshot("injuries", (lambda: truncated_csv(Path(a.data_dir) / f"injuries_{s}.csv", s, w)), kick, hz, root=root_i, now=cutoff - timedelta(hours=1))
        SN.take_snapshot("weekly_rosters", (lambda b=buf.getvalue().encode(): b), kick, hz, root=root_i, now=cutoff - timedelta(hours=1))
        SN.take_snapshot("depth_charts", (lambda c=cutoff: truncated_csv(Path(a.data_dir) / f"depth_charts_{s}.csv", s, w, c, "dt")), kick, hz, root=root_i, now=cutoff - timedelta(hours=1))
    st90 = ST.Store(base / "inactive2", "forecasts")
    r = one(st90, root_i, "T90")
    x = [z for z in FC.read_forecasts(st90) if z["player_id"] == starter and z["outcome"] == "rec_yds"]
    st24 = ST.Store(base / "inactive2b", "forecasts")
    r24 = one(st24, root_i, "T24")
    x24 = [z for z in FC.read_forecasts(st24) if z["player_id"] == starter and z["outcome"] == "rec_yds"]
    R["last_minute_inactive"] = {"player": starter, "T90_p_active": x[0]["p_active"] if x else None, "T90_mean_rec_yds": x[0]["mean"] if x else None,
                                 "T90_p90": x[0]["quantiles"]["p90"] if x else None, "T24_p_active": x24[0]["p_active"] if x24 else None, "T24_mean_rec_yds": x24[0]["mean"] if x24 else None,
                                 "pass": bool(x) and x[0]["p_active"] == 0.0 and x[0]["mean"] == 0.0 and x[0]["quantiles"]["p95"] == 0.0 and bool(x24) and x24[0]["p_active"] > 0}
    # 11. depth-chart file missing (2025+ requires it)
    root, st = fresh("nodepth", skip=("depth_charts",))
    r = one(st, root, "T24")
    R["depth_chart_missing"] = {"status": r["status"], "reason": r.get("reason"), "records": len(st.all_records()), "pass": r["status"] == "skipped" and not st.all_records()}
    # 12. stats grading file revised: a new revision is appended, the original score and the forecasts are untouched
    root, st = fresh("revised")
    one(st, root, "T24")
    sst = ST.Store(base / "revised", "scores")
    stats_b = (Path(a.data_dir) / f"stats_player_week_{s}.csv").read_bytes(); games_b = (Path(a.data_dir) / "games.csv").read_bytes()
    g1 = SCORE.grade(st, sst, stats_b, games_b)
    rows = list(csv.DictReader(io.StringIO(stats_b.decode())))
    for r_ in rows:
        if r_["player_id"] == starter and int(r_["week"]) == w:
            r_["receiving_yards"] = str(float(r_["receiving_yards"] or 0) + 25)
    buf = io.StringIO(); wr = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n"); wr.writeheader(); wr.writerows(rows)
    before_f = record_hashes_safe(st)
    g2 = SCORE.grade(st, sst, buf.getvalue().encode(), games_b)
    R["grading_file_revised"] = {"first": g1, "revised": g2, "forecasts_unchanged": record_hashes_safe(st) == before_f,
                                 "pass": g1["graded_new"] > 0 and g2["revisions_appended"] > 0 and record_hashes_safe(st) == before_f and len(sst.all_records()) == g1["graded_new"] + g2["graded_new"]}
    # 13. no snapshot at all for a future game (nothing retrievable yet): skipped
    r = FC.run_game(bundle, loader, D, names, ST.Store(base / "none", "forecasts"), s, w, A, B, "T24", base / "nothing", "chaos")
    R["no_snapshots_at_all"] = {"status": r["status"], "pass": r["status"] == "skipped"}
    R["all_pass"] = all(v.get("pass") for v in R.values() if isinstance(v, dict))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(R, open(Path(a.out) / "chaos_results.json", "w"), indent=1, default=float)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in R.items()}, indent=1))
    return R


def record_hashes_safe(store):
    return {r["id"]: ST.sha(ST.canon({k: v for k, v in r.items() if not k.startswith("_")})) for r in FC.read_forecasts(store)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/tmp/nflcsv")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--adj", nargs="+", required=True)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--week", type=int, default=10)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--chaos", action="store_true")
    a = ap.parse_args()
    chaos(a) if a.chaos else dry_run(a)


if __name__ == "__main__":
    main()
