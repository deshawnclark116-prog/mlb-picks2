"""
NFL_PHASE1D_DRYRUN  (Phase 1D)  -- multi-week TIME-TRAVEL operational validation of the complete runner on BURNED weeks

Weeks were fixed in nfl_models/nfl_player_outcome_phase1d/dry_run_plan.json BEFORE any of these runs. Outcome performance is never used for tuning; the scores
produced here only exercise the score logger, the baselines and the gate evaluators (machinery), they are not a verdict and not a holdout.

Per week (one process each; several weeks may run concurrently against one content-addressed store and one ledger set):
  A   the complete runner, T24 and T90, every game: snapshot retrieval (every source), schedule cutoff, weekly walk-forward refit, LiveLoader, forecast, baselines, statuses
  B   idempotent rerun of A into the same store: 0 new records, all duplicates verified, store bytes unchanged
  C   crash mid-run in a fresh store then restart: converges to the reference bytes
  D   deterministic reproduction from scratch (no cache) into a fresh store: identical bytes
  E1  corrupt TARGET-game rows in the SOURCE files: the snapshot content ids and the forecast bytes must not change
  E2  hand the runner a CONTAMINATED snapshot that contains the (corrupted) target-game rows: distribution outputs must still be identical (the loader is as-of correct)
  E3  corrupt a PRIOR completed game of a team in the subset: its forecasts SHOULD change (real historical sensitivity)
  F   grading (official files), score idempotency, forecasts untouched, baselines graded, board, provenance / snapshot verification, schedule-cutoff audit

  python -u nfl_phase1d_dryrun.py --root ROOT --cas CAS --season 2025 --week 11 [--n 25000]
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

import nfl_phase1_board as BD
import nfl_phase1_forecast as FC
import nfl_phase1_score as SCORE
import nfl_phase1_store as ST
import nfl_phase1d_cas as CAS
import nfl_phase1d_runner as RN
import nfl_phase1d_schedule as SCH

REPO = Path(__file__).resolve().parent
OUT = REPO / "nfl_models" / "nfl_player_outcome_phase1d" / "dry_run"


def rec_hashes(store, s=None, w=None):
    out = {}
    for r in FC.read_forecasts(store):
        if (s is None or r["season"] == s) and (w is None or r["week"] == w):
            out[r["id"]] = ST.sha(ST.canon({k: v for k, v in r.items() if not k.startswith("_")}))
    return out


def content_view(r):
    """The forecast content (distribution + inputs actually used), without the provenance of which snapshot set carried it."""
    drop = {"input_snapshots"}
    return ST.sha(ST.canon({k: v for k, v in r.items() if not k.startswith("_") and k not in drop}))


NUM_COLS_STATS = ["completions", "attempts", "passing_yards", "passing_tds", "carries", "rushing_yards", "rushing_tds", "receptions", "targets", "receiving_yards", "receiving_tds"]


def perturb_dir(src, dst, season, week, mode, teams=None, seed=7):
    """Copy of the data directory (symlinks for unchanged files) with rows corrupted. mode 'target': every game of (season, week); 'prior': the games of `teams` in
    (season, week). Corrupts stats, play-by-play yards / outcomes, snap counts and the schedule result columns of those games."""
    src, dst = Path(src), Path(dst)
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    games = list(csv.DictReader(open(src / "games.csv", newline="", encoding="utf-8")))
    hit_games = {r["game_id"] for r in games if r["season"] == str(season) and r["week"] == str(week) and r["game_type"] == "REG"
                 and (mode == "target" or r["home_team"] in (teams or ()) or r["away_team"] in (teams or ()))}
    for f in src.iterdir():
        name = f.name
        touch = name in (f"stats_player_week_{season}.csv", f"snap_counts_{season}.csv", f"pbp_{season}.csv.gz", "games.csv", f"participation_{season}.csv", f"ftn_{season}.csv")
        if not touch:
            os.symlink(f.resolve(), dst / name)
            continue
        if name == "games.csv":
            hdr = list(games[0].keys())
            for r in games:
                if r["game_id"] in hit_games:
                    r["home_score"], r["away_score"] = str(int(rng.integers(0, 45))), str(int(rng.integers(0, 45)))
                    r["result"] = str(int(r["home_score"]) - int(r["away_score"])); r["total"] = str(int(r["home_score"]) + int(r["away_score"]))
            buf = io.StringIO(); wr = csv.DictWriter(buf, fieldnames=hdr, lineterminator="\n"); wr.writeheader(); wr.writerows(games)
            (dst / name).write_bytes(buf.getvalue().encode())
            continue
        gz = name.endswith(".gz")
        text = gzip.decompress(f.read_bytes()).decode() if gz else f.read_text(encoding="utf-8")
        rdr = csv.DictReader(io.StringIO(text, newline=""))
        hdr = rdr.fieldnames
        rows = []
        for r in rdr:
            gid = r.get("game_id") or r.get("nflverse_game_id")
            if gid in hit_games:
                if name.startswith("stats_player_week"):
                    for c in NUM_COLS_STATS:
                        if c in r and r[c] not in ("", "NA"):
                            r[c] = str(max(0.0, float(r[c]) * rng.uniform(0, 3) + rng.integers(0, 15)))
                elif name.startswith("snap_counts"):
                    for c in ("offense_snaps", "defense_snaps"):
                        if r.get(c) not in ("", "NA", None):
                            r[c] = str(float(rng.integers(0, 70)))
                elif name.startswith("pbp"):
                    for c in ("yards_gained", "rushing_yards", "receiving_yards", "air_yards", "yards_after_catch"):
                        if r.get(c) not in ("", "NA", None):
                            r[c] = str(int(rng.integers(-5, 60)))
                    for c in ("touchdown", "rush_touchdown", "pass_touchdown", "interception", "complete_pass", "sack"):
                        if r.get(c) in ("0", "1"):
                            r[c] = str(int(rng.integers(0, 2)))
            rows.append(r)
        buf = io.StringIO(); wr = csv.DictWriter(buf, fieldnames=hdr, lineterminator="\n"); wr.writeheader(); wr.writerows(rows)
        b = buf.getvalue().encode()
        (dst / name).write_bytes(CAS._gz(b) if gz else b)
    return dst, sorted(hit_games)


def dry_week(a):
    t0 = time.time()
    s, w = a.season, a.week
    root = Path(a.root)
    log = lambda m: print(f"[{s} wk{w:02d}] {m}", flush=True)
    r = RN.Runner(root, a.data_dir, a.n, log=log, cas_root=a.cas)
    plan = r.plan_week(s, w)
    if a.limit_kick_groups:                                   # test mode: only the first K kickoff groups of the week
        keep = sorted({v[2] for v in plan.values()})[:a.limit_kick_groups]
        plan = {g: v for g, v in plan.items() if v[2] in keep}
    games_all = set(plan) if a.limit_kick_groups else None
    part_path = OUT / f"dry_run_{s}_wk{w:02d}.partial.json"
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(part_path.read_text()) if (a.resume and part_path.exists()) else {}
    save = lambda res_: part_path.write_text(json.dumps(res_, indent=1, default=str))
    kicks = sorted({v[2] for v in plan.values()})
    sub = {g for g, v in plan.items() if v[2] == kicks[0]}
    res = {"label": "TIME-TRAVEL OPERATIONAL VALIDATION on a BURNED week (not a holdout; no tuning)", "season": s, "week": w, "n_games": len(plan), "subset_games": sorted(sub),
           "n_draws": a.n}
    fstore = ST.Store(root, "forecasts")
    # ---- A
    logsA = [] if "run_A" in prior else r.run_week(s, w, "run-A", games=games_all)   # (plan is complete unless test mode)
    ref = rec_hashes(fstore, s, w)
    ok = [l for l in logsA if l["status"] == RN.STATUS_OK]
    if "run_A" in prior:
        res["run_A"] = prior["run_A"]
    else:
        res["run_A"] = {"game_horizons_expected": 2 * len(plan), "success": len(ok), "safe_explicit_failures": [l for l in logsA if l["status"] != RN.STATUS_OK], "records": len(ref),
                        "model_versions": sorted({l["model_version"] for l in ok}), "artifact_bundles": sorted({l["artifact_bundle_sha256"] for l in ok}), "seconds": round(time.time() - t0)}
    save(res)
    log(f"A done {res['run_A']['success']}/{res['run_A']['game_horizons_expected']} {res['run_A']['seconds']}s")
    # ---- B
    if "idempotent_rerun" in prior:
        res["idempotent_rerun"] = prior["idempotent_rerun"]
    else:
        logsB = r.run_week(s, w, "run-B", games=games_all)
        res["idempotent_rerun"] = {"new_records_written": sum(l.get("written", 0) for l in logsB), "verified_duplicates": sum(l.get("verified_duplicates", 0) for l in logsB),
                                   "store_unchanged": rec_hashes(fstore, s, w) == ref, "seconds": round(time.time() - t0)}
    save(res)
    log(f"B done {res['idempotent_rerun']}")
    sub_recs = [x for x in FC.read_forecasts(fstore) if x["season"] == s and x["week"] == w and x["game_id"] in sub]
    sub_ids = {x["id"] for x in sub_recs}
    ref_sub = {i: h for i, h in ref.items() if i in sub_ids}
    ref_full = {x["id"]: content_view(x) for x in sub_recs}
    sub_games = set(sub)

    def stage(key, fn):
        if key in prior:
            res[key] = prior[key]
        else:
            res[key] = fn()
        save(res)
        log(f"{key} done")

    # ---- C crash / restart
    def stage_c():
        crashed = False
        try:
            r.run_week(s, w, "run-C", kind="forecasts_crash", games=sub_games, crash_after=1)
        except RuntimeError:
            crashed = True
        cstore = ST.Store(root, "forecasts_crash")
        partial = len(rec_hashes(cstore, s, w))
        r.run_week(s, w, "run-D", kind="forecasts_crash", games=sub_games)
        return {"crashed": crashed, "records_after_crash": partial, "records_after_restart": len(rec_hashes(cstore, s, w)), "identical_to_reference": rec_hashes(cstore, s, w) == ref_sub,
                "tmp_files_left": cstore.recover()}
    stage("crash_restart", stage_c)

    # ---- D reproduction (no cache anywhere: complete recompute of every derived object into a fresh store)
    def stage_d():
        r.run_week(s, w, "run-E", kind="forecasts_repro", games=sub_games)
        return {"identical_bytes_to_reference": rec_hashes(ST.Store(root, "forecasts_repro"), s, w) == ref_sub, "n": len(ref_sub)}
    stage("reproduction", stage_d)

    # ---- E1 corrupt target rows in the source files
    pdir, hit = perturb_dir(a.data_dir, Path(a.work) / f"perturb_target_{s}_{w}", s, w, "target")

    def stage_e1():
        rp = RN.Runner(root, str(pdir), a.n, log=log, cas_root=a.cas)
        same_content, err = True, None
        try:
            rp.run_week(s, w, "run-P1", kind="forecasts_pert_source", games=sub_games)
        except ST.HardError as e:
            same_content, err = False, str(e)
        pstore = ST.Store(root, "forecasts_pert_source")
        return {"snapshot_content_ids_unchanged": same_content, "error": err, "identical_bytes_to_reference": rec_hashes(pstore, s, w) == ref_sub, "corrupted_games": hit}
    stage("perturbation_target_source", stage_e1)

    # ---- E2 contaminated snapshot (target rows present, corrupted)
    def stage_e2():
        rc = RN.Runner(root, str(pdir), a.n, log=log, cas_root=a.cas, contaminate=(s, w), group_suffix="-contaminated")
        rc.run_week(s, w, "run-P2", kind="forecasts_contaminated", games=sub_games)
        cst = ST.Store(root, "forecasts_contaminated")
        cv = {x["id"]: content_view(x) for x in FC.read_forecasts(cst) if x["season"] == s and x["week"] == w}
        return {"n": len(ref_full), "identical_distribution_outputs": cv == ref_full, "n_differing": sum(1 for k in ref_full if cv.get(k) != ref_full[k]),
                "note": "the snapshot handed to the loader contained corrupted target-game rows"}
    stage("perturbation_contaminated_snapshot", stage_e2)

    # ---- E3 corrupt a prior completed game
    def stage_e3():
        first_game = sorted(sub)[0]
        home0, away0 = plan[first_game][0], plan[first_game][1]
        played_prev = {t for r_ in r.tt.games if r_["season"] == str(s) and r_["week"] == str(w - 1) and r_["game_type"] == "REG" for t in (r_["home_team"], r_["away_team"])}
        home = home0 if home0 in played_prev else away0                                # a team that actually played the previous week (not on a bye)
        pdir2, hit2 = perturb_dir(a.data_dir, Path(a.work) / f"perturb_prior_{s}_{w}", s, w - 1, "prior", teams={home})
        rq = RN.Runner(root, str(pdir2), a.n, log=log, cas_root=a.cas, group_suffix="-priorpert")
        rq.run_week(s, w, "run-P3", kind="forecasts_prior_pert", games={first_game})
        qst = ST.Store(root, "forecasts_prior_pert")
        newf = {x["id"]: x for x in FC.read_forecasts(qst) if x["game_id"] == first_game}
        oldf = {x["id"]: x for x in FC.read_forecasts(fstore) if x["game_id"] == first_game}
        d_aff, d_oth = [], []
        for i, x in newf.items():
            o = oldf.get(i)
            if o is None:
                continue
            (d_aff if x["team"] == home else d_oth).append(abs(x["mean"] - o["mean"]))
        return {"corrupted_prior_games": hit2, "affected_team": home, "n_rows_affected_team": len(d_aff), "mean_abs_change_in_forecast_mean_affected_team": float(np.mean(d_aff)) if d_aff else None,
                "n_rows_other_team": len(d_oth), "mean_abs_change_other_team": float(np.mean(d_oth)) if d_oth else None,
                "forecast_ids_unchanged_but_bytes_differ": sum(1 for i, x in newf.items() if i in oldf and ST.canon(x) != ST.canon(oldf[i])), "n_compared": len(newf),
                "note": "forecast ids do not depend on inputs, so these were written to a separate store; the sensitivity is the change in the distributions"}
    if w > 1:
        stage("perturbation_prior_history", stage_e3)

    # ---- F grading, baselines, board, provenance, schedule audit
    D_ = Path(a.data_dir)
    stats_b = (D_ / f"stats_player_week_{s}.csv").read_bytes(); games_b = (D_ / "games.csv").read_bytes(); snap_b = (D_ / f"snap_counts_{s}.csv").read_bytes(); pl_b = (D_ / "players.csv").read_bytes()
    sstore = ST.Store(root, "scores", lock_timeout=7200)
    g1 = SCORE.grade(fstore, sstore, stats_b, games_b, snap_bytes=snap_b, players_bytes=pl_b)
    g2 = SCORE.grade(fstore, sstore, stats_b, games_b, snap_bytes=snap_b, players_bytes=pl_b)
    bstore = ST.Store(root, "baselines", lock_timeout=7200); bsstore = ST.Store(root, "baseline_scores", lock_timeout=7200)
    b1 = SCORE.grade(bstore, bsstore, stats_b, games_b, snap_bytes=snap_b, players_bytes=pl_b)
    b2 = SCORE.grade(bstore, bsstore, stats_b, games_b, snap_bytes=snap_b, players_bytes=pl_b)
    res["grading"] = {"forecasts_first": g1, "forecasts_rerun": g2, "baselines_first": b1, "baselines_rerun": b2, "forecasts_unchanged_after_grading": rec_hashes(fstore, s, w) == ref}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"shadow_board_{s}_wk{w:02d}_T24_sample.md").write_text("\n".join(BD.build(fstore, s, w, "T24").splitlines()[:120]) + "\n")
    store = r.store; led = r.ledger
    sets = [x for x in led.read() if x.get("type") == "set" and x["group"].startswith(f"{s}_{w:02d}_") and "-" not in x["group"].split("Z")[-1]]
    prob = []
    for st_ in sets:
        prob += CAS.verify_set(store, led, st_, CAS.logical_files(data_dir=a.data_dir))
    used = {x["input_snapshots"]["snapshot_set_id"] for x in FC.read_forecasts(fstore) if x["season"] == s and x["week"] == w}
    known = {x["set_id"] for x in led.read() if x.get("type") == "set"}
    late = [x["set_id"] for x in sets if CAS.parse_iso(x["retrieval_ts"]) > CAS.parse_iso(x["cutoff"])]
    stat_rows = [x for x in r.status.read() if x["season"] == s and x["week"] == w and x["run_id"] == "run-A"]
    res["provenance_and_schedule"] = {"snapshot_sets_verified": len(sets), "problems": prob[:10], "n_problems": len(prob), "forecast_sets_all_in_manifest": used <= known,
                                      "sets_retrieved_after_cutoff": late, "status_rows_run_A": len(stat_rows),
                                      "unexplained_missing_game_horizons": 2 * len(plan) - len(stat_rows), "schedule_revisions_recorded": len([x for x in r.sched.rows() if x["game_id"] in plan])}
    res["cas"] = {"blobs": store.blob_count(), "manifest_file_rows": sum(1 for x in led.read() if x.get("type") == "file"), "note": "identical bytes are stored once and referenced by every set"}
    res["seconds_total"] = round(time.time() - t0)
    (OUT / f"dry_run_{s}_wk{w:02d}.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"finished in {res['seconds_total']}s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cas", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--data-dir", default="/tmp/nflcsv")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--resume", action="store_true", help="skip stages already recorded in the partial result file")
    ap.add_argument("--limit-kick-groups", type=int, default=0, help="TEST MODE: only the first K kickoff groups of the week")
    dry_week(ap.parse_args())


if __name__ == "__main__":
    main()
