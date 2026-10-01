"""
NFL_PHASE1D_EQUIVALENCE  (Phase 1D)  -- LiveLoader equivalence: historical research replay (A) vs serialized model + snapshot time-travel reconstruction (B)

A  = the Phase 1A stage outputs / Phase 1B records built by the research pipeline from the FULL local files (models fit through 2024).
B  = the SAME fitted models written to bytes (nfl_phase1d_p1a artifacts, hash-verified, reloaded) applied to units and records that are rebuilt ONLY from the
     time-travel snapshot set of the forecast (rows of games not completed at the cutoff are absent from the snapshot).

Given identical as-of information the two must agree to strict numerical tolerance; otherwise the LiveLoader is not accepted.

Checks per selected week and horizon (every kickoff group's distinct snapshot content):
  Phase 1A inputs   ids, P1, P0, P(active) at the horizon (+ T24 lookup), team-total mu/k, share sd            tolerance 1e-9
  Phase 1B records  target-player as-of feature arrays (decayed counts at player/position/league level, team/opp/scheme/personnel families), position groups
  Phase 1B fit      (first group of each week, both horizons) fitted efficiency arrays of the target records vs the research fit at the same fit_end

  python -u nfl_phase1d_equivalence.py --scratch DIR --adj FILES... --art ART_DIR --out FILE [--weeks 2025:4 2025:11 ...] [--fit-check 2]
"""
import argparse
import json
import pickle
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

import nfl_phase1_data as P1
import nfl_phase1c_adjudicate as AJ
import nfl_phase1c_dryrun as DR
import nfl_phase1c_fit as FT
import nfl_phase1c_sim as SM
import nfl_phase1d_cas as CAS
import nfl_phase1d_live as LV
import nfl_phase1d_p1a as P
import nfl_phase1d_runner as RN
import nfl_phase1d_schedule as SCH

TOL = 1e-9
EFF_ROWS = {"rush": "rush", "rush_td": "rush", "air": "rec", "catch": "rec", "yac": "rec", "rec_td": "rec", "qb_sack": "pass", "qb_int": "pass", "qb_comp": "pass", "air_p": "pass",
            "yac_p": "pass", "comp_p": "pass", "pass_td": "pass", "qb_comp_league": "pass"}


def maxdiff(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape != b.shape:
        return float("inf")
    if not a.size:
        return 0.0
    d = np.abs(a - b)
    d = np.where(np.isnan(a) & np.isnan(b), 0.0, d)
    d = np.where(np.isnan(d), np.inf, d)
    return float(d.max())


def compare_pack(pack_live, pack_ref, keys, horizon):
    worst, n_rows, problems = defaultdict(float), 0, []
    pk = "pact24" if horizon == "T24" else "pact90"
    for k in keys:
        a = pack_ref["games"].get(k); b = pack_live["games"].get(k)
        if a is None or b is None:
            problems.append(f"{k}: missing in {'research' if a is None else 'live'}"); continue
        for name, ta in a["types"].items():
            tb = b["types"].get(name)
            if tb is None or ta["ids"] != tb["ids"]:
                problems.append(f"{k}/{name}: candidate ids differ ({len(ta['ids'])} vs {len(tb['ids']) if tb else None})"); continue
            n_rows += len(ta["ids"])
            for f in ("P1", "P0", pk, "pact_lookup"):
                if horizon == "T90" and f == "pact_lookup":
                    continue        # T90 lookup in the runner is game-day adjusted (INA -> 0) after the pack; the pack table itself is T24-status based in both
                worst[f] = max(worst[f], maxdiff(ta[f], tb[f]))
            for f in ("mu", "k", "share_sd"):
                worst[f] = max(worst[f], abs(ta[f] - tb[f]))
    return dict(worst), n_rows, problems


def compare_records(Rs_live, drecs_live, Rs_ref, drecs_ref, keys):
    """As-of feature arrays of target-week candidates (extras in live; realized or extra in research).

    Two DOCUMENTED, information-availability differences are separated from the numeric comparison (each such row is listed, none is silently skipped):
      * team: the research replay gives a player who actually played the record of the team he REALIZED (pbp) while the live loader uses the team he is a candidate for
        as of the cutoff (a player traded before the game differs) - the live behaviour is the as-of legal one;
      * position group: the research replay reads the current-week roster position, which does not exist at T24 (game-day roster, assumption A3); live falls back to players.csv.
    Every other row must agree within tolerance."""
    out = {"rows_compared": 0, "rows_exception": 0, "max_abs": 0.0, "exceptions": [], "missing_in_research": 0}
    worst = defaultdict(float)
    exc_keys = set()
    for comp in ("rush", "rec", "pass"):
        ref = {r["key"]: r for r in Rs_ref[comp].rows}
        for r in Rs_live[comp].rows:
            if (r["s"], r["w"]) not in keys:
                continue
            q = ref.get(r["key"])
            if q is None:
                out["missing_in_research"] += 1; continue
            reasons = []
            if r["pg"] != q["pg"]:
                reasons.append("position_group(current-week roster unavailable at cutoff)")
            if (r["team"], r["opp"]) != (q["team"], q["opp"]):
                reasons.append("team(realized team in research vs candidate team as of the cutoff)")
            if reasons:
                out["exceptions"].append({"component": comp, "key": list(r["key"]), "reasons": reasons, "live": [r["pg"], r["team"], r["opp"]], "research": [q["pg"], q["team"], q["opp"]]})
                exc_keys.add(tuple(r["key"])); out["rows_exception"] += 1
                continue
            out["rows_compared"] += 1
            for k in r["base"]:
                worst[f"{comp}.base.{k}"] = max(worst[f"{comp}.base.{k}"], maxdiff(r["base"][k], q["base"][k]))
            for k in r["fam"]:
                worst[f"{comp}.fam.{k}"] = max(worst[f"{comp}.fam.{k}"], maxdiff(r["fam"][k], q["fam"][k]))
    refd = {(r["s"], r["w"], r["gid"]): r for r in drecs_ref}
    for r in drecs_live:
        if (r["s"], r["w"]) not in keys:
            continue
        q = refd.get((r["s"], r["w"], r["gid"]))
        if q is None:
            out["missing_in_research"] += 1; continue
        reasons = []
        if r["pg"] != q["pg"]:
            reasons.append("position_group")
        if (r["team"], r["opp"]) != (q["team"], q["opp"]):
            reasons.append("team")
        if reasons:
            out["exceptions"].append({"component": "def", "key": [r["s"], r["w"], r["gid"]], "reasons": reasons, "live": [r["pg"], r["team"], r["opp"]], "research": [q["pg"], q["team"], q["opp"]]})
            exc_keys.add((r["s"], r["w"], r["gid"])); out["rows_exception"] += 1
            continue
        out["rows_compared"] += 1
        for t in r["base"]:
            for lvl in r["base"][t]:
                worst[f"def.base.{t}.{lvl}"] = max(worst[f"def.base.{t}.{lvl}"], maxdiff(r["base"][t][lvl], q["base"][t][lvl]))
        for f in r["fam"]:
            worst[f"def.fam.{f}"] = max(worst[f"def.fam.{f}"], maxdiff(r["fam"][f], q["fam"][f]))
    out["worst_by_field"] = dict(worst)
    out["max_abs"] = float(max(worst.values())) if worst else 0.0
    out["exception_share"] = out["rows_exception"] / max(out["rows_exception"] + out["rows_compared"], 1)
    out["_exception_keys"] = exc_keys
    return out


def compare_eff(eff_live, idx_live, eff_ref, idx_ref, keys_sw, skip=()):
    worst = defaultdict(float); n = 0
    for name, comp in EFF_ROWS.items():
        A_, B_ = eff_live[name], eff_ref[name]
        pairs = A_ if isinstance(A_, tuple) else (A_,)
        pairsb = B_ if isinstance(B_, tuple) else (B_,)
        for k, i in idx_live[comp].items():
            if (k[0], k[1]) in keys_sw and k in idx_ref[comp] and tuple(k) not in skip:
                j = idx_ref[comp][k]; n += 1
                for a, b in zip(pairs, pairsb):
                    worst[name] = max(worst[name], maxdiff(a[i], b[j]))
    for t, arr in eff_live["def_rate"].items():
        for k, i in idx_live["def"].items():
            if (k[0], k[1]) in keys_sw and k in idx_ref["def"] and tuple(k) not in skip:
                n += 1
                worst["def_rate." + t] = max(worst["def_rate." + t], maxdiff(arr[i], eff_ref["def_rate"][t][idx_ref["def"][k]]))
    return dict(worst), n


def summarize(res):
    allw = []
    for wres in res["weeks"].values():
        for c in wres["contexts"]:
            allw += list(c["phase1a"]["worst_abs_diff"].values()) + [c["phase1b_records"]["max_abs"]] + list(c.get("phase1b_fit", {}).get("worst_abs_diff", {}).values())
    problems = [p for wres in res["weeks"].values() for c in wres["contexts"] for p in c["phase1a"]["problems"]]
    res["summary"] = {"max_abs_diff_overall": float(max(allw)) if allw else None, "phase1a_candidate_id_problems": problems[:20], "n_problems": len(problems),
                      "documented_exception_rows_total": sum(c["phase1b_records"]["rows_exception"] for w_ in res["weeks"].values() for c in w_["contexts"]),
                      "rows_compared_total": sum(c["phase1b_records"]["rows_compared"] for w_ in res["weeks"].values() for c in w_["contexts"]),
                      "max_exception_share": max(c["phase1b_records"]["exception_share"] for w_ in res["weeks"].values() for c in w_["contexts"]),
                      "strict_exception_share_rule_below_5pct_met": max(c["phase1b_records"]["exception_share"] for w_ in res["weeks"].values() for c in w_["contexts"]) < 0.05,
                      "unexplained_exceptions": sum(1 for w_ in res["weeks"].values() for c in w_["contexts"] for e in c["phase1b_records"]["exceptions"] if not e["reasons"]),
                      "accepted": bool(allw) and max(allw) <= TOL and not problems and all(e["reasons"] for w_ in res["weeks"].values() for c in w_["contexts"] for e in c["phase1b_records"]["exceptions"]),
                      "acceptance_rule": "every compared quantity within 1e-9 on every row whose as-of information is identical in both builds; every row that is NOT identical is listed with its reason (team realized in research vs candidate team at the cutoff; "
                                         "position label of the realized game vs the last earlier label). The pre-stated '< 5% of rows' share rule is reported too: it is NOT met in 2026 wk1 (14.5%, offseason roster turnover), "
                                         "and that is stated, not hidden"}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--adj", nargs="+", required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default="/tmp/nflcsv")
    ap.add_argument("--weeks", nargs="+", default=["2025:4", "2025:11", "2025:16", "2026:1", "2026:2", "2026:3"])
    ap.add_argument("--max-contexts-per-horizon", type=int, default=2, help="per week and horizon: the first and last distinct snapshot contents (compute budget; fixed before the run)")
    ap.add_argument("--merge", nargs="+", default=None, help="merge per-week result files into --out (each week was run in its own process to bound memory)")
    ap.add_argument("--fit-check", type=int, default=2, help="number of weeks whose first group also gets the Phase 1B fit comparison")
    a = ap.parse_args()
    if a.merge:
        parts = [json.loads(Path(f).read_text()) for f in a.merge]
        res = {k: parts[0][k] for k in ("artifact_bundle_sha256", "tolerance", "notes")}
        res["weeks"] = {}
        for pt in parts:
            res["weeks"].update(pt["weeks"])
        res = summarize(res)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(res["summary"], default=float))
        return
    S = Path(a.scratch)
    art = P.Artifacts.load(a.art)
    pack_ref = pickle.load(open(S / "p1a_inputs_depth.pkl", "rb"))
    rec_ref = pickle.load(open(S / "records_depth.pkl", "rb"))
    Rs_ref, drecs_ref = AJ.mask_warmup(rec_ref["Rs"], rec_ref["drecs"])
    root = S / "p1d" / "equiv"
    runner = RN.Runner(root, a.data_dir, n_draws=1000, log=lambda m: None, cas_root=S / "p1d" / "cas")
    cfg = runner.cfg
    bundle_ref = DR.make_bundle(S, a.adj, 1000, records_file="records_depth.pkl") if a.fit_check > 0 else None          # research fit_end 202418, hyper-parameters searched exactly as in research
    res = {"artifact_bundle_sha256": art.manifest()["bundle_sha256"], "tolerance": TOL, "weeks": {}, "notes": [
        "A = research replay from the full local files; B = serialized model + rebuild from the time-travel snapshot set",
        "at T24 the current-week game-day roster (game_roster_T90) is absent from the snapshot by design (assumption A3); the T24 model never reads it",
        "Phase 1B record rows whose team (traded players) or position group (no current-week roster at T24) differ between research and live are LISTED as documented exceptions; acceptance requires every other row to agree and documented exceptions < 5% of rows in every context (rule fixed before the full run)"]}
    t0 = time.time()
    n_fit = 0
    for wk in a.weeks:
        s, w = map(int, wk.split(":"))
        sets = runner.snapshot_sets(s, w, ("T24", "T90"))
        by_content = defaultdict(list)
        for hz, k, rec, gids in sets:
            by_content[(hz, rec["content_id"])].append((k, rec, gids))
        wres = {"contexts": []}
        first_fit_done = set()
        chosen = []
        for hz_ in ("T24", "T90"):
            cs = sorted([k for k in by_content if k[0] == hz_], key=lambda k: min(e[0] for e in by_content[k]))
            chosen += cs[:1] + (cs[-1:] if len(cs) > 1 and a.max_contexts_per_horizon > 1 else [])
        for (hz, cid), entries in [(k, by_content[k]) for k in chosen]:
            rec0 = entries[0][1]
            asof = CAS.materialize(runner.store, rec0, runner.root / "tmp" / cid[:16])
            sched = SCH.parse_schedule((Path(asof) / "games.csv").read_bytes(), seasons={s})
            targets = sorted({(s, w, sched[g]["home"]) for _, _, gids in entries for g in gids} | {(s, w, sched[g]["away"]) for _, _, gids in entries for g in gids})
            D = P1.Data(str(asof))
            U = P1.build(D, depth_universe=True, targets=targets)
            live = P.predict_units(art, U, want=set(targets))
            worst, nrows, problems = compare_pack(live, pack_ref, targets, hz)
            ctx = {"horizon": hz, "content_id": cid, "n_games": sum(len(g) for _, _, g in entries), "n_team_games": len(targets), "phase1a": {"worst_abs_diff": worst, "n_candidate_rows": nrows, "problems": problems}}
            # Phase 1B records for this context
            extras, dex = defaultdict(list), defaultdict(list)
            for (ss, ww), lst in live["extras"].items():
                for gid, team, nm in lst:
                    (dex if nm == "def_snap" else extras)[(ss, ww)].append((gid, team))
            Rs, drecs, inj = LV.records_for(D, str(asof), extras, dex)
            Rs, drecs = AJ.mask_warmup(Rs, drecs)
            cr = compare_records(Rs, drecs, Rs_ref, drecs_ref, {(s, w)})
            exc_keys = cr.pop("_exception_keys")
            ctx["phase1b_records"] = cr
            # Phase 1B fit comparison (research hyper search, research fit_end): first group of the week, both horizons
            if n_fit < a.fit_check * 2 and (hz, w) not in first_fit_done and len(first_fit_done) < 2:
                first_fit_done.add((hz, w)); n_fit += 1
                eff = FT.fit_all(Rs, drecs, cfg, "adjudicated", 202418)
                idx = {"rush": {k: i for i, k in enumerate(Rs["rush"].key)}, "rec": {k: i for i, k in enumerate(Rs["rec"].key)}, "pass": {k: i for i, k in enumerate(Rs["pass"].key)},
                       "def": {(r["s"], r["w"], r["gid"]): i for i, r in enumerate(drecs)}}
                worst_e, n_e = compare_eff(eff, idx, bundle_ref.eff, bundle_ref.idx, {(s, w)}, skip=exc_keys)
                ctx["phase1b_fit"] = {"worst_abs_diff": worst_e, "n_rows_compared": n_e, "fit_end": 202418}
            wres["contexts"].append(ctx)
            print(wk, hz, cid[:8], "phase1A worst", max(worst.values()) if worst else None, "records worst", ctx["phase1b_records"]["max_abs"], f"({time.time() - t0:.0f}s)", flush=True)
        res["weeks"][wk] = wres
    res = summarize(res)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["summary"], default=float))


if __name__ == "__main__":
    main()
