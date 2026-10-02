"""
NFL_PHASE1E_V2_EQUIV (Phase 1E) -- v2 forward-logger time-travel equivalence + leakage perturbation tests on burned weeks.

For each burned week: the logger's candidate features / point predictions are computed from the files a provider WOULD have served at the T24 cutoff (TimeTravelSource: no row of
any game that had not completed by the cutoff) and compared with the honest implementation (full-data v3.Replayer rows of the same week predicted with the same frozen artifact).
Perturbations: (1) the target week's outcomes are placed in the files (contaminated) and altered -> the forecast must not change; (2) an earlier game of one player is altered -> his forecast must change.

  python -u nfl_phase1e_v2_equiv.py --data-dir /tmp/nflcsv --out nfl_models/nfl_player_outcome_phase1e/v2_time_travel_equivalence.json
"""
import argparse
import csv
import io
import json
import os
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np

import nfl_phase1d_cas as CAS
import nfl_phase1d_schedule as SCH
import nfl_phase1e_v2 as V2
import nfl_yardage_v2 as v2
import nfl_yardage_v3 as v3

WEEKS = [(2025, 2), (2025, 7), (2025, 13), (2025, 17), (2026, 3)]
TOL = 1e-9


def asof_dir(data_dir, s, w, dest, horizon="T24"):
    """Directory with exactly what the providers would have served at the cutoff of the EARLIEST kickoff of (s, w) (the most restrictive; every game of the week is later)."""
    tt = CAS.TimeTravelSource(data_dir)
    kicks = [k for g, k in tt.kick.items() if g.startswith(f"{s}_{w:02d}_")]
    cutoff = SCH.forecast_cutoff(min(kicks), horizon)
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    done = tt.completed_ids(cutoff)
    raw = (Path(data_dir) / "games.csv").read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    header = list(rows[0].keys())
    for r in rows:
        if r["game_id"] not in done:
            for c in CAS.BLANK_UNTIL_COMPLETE:
                if c in r:
                    r[c] = ""
    (dest / "games.csv").write_bytes(CAS._csv_bytes(header, rows))                     # RAW schedule WITH the market columns (v2 recipe input), outcomes blanked
    for sx in V2.SEASONS:
        for tpl in ("stats_player_week_{s}.csv", "snap_counts_{s}.csv"):
            name = tpl.format(s=sx)
            if (Path(data_dir) / name).exists():
                (dest / name).write_bytes(tt.bytes_for(name, cutoff, (s, w), horizon)[0])
    return dest, cutoff


def preds_for(frozen, rows, oc):
    elig = [(m, f) for m, f in rows[oc] if m["eligible"]]
    if not elig:
        return {}
    X = np.array([[f.get(c) if f.get(c) is not None else np.nan for c in v2.FEATURES] for _, f in elig], dtype=np.float32)
    p = frozen.predict(oc, X)
    return {(m["game_id"], m["player_id"]): float(x) for (m, _), x in zip(elig, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/tmp/nflcsv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frozen = V2.Frozen()
    d3_full = v3.Data(a.data_dir, V2.SEASONS)
    honest = {}
    for oc, market in V2.MARKETS.items():                                          # honest implementation: full-data replay rows, same frozen artifact
        rows = v3.Replayer(d3_full, market).replay()
        X = np.array([[r[1].get(c) if r[1].get(c) is not None else np.nan for c in v2.FEATURES] for r in rows], dtype=np.float32)
        p = frozen.predict(oc, X)
        honest[oc] = {(r[0]["s"], r[0]["w"], r[0]["team"], r[0]["pid"]): float(x) for r, x in zip(rows, p)}
    res = {"tolerance": TOL, "weeks": {}, "perturbation": {}}
    tmp = Path(tempfile.mkdtemp(prefix="v2eq_"))
    for (s, w) in WEEKS:
        d, cutoff = asof_dir(a.data_dir, s, w, tmp / f"asof_{s}_{w}")
        rows, _ = V2.candidate_rows(str(d), s, w)
        wk = {"cutoff_used": SCH.iso(cutoff)}
        for oc in V2.MARKETS:
            lp = preds_for(frozen, rows, oc)
            team_of = {(m["game_id"], m["player_id"]): m["team"] for m, _ in rows[oc]}
            hp = {(k[3], k[2]): v for k, v in honest[oc].items() if k[0] == s and k[1] == w}          # (pid, team) -> pred
            lp_pt = {(pid, team_of[(g, pid)]): v for (g, pid), v in lp.items()}
            common = sorted(set(hp) & set(lp_pt))
            diffs = np.array([abs(hp[k] - lp_pt[k]) for k in common])
            missing = sorted(set(hp) - set(lp_pt))
            def prev_team(pid):
                ks = sorted((k for k, x in d3_full.stats.items() if pid in x and (k[0], k[1]) < (s, w)), key=lambda k: (k[0], k[1]))
                return ks[-1][2] if ks else None
            expl = [{"player_id": pid, "target_team": tm, "previous_game_team": prev_team(pid)} for pid, tm in missing]
            unexplained = [e for e in expl if e["previous_game_team"] == e["target_team"]]
            wk[oc] = {"n_honest_rows": len(hp), "n_logger_candidates": len(lp_pt), "n_honest_rows_missing_from_logger": len(missing), "missing_rows_are_team_changers": expl, "n_unexplained_missing": len(unexplained),
                      "n_compared": len(common), "max_abs_diff": float(diffs.max()) if len(diffs) else None, "all_within_tol": bool(len(diffs) and diffs.max() <= TOL and not unexplained)}
        res["weeks"][f"{s}_wk{w}"] = wk
    # ---- leakage perturbations on 2025 wk13
    s, w = 2025, 13
    base_dir, cutoff = asof_dir(a.data_dir, s, w, tmp / "base")
    base_rows, _ = V2.candidate_rows(str(base_dir), s, w)
    base = {oc: preds_for(frozen, base_rows, oc) for oc in V2.MARKETS}
    full_dir = tmp / "contaminated"; full_dir.mkdir()
    for f in ("games.csv", f"stats_player_week_{s}.csv", f"snap_counts_{s}.csv"):
        shutil.copy(Path(a.data_dir) / f, full_dir / f)
    for sx in V2.SEASONS:
        for tpl in ("stats_player_week_{s}.csv", "snap_counts_{s}.csv"):
            n = tpl.format(s=sx)
            if not (full_dir / n).exists() and (base_dir / n).exists():
                shutil.copy(base_dir / n, full_dir / n)
    # (1a) contaminated with the REAL outcomes of the target week and later; (1b) the same outcomes multiplied / zeroed
    def rewrite(path, fn):
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        hdr = list(rows[0].keys())
        for r in rows:
            fn(r)
        Path(path).write_bytes(CAS._csv_bytes(hdr, rows))
    def scramble(r):
        if r["season"] == str(s) and int(r["week"]) >= w:
            for c in ("carries", "rushing_yards", "targets", "receiving_yards", "receptions"):
                if r.get(c) not in (None, ""):
                    r[c] = str(float(r[c]) * 7 + 13)
    c1 = {oc: preds_for(frozen, V2.candidate_rows(str(full_dir), s, w)[0], oc) for oc in V2.MARKETS}
    rewrite(full_dir / f"stats_player_week_{s}.csv", scramble)
    c2 = {oc: preds_for(frozen, V2.candidate_rows(str(full_dir), s, w)[0], oc) for oc in V2.MARKETS}
    pert = {}
    for oc in V2.MARKETS:
        keys = sorted(base[oc])
        pert[oc] = {"n": len(keys), "contaminated_real_outcomes_max_abs_diff": max(abs(c1[oc][k] - base[oc][k]) for k in keys), "contaminated_scrambled_outcomes_max_abs_diff": max(abs(c2[oc][k] - base[oc][k]) for k in keys),
                    "unchanged": bool(max(abs(c1[oc][k] - base[oc][k]) for k in keys) <= TOL and max(abs(c2[oc][k] - base[oc][k]) for k in keys) <= TOL)}
    # (2) prior history changed -> the later forecast changes
    hist_dir = tmp / "histpert"; shutil.copytree(base_dir, hist_dir, copy_function=shutil.copy)
    target = sorted(base["rush_yds"], key=lambda k: -base["rush_yds"][k])[0]
    pid = target[1]
    def hist_change(r):
        if r["player_id"] == pid and r["season"] == str(s) and int(r["week"]) < w:
            for c in ("carries", "rushing_yards"):
                if r.get(c) not in (None, ""):
                    r[c] = str(float(r[c]) * 3 + 20)
    rewrite(hist_dir / f"stats_player_week_{s}.csv", hist_change)
    h = {oc: preds_for(frozen, V2.candidate_rows(str(hist_dir), s, w)[0], oc) for oc in V2.MARKETS}
    changed = abs(h["rush_yds"][target] - base["rush_yds"][target])
    others = [abs(h["rush_yds"][k] - base["rush_yds"][k]) for k in base["rush_yds"] if k[1] != pid]
    pert["prior_history_change"] = {"player_id": pid, "forecast_before": base["rush_yds"][target], "forecast_after": h["rush_yds"][target], "abs_change": float(changed), "changed": bool(changed > 1e-6),
                                    "n_other_players_changed_via_opponent_features": int(sum(o > 1e-9 for o in others))}
    res["perturbation"] = pert
    res["pass"] = all(v[oc]["all_within_tol"] for v in res["weeks"].values() for oc in V2.MARKETS) and all(pert[oc]["unchanged"] for oc in V2.MARKETS) and pert["prior_history_change"]["changed"]
    Path(a.out).write_text(json.dumps(res, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({"pass": res["pass"], "weeks": {k: {oc: (v[oc]["max_abs_diff"], v[oc]["n_compared"], v[oc]["n_honest_rows_missing_from_logger"]) for oc in V2.MARKETS} for k, v in res["weeks"].items()}, "perturbation": pert}, indent=1, default=float))


if __name__ == "__main__":
    main()
