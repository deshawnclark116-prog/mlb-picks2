#!/usr/bin/env python3
"""NHL V2 Phase1A-SOG forward grader (research only). Scores a FORECAST ledger row only after the game is official-final.

Official SOG = stats-REST skater summary 'shots' (regular season; shootout shots are never credited). A forecast player absent from the official rows has played=0 and SOG=0
(the B2 target is unconditional on playing). RAW ledger = all graded forecasts; CLEAN ledger = RAW minus documented abnormal exits listed in censor_registry.jsonl
(in-game injury with abnormal lost ice time, ejection/misconduct) - never poor shooting, ordinary benching, demotion, score effects or coaching decisions."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nhl_v2_phase1a_sog_acquire as A
import nhl_v2_phase1a_sog_forward as F
import nhl_v2_phase1a_sog_model as M

UTC = timezone.utc
GRADES = F.FWD / "graded.jsonl"
CENSOR = F.FWD / "censor_registry.jsonl"
ALLOWED_CENSOR_REASONS = ("IN_GAME_INJURY_ABNORMAL_LOST_ICE_TIME", "EJECTION_OR_MISCONDUCT_ABNORMAL_EXIT")


def load_censors(path=CENSOR):
    out = {}
    if Path(path).exists():
        for l in Path(path).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("reason") not in ALLOWED_CENSOR_REASONS or not r.get("evidence"):
                    raise ValueError("censor entry needs an allowed reason and documented evidence: %s" % r)
                out[r["forecast_id"]] = r
    return out


def official_game(fetcher, game_id, date):
    """Returns (state, {player_id: {sog, toi, pp_toi, shifts, team_id}}, meta) from schedule + stats-REST for that date."""
    sraw, smeta = fetcher("%s/schedule/%s" % (A.API, date))
    sched = json.loads(sraw)
    st = None; game = None
    for day in sched["gameWeek"]:
        for g in day["games"]:
            if g["id"] == game_id:
                st = g["gameState"]; game = g
    info = {"official_start_utc": game["startTimeUTC"] if game else None, "schedule_state": (game or {}).get("gameScheduleState", "OK")}
    if info["schedule_state"] != "OK":
        return st, None, {"schedule": smeta, **info}
    if st not in ("OFF", "FINAL"):
        return st, None, {"schedule": smeta, **info}
    w = A.acquire_window(date, date, fetcher=fetcher)
    games, rows = A.assemble([w])
    if game_id not in games:
        return st, None, {"schedule": smeta, **info}
    rr = {r["player_id"]: {"sog": r["sog"], "toi": r["toi_sec"], "pp_toi": r["pp_toi_sec"], "shifts": r["shifts"], "team_id": r["team_id"]} for r in rows if r["game_id"] == game_id}
    team_sog = {}
    for p in rr.values():
        team_sog[p["team_id"]] = team_sog.get(p["team_id"], 0) + p["sog"]
    return st, {"players": rr, "team_sog": team_sog}, {"schedule": smeta, "summary": w["provenance"]["summary"], "timeonice": w["provenance"]["timeonice"], **info}


def grade(fetcher=A.fetch, now_fn=lambda: datetime.now(UTC), ledger=None, grades_path=GRADES, log=print):
    ledger = ledger or F.Ledger()
    ledger.verify()
    done = set()
    gl = F.Ledger(grades_path)
    for r in gl.rows():
        done.add(r["forecast_id"])
    games = {}
    for r in ledger.rows():
        if r.get("record_type") == "FORECAST" and r["forecast_id"] not in done:
            games.setdefault(r["game_id"], []).append(r)
    n = 0
    for gid, fr in sorted(games.items()):
        date = fr[0]["scheduled_start"][:10]
        # the schedule day key can be the previous local date for late games: try the UTC date and the day before
        res = None
        for d in (date, (datetime.fromisoformat(date) - __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")):
            st, data, meta = official_game(fetcher, gid, d)
            if meta.get("schedule_state", "OK") != "OK" or data is not None:
                res = (st, data, meta); break
        if res is None:
            log("not final yet", gid); continue
        st, data, meta = res
        for r in fr:
            if meta.get("schedule_state", "OK") != "OK" or meta.get("official_start_utc") != r["scheduled_start"]:
                reason = "POSTPONED_OR_CANCELLED" if meta.get("schedule_state", "OK") != "OK" else "SCHEDULE_START_CHANGED_NO_LINKAGE"
                gl.append({"record_type": "UNGRADED", "forecast_id": r["forecast_id"], "forecast_row_hash": r["row_hash"], "game_id": gid, "player_id": r["player_id"], "forecast_horizon": r["forecast_horizon"],
                           "graded_at": F.iso(now_fn()), "reason": reason, "forecast_scheduled_start": r["scheduled_start"], "official_start_utc": meta.get("official_start_utc"), "schedule_state": meta.get("schedule_state", "OK")})
                n += 1
                continue
            p = data["players"].get(r["player_id"])
            played = p is not None
            actual = int(p["sog"]) if played else 0
            team_id = p["team_id"] if played else None
            rec = {"record_type": "GRADE", "forecast_id": r["forecast_id"], "forecast_row_hash": r["row_hash"], "game_id": gid, "player_id": r["player_id"], "forecast_horizon": r["forecast_horizon"],
                   "graded_at": F.iso(now_fn()), "played": played, "actual_sog": actual,
                   "diagnostics_postgame_only": {"actual_toi_seconds": p["toi"] if played else None, "actual_pp_toi_seconds": p["pp_toi"] if played else None, "actual_shifts": p["shifts"] if played else None,
                                                 "actual_team_sog": data["team_sog"].get(team_id) if team_id is not None else None},
                   "official_source_hashes": {k: v["sha256"] for k, v in meta.items() if isinstance(v, dict) and "sha256" in v}, "game_state": st}
            gl.append(rec); n += 1
    return n


def classify_miss(fc, gr):
    """Large-miss forensics (diagnostics only; fixed ratios from the protocol). Returns a class or None when the miss is not large."""
    err = abs(gr["actual_sog"] - fc["expected_sog"])
    if err <= 3:
        return None
    rc, d = fc["receipt"], gr["diagnostics_postgame_only"]
    flags = []
    if not gr["played"] or (d["actual_toi_seconds"] is not None and rc["toi_seconds"]["recent3"] and d["actual_toi_seconds"] < 0.5 * rc["toi_seconds"]["recent3"]):
        flags.append("AVAILABILITY")
    elif d["actual_toi_seconds"] and rc["toi_seconds"]["recent3"]:
        r = d["actual_toi_seconds"] / rc["toi_seconds"]["recent3"]
        if r >= 1.25 or r <= 0.75:
            flags.append("TOI_OPPORTUNITY")
    if gr["played"] and d["actual_toi_seconds"] and rc["sog_per60_app10"]:
        rate = 3600.0 * gr["actual_sog"] / d["actual_toi_seconds"]
        q = rate / rc["sog_per60_app10"] if rc["sog_per60_app10"] > 0 else None
        if q is not None and (q >= 1.5 or q <= 0.5):
            flags.append("SHOT_RATE")
    if d["actual_team_sog"] is not None and rc["team_sog_for_mean5"]:
        q = d["actual_team_sog"] / rc["team_sog_for_mean5"]
        if q >= 1.25 or q <= 0.75:
            flags.append("TEAM_ENVIRONMENT")
    if gr["played"] and d["actual_pp_toi_seconds"] is not None and rc["pp_toi_recent3_seconds"] is not None and d["actual_pp_toi_seconds"] >= 120 and d["actual_pp_toi_seconds"] >= 2 * max(rc["pp_toi_recent3_seconds"], 1):
        flags.append("PP_DEPLOYMENT")
    return "UNRESOLVED" if not flags else flags[0] if len(flags) == 1 else "MULTIPLE"


def join(ledger=None, grades_path=GRADES, censor_path=CENSOR):
    ledger = ledger or F.Ledger()
    fc = {r["forecast_id"]: r for r in ledger.rows() if r.get("record_type") == "FORECAST"}
    cens = load_censors(censor_path)
    out = []
    for g in F.Ledger(grades_path).rows():
        if g.get("record_type") != "GRADE":
            continue
        f = fc.get(g["forecast_id"])
        if f is None or f["row_hash"] != g["forecast_row_hash"]:
            raise ValueError("grade does not match an immutable forecast row: %s" % g["forecast_id"])
        out.append((f, g, g["forecast_id"] in cens))
    return out


def _base_rows(sel, lock, population):
    rates = lock["base_rates"][population]
    return np.array([rates[f["receipt"]["position"]] for f, g in sel], float)


def _poisson_summary(mean, y, gid, pid, base, name):
    mf = np.maximum(np.asarray(mean, float), 0.05)
    return M.summarize("poisson", {"mu": mf}, y, name, gid, pid, np.asarray(mean, float), base)


def evaluate(pairs, lock, horizon, clean=False, population="FULL"):
    """Forward metrics for one horizon/population on graded pairs (RAW unless clean=True). Comparator forecasts were stored at forecast time."""
    sel = [(f, g) for f, g, c in pairs if f["forecast_horizon"] == horizon and not (clean and c) and (population == "FULL" or f["meaningful_expected_participant"])]
    if not sel:
        return None
    y = np.array([g["actual_sog"] for f, g in sel]); mu = np.array([f["expected_sog"] for f, g in sel]); alpha = sel[0][0]["dispersion"]
    gid = np.array([f["game_id"] for f, g in sel]); pid = np.array([f["player_id"] for f, g in sel])
    base = _base_rows(sel, lock, population)
    s, rows = M.summarize("nb2", {"mu": mu, "alpha": alpha}, y, "B2", gid, pid, mu, base)
    out = {"horizon": horizon, "population": population, "ledger": "CLEAN" if clean else "RAW", "V2_B2": s}
    ok = np.array([bool(f["comparators"]["comparator_eligible"]) and f["comparators"]["human_frozen_mean"] is not None and f["comparators"]["simple_prior10_mean"] is not None for f, g in sel])
    if ok.any():
        idx = np.where(ok)[0]
        sub = [sel[i] for i in idx]
        s2, r2 = M.summarize("nb2", {"mu": mu[idx], "alpha": alpha}, y[idx], "B2", gid[idx], pid[idx], mu[idx], base[idx])
        comp = {"identical_rows": int(ok.sum()), "V2_B2": s2}
        starts = {}
        for f, g in sub:
            starts[f["game_id"]] = int(datetime.fromisoformat(f["scheduled_start"].replace("Z", "+00:00")).timestamp())
        ug = np.unique(gid[idx]); wk = M.D.week_index(np.array([starts[int(x)] for x in ug]))
        for nm, key in (("human_frozen", "human_frozen_mean"), ("simple_prior10_mean", "simple_prior10_mean")):
            m = np.array([f["comparators"][key] for f, g in sub], float)
            sc, rc = _poisson_summary(m, y[idx], gid[idx], pid[idx], base[idx], nm)
            comp[nm] = sc
            delta = r2["crps_pg"] - rc["crps_pg"]
            comp["paired_crps_V2_B2_minus_" + nm] = dict(M.blocked_bootstrap(delta, wk), relative_crps_change=float(delta.mean() / sc["crps_macro_game"]))
        out["comparators_identical_rows"] = comp
    return out


def production_join(pairs, picks_log_text, seconds_before_cutoff=0):
    """Production SOG>=3 probability (prob_over of shots_on_goal* picks) for forecast rows, accepted only if logged strictly before the cutoff."""
    log = {}
    for l in picks_log_text.splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if str(r.get("market", "")).startswith("shots_on_goal"):
            log[(r["player_id"], r["game_date"])] = (r["prob_over"], r["logged_at"])
    out = {}
    for f, g, c in pairs:
        k = (f["player_id"], f.get("schedule_date"))
        if k in log:
            p, at = log[k]
            if datetime.fromisoformat(at.replace("Z", "+00:00")) < datetime.fromisoformat(f["cutoff_at"].replace("Z", "+00:00")):
                out[f["forecast_id"]] = p
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grade"])
    ap.parse_args()
    print("graded", grade())
