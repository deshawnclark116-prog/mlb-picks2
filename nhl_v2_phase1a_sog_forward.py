#!/usr/bin/env python3
"""NHL V2 Phase1A-SOG forward forecaster, append-only hash-chained ledger and engine-lock verification (research only).

A forecast for (game, horizon) is valid only if it is generated inside [cutoff - WINDOW_S, cutoff], every retrieval it uses completed at or before the cutoff,
the cutoff is at/after the lock's eligible_from_cutoff_utc, and the running code/models match the engine lock. Missed windows are recorded, never backfilled.
No betting-market inputs. No simulation. Coefficients are frozen; completed 2026 games feed rolling history features only."""
import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nhl_v2_phase1a_sog_acquire as A
import nhl_v2_phase1a_sog_compare as C
import nhl_v2_phase1a_sog_data as D
import nhl_v2_phase1a_sog_model as M
from nhl_v2_phase1a_sog_quality import meaningful_mask

REPO = Path(__file__).resolve().parent
OUT = D.OUT
LOCK = OUT / "phase1a_sog_engine_lock.json"
FWD = OUT / "phase1a_sog_forward"
LEDGER = FWD / "ledger.jsonl"
BLOBS = FWD / "blobs"
UTC = timezone.utc
HORIZONS = ("T24H", "T90", "T30")
HORIZON_MIN = {"T24H": 1440, "T90": 90, "T30": 30}
WINDOW_S = 900
LOCK_CODE_FILES = ("nhl_v2_phase1a_sog_acquire.py", "nhl_v2_phase1a_sog_data.py", "nhl_v2_phase1a_sog_model.py", "nhl_v2_phase1a_sog_compare.py", "nhl_v2_phase1a_sog_quality.py",
                   "nhl_v2_phase1a_sog_forward.py", "nhl_v2_phase1a_sog_grade.py")
GENESIS = "0" * 64


class LockError(RuntimeError):
    pass


def iso(t):
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso(s):
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(UTC)


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ append-only hash-chained ledger
class Ledger:
    def __init__(self, path=LEDGER):
        self.path = Path(path)

    def rows(self):
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    @staticmethod
    def row_hash(rec):
        return sha_text(canon({k: v for k, v in rec.items() if k != "row_hash"}))

    def verify(self):
        prev = GENESIS; n = 0
        for r in self.rows():
            if r.get("prev_hash") != prev or r.get("row_hash") != self.row_hash(r):
                raise LockError("ledger chain broken at row %d" % n)
            prev = r["row_hash"]; n += 1
        return n

    def append(self, rec):
        rows = self.rows()
        prev = rows[-1]["row_hash"] if rows else GENESIS
        rec = dict(rec, prev_hash=prev, seq=len(rows))
        rec["row_hash"] = self.row_hash(rec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(canon(rec) + "\n")
        return rec

    def keys(self):
        return {(r["game_id"], r["forecast_horizon"]) for r in self.rows() if r.get("record_type") in ("FORECAST", "MISSED_CUTOFF", "INVALID_LATE")}


# ------------------------------------------------------------------ lock
def verify_lock(lock_path=LOCK, repo=REPO):
    lock = json.loads(Path(lock_path).read_text())
    bad = []
    for f, h in lock["code_sha256"].items():
        if sha_file(repo / f) != h:
            bad.append(f)
    for f, h in lock["model_sha256"].items():
        if sha_file(OUT / "phase1a_sog_models" / f) != h:
            bad.append("model:" + f)
    for f, h in lock["protocol_sha256"].items():
        if sha_file(OUT / f) != h:
            bad.append(f)
    if bad:
        raise LockError("engine lock mismatch: %s" % bad)
    return lock


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


# ------------------------------------------------------------------ planning
def cutoff_of(start_utc, horizon):
    return parse_iso(start_utc) - timedelta(minutes=HORIZON_MIN[horizon])


def plan(now, games, lock, ledger_keys):
    """Pending (game_id, horizon, cutoff) whose window is open now: cutoff - WINDOW_S <= now <= cutoff, cutoff >= lock eligible_from, game not final."""
    eligible_from = parse_iso(lock["eligible_from_cutoff_utc"])
    out = []
    for gid, g in games.items():
        if g.get("state") not in ("FUT", "PRE"):
            continue
        for h in HORIZONS:
            c = cutoff_of(g["game_start_utc"], h)
            if c < eligible_from or (gid, h) in ledger_keys:
                continue
            if c - timedelta(seconds=WINDOW_S) <= now <= c:
                out.append((gid, h, c))
    return sorted(out, key=lambda x: (x[2], x[0], x[1]))


def missed(now, games, lock, ledger_keys):
    eligible_from = parse_iso(lock["eligible_from_cutoff_utc"])
    out = []
    for gid, g in games.items():
        if g.get("state") not in ("FUT", "PRE", "LIVE", "CRIT", "OFF", "FINAL"):
            continue
        for h in HORIZONS:
            c = cutoff_of(g["game_start_utc"], h)
            if eligible_from <= c < now and (gid, h) not in ledger_keys:
                out.append((gid, h, c))
    return sorted(out, key=lambda x: (x[2], x[0], x[1]))


# ------------------------------------------------------------------ live data
def fetch_json(fetcher, url):
    raw, meta = fetcher(url)
    return json.loads(raw), meta, raw


def store_blob(raw, sha, blobs=BLOBS):
    blobs.mkdir(parents=True, exist_ok=True)
    p = blobs / (sha + ".gz")
    if not p.exists():
        bio = gzip.compress(raw, mtime=0)
        p.write_bytes(bio)


def live_season(fetcher, now, season=2026, lead_days=5):
    """All 2026 regular-season games known to the schedule (any state) + player rows of completed games. Returns (games, rows, provenance)."""
    games, provenance, wins = {}, [], []
    for lo, hi in A.window_list(season):
        if parse_iso(lo + "T00:00:00Z") > now + timedelta(days=lead_days):
            break
        wins.append((lo, hi))
    completed_games = {}
    all_rows = []
    for lo, hi in wins:
        sched, smeta, sraw = fetch_json(fetcher, "%s/schedule/%s" % (A.API, lo))
        provenance.append(dict(smeta, kind="schedule", window=[lo, hi]))
        store_blob(sraw, smeta["sha256"])
        window_games = []
        for day in sched["gameWeek"]:
            if lo <= day["date"] <= hi:
                for g in day["games"]:
                    if g["gameType"] == 2:
                        rec = {"game_id": g["id"], "game_start_utc": g["startTimeUTC"], "home_abbrev": g["homeTeam"]["abbrev"], "away_abbrev": g["awayTeam"]["abbrev"],
                               "home_team_id": g["homeTeam"]["id"], "away_team_id": g["awayTeam"]["id"], "date": day["date"], "state": g["gameState"]}
                        games[g["id"]] = rec; window_games.append(rec)
        if parse_iso(lo + "T00:00:00Z") > now:
            continue
        w = {"window": [lo, hi], "games": [dict(g) for g in window_games if g["state"] in ("OFF", "FINAL")], "provenance": {}}
        for name, path in (("summary", "/skater/summary"), ("timeonice", "/skater/timeonice")):
            d, meta, raw = fetch_json(fetcher, A.stats_url(path, lo, hi))
            w[name] = d["data"]
            provenance.append(dict(meta, kind=name, window=[lo, hi], reported_total=d["total"], returned_rows=len(d["data"]), guard_ok=bool(d["total"] < 10000 and len(d["data"]) == d["total"])))
            store_blob(raw, meta["sha256"])
        cg, rr = A.assemble([w])
        completed_games.update(cg); all_rows += rr
    return games, all_rows, provenance


def extract_availability(landing, rail, boxscore):
    """Pregame roster observation, NEVER certified: {player_id: state}. rosterSpots / scratches are read if present."""
    spots, scratches, names = {}, set(), {}
    for payload in (landing, boxscore):
        for k in ("rosterSpots",):
            for s in (payload or {}).get(k, []) or []:
                pid = s.get("playerId")
                if pid:
                    spots[pid] = s.get("teamId"); nm = s.get("firstName", {}).get("default", "") + " " + s.get("lastName", {}).get("default", "")
                    names[pid] = nm.strip()
    gi = (rail or {}).get("gameInfo", {}) or {}
    for side in ("awayTeam", "homeTeam"):
        for s in (gi.get(side, {}) or {}).get("scratches", []) or []:
            pid = s.get("id") or s.get("playerId")
            if pid:
                scratches.add(pid)
                nm = (s.get("firstName", {}) or {}).get("default", "") + " " + (s.get("lastName", {}) or {}).get("default", "")
                if nm.strip():
                    names[pid] = nm.strip()
    return spots, scratches, names


def roster_names(fetcher, abbrev, provenance):
    try:
        d, meta, raw = fetch_json(fetcher, "%s/roster/%s/current" % (A.API, abbrev))
    except Exception:
        return {}
    provenance.append(dict(meta, kind="roster", team=abbrev)); store_blob(raw, meta["sha256"])
    out = {}
    for k in ("forwards", "defensemen"):
        for p in d.get(k, []):
            out[p["id"]] = (p.get("firstName", {}).get("default", "") + " " + p.get("lastName", {}).get("default", "")).strip()
    return out


# ------------------------------------------------------------------ the forecast run
def run(now_fn=lambda: datetime.now(UTC), fetcher=A.fetch, dry_run=False, ledger=None, frozen=None, log=print):
    ledger = ledger or Ledger()
    lock = verify_lock()
    ledger.verify()
    t0 = now_fn()
    games_live, rows_live, prov = live_season(fetcher, t0)
    keys = ledger.keys()
    pend = plan(t0, games_live, lock, keys)
    miss = missed(t0, games_live, lock, keys)
    log("pending", len(pend), "missed", len(miss))
    run_id = "run-%s" % iso(t0)
    sha = git_sha()
    wrote = 0
    for gid, h, c in miss:
        g = games_live[gid]
        if not dry_run:
            ledger.append({"record_type": "MISSED_CUTOFF", "run_id": run_id, "engine_version": M.ENGINE_VERSION, "git_sha": sha, "game_id": gid, "forecast_horizon": h, "scheduled_start": g["game_start_utc"],
                           "cutoff_at": iso(c), "observed_at": iso(t0), "status": "MISSED_CUTOFF", "reason": "no valid forecast was generated inside the window; never backfilled"})
        wrote += 1
    if not pend:
        return {"forecasts": 0, "missed": len(miss)}
    games0, rows0 = frozen if frozen else D.load_frozen()
    games = dict(games0)
    for gid, g in games_live.items():
        games[gid] = {k: v for k, v in g.items() if k != "state"}
    rows = list(rows0) + rows_live
    const = lock["comparator_constants"]
    names_cache = {}
    fc = 0
    for h in sorted({p[1] for p in pend}):
        art = json.loads((OUT / "phase1a_sog_models" / ("engine_%s.json" % h)).read_text())
        targets = [p for p in pend if p[1] == h]
        tgids = sorted({p[0] for p in targets})
        tab, _ = D.build_rows(games, rows, target_gids=tgids, horizon_min=HORIZON_MIN[h], labels=False)
        if not len(tab["player_id"]):
            continue
        mu = M.predict_mu(art, tab); alpha = art["nb2"]["alpha"]
        summ = M.distribution_summary(mu, alpha)
        cf = C.comparator_features(games, rows, tab, HORIZON_MIN[h])
        mu_h = C.human_mu(cf, {k: {kk: float(vv) for kk, vv in v.items()} for k, v in const.items()}); el = C.eligible(cf)
        mmask = meaningful_mask(tab)
        raw_hashes = {"history": [p["sha256"] for p in prov if p["kind"] in ("summary", "timeonice", "schedule")]}
        for gid in tgids:
            g = games[gid]; c = cutoff_of(g["game_start_utc"], h)
            avail = {}
            payload_meta = []
            for ep in ("landing", "right-rail", "boxscore"):
                try:
                    d, meta, raw = fetch_json(fetcher, "%s/gamecenter/%d/%s" % (A.API, gid, ep))
                    avail[ep] = d; payload_meta.append({"endpoint": ep, "sha256": meta["sha256"], "retrieval_started_utc": meta["retrieval_started_utc"], "retrieval_completed_utc": meta["retrieval_completed_utc"]})
                    store_blob(raw, meta["sha256"])
                except Exception as e:                                          # noqa
                    avail[ep] = None; payload_meta.append({"endpoint": ep, "error": str(e)[:120]})
            spots, scratches, nm_av = extract_availability(avail.get("landing"), avail.get("right-rail"), avail.get("boxscore"))
            for ab in (g["home_abbrev"], g["away_abbrev"]):
                if ab not in names_cache:
                    names_cache[ab] = roster_names(fetcher, ab, prov)
            t_end = now_fn()
            late = t_end > c or any(parse_iso(pm["retrieval_completed_utc"]) > c for pm in payload_meta if "retrieval_completed_utc" in pm) or any(parse_iso(p["retrieval_completed_utc"]) > c for p in prov)
            idx = np.where(tab["game_id"] == gid)[0]
            if late:
                if not dry_run:
                    ledger.append({"record_type": "INVALID_LATE", "run_id": run_id, "engine_version": M.ENGINE_VERSION, "git_sha": sha, "game_id": gid, "forecast_horizon": h, "scheduled_start": g["game_start_utc"], "cutoff_at": iso(c),
                                   "observed_at": iso(t_end), "status": "CAPTURED_LATE_AUDIT_ONLY", "payloads": payload_meta, "reason": "retrieval or generation completed after the cutoff: never used as pregame information"})
                continue
            for i in idx:
                pid = int(tab["player_id"][i]); team = int(tab["team_id"][i])
                a_abbrev = g["home_abbrev"] if team == g["home_team_id"] else g["away_abbrev"]; o_abbrev = g["away_abbrev"] if team == g["home_team_id"] else g["home_abbrev"]
                if not avail.get("landing") and not avail.get("boxscore"):
                    state = "NOT_CAPTURED"
                elif pid in scratches:
                    state = "SCRATCH_LISTED_OBSERVED_NOT_CERTIFIED"
                elif spots:
                    state = "IN_ROSTER_SPOTS_OBSERVED_NOT_CERTIFIED" if pid in spots else "ABSENT_FROM_ROSTER_SPOTS_OBSERVED_NOT_CERTIFIED"
                else:
                    state = "NO_ROSTER_DATA_PUBLISHED_YET"
                feat = {f: (None if np.isnan(tab[f][i]) else float(tab[f][i])) for f in D.FEATURES}
                snap = {"features": feat, "hist10": [int(x) for x in tab["hist10"][i]], "plays10": int(tab["plays10"][i]), "den10": int(tab["den10"][i])}
                rec = {"record_type": "FORECAST", "forecast_id": "%d-%s-%d" % (gid, h, pid), "run_id": run_id, "engine_version": M.ENGINE_VERSION, "git_sha": sha, "lock_sha256": sha_file(LOCK),
                       "game_id": gid, "schedule_date": g.get("date"), "player_id": pid, "team": a_abbrev, "opponent": o_abbrev, "scheduled_start": g["game_start_utc"], "forecast_horizon": h, "cutoff_at": iso(c), "generated_at": iso(t_end),
                       "candidate_status": "CANDIDATE_MEANINGFUL" if bool(mmask[i]) else "CANDIDATE_NOT_MEANINGFUL", "meaningful_expected_participant": bool(mmask[i]),
                       "availability_state": state, "availability_confidence": "NOT_CERTIFIED", "availability_used_in_forecast": False,
                       "feature_snapshot_sha256": sha_text(canon(snap)), "raw_source_hashes": {"history": raw_hashes["history"], "game_payloads": payload_meta},
                       "expected_sog": float(summ["mean"][i]), "median_sog": float(summ["median"][i]), "variance": float(summ["variance"][i]), "dispersion": float(alpha),
                       "P1": float(summ["p_ge"][i, 0]), "P2": float(summ["p_ge"][i, 1]), "P3": float(summ["p_ge"][i, 2]), "P4": float(summ["p_ge"][i, 3]), "P5": float(summ["p_ge"][i, 4]),
                       "comparators": {"simple_prior10_mean": None if np.isnan(cf["m10"][i]) else float(cf["m10"][i]), "human_frozen_mean": None if np.isnan(mu_h[i]) else float(mu_h[i]), "comparator_eligible": bool(el[i])},
                       "receipt": {"player_name": names_cache.get(a_abbrev, {}).get(pid) or nm_av.get(pid) or None, "position": "F" if tab["POS_F"][i] == 1 else "D" if tab["POS_D"][i] == 1 else "U",
                                   "recent_participation": {"plays_last10_team_games": int(tab["plays10"][i]), "of": int(tab["den10"][i]), "team_games_since_appearance": feat["TEAM_GAMES_SINCE_APPEARANCE"]},
                                   "recent_sog_history_last10_appearances": snap["hist10"], "sog_per60_app10": feat["SOG_PER60_APP10"],
                                   "toi_seconds": {"recent3": feat["TOI_MEAN_CT_APP3"], "recent10": feat["TOI_MEAN_CT_APP10"], "trend_3_minus_10": feat["TOI_DELTA_CT_3_10"]},
                                   "pp_toi_recent3_seconds": feat["PP_TOI_MEAN_CT_APP3"], "pp_allocation_share_recent3_prior_games": feat["PP_ALLOC_SHARE_MEAN_CT_APP3"], "shifts_recent3": feat["SHIFT_MEAN_CT_APP3"],
                                   "team_sog_for_mean5": feat["TEAM_SOG_FOR_MEAN5"], "opp_sog_allowed_mean5": feat["OPP_SOG_ALLOWED_MEAN5"], "rest_hours": feat["TEAM_REST_HOURS"], "back_to_back": bool(feat["BACK_TO_BACK"]),
                                   "home": bool(feat["IS_HOME"]), "model_log_mean": float(np.log(mu[i])), "nb_dispersion": float(alpha), "source_hashes_ref": "raw_source_hashes",
                                   "target_game_lineup_or_pp_state_used": False}}
                if not dry_run:
                    ledger.append(rec)
                fc += 1
    return {"forecasts": fc, "missed": len(miss), "run_id": run_id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "verify", "dry-run"])
    a = ap.parse_args()
    if a.cmd == "verify":
        verify_lock(); print("lock ok; ledger rows", Ledger().verify()); return
    print(run(dry_run=(a.cmd == "dry-run")))


if __name__ == "__main__":
    main()
