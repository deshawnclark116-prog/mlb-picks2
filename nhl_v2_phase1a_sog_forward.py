#!/usr/bin/env python3
"""NHL V2 Phase1A-SOG forward forecaster (engine v1.1: decision key includes the scheduled start; per-run source manifests; hard source-completeness gate), append-only hash-chained ledger and engine-lock verification (research only).

A forecast for (game, horizon, scheduled_start) is valid only if it is generated inside [cutoff - WINDOW_S, cutoff], every retrieval it uses completed at or before the cutoff,
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
LOCK = OUT / "phase1a_sog_engine_lock_v1_1.json"
LOCK_V1_0 = OUT / "phase1a_sog_engine_lock.json"
ENGINE_VERSION = "nhl-v2-sog-b2-1.1"
FWD = OUT / "phase1a_sog_forward"
LEDGER = FWD / "ledger.jsonl"
BLOBS = FWD / "blobs"
MANIFESTS = FWD / "manifests"
UTC = timezone.utc
HORIZONS = ("T24H", "T90", "T30")
HORIZON_MIN = {"T24H": 1440, "T90": 90, "T30": 30}
WINDOW_S = 900
PREDICTIVE_KINDS = ("schedule", "summary", "timeonice")
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

    KEY_TYPES = ("FORECAST", "MISSED_CUTOFF", "MISSED_REVISED_CUTOFF", "INVALID_LATE", "SOURCE_INCOMPLETE_NO_FORECAST", "SOURCE_FETCH_FAILED_NO_FORECAST")

    def keys(self):
        """Decision keys that are already consumed: (game_id, forecast_horizon, scheduled_start_utc)."""
        return {(r["game_id"], r["forecast_horizon"], r["scheduled_start"]) for r in self.rows() if r.get("record_type") in self.KEY_TYPES}

    def last_known(self):
        """game_id -> (scheduled_start, schedule_state) from the most recent row that carries a start."""
        out = {}
        for r in self.rows():
            if "scheduled_start" in r and r.get("record_type") in self.KEY_TYPES + ("DECISION_STATE",):
                out[r["game_id"]] = (r["scheduled_start"], r.get("schedule_state", "OK"))
        return out

    def starts_for(self, game_id, horizon):
        return {r["scheduled_start"] for r in self.rows() if r.get("record_type") in self.KEY_TYPES and r["game_id"] == game_id and r["forecast_horizon"] == horizon}


def compact_start(start):
    return start.replace("-", "").replace(":", "").replace("Z", "")


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
    """Pending (game_id, horizon, cutoff) whose window is open now: cutoff - WINDOW_S <= now <= cutoff, cutoff >= lock eligible_from, game not started,
    schedule state OK, and the decision key (game_id, horizon, scheduled_start) not yet consumed."""
    eligible_from = parse_iso(lock["eligible_from_cutoff_utc"])
    out = []
    for gid, g in games.items():
        if g.get("state") not in ("FUT", "PRE") or g.get("schedule_state", "OK") != "OK":
            continue
        for h in HORIZONS:
            c = cutoff_of(g["game_start_utc"], h)
            if c < eligible_from or (gid, h, g["game_start_utc"]) in ledger_keys:
                continue
            if c - timedelta(seconds=WINDOW_S) <= now <= c:
                out.append((gid, h, c))
    return sorted(out, key=lambda x: (x[2], x[0], x[1]))


def missed(now, games, lock, ledger_keys, ledger=None):
    """(game_id, horizon, cutoff, status): the current-start window passed without a consumed key. MISSED_REVISED_CUTOFF when the ledger already holds a row for the
    same game and horizon under a different start, else MISSED_CUTOFF. Postponed/cancelled games are not missed (no decision exists)."""
    eligible_from = parse_iso(lock["eligible_from_cutoff_utc"])
    out = []
    for gid, g in games.items():
        if g.get("schedule_state", "OK") != "OK":
            continue
        for h in HORIZONS:
            c = cutoff_of(g["game_start_utc"], h)
            if eligible_from <= c < now and (gid, h, g["game_start_utc"]) not in ledger_keys:
                other = bool(ledger) and any(st != g["game_start_utc"] for st in ledger.starts_for(gid, h))
                out.append((gid, h, c, "MISSED_REVISED_CUTOFF" if other else "MISSED_CUTOFF"))
    return sorted(out, key=lambda x: (x[2], x[0], x[1]))


def schedule_state_changes(games, ledger):
    """DECISION_STATE records for games the ledger already knows whose start was revised or that became postponed/cancelled."""
    known = ledger.last_known()
    out = []
    for gid, (start, state) in sorted(known.items()):
        g = games.get(gid)
        if g is None:
            continue
        cur_state = g.get("schedule_state", "OK")
        if g["game_start_utc"] != start:
            out.append({"game_id": gid, "state": "SCHEDULE_REVISED", "previous_scheduled_start": start, "scheduled_start": g["game_start_utc"], "schedule_state": cur_state})
        elif cur_state != "OK" and state == "OK":
            out.append({"game_id": gid, "state": "POSTPONED_OR_CANCELLED_OBSERVED", "previous_scheduled_start": start, "scheduled_start": g["game_start_utc"], "schedule_state": cur_state})
    return out


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


def source_entry(kind, meta, window=None, **extra):
    e = {"kind": kind, "predictive": kind in PREDICTIVE_KINDS, "url": meta.get("url"), "retrieval_started_utc": meta.get("retrieval_started_utc"), "retrieval_completed_utc": meta.get("retrieval_completed_utc"),
         "sha256": meta.get("sha256"), "bytes": meta.get("bytes"), "http_status": meta.get("http_status"), "completeness_status": "COMPLETE"}
    if window is not None:
        e["window"] = list(window)
    e.update(extra)
    return e


def failed_entry(kind, url, window, status, error, **extra):
    e = {"kind": kind, "predictive": kind in PREDICTIVE_KINDS, "url": url, "window": list(window), "completeness_status": status, "error": str(error)[:200], "sha256": None}
    e.update(extra)
    return e


def live_season(fetcher, now, season=2026, lead_days=5):
    """All 2026 regular-season games known to the schedule (any state) + player rows of completed games.
    Returns (games, rows, sources, failures). Every predictive source is recorded with full retrieval provenance; failures are returned, never swallowed."""
    games, sources, failures, wins = {}, [], [], []
    for lo, hi in A.window_list(season):
        if parse_iso(lo + "T00:00:00Z") > now + timedelta(days=lead_days):
            break
        wins.append((lo, hi))
    all_rows = []
    completed_ids = set()
    for lo, hi in wins:
        url = "%s/schedule/%s" % (A.API, lo)
        try:
            sched, smeta, sraw = fetch_json(fetcher, url)
        except Exception as e:                                                  # noqa
            failures.append(failed_entry("schedule", url, (lo, hi), "FETCH_FAILED", e)); continue
        store_blob(sraw, smeta["sha256"])
        try:
            window_games = []
            for day in sched["gameWeek"]:
                if lo <= day["date"] <= hi:
                    for g in day["games"]:
                        if g["gameType"] == 2:
                            rec = {"game_id": g["id"], "game_start_utc": g["startTimeUTC"], "home_abbrev": g["homeTeam"]["abbrev"], "away_abbrev": g["awayTeam"]["abbrev"],
                                   "home_team_id": g["homeTeam"]["id"], "away_team_id": g["awayTeam"]["id"], "date": day["date"], "state": g["gameState"], "schedule_state": g.get("gameScheduleState", "OK")}
                            games[g["id"]] = rec; window_games.append(rec)
            sources.append(source_entry("schedule", smeta, (lo, hi), n_games=len(window_games)))
        except Exception as e:                                                  # noqa
            failures.append(failed_entry("schedule", url, (lo, hi), "INCOMPLETE", "malformed schedule payload: %r" % e, sha256=smeta["sha256"])); continue
        if parse_iso(lo + "T00:00:00Z") > now:
            continue
        w = {"window": [lo, hi], "games": [dict(g) for g in window_games if g["state"] in ("OFF", "FINAL")], "provenance": {}}
        ok = True
        for name, path in (("summary", "/skater/summary"), ("timeonice", "/skater/timeonice")):
            u = A.stats_url(path, lo, hi)
            try:
                d, meta, raw = fetch_json(fetcher, u)
            except Exception as e:                                              # noqa
                failures.append(failed_entry(name, u, (lo, hi), "FETCH_FAILED", e)); ok = False; continue
            store_blob(raw, meta["sha256"])
            guard = bool(d["total"] < 10000 and len(d["data"]) == d["total"] and meta.get("http_status") == 200)
            ent = source_entry(name, meta, (lo, hi), reported_total=d["total"], returned_rows=len(d["data"]))
            if not guard:
                ent["completeness_status"] = "INCOMPLETE"
                failures.append(dict(ent, error="guard failed: reported_total %s, returned_rows %s, http %s" % (d["total"], len(d["data"]), meta.get("http_status"))))
                ok = False
            sources.append(ent)
            w[name] = d["data"]
        if not ok:
            continue
        try:
            cg, rr = A.assemble([w])
        except Exception as e:                                                  # noqa
            failures.append(failed_entry("assembly", "", (lo, hi), "INCOMPLETE", "assembly invariant failed: %r" % e)); continue
        by_game = {}
        for r in rr:
            by_game.setdefault(r["game_id"], set()).add(r["team_id"])
        for g in window_games:
            if g["state"] in ("OFF", "FINAL") and parse_iso(g["game_start_utc"]) <= now - timedelta(minutes=D.GAME_MAX_MINUTES):
                if by_game.get(g["game_id"], set()) != {g["home_team_id"], g["away_team_id"]}:
                    failures.append(failed_entry("game_rows", "", (lo, hi), "INCOMPLETE", "completed game %s has no player rows for both teams" % g["game_id"], game_id=g["game_id"]))
        all_rows += rr
    return games, all_rows, sources, failures


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
def build_manifest(run_id, generated_at, cutoffs, sources):
    m = {"run_id": run_id, "generated_at": iso(generated_at), "decision_cutoffs": sorted(cutoffs), "sources": sources, "engine_version": ENGINE_VERSION}
    m["manifest_sha256"] = sha_text(canon({k: v for k, v in m.items() if k != "manifest_sha256"}))
    return m


def store_manifest(m, manifests=None):
    d = Path(manifests or MANIFESTS)
    d.mkdir(parents=True, exist_ok=True)
    p = d / (m["manifest_sha256"] + ".json")
    if not p.exists():
        p.write_text(canon(m) + "\n", encoding="utf-8")
    return m["manifest_sha256"]


def audit_ledger(ledger=None, manifests=None, blobs=None):
    """Proof from committed artifacts: for every FORECAST row, every predictive source of its manifest is COMPLETE, its blob exists with a matching sha256, and it was
    retrieved no later than the row's cutoff. Returns a list of violations (empty = proven)."""
    ledger = ledger or Ledger()
    md = Path(manifests or MANIFESTS); bd = Path(blobs or BLOBS)
    bad = []
    for r in ledger.rows():
        if r.get("record_type") != "FORECAST":
            continue
        mp = md / (r.get("source_manifest_sha256", "missing") + ".json")
        if not mp.exists():
            bad.append((r["forecast_id"], "manifest missing")); continue
        m = json.loads(mp.read_text())
        if sha_text(canon({k: v for k, v in m.items() if k != "manifest_sha256"})) != m["manifest_sha256"] or m["manifest_sha256"] != r["source_manifest_sha256"]:
            bad.append((r["forecast_id"], "manifest hash mismatch")); continue
        c = parse_iso(r["cutoff_at"])
        n_pred = 0
        for s_ in m["sources"]:
            if not s_.get("predictive"):
                continue
            n_pred += 1
            if s_.get("completeness_status") != "COMPLETE" or not s_.get("sha256") or s_.get("http_status") != 200:
                bad.append((r["forecast_id"], "predictive source not COMPLETE: %s" % s_.get("kind"))); continue
            if parse_iso(s_["retrieval_completed_utc"]) > c:
                bad.append((r["forecast_id"], "predictive source retrieved after cutoff: %s" % s_.get("kind")))
            bp = bd / (s_["sha256"] + ".gz")
            if not bp.exists() or hashlib.sha256(gzip.decompress(bp.read_bytes())).hexdigest() != s_["sha256"]:
                bad.append((r["forecast_id"], "blob missing or hash mismatch: %s" % s_["sha256"][:12]))
        if n_pred == 0:
            bad.append((r["forecast_id"], "no predictive sources in manifest"))
    return bad


def run(now_fn=lambda: datetime.now(UTC), fetcher=A.fetch, dry_run=False, ledger=None, frozen=None, log=print):
    ledger = ledger or Ledger()
    lock = verify_lock()
    ledger.verify()
    t0 = now_fn()
    games_live, rows_live, sources, failures = live_season(fetcher, t0)
    keys = ledger.keys()
    pend = plan(t0, games_live, lock, keys)
    miss = missed(t0, games_live, lock, keys, ledger)
    changes = schedule_state_changes(games_live, ledger)
    log("pending", len(pend), "missed", len(miss), "state-changes", len(changes), "source-failures", len(failures))
    run_id = "run-%s" % iso(t0)
    sha = git_sha()
    if not (pend or miss or changes):
        return {"forecasts": 0, "missed": 0, "state_changes": 0}
    common = {"run_id": run_id, "engine_version": ENGINE_VERSION, "git_sha": sha, "lock_sha256": sha_file(LOCK)}

    def write(rec):
        if not dry_run:
            ledger.append(rec)

    for ch in changes:
        write(dict(common, record_type="DECISION_STATE", observed_at=iso(t0), **ch))
    # ---- phase 1: fetch target-game availability payloads (non-predictive) only when a normal forecast is possible
    avail, game_meta, names_cache = {}, {}, {}
    if pend and not failures:
        for gid in sorted({p[0] for p in pend}):
            g = games_live[gid]; av = {}; metas = []
            for ep in ("landing", "right-rail", "boxscore"):
                url = "%s/gamecenter/%d/%s" % (A.API, gid, ep)
                try:
                    d, meta, raw = fetch_json(fetcher, url)
                    av[ep] = d; store_blob(raw, meta["sha256"])
                    e = source_entry("game_" + ep.replace("-", "_"), meta, game_id=gid); metas.append(e)
                except Exception as ex:                                         # noqa
                    av[ep] = None; metas.append(failed_entry("game_" + ep.replace("-", "_"), url, (g["game_start_utc"],) * 2, "FETCH_FAILED", ex, game_id=gid))
            avail[gid] = (av, metas); sources.extend(metas)
            for ab in (g["home_abbrev"], g["away_abbrev"]):
                if ab not in names_cache:
                    prov = []
                    names_cache[ab] = roster_names(fetcher, ab, prov)
                    for pm in prov:
                        sources.append(source_entry("roster", pm, team=ab))
    all_sources = sources + failures
    cutoffs = sorted({iso(c) for _, _, c in pend} | {iso(c) for _, _, c, _ in miss})
    manifest = build_manifest(run_id, t0, cutoffs, all_sources)
    if not dry_run:
        store_manifest(manifest)
    msha = manifest["manifest_sha256"]
    wrote = 0
    for gid, h, c, status in miss:
        g = games_live[gid]
        write(dict(common, record_type=status, game_id=gid, forecast_horizon=h, scheduled_start=g["game_start_utc"], schedule_state=g.get("schedule_state", "OK"), cutoff_at=iso(c), observed_at=iso(t0),
                   status=status, source_manifest_sha256=msha, reason="no valid forecast was generated inside the window; never backfilled"))
        wrote += 1
    if not pend:
        return {"forecasts": 0, "missed": wrote, "state_changes": len(changes), "run_id": run_id}
    if failures:
        fetch_failed = any(f_["completeness_status"] == "FETCH_FAILED" for f_ in failures)
        status = "SOURCE_FETCH_FAILED_NO_FORECAST" if fetch_failed else "SOURCE_INCOMPLETE_NO_FORECAST"
        for gid, h, c in pend:
            g = games_live[gid]
            write(dict(common, record_type=status, game_id=gid, forecast_horizon=h, scheduled_start=g["game_start_utc"], schedule_state=g.get("schedule_state", "OK"), cutoff_at=iso(c), observed_at=iso(now_fn()),
                       status=status, source_manifest_sha256=msha,
                       failed_sources=[{k: f_.get(k) for k in ("kind", "window", "url", "completeness_status", "error", "sha256", "game_id")} for f_ in failures],
                       reason="a required predictive source was incomplete or unavailable; no forecast is generated from partial history"))
        return {"forecasts": 0, "missed": wrote, "no_forecast": len(pend), "status": status, "run_id": run_id}
    games0, rows0 = frozen if frozen else D.load_frozen()
    games = dict(games0)
    for gid, g in games_live.items():
        games[gid] = {k: v for k, v in g.items() if k not in ("state", "schedule_state")}
    rows = list(rows0) + rows_live
    const = lock["comparator_constants"]
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
        for gid in tgids:
            g = games_live[gid]; c = cutoff_of(g["game_start_utc"], h)
            av, metas = avail[gid]
            spots, scratches, nm_av = extract_availability(av.get("landing"), av.get("right-rail"), av.get("boxscore"))
            t_end = now_fn()
            late = t_end > c or any(s_.get("retrieval_completed_utc") and parse_iso(s_["retrieval_completed_utc"]) > c for s_ in all_sources if s_.get("game_id") in (None, gid))
            idx = np.where(tab["game_id"] == gid)[0]
            if late:
                write(dict(common, record_type="INVALID_LATE", game_id=gid, forecast_horizon=h, scheduled_start=g["game_start_utc"], schedule_state=g.get("schedule_state", "OK"), cutoff_at=iso(c), observed_at=iso(t_end),
                           status="CAPTURED_LATE_AUDIT_ONLY", source_manifest_sha256=msha, reason="retrieval or generation completed after the cutoff: never used as pregame information"))
                continue
            for i in idx:
                pid = int(tab["player_id"][i]); team = int(tab["team_id"][i])
                a_abbrev = g["home_abbrev"] if team == g["home_team_id"] else g["away_abbrev"]; o_abbrev = g["away_abbrev"] if team == g["home_team_id"] else g["home_abbrev"]
                if not av.get("landing") and not av.get("boxscore"):
                    state = "NOT_CAPTURED"
                elif pid in scratches:
                    state = "SCRATCH_LISTED_OBSERVED_NOT_CERTIFIED"
                elif spots:
                    state = "IN_ROSTER_SPOTS_OBSERVED_NOT_CERTIFIED" if pid in spots else "ABSENT_FROM_ROSTER_SPOTS_OBSERVED_NOT_CERTIFIED"
                else:
                    state = "NO_ROSTER_DATA_PUBLISHED_YET"
                feat = {f: (None if np.isnan(tab[f][i]) else float(tab[f][i])) for f in D.FEATURES}
                snap = {"features": feat, "hist10": [int(x) for x in tab["hist10"][i]], "plays10": int(tab["plays10"][i]), "den10": int(tab["den10"][i])}
                rec = dict(common, record_type="FORECAST", forecast_id="%d-%s-%s-%d" % (gid, h, compact_start(g["game_start_utc"]), pid), game_id=gid, schedule_date=g.get("date"), player_id=pid, team=a_abbrev, opponent=o_abbrev,
                           scheduled_start=g["game_start_utc"], schedule_state=g.get("schedule_state", "OK"), forecast_horizon=h, cutoff_at=iso(c), generated_at=iso(t_end), source_manifest_sha256=msha,
                           candidate_status="CANDIDATE_MEANINGFUL" if bool(mmask[i]) else "CANDIDATE_NOT_MEANINGFUL", meaningful_expected_participant=bool(mmask[i]),
                           availability_state=state, availability_confidence="NOT_CERTIFIED", availability_used_in_forecast=False,
                           feature_snapshot_sha256=sha_text(canon(snap)), raw_source_hashes={"predictive_history": [s_["sha256"] for s_ in sources if s_.get("predictive")], "game_payloads": [{k: s_.get(k) for k in ("kind", "sha256", "retrieval_started_utc", "retrieval_completed_utc", "completeness_status")} for s_ in metas]},
                           expected_sog=float(summ["mean"][i]), median_sog=float(summ["median"][i]), variance=float(summ["variance"][i]), dispersion=float(alpha),
                           P1=float(summ["p_ge"][i, 0]), P2=float(summ["p_ge"][i, 1]), P3=float(summ["p_ge"][i, 2]), P4=float(summ["p_ge"][i, 3]), P5=float(summ["p_ge"][i, 4]),
                           comparators={"simple_prior10_mean": None if np.isnan(cf["m10"][i]) else float(cf["m10"][i]), "human_frozen_mean": None if np.isnan(mu_h[i]) else float(mu_h[i]), "comparator_eligible": bool(el[i])},
                           receipt={"player_name": names_cache.get(a_abbrev, {}).get(pid) or nm_av.get(pid) or None, "position": "F" if tab["POS_F"][i] == 1 else "D" if tab["POS_D"][i] == 1 else "U",
                                    "recent_participation": {"plays_last10_team_games": int(tab["plays10"][i]), "of": int(tab["den10"][i]), "team_games_since_appearance": feat["TEAM_GAMES_SINCE_APPEARANCE"]},
                                    "recent_sog_history_last10_appearances": snap["hist10"], "sog_per60_app10": feat["SOG_PER60_APP10"],
                                    "toi_seconds": {"recent3": feat["TOI_MEAN_CT_APP3"], "recent10": feat["TOI_MEAN_CT_APP10"], "trend_3_minus_10": feat["TOI_DELTA_CT_3_10"]},
                                    "pp_toi_recent3_seconds": feat["PP_TOI_MEAN_CT_APP3"], "pp_allocation_share_recent3_prior_games": feat["PP_ALLOC_SHARE_MEAN_CT_APP3"], "shifts_recent3": feat["SHIFT_MEAN_CT_APP3"],
                                    "team_sog_for_mean5": feat["TEAM_SOG_FOR_MEAN5"], "opp_sog_allowed_mean5": feat["OPP_SOG_ALLOWED_MEAN5"], "rest_hours": feat["TEAM_REST_HOURS"], "back_to_back": bool(feat["BACK_TO_BACK"]),
                                    "home": bool(feat["IS_HOME"]), "model_log_mean": float(np.log(mu[i])), "nb_dispersion": float(alpha), "source_hashes_ref": "source_manifest_sha256",
                                    "target_game_lineup_or_pp_state_used": False})
                write(rec)
                fc += 1
    return {"forecasts": fc, "missed": wrote, "state_changes": len(changes), "run_id": run_id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "verify", "dry-run"])
    a = ap.parse_args()
    if a.cmd == "verify":
        verify_lock(); n = Ledger().verify(); bad = audit_ledger()
        if bad:
            raise LockError("provenance audit failed: %s" % bad[:5])
        print("lock ok; ledger rows", n, "; provenance audit clean"); return
    print(run(dry_run=(a.cmd == "dry-run")))


if __name__ == "__main__":
    main()
