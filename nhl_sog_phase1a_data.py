"""
NHL_SOG_PHASE1A_DATA -- frozen historical dataset (Phase 0B M2 acquisition), data-quality gates and the leak-free prediction-row builder for the Phase 1A SOG distribution engine.

Acquisition (M2 only): api-web schedule + stats-REST skater/summary + skater/timeonice over weekly gameDate windows, regular season, no team filter, guard `reported total < 10000 and rows == total`.
The research input is the committed frozen per-season gzip JSONL files; everything downstream reads ONLY those files.
"""
import argparse
import bisect
import gzip
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import nhl_outcome_contract as CT

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_outcome_engine"
DATA = OUT / "phase1a_data"
API = "https://api-web.nhle.com/v1"
ST = "https://api.nhle.com/stats/rest/en"
UTC = timezone.utc
SEASONS = list(range(2017, 2026))
TARGET_SEASONS = list(range(2018, 2026))
FROZEN_FIELDS = ["game_id", "season_start_year", "game_start_utc", "team_id", "team_abbrev", "opponent", "player_id", "position", "sog", "toi_sec", "ev_toi_sec", "pp_toi_sec", "sh_toi_sec", "shifts"]
EXTRA_FIELDS = ["ot_toi_sec"]
MAX_SOG = 60
SPORTSBOOK_WORDS = ("odds", "line", "spread", "moneyline", "over_under", "book", "price", "projection")


# ------------------------------------------------------------------ acquisition
def window_list(season):
    """Weekly 7-day windows Sep 28 .. ~Jun 1 of the season."""
    lo = datetime(season, 9, 28)
    end = datetime(season + 1, 6, 1)
    out = []
    while lo <= end:
        out.append((lo.strftime("%Y-%m-%d"), (lo + timedelta(days=6)).strftime("%Y-%m-%d")))
        lo += timedelta(days=7)
    return out


def fetch(url, cache):
    key = hashlib.sha256(url.encode()).hexdigest()
    f, m = Path(cache) / f"{key}.bin", Path(cache) / f"{key}.meta.json"
    if f.exists() and m.exists():
        return f.read_bytes(), json.loads(m.read_text()), True
    for attempt in range(4):
        try:
            t0 = time.time()
            r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nhl-phase1a"}), timeout=120)
            raw = r.read()
            meta = {"url": url, "retrieved_at_utc": datetime.now(UTC).isoformat(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "seconds": round(time.time() - t0, 2), "http_status": r.status}
            f.write_bytes(raw); m.write_text(json.dumps(meta))
            return raw, meta, False
        except Exception as e:                                                  # noqa
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def stats_url(path, lo, hi):
    cay = f'gameDate>="{lo}" and gameDate<="{hi}" and gameTypeId=2'
    return ST + path + "?" + urllib.parse.urlencode({"isGame": "true", "limit": "-1", "cayenneExp": cay}, quote_via=urllib.parse.quote)


def acquire_window(args):
    lo, hi, cache = args
    sraw, smeta, _ = fetch(f"{API}/schedule/{lo}", cache)
    sched = json.loads(sraw)
    games = []
    for day in sched["gameWeek"]:
        if lo <= day["date"] <= hi:
            for g in day["games"]:
                if g["gameType"] == 2 and g["gameState"] in ("OFF", "FINAL"):
                    games.append({"game_id": g["id"], "game_start_utc": g["startTimeUTC"], "home_abbrev": g["homeTeam"]["abbrev"], "away_abbrev": g["awayTeam"]["abbrev"], "home_team_id": g["homeTeam"]["id"], "away_team_id": g["awayTeam"]["id"], "date": day["date"]})
    out = {"window": [lo, hi], "games": games, "provenance": {"schedule": {k: smeta[k] for k in ("url", "retrieved_at_utc", "sha256", "bytes")}}}
    for name, path in (("summary", "/skater/summary"), ("timeonice", "/skater/timeonice")):
        url = stats_url(path, lo, hi)
        raw, meta, _ = fetch(url, cache)
        d = json.loads(raw)
        out[name] = d["data"]
        out["provenance"][name] = {"url": url, "retrieved_at_utc": meta["retrieved_at_utc"], "sha256": meta["sha256"], "bytes": meta["bytes"], "reported_total": d["total"], "returned_rows": len(d["data"]),
                                   "guard_ok": d["total"] < 10000 and len(d["data"]) == d["total"]}
    return out


def assemble(windows):
    """windows -> (games {id: dict}, rows [dict]) with explicit missingness (None) and no silent zero filling."""
    games, rows, seen = {}, [], set()
    for w in windows:
        for g in w["games"]:
            games[g["game_id"]] = g
    for w in windows:
        tio = {(r["gameId"], r["playerId"]): r for r in w["timeonice"]}
        for r in w["summary"]:
            g = games.get(r["gameId"])
            if g is None:
                raise RuntimeError(f"summary row of game {r['gameId']} has no completed regular-season game in the schedule")
            team = r["teamAbbrev"]
            tid = g["home_team_id"] if team == g["home_abbrev"] else g["away_team_id"] if team == g["away_abbrev"] else None
            if tid is None:
                raise RuntimeError(f"team {team} is not a participant of game {r['gameId']}")
            key = (r["gameId"], tid, r["playerId"])
            if key in seen:
                raise RuntimeError(f"duplicate skater-game row {key}")
            seen.add(key)
            t = tio.get((r["gameId"], r["playerId"]), {})
            rows.append({"game_id": r["gameId"], "season_start_year": int(str(r["gameId"])[:4]), "game_start_utc": g["game_start_utc"], "team_id": tid, "team_abbrev": team, "opponent": r["opponentTeamAbbrev"], "player_id": r["playerId"],
                         "position": r.get("positionCode"), "sog": r["shots"], "toi_sec": t.get("timeOnIce"), "ev_toi_sec": t.get("evTimeOnIce"), "pp_toi_sec": t.get("ppTimeOnIce"), "sh_toi_sec": t.get("shTimeOnIce"),
                         "ot_toi_sec": t.get("otTimeOnIce"), "shifts": t.get("shifts")})
    rows.sort(key=lambda r: (r["game_start_utc"], r["game_id"], r["team_id"], r["player_id"]))
    return games, rows


def det_gzip(text):
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", mtime=0, compresslevel=9) as g:
        g.write(text.encode())
    return bio.getvalue()


def write_frozen(games, rows, out_dir=DATA):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for s in SEASONS:
        srows = [r for r in rows if r["season_start_year"] == s]
        sgames = sorted((g for g in games.values() if int(str(g["game_id"])[:4]) == s), key=lambda g: (g["game_start_utc"], g["game_id"]))
        for kind, items, name in (("skater_games", srows, f"skater_games_{s}.jsonl.gz"), ("games", sgames, f"games_{s}.jsonl.gz")):
            text = "".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in items)
            b = det_gzip(text)
            (out_dir / name).write_bytes(b)
            files[name] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "rows": len(items), "season": s, "kind": kind}
    return files


def read_jsonl_gz(p):
    return [json.loads(l) for l in gzip.decompress(Path(p).read_bytes()).decode().splitlines() if l]


def load_frozen(data_dir=DATA):
    games, rows = {}, []
    for s in SEASONS:
        for g in read_jsonl_gz(Path(data_dir) / f"games_{s}.jsonl.gz"):
            games[g["game_id"]] = g
        rows += read_jsonl_gz(Path(data_dir) / f"skater_games_{s}.jsonl.gz")
    return games, rows


def verify_manifest(manifest_path=OUT / "phase1a_data_manifest.json", data_dir=DATA):
    m = json.loads(Path(manifest_path).read_text())
    bad = [n for n, f in m["files"].items() if hashlib.sha256((Path(data_dir) / n).read_bytes()).hexdigest() != f["sha256"]]
    return bad, m


def acquire(cache, workers=4, log=print):
    t0 = time.time()
    jobs = [(lo, hi, cache) for s in SEASONS for lo, hi in window_list(s)]
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(acquire_window, jobs))
    games, rows = assemble(results)
    files = write_frozen(games, rows)
    prov = [{"window": r["window"], **{k: v for k, v in r["provenance"].items()}} for r in results]
    bad = [p for p in prov for k in ("summary", "timeonice") if not p[k]["guard_ok"]]
    manifest = {"dataset": "nhl-sog-phase1a-frozen-v1", "method": "Phase 0B M2 (schedule + stats-REST skater/summary + skater/timeonice, weekly gameDate windows, regular season, no team filter)", "seasons": SEASONS,
                "frozen_fields": FROZEN_FIELDS + EXTRA_FIELDS, "files": files, "n_windows": len(prov), "n_requests": 3 * len(prov), "total_bytes_downloaded": sum(p[k]["bytes"] for p in prov for k in ("schedule", "summary", "timeonice")),
                "guard_failures": [{"window": p["window"]} for p in bad], "windows": prov, "acquisition_seconds": round(time.time() - t0, 1), "retrieved_at_utc_range": [min(p["summary"]["retrieved_at_utc"] for p in prov), max(p["summary"]["retrieved_at_utc"] for p in prov)]}
    manifest["manifest_content_sha256"] = hashlib.sha256(json.dumps({k: manifest[k] for k in ("files", "windows")}, sort_keys=True).encode()).hexdigest()
    (OUT / "phase1a_data_manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"games {len(games)} rows {len(rows)} windows {len(prov)} guard failures {len(bad)} seconds {manifest['acquisition_seconds']}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["acquire"])
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    acquire(a.cache)


# ================================================================== data-quality gates
def epoch(s):
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp())


def toi_classes(rows):
    """TOI quality per season: all components present / missing component(s) / present but inconsistent (EV+PP+SH(+OT) vs total within 1 s)."""
    out = {}
    for s in SEASONS:
        rs = [r for r in rows if r["season_start_year"] == s]
        full = [r for r in rs if all(r[k] is not None for k in ("toi_sec", "ev_toi_sec", "pp_toi_sec", "sh_toi_sec"))]
        incons = []
        for r in full:
            tot = r["ev_toi_sec"] + r["pp_toi_sec"] + r["sh_toi_sec"]
            if abs(tot - r["toi_sec"]) > 1 and abs(tot + (r["ot_toi_sec"] or 0) - r["toi_sec"]) > 1:
                incons.append(r)
        out[s] = {"rows": len(rs), "all_components_present": len(full), "missing_components": len(rs) - len(full), "present_but_inconsistent": len(incons),
                  "inconsistent_fraction_of_fully_observed": round(len(incons) / len(full), 6) if full else None, "block_threshold": 0.002, "pass": (len(incons) / len(full) if full else 0) <= 0.002}
    return out


def quality_gates(games, rows, manifest):
    g = {}
    keys = [(r["game_id"], r["team_id"], r["player_id"]) for r in rows]
    g["unique_game_team_player_rows"] = len(keys) == len(set(keys))
    g["every_row_game_in_schedule"] = all(r["game_id"] in games for r in rows)
    g["every_target_game_has_rows"] = all(any(True for _ in [0]) for _ in [0]) and len({r["game_id"] for r in rows}) == len(games)
    g["sog_nonnegative_integer"] = all(isinstance(r["sog"], int) and r["sog"] >= 0 for r in rows)
    g["sog_le_60"] = max(r["sog"] for r in rows) <= MAX_SOG
    g["toi_shift_nonnegative_where_present"] = all((r[k] is None or r[k] >= 0) for r in rows for k in ("toi_sec", "ev_toi_sec", "pp_toi_sec", "sh_toi_sec", "shifts"))
    g["player_id_numeric"] = all(isinstance(r["player_id"], int) for r in rows)
    g["no_sportsbook_columns"] = not any(any(w in k.lower() for w in SPORTSBOOK_WORDS) for k in rows[0].keys())
    g["m2_guards_all_windows"] = not manifest["guard_failures"] and all(w[k]["guard_ok"] for w in manifest["windows"] for k in ("summary", "timeonice"))
    g["ordered_by_utc_start_not_synthetic_week"] = all(rows[i]["game_start_utc"] <= rows[i + 1]["game_start_utc"] for i in range(len(rows) - 1)) and "week" not in rows[0]
    g["positions_known_codes"] = {r["position"] for r in rows} <= {"C", "L", "R", "D", None}
    return g


# ================================================================== prediction-row builder (Phase 0B candidate universe, T = start - 90 min, source start + 210 min <= T)
FEATURES = ["POS_F", "POS_D", "POS_UNKNOWN", "IS_HOME", "TEAM_REST_HOURS", "BACK_TO_BACK", "PLAYED_LAST1", "PLAY_RATE_TG3", "PLAY_RATE_TG10", "PLAY_DEN_TG3", "PLAY_DEN_TG10", "TEAM_GAMES_SINCE_APPEARANCE",
            "DAYS_SINCE_LAST_APPEARANCE", "N_CURRENT_SEASON_TEAM_GAMES_OBS", "N_CURRENT_SEASON_PLAYER_APPEARANCES", "SOG_MEAN_APP5", "SOG_MEAN_APP10", "SOG_SD_APP10", "SOG_PER60_APP10", "N_SKILL_APPEARANCES_10",
            "TOI_MEAN_CT_APP3", "TOI_MEAN_CT_APP10", "TOI_DELTA_CT_3_10", "PP_TOI_MEAN_CT_APP3", "PP_ALLOC_SHARE_MEAN_CT_APP3", "PP_ALLOC_SHARE_MEAN_CT_APP10", "PP_ALLOC_SHARE_DELTA_CT_3_10", "SHIFT_MEAN_CT_APP3",
            "SHIFT_MEAN_CT_APP10", "SHIFT_DELTA_CT_3_10", "N_ROLE_APPEARANCES_10", "TEAM_SOG_FOR_MEAN5", "OPP_SOG_ALLOWED_MEAN5"]
BINARY = {"POS_F", "POS_D", "POS_UNKNOWN", "IS_HOME", "BACK_TO_BACK", "PLAYED_LAST1"}
CUTOFF_BACK_S = (90 + CT.GAME_MAX_MINUTES) * 60           # a source game is usable iff its start <= target start - 300 min
REST_CAP = 240.0


def nanmean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else math.nan


def pos_class(code):
    return "F" if code in ("C", "L", "R", "F", "W") else "D" if code == "D" else "U"


def build_prediction_rows(games, rows, target_seasons=TARGET_SEASONS):
    """Candidate rows + labels. Every feature is computed from rows of games with start <= target_start - 300 min (so source_start + 210 min <= T and the target game itself is excluded by construction);
    target-game rows are read ONLY after the features of the team-game are complete, for labels and coverage."""
    gstart = {gid: epoch(g["game_start_utc"]) for gid, g in games.items()}
    team_games = defaultdict(list)                                           # team -> [(start, game_id)] from the SCHEDULE
    for gid, g in games.items():
        for tid in (g["home_team_id"], g["away_team_id"]):
            team_games[tid].append((gstart[gid], gid))
    for t in team_games:
        team_games[t].sort()
    team_starts = {t: [x[0] for x in v] for t, v in team_games.items()}
    game_players = defaultdict(set)                                          # (game, team) -> {player}
    team_sog = defaultdict(int); team_pp = defaultdict(int)
    for r in rows:
        game_players[(r["game_id"], r["team_id"])].add(r["player_id"])
        team_sog[(r["game_id"], r["team_id"])] += r["sog"]
        team_pp[(r["game_id"], r["team_id"])] += r["pp_toi_sec"] or 0
    sog_against = {}
    for gid, g in games.items():
        h, a = g["home_team_id"], g["away_team_id"]
        sog_against[(gid, h)] = team_sog[(gid, a)]; sog_against[(gid, a)] = team_sog[(gid, h)]
    apps = defaultdict(list)                                                 # player -> chronological appearances (any team)
    for r in rows:
        s = gstart[r["game_id"]]
        share = (r["pp_toi_sec"] / team_pp[(r["game_id"], r["team_id"])]) if (r["pp_toi_sec"] is not None and team_pp[(r["game_id"], r["team_id"])] > 0) else None
        apps[r["player_id"]].append((s, r["game_id"], r["team_id"], r["position"], r["sog"], r["toi_sec"], r["pp_toi_sec"], r["shifts"], share))
    for p in apps:
        apps[p].sort(key=lambda a: (a[0], a[1]))
    app_starts = {p: [a[0] for a in v] for p, v in apps.items()}
    by_game_team = defaultdict(list)
    for r in rows:
        by_game_team[(r["game_id"], r["team_id"])].append(r)
    cols = {k: [] for k in ("game_id", "team_id", "opp_id", "player_id", "season", "start", "cutoff", "played", "sog", "plays10", "den10", "hist10")}
    feats = {k: [] for k in FEATURES}
    cov = {s: {"team_games": 0, "candidate_rows": 0, "played_rows": 0, "actual_target_skaters": 0, "observable": 0, "unobservable": 0, "actual_SOG": 0, "observable_SOG": 0, "unobservable_diagnostics": defaultdict(int)} for s in target_seasons}
    for gid in sorted(games, key=lambda g: (gstart[g], g)):
        g = games[gid]
        season = int(str(gid)[:4])
        if season not in target_seasons:
            continue
        s0 = gstart[gid]; limit = s0 - CUTOFF_BACK_S
        for team, opp, home in ((g["home_team_id"], g["away_team_id"], 1), (g["away_team_id"], g["home_team_id"], 0)):
            ti = bisect.bisect_right(team_starts[team], limit)               # allowed prior team games = first ti schedule entries (target game excluded: its start > limit)
            tg = team_games[team][:ti]
            last10 = [x[1] for x in tg[-10:]]; last3 = last10[-3:]; last1 = last10[-1:]
            cand = sorted({p for gg in last10 for p in game_players[(gg, team)]})
            pos_idx = team_games[team].index((s0, gid)) if (s0, gid) in team_games[team] else None
            prev_start = team_games[team][pos_idx - 1][0] if pos_idx else None
            rest = (s0 - prev_start) / 3600.0 if prev_start else math.nan
            # environment
            tf5 = nanmean([team_sog[(x[1], team)] for x in tg[-5:]]) if tg else math.nan
            oi = bisect.bisect_right(team_starts[opp], limit)
            og = team_games[opp][:oi]
            oa5 = nanmean([sog_against[(x[1], opp)] for x in og[-5:]]) if og else math.nan
            n_season_tg = sum(1 for x in tg if int(str(x[1])[:4]) == season)
            for pid in cand:
                pa = apps[pid]; k = bisect.bisect_right(app_starts[pid], limit)
                allowed = pa[:k]                                              # all-team appearances completed before T
                ct = [a for a in allowed if a[2] == team]                      # current-team appearances only
                last_ct = ct[-1]
                pos = pos_class(last_ct[3])                                    # latest PRIOR same-team appearance
                plays = [1 if pid in game_players[(gg, team)] else 0 for gg in last10]
                den10 = len(last10); den3 = len(last3)
                plays3 = sum(plays[-3:]); plays10 = sum(plays)
                since = 0
                for gg in reversed([x[1] for x in tg]):
                    if pid in game_players[(gg, team)]:
                        break
                    since += 1
                skill = allowed[-10:]
                sogs = [a[4] for a in skill]
                tois = [(a[4], a[5]) for a in skill if a[5]]
                per60 = 3600.0 * sum(x for x, _ in tois) / sum(t for _, t in tois) if tois and sum(t for _, t in tois) > 0 else math.nan
                c3, c10 = ct[-3:], ct[-10:]
                toi3, toi10 = nanmean([a[5] for a in c3]), nanmean([a[5] for a in c10])
                pp3 = nanmean([a[6] for a in c3])
                sh3, sh10 = nanmean([a[7] for a in c3]), nanmean([a[7] for a in c10])
                al3, al10 = nanmean([a[8] for a in c3]), nanmean([a[8] for a in c10])
                d = lambda a, b: a - b if not (math.isnan(a) or math.isnan(b)) else math.nan
                f = {"POS_F": float(pos == "F"), "POS_D": float(pos == "D"), "POS_UNKNOWN": float(pos == "U"), "IS_HOME": float(home), "TEAM_REST_HOURS": rest, "BACK_TO_BACK": float(rest < 36) if not math.isnan(rest) else 0.0,
                     "PLAYED_LAST1": float(plays[-1]) if plays else 0.0, "PLAY_RATE_TG3": plays3 / den3 if den3 else math.nan, "PLAY_RATE_TG10": plays10 / den10 if den10 else math.nan, "PLAY_DEN_TG3": float(den3), "PLAY_DEN_TG10": float(den10),
                     "TEAM_GAMES_SINCE_APPEARANCE": float(since), "DAYS_SINCE_LAST_APPEARANCE": (s0 - last_ct[0]) / 86400.0, "N_CURRENT_SEASON_TEAM_GAMES_OBS": float(n_season_tg),
                     "N_CURRENT_SEASON_PLAYER_APPEARANCES": float(sum(1 for a in allowed if int(str(a[1])[:4]) == season)),
                     "SOG_MEAN_APP5": nanmean(sogs[-5:]), "SOG_MEAN_APP10": nanmean(sogs), "SOG_SD_APP10": (float(np.std(sogs, ddof=1)) if len(sogs) >= 2 else math.nan), "SOG_PER60_APP10": per60, "N_SKILL_APPEARANCES_10": float(len(sogs)),
                     "TOI_MEAN_CT_APP3": toi3, "TOI_MEAN_CT_APP10": toi10, "TOI_DELTA_CT_3_10": d(toi3, toi10), "PP_TOI_MEAN_CT_APP3": pp3, "PP_ALLOC_SHARE_MEAN_CT_APP3": al3, "PP_ALLOC_SHARE_MEAN_CT_APP10": al10, "PP_ALLOC_SHARE_DELTA_CT_3_10": d(al3, al10),
                     "SHIFT_MEAN_CT_APP3": sh3, "SHIFT_MEAN_CT_APP10": sh10, "SHIFT_DELTA_CT_3_10": d(sh3, sh10), "N_ROLE_APPEARANCES_10": float(len(c10)), "TEAM_SOG_FOR_MEAN5": tf5, "OPP_SOG_ALLOWED_MEAN5": oa5}
                for kf in FEATURES:
                    feats[kf].append(f[kf])
                cols["hist10"].append(sogs + [-1] * (10 - len(sogs)))
                cols["plays10"].append(plays10); cols["den10"].append(den10)
                for kk, vv in (("game_id", gid), ("team_id", team), ("opp_id", opp), ("player_id", pid), ("season", season), ("start", s0), ("cutoff", s0 - 5400)):
                    cols[kk].append(vv)
                cols["played"].append(0); cols["sog"].append(0)               # labels filled below, AFTER all features of this team-game exist
            # ---- labels / coverage (target rows read only now) ----
            first = len(cols["game_id"]) - len(cand)
            actual = {r["player_id"]: r for r in by_game_team.get((gid, team), [])}
            cset = set(cand)
            for i, pid in enumerate(cand):
                a = actual.get(pid)
                if a is not None:
                    cols["played"][first + i] = 1; cols["sog"][first + i] = a["sog"]
            c = cov[season]
            c["team_games"] += 1; c["candidate_rows"] += len(cand); c["played_rows"] += sum(1 for p in cand if p in actual)
            c["actual_target_skaters"] += len(actual); c["observable"] += sum(1 for p in actual if p in cset); c["unobservable"] += sum(1 for p in actual if p not in cset)
            c["actual_SOG"] += sum(a["sog"] for a in actual.values()); c["observable_SOG"] += sum(a["sog"] for p, a in actual.items() if p in cset)
            for p, a in actual.items():
                if p not in cset:
                    k = bisect.bisect_right(app_starts.get(p, []), limit)
                    al = apps.get(p, [])[:k]
                    c["unobservable_diagnostics"][("NO_LOADED_HISTORY" if not al else "STALE_SAME_TEAM_HISTORY_OUTSIDE_MEMBERSHIP_WINDOW" if any(x[2] == team for x in al) else "PRIOR_NHL_HISTORY_ELSEWHERE")] += 1
    out = {k: np.array(v) for k, v in cols.items() if k != "hist10"}
    out["hist10"] = np.array(cols["hist10"], dtype=np.int16).reshape(-1, 10)
    for k in FEATURES:
        out[k] = np.array(feats[k], dtype=np.float64)
    for s in cov:
        cov[s]["unobservable_diagnostics"] = dict(cov[s]["unobservable_diagnostics"])
    return out, cov


def table_hash(tab):
    h = hashlib.sha256()
    for k in sorted(tab):
        h.update(k.encode()); h.update(np.ascontiguousarray(tab[k]).tobytes())
    return h.hexdigest()


def week_index(start_sec):
    return (np.asarray(start_sec) - 345600) // 604800            # Monday 00:00 UTC calendar weeks
