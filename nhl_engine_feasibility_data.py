"""
NHL_ENGINE_FEASIBILITY_DATA -- frozen official label tables for the Goalie G0 / Scoring S0 / Team-game T0 feasibility audits (no model, no sportsbook data).
Sources (all official NHL): stats-REST goalie/summary, stats-REST skater/summary, api-web schedule. Weekly gameDate windows, regular season, same M2 guards as the frozen Phase 1A dataset
(reported total < 10000 and rows == total). Every table is a deterministic gzip jsonl committed with its sha256.
"""
import gzip
import hashlib
import io
import json
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nhl_sog_phase1a_data as D

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_outcome_engine"
SEASONS = list(range(2017, 2026))
UA = {"User-Agent": "nhl-engine-feasibility"}
GOALIE_FIELDS = ["gameId", "playerId", "teamAbbrev", "opponentTeamAbbrev", "homeRoad", "gameDate", "gamesStarted", "gamesPlayed", "saves", "shotsAgainst", "goalsAgainst", "timeOnIce", "shutouts", "wins", "losses", "otLosses"]
SKATER_FIELDS = ["gameId", "playerId", "teamAbbrev", "positionCode", "goals", "assists", "points", "evGoals", "evPoints", "ppGoals", "ppPoints", "shGoals", "shPoints", "otGoals", "gameWinningGoals", "shots", "timeOnIcePerGame"]


def fetch_json(url, cache):
    key = hashlib.sha256(url.encode()).hexdigest()
    f, m = Path(cache) / f"{key}.bin", Path(cache) / f"{key}.meta.json"
    if f.exists() and m.exists():
        return json.loads(f.read_bytes()), json.loads(m.read_text())
    last = None
    for a in range(4):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120)
            raw = r.read()
            meta = {"url": url, "retrieved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "http_status": r.status}
            f.write_bytes(raw); m.write_text(json.dumps(meta))
            return json.loads(raw), meta
        except Exception as e:                                                  # noqa
            last = e; time.sleep(2 * (a + 1))
    raise RuntimeError(f"{url}: {last}")


def det_gzip(text):
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", mtime=0, compresslevel=9) as g:
        g.write(text.encode())
    return bio.getvalue()


def acquire(cache, workers=4, log=print):
    jobs = [(s, lo, hi) for s in SEASONS for lo, hi in D.window_list(s)]

    def one(j):
        s, lo, hi = j
        g, gm = fetch_json(D.stats_url("/goalie/summary", lo, hi), cache)
        k, km = fetch_json(D.stats_url("/skater/summary", lo, hi), cache)
        sch, sm = fetch_json(f"{D.API}/schedule/{lo}", cache)
        return s, lo, hi, g, gm, k, km, sch, sm
    goalie, skater, teamg, prov = defaultdict(list), defaultdict(list), {}, []
    guard_fail = 0
    with ThreadPoolExecutor(workers) as ex:
        for s, lo, hi, g, gm, k, km, sch, sm in ex.map(one, jobs):
            for name, d, m in (("goalie", g, gm), ("skater", k, km)):
                ok = d["total"] < 10000 and len(d["data"]) == d["total"]
                guard_fail += (not ok)
                prov.append({"window": [lo, hi], "report": name, "reported_total": d["total"], "rows": len(d["data"]), "guard_ok": ok, "sha256": m["sha256"], "bytes": m["bytes"], "retrieved_at_utc": m["retrieved_at_utc"]})
            prov.append({"window": [lo, hi], "report": "schedule", "sha256": sm["sha256"], "bytes": sm["bytes"], "retrieved_at_utc": sm["retrieved_at_utc"]})
            for r in g["data"]:
                goalie[int(str(r["gameId"])[:4])].append({f: r.get(f) for f in GOALIE_FIELDS})
            for r in k["data"]:
                skater[int(str(r["gameId"])[:4])].append({f: r.get(f) for f in SKATER_FIELDS})
            for day in sch["gameWeek"]:
                if not (lo <= day["date"] <= hi):
                    continue
                for gm_ in day["games"]:
                    if gm_["gameType"] != 2 or gm_["gameState"] not in ("OFF", "FINAL"):
                        continue
                    teamg[gm_["id"]] = {"gameId": gm_["id"], "startTimeUTC": gm_["startTimeUTC"], "homeTeamId": gm_["homeTeam"]["id"], "awayTeamId": gm_["awayTeam"]["id"], "homeAbbrev": gm_["homeTeam"]["abbrev"], "awayAbbrev": gm_["awayTeam"]["abbrev"],
                                        "homeScore": gm_["homeTeam"].get("score"), "awayScore": gm_["awayTeam"].get("score"), "lastPeriodType": (gm_.get("gameOutcome") or {}).get("lastPeriodType"), "gameState": gm_["gameState"]}
    log(f"guard failures {guard_fail}; goalie rows {sum(map(len, goalie.values()))} skater rows {sum(map(len, skater.values()))} games {len(teamg)}")
    return goalie, skater, teamg, prov, guard_fail


def write_tables(goalie, skater, teamg, prov):
    files = {}
    specs = [("phase_goalie_g0_data", "goalie_games", goalie), ("phase_scoring_s0_data", "skater_scoring", skater)]
    for d, stem, tab in specs:
        dd = OUT / d; dd.mkdir(exist_ok=True)
        for s in SEASONS:
            rows = sorted(tab.get(s, []), key=lambda r: (r["gameId"], r["playerId"]))
            b = det_gzip("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows))
            (dd / f"{stem}_{s}.jsonl.gz").write_bytes(b)
            files[f"{d}/{stem}_{s}.jsonl.gz"] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "rows": len(rows)}
    dd = OUT / "phase_team_game_t0_data"; dd.mkdir(exist_ok=True)
    for s in SEASONS:
        rows = sorted((r for r in teamg.values() if int(str(r["gameId"])[:4]) == s), key=lambda r: r["gameId"])
        b = det_gzip("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows))
        (dd / f"team_games_{s}.jsonl.gz").write_bytes(b)
        files[f"phase_team_game_t0_data/team_games_{s}.jsonl.gz"] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "rows": len(rows)}
    man = {"dataset": "nhl-engine-feasibility-labels-v1", "files": files, "n_requests": len(prov), "total_bytes_downloaded": sum(p["bytes"] for p in prov), "windows": prov}
    man["manifest_content_sha256"] = hashlib.sha256(json.dumps({k: man[k] for k in ("files",)}, sort_keys=True).encode()).hexdigest()
    (OUT / "nhl_engine_feasibility_data_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


def read_table(rel):
    return [json.loads(l) for l in gzip.decompress((OUT / rel).read_bytes()).decode().splitlines() if l]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); a = ap.parse_args()
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    g, s, t, p, gf = acquire(a.cache)
    m = write_tables(g, s, t, p)
    print("files", len(m["files"]), "requests", m["n_requests"], "bytes", m["total_bytes_downloaded"], "guard failures", gf)
