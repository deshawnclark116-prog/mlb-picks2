"""
NHL_SOG_PHASE1B_ADJUDICATE -- source adjudication for the Phase 1B-A attempt extension (protocol: phase1b_attempt_source_adjudication_protocol.json, committed first).
Steps: A boxscore vs frozen official SOG for every mismatch player-game; B Stats REST skater/realtime acquisition (official individual attempt components);
C official components vs API PBP components for every source player-game; D official HTML play-by-play adjudication (mismatch games + deterministic controls); E decision.
No model is fit here. No tolerated mismatch rate exists anywhere.
"""
import argparse
import gzip
import hashlib
import json
import re
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nhl_sog_phase1a_data as D
import nhl_sog_phase1b_attempts as AT

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_outcome_engine"
UA = {"User-Agent": "nhl-phase1b-a-adjudication"}
SOURCE_SEASONS = list(range(2017, 2024))
CONTROL_TAG = "nhl-phase1b-a-adjudication-control-v1"
UTC = timezone.utc


def get_json(url, cache, retries=4):
    key = hashlib.sha256(url.encode()).hexdigest()
    f, m = Path(cache) / f"{key}.bin", Path(cache) / f"{key}.meta.json"
    if f.exists() and m.exists():
        return json.loads(f.read_bytes()), json.loads(m.read_text())
    last = None
    for a in range(retries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120)
            raw = r.read()
            meta = {"url": url, "retrieved_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "http_status": r.status}
            f.write_bytes(raw); m.write_text(json.dumps(meta))
            return json.loads(raw), meta
        except Exception as e:                                                  # noqa
            last = e; time.sleep(2 * (a + 1))
    raise RuntimeError(f"{url}: {last}")


def get_text(url, cache, retries=4):
    key = hashlib.sha256(url.encode()).hexdigest()
    f, m = Path(cache) / f"{key}.html", Path(cache) / f"{key}.meta.json"
    if f.exists() and m.exists():
        return f.read_text(errors="replace"), json.loads(m.read_text())
    last = None
    for a in range(retries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120)
            raw = r.read()
            meta = {"url": url, "retrieved_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "http_status": r.status}
            f.write_bytes(raw); m.write_text(json.dumps(meta))
            return raw.decode("utf-8", errors="replace"), meta
        except Exception as e:                                                  # noqa
            last = e; time.sleep(2 * (a + 1))
    return None, {"url": url, "error": f"{type(last).__name__}: {last}"[:200]}


# ------------------------------------------------------------------ step A
def step_a(mismatches, cache, workers=6):
    games = sorted({m["game_id"] for m in mismatches})

    def one(g):
        d, meta = get_json(f"{AT.API}/gamecenter/{g}/boxscore", cache)
        pl = {p["playerId"]: p.get("sog") for side in ("awayTeam", "homeTeam") for pos in ("forwards", "defense") for p in d["playerByGameStats"][side][pos]}
        return g, pl, meta
    with ThreadPoolExecutor(workers) as ex:
        res = {g: (pl, meta) for g, pl, meta in ex.map(one, games)}
    classes, rows = Counter(), []
    for m in mismatches:
        bx = res[m["game_id"]][0].get(m["player_id"])
        cls = "BOXSCORE_EQUALS_FROZEN_OFFICIAL" if bx == m["official_sog"] else "BOXSCORE_EQUALS_PBP" if bx == m["pbp_sog"] else "BOXSCORE_EQUALS_NEITHER"
        classes[cls] += 1
        rows.append({**{k: m[k] for k in ("game_id", "player_id", "official_sog", "pbp_sog")}, "boxscore_sog": bx, "class": cls})
    return {"n_mismatch_player_games": len(mismatches), "n_games": len(games), "classes": dict(classes), "rows": rows}


# ------------------------------------------------------------------ step B
def step_b_probe(cache):
    cfg, meta = get_json("https://api.nhle.com/stats/rest/en/config", cache)
    rep = cfg["playerReportData"]
    out = {"config_url": meta["url"], "reports": {}}
    for name, v in sorted(rep.items()):
        items = (v.get("game") or {}).get("displayItems") or []
        hits = [i for i in items if re.search(r"attempt|missed|blocked|sat|shot", i, re.I)]
        out["reports"][name] = {"game_level_fields_matching_attempt|missed|blocked|sat|shot": hits}
    return out


def step_b_acquire(cache, workers=4, log=print):
    jobs = [(lo, hi) for s in SOURCE_SEASONS for lo, hi in D.window_list(s)]

    def one(j):
        lo, hi = j
        url = D.stats_url("/skater/realtime", lo, hi)
        d, meta = get_json(url, cache)
        return lo, hi, d, meta
    rows, windows = {}, []
    with ThreadPoolExecutor(workers) as ex:
        for lo, hi, d, meta in ex.map(one, jobs):
            guard = d["total"] < 10000 and len(d["data"]) == d["total"]
            windows.append({"window": [lo, hi], "reported_total": d["total"], "returned_rows": len(d["data"]), "guard_ok": guard, "sha256": meta["sha256"], "bytes": meta["bytes"], "retrieved_at_utc": meta["retrieved_at_utc"]})
            for r in d["data"]:
                rows[(r["gameId"], r["playerId"])] = {"missed": r.get("missedShots"), "blocked_own": r.get("shotAttemptsBlocked"), "blocks_made": r.get("blockedShots"), "total_attempts": r.get("totalShotAttempts")}
    log(f"realtime windows {len(windows)} rows {len(rows)} guard failures {sum(1 for w in windows if not w['guard_ok'])}")
    return rows, windows


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


# ------------------------------------------------------------------ step C
def step_c(attempt_rows, realtime, frozen_sog):
    cov = defaultdict(lambda: {"rows": 0, "realtime_present": 0, "both_integer_nonnegative": 0})
    comp = {"missed": {"compared": 0, "mismatch": 0, "examples": []}, "blocked": {"compared": 0, "mismatch": 0, "examples": []}}
    dis_games = defaultdict(list)
    for (g, p), r in attempt_rows.items():
        s = int(str(g)[:4])
        c = cov[s]; c["rows"] += 1
        rt = realtime.get((g, p))
        if rt is None:
            continue
        c["realtime_present"] += 1
        if is_int(rt["missed"]) and is_int(rt["blocked_own"]):
            c["both_integer_nonnegative"] += 1
        else:
            continue
        for k, mine, theirs in (("missed", r["missed_attempts"], rt["missed"]), ("blocked", r["blocked_attempts"], rt["blocked_own"])):
            comp[k]["compared"] += 1
            if mine != theirs:
                comp[k]["mismatch"] += 1; dis_games[g].append({"player_id": p, "component": k, "pbp": mine, "official_realtime": theirs})
                if len(comp[k]["examples"]) < 15:
                    comp[k]["examples"].append({"game_id": g, "player_id": p, "pbp": mine, "official_realtime": theirs})
    gate_b = all(c["rows"] == c["both_integer_nonnegative"] for c in cov.values())
    return {"coverage_by_season": {str(s): v for s, v in sorted(cov.items())}, "gate_B_passes": bool(gate_b), "component_vs_pbp": comp, "games_with_component_disagreement": sorted(dis_games), "disagreements_sample": {str(g): v[:5] for g, v in list(dis_games.items())[:10]}}


def pick_controls(game_ids, exclude, n):
    pool = sorted((g for g in game_ids if g not in exclude), key=lambda g: (hashlib.sha256(f"{CONTROL_TAG}|{g}".encode()).hexdigest(), g))
    return pool[:n]


# ------------------------------------------------------------------ step D: official HTML play-by-play
HTML_TEAM_FIX = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL"}


def html_url(game_id):
    s = int(str(game_id)[:4])
    return f"https://www.nhl.com/scores/htmlreports/{s}{s + 1}/PL{str(game_id)[4:]:0>6}.HTM"


def _cells6(chunk):
    import html as _h
    cs = re.findall(r"<td[^>]*>(.*?)</td>", chunk, re.S)[:6]
    return [re.sub(r"\s+", " ", _h.unescape(re.sub(r"<.*?>", " ", c))).strip() for c in cs]


def parse_html_events(text):
    """-> list of (period:int, event, team_token, number:int) for SHOT / GOAL / MISS / BLOCK rows (the SHOOTER is credited, never the blocker); period > 4 (shootout) is excluded."""
    out, unparsed = [], []
    for chunk in re.split(r'<tr[^>]*class="\s*(?:odd|even)Color"[^>]*>', text)[1:]:      # row markup differs across seasons (id="PL-n" vs a tab-prefixed class)
        c = _cells6(chunk)
        if len(c) < 6 or c[4] not in ("SHOT", "GOAL", "MISS", "BLOCK"):
            continue
        try:
            period = int(c[1])
        except ValueError:
            unparsed.append(c); continue
        if period > 4:
            continue                                                   # shootout (regular season has one OT period = period 4)
        d = c[5]
        m = re.match(r"^([A-Z.]+) ONGOAL - #(\d+)\b", d) if c[4] == "SHOT" else re.match(r"^([A-Z.]+) #(\d+)\b", d)
        if not m:
            unparsed.append(c); continue
        out.append((period, c[4], m.group(1), int(m.group(2))))
    return out, unparsed


def roster_map(pbp):
    away, home = pbp["awayTeam"], pbp["homeTeam"]
    tokens = {away["abbrev"]: away["id"], home["abbrev"]: home["id"]}
    by_num = {(s["teamId"], s["sweaterNumber"]): s["playerId"] for s in pbp.get("rosterSpots", []) if "sweaterNumber" in s}
    return tokens, by_num


def html_player_counts(text, pbp):
    events, unparsed = parse_html_events(text)
    tokens, by_num = roster_map(pbp)
    counts, unmapped = defaultdict(lambda: {"sog": 0, "missed": 0, "blocked": 0}), []
    for period, ev, tok, num in events:
        t = tok.replace(".", "")
        abbr = HTML_TEAM_FIX.get(t, t)
        tid = tokens.get(abbr)
        pid = by_num.get((tid, num)) if tid is not None else None
        if pid is None:
            unmapped.append({"event": ev, "team_token": tok, "number": num, "period": period}); continue
        k = {"SHOT": "sog", "GOAL": "sog", "MISS": "missed", "BLOCK": "blocked"}[ev]
        counts[pid][k] += 1
    return dict(counts), unmapped, unparsed


def step_d(game_ids, attempt_rows, frozen_sog, cache, pbp_cache, workers=6, log=print):
    def one(g):
        txt, meta = get_text(html_url(g), cache)
        return g, txt, meta
    with ThreadPoolExecutor(workers) as ex:
        fetched = {g: (t, m) for g, t, m in ex.map(one, game_ids)}
    by_game = defaultdict(dict)
    for (g, p), r in attempt_rows.items():
        if g in fetched:
            by_game[g][p] = r
    games_out, summary = {}, Counter()
    for g in sorted(game_ids):
        txt, meta = fetched[g]
        if txt is None:
            games_out[g] = {"error": meta.get("error"), "html_unavailable": True}; summary["html_unavailable_games"] += 1; continue
        pbp = json.loads((Path(pbp_cache) / (hashlib.sha256(f"{AT.API}/gamecenter/{g}/play-by-play".encode()).hexdigest() + ".bin")).read_bytes())
        hc, unmapped, unparsed = html_player_counts(txt, pbp)
        rec = {"html_sha256": meta["sha256"], "html_bytes": meta["bytes"], "unmapped_events": len(unmapped), "unparsed_rows": len(unparsed), "players": {}}
        if unmapped:
            rec["unmapped_sample"] = unmapped[:5]
        n_pg = 0
        for p, r in sorted(by_game[g].items()):
            h = hc.get(p, {"sog": 0, "missed": 0, "blocked": 0})
            off = frozen_sog[(g, p)]
            n_pg += 1
            summary["player_games"] += 1
            summary["html_sog_eq_official"] += int(h["sog"] == off)
            summary["html_sog_eq_pbp"] += int(h["sog"] == r["sog_from_pbp"])
            summary["html_missed_eq_pbp"] += int(h["missed"] == r["missed_attempts"])
            summary["html_blocked_eq_pbp"] += int(h["blocked"] == r["blocked_attempts"])
            if h["sog"] != off or h["missed"] != r["missed_attempts"] or h["blocked"] != r["blocked_attempts"]:
                rec["players"][str(p)] = {"official_sog": off, "pbp": [r["sog_from_pbp"], r["missed_attempts"], r["blocked_attempts"]], "html": [h["sog"], h["missed"], h["blocked"]]}
        extra = [p for p in hc if p not in by_game[g]]
        rec["html_players_not_in_frozen_skater_table"] = {str(p): hc[p] for p in extra}
        summary["games"] += 1; summary["unmapped_events"] += len(unmapped); summary["unparsed_rows"] += len(unparsed)
        games_out[g] = rec
    return games_out, dict(summary)
