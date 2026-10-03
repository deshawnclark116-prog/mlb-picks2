"""
NHL_SOG_PHASE1B_ATTEMPTS -- Phase 1B-A: deterministic bounded target sample, minimal PBP acquisition, derived per player-game attempt table, PBP quality checks and the four registered lagged attempt features.
Registered in phase1b_attempt_signal_protocol.json. Targets 2018-2023 only; 2024/2025 are never read as targets here. No model code in this file.
"""
import bisect
import gzip
import hashlib
import io
import json
import math
import random
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import nhl_outcome_contract as CT
import nhl_sog_phase1a_data as D

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_outcome_engine"
ADATA = OUT / "phase1b_attempt_data"
API = D.API
SAMPLE_TAG = "nhl-phase1b-a-v1"
SAMPLE_SEASONS = [2018, 2019, 2020, 2021, 2022, 2023]
ALLOWED_TARGET_SEASONS = tuple(SAMPLE_SEASONS)
PER_SEASON = 200
NEW_FEATURES = ["ATTEMPTS_MEAN_APP5", "ATTEMPTS_MEAN_APP10", "ATTEMPTS_PER60_APP10", "ON_NET_RATE_APP10"]
MAX_UNOBTAINABLE_FRACTION = 0.005
UTC = timezone.utc


def assert_allowed_seasons(seasons):
    bad = [s for s in seasons if s not in ALLOWED_TARGET_SEASONS]
    if bad:
        raise ValueError(f"seasons {bad} are excluded from Phase 1B-A (2018-2023 targets only)")


# ------------------------------------------------------------------ sampling (outcome independent)
def sample_key(game_id):
    return hashlib.sha256(f"{SAMPLE_TAG}|{game_id}".encode()).hexdigest()


def sample_games(games, seasons=SAMPLE_SEASONS, per_season=PER_SEASON):
    """games: {game_id: {...}} or iterable of game ids. Uses ONLY the game id (never an outcome). Returns {season: [game_id,...] in hash order}."""
    assert_allowed_seasons(seasons)
    ids = list(games)
    out = {}
    for s in seasons:
        pool = [g for g in ids if int(str(g)[:4]) == s]
        pool.sort(key=lambda g: (sample_key(g), g))
        out[s] = pool[:per_season]
    return out


def month_distribution(games, sample):
    out = {}
    for s, ids in sample.items():
        c = Counter(games[g]["game_start_utc"][:7] for g in ids)
        allc = Counter(g["game_start_utc"][:7] for gid, g in games.items() if int(str(gid)[:4]) == s)
        out[s] = {"sampled_by_month": dict(sorted(c.items())), "all_games_by_month": dict(sorted(allc.items())), "n_sampled": len(ids), "n_all": sum(allc.values())}
    return out


# ------------------------------------------------------------------ PBP parsing
def pbp_counts(events):
    """{player_id: {'sog','missed','blocked','attempts'}} from PBP events: shot-on-goal + goal (SOG), missed-shot, blocked-shot (shooter, never the blocker); shootout (periodType SO) excluded. Order invariant."""
    c = CT.shot_counts_from_events(events)
    out = {}
    for pid, cnt in c.items():
        out[pid] = {"sog": CT.sog(cnt), "missed": cnt["missed-shot"], "blocked": cnt["blocked-shot"], "attempts": CT.attempts(cnt)}
    return out


def shootout_events_excluded(events):
    return sum(1 for e in events if e.get("typeDescKey") in ("shot-on-goal", "goal", "missed-shot", "blocked-shot") and (e.get("periodDescriptor") or {}).get("periodType") == "SO")


def game_player_rows(game_id, pbp, frozen_players):
    """One row per frozen skater of the game. Returns (rows, extras): extras = shooters absent from the frozen skater table, classified goalie / non_goalie."""
    cnt = pbp_counts(pbp.get("plays", []))
    goalies = {s["playerId"] for s in pbp.get("rosterSpots", []) if s.get("positionCode") == "G"}
    rows = []
    for pid in sorted(frozen_players):
        c = cnt.get(pid, {"sog": 0, "missed": 0, "blocked": 0, "attempts": 0})
        rows.append({"game_id": game_id, "player_id": pid, "sog_from_pbp": c["sog"], "missed_attempts": c["missed"], "blocked_attempts": c["blocked"], "shot_attempts": c["attempts"]})
    extras = {"goalie_shooters": sorted(p for p in cnt if p not in frozen_players and p in goalies), "non_goalie_unmatched_shooters": sorted(p for p in cnt if p not in frozen_players and p not in goalies and cnt[p]["sog"] > 0),
              "other_unmatched_attempt_only": sorted(p for p in cnt if p not in frozen_players and p not in goalies and cnt[p]["sog"] == 0)}
    return rows, extras


# ------------------------------------------------------------------ history windows (mirror of the Phase 1A builder rule)
def build_appearances(rows):
    gstart = {}
    apps = defaultdict(list)
    for r in rows:
        s = D.epoch(r["game_start_utc"])
        apps[r["player_id"]].append((s, r["game_id"], r["toi_sec"], r["sog"]))
    for p in apps:
        apps[p].sort(key=lambda a: (a[0], a[1]))
    return apps, {p: [a[0] for a in v] for p, v in apps.items()}


def window_for(apps, app_starts, pid, target_start, target_game_id, n=10):
    """The last <=n all-team appearances of `pid` with source_start <= target_start - 300 min (source_start + 210 min <= T). The target game is excluded by construction (and asserted)."""
    limit = target_start - D.CUTOFF_BACK_S
    k = bisect.bisect_right(app_starts.get(pid, []), limit)
    w = apps.get(pid, [])[:k][-n:]
    assert all(a[1] != target_game_id and a[0] <= limit for a in w), "target game / future game leaked into a history window"
    return w


def required_source_games(tab, apps, app_starts):
    need = set()
    for gid, pid, st in zip(tab["game_id"], tab["player_id"], tab["start"]):
        for a in window_for(apps, app_starts, int(pid), int(st), int(gid)):
            need.add(a[1])
    return need


# ------------------------------------------------------------------ the four registered features
def _mean(xs):
    return sum(xs) / len(xs) if xs else math.nan


def attempt_features_for_window(window, attempts):
    """window: list of (start, game_id, toi_sec, sog) ordered chronologically (last <=10 appearances); attempts: {(game_id, player_id) or game_id-keyed lookup via callable}.
    `attempts(game_id)` -> dict with shot_attempts / sog_from_pbp for THIS player-game or None when unavailable. Returns the four features (NaN = missing)."""
    have = [(a, attempts(a[1])) for a in window]
    have = [(a, r) for a, r in have if r is not None]                                  # appearances with attempt data
    att = [r["shot_attempts"] for a, r in have]
    sogs = [r["sog_official"] if "sog_official" in r else r["sog_from_pbp"] for a, r in have]
    f5 = _mean(att[-5:])
    f10 = _mean(att)
    usable = [(r["shot_attempts"], a[2]) for a, r in have if a[2] and a[2] > 0]                # TOI present and > 0
    tsum = sum(t for _, t in usable)
    per60 = 3600.0 * sum(x for x, _ in usable) / tsum if usable and tsum > 0 else math.nan  # ratio of sums over the SAME usable appearances
    tot_att = sum(att)
    on_net = (sum(sogs) / tot_att) if tot_att > 0 else math.nan                          # zero attempts -> missing, never zero
    return {"ATTEMPTS_MEAN_APP5": f5, "ATTEMPTS_MEAN_APP10": f10, "ATTEMPTS_PER60_APP10": per60, "ON_NET_RATE_APP10": on_net}


def build_attempt_features(tab, apps, app_starts, attempt_rows):
    """attempt_rows: {(game_id, player_id): row}. Adds the four feature arrays (same row order as `tab`) -> dict."""
    out = {k: np.full(len(tab["game_id"]), np.nan) for k in NEW_FEATURES}
    for i, (gid, pid, st) in enumerate(zip(tab["game_id"], tab["player_id"], tab["start"])):
        pid = int(pid)
        w = window_for(apps, app_starts, pid, int(st), int(gid))
        f = attempt_features_for_window(w, lambda g, pid=pid: attempt_rows.get((g, pid)))
        for k in NEW_FEATURES:
            out[k][i] = f[k]
    return out


# ------------------------------------------------------------------ acquisition
def fetch_pbp(game_id, cache, retries=4):
    url = f"{API}/gamecenter/{game_id}/play-by-play"
    key = hashlib.sha256(url.encode()).hexdigest()
    f, m = Path(cache) / f"{key}.bin", Path(cache) / f"{key}.meta.json"
    if f.exists() and m.exists():
        return f.read_bytes(), json.loads(m.read_text()), True
    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nhl-phase1b-a"}), timeout=120)
            raw = r.read()
            meta = {"game_id": game_id, "url": url, "retrieved_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "seconds": round(time.time() - t0, 3), "http_status": r.status}
            f.write_bytes(raw); m.write_text(json.dumps(meta))
            return raw, meta, False
        except Exception as e:                                                      # noqa
            last = e
            if getattr(e, "code", None) in (404, 410):
                break
            time.sleep(2 * (attempt + 1))
    return None, {"game_id": game_id, "url": url, "http_status": getattr(last, "code", None), "error": f"{type(last).__name__}: {last}"[:200], "unobtainable": True}, False


def acquire(game_ids, cache, workers=8, log=print):
    Path(cache).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ids = sorted(game_ids)
    results = {}
    done = [0]

    def one(g):
        raw, meta, cached = fetch_pbp(g, cache)
        done[0] += 1
        if done[0] % 250 == 0:
            log(f"  fetched {done[0]}/{len(ids)} t={time.time() - t0:.0f}s")
        return g, raw, meta, cached
    with ThreadPoolExecutor(workers) as ex:
        for g, raw, meta, cached in ex.map(one, ids):
            results[g] = (raw, meta, cached)
    return results, round(time.time() - t0, 1)


# ------------------------------------------------------------------ derived dataset + quality gate
def derive(pbp_results, frozen_by_game):
    """-> (attempt_rows list, per-game info, quality summary). frozen_by_game: {game_id: {player_id: official_sog}}."""
    rows, per_game, mism, ex_goalie, ex_other = [], {}, [], 0, []
    q = {"games_checked": 0, "player_games_checked": 0, "exact_matches": 0, "mismatches": 0, "mismatch_examples": [], "shootout_events_excluded": 0, "goalie_shooters_excluded": 0, "non_goalie_unmatched_shooters": [],
         "shot_attempts_ge_sog_violations": 0, "negative_or_noninteger_violations": 0, "duplicate_rows": 0, "order_invariance_failures": 0, "unobtainable_games": [], "games_with_events": 0}
    seen = set()
    for g in sorted(pbp_results):
        raw, meta, _ = pbp_results[g]
        if raw is None:
            q["unobtainable_games"].append(g); continue
        pbp = json.loads(raw)
        events = pbp.get("plays", [])
        rs, extras = game_player_rows(g, pbp, set(frozen_by_game[g]))
        shuf = list(events); random.Random(g).shuffle(shuf)                                  # order-invariance check with a deterministic shuffle
        if pbp_counts(shuf) != pbp_counts(events):
            q["order_invariance_failures"] += 1
        q["games_checked"] += 1
        q["games_with_events"] += int(len(events) > 0)
        q["shootout_events_excluded"] += shootout_events_excluded(events)
        q["goalie_shooters_excluded"] += len(extras["goalie_shooters"])
        for pid in extras["non_goalie_unmatched_shooters"]:
            q["non_goalie_unmatched_shooters"].append({"game_id": g, "player_id": pid})
        for r in rs:
            k = (r["game_id"], r["player_id"])
            if k in seen:
                q["duplicate_rows"] += 1
            seen.add(k)
            q["player_games_checked"] += 1
            official = frozen_by_game[g][r["player_id"]]
            if r["sog_from_pbp"] == official:
                q["exact_matches"] += 1
            else:
                q["mismatches"] += 1
                if len(q["mismatch_examples"]) < 25:
                    q["mismatch_examples"].append({"game_id": g, "player_id": r["player_id"], "pbp_sog": r["sog_from_pbp"], "official_sog": official})
            if r["shot_attempts"] < r["sog_from_pbp"]:
                q["shot_attempts_ge_sog_violations"] += 1
            comps = (r["sog_from_pbp"], r["missed_attempts"], r["blocked_attempts"], r["shot_attempts"])
            if any((not isinstance(x, int)) or x < 0 for x in comps) or r["shot_attempts"] != r["sog_from_pbp"] + r["missed_attempts"] + r["blocked_attempts"]:
                q["negative_or_noninteger_violations"] += 1
            rows.append(r)
        per_game[g] = {"n_players": len(rs), "n_events": len(events)}
    q["mismatches"] += len(q["non_goalie_unmatched_shooters"])                              # an unmatched non-goalie SOG shooter is a genuine attribution mismatch
    q["unobtainable_fraction"] = len(q["unobtainable_games"]) / max(1, len(pbp_results))
    return rows, per_game, q


def quality_pass(q):
    return bool(q["mismatches"] == 0 and q["shot_attempts_ge_sog_violations"] == 0 and q["negative_or_noninteger_violations"] == 0 and q["duplicate_rows"] == 0 and q["order_invariance_failures"] == 0
                and q["unobtainable_fraction"] <= MAX_UNOBTAINABLE_FRACTION and q["games_checked"] > 0)


def det_gzip_text(text):
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", mtime=0, compresslevel=9) as g:
        g.write(text.encode())
    return bio.getvalue()


def write_frozen(attempt_rows, provenance, out_dir=ADATA):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    by_season = defaultdict(list)
    for r in sorted(attempt_rows, key=lambda r: (r["game_id"], r["player_id"])):
        by_season[int(str(r["game_id"])[:4])].append(r)
    for s, rs in sorted(by_season.items()):
        b = det_gzip_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rs))
        n = f"attempts_{s}.jsonl.gz"; (out_dir / n).write_bytes(b)
        files[n] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "rows": len(rs), "season": s, "games": len({r["game_id"] for r in rs})}
    pb = det_gzip_text("".join(json.dumps(p, sort_keys=True, separators=(",", ":")) + "\n" for p in sorted(provenance, key=lambda p: p["game_id"])))
    (out_dir / "provenance.jsonl.gz").write_bytes(pb)
    files["provenance.jsonl.gz"] = {"sha256": hashlib.sha256(pb).hexdigest(), "bytes": len(pb), "rows": len(provenance)}
    return files


def load_attempt_rows(out_dir=ADATA, pattern="attempts_*.jsonl.gz"):
    out = {}
    for p in sorted(Path(out_dir).glob(pattern)):
        for l in gzip.decompress(p.read_bytes()).decode().splitlines():
            if l:
                r = json.loads(l)
                out[(r["game_id"], r["player_id"])] = r
    return out
