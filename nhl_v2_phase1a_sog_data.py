"""NHL V2 Phase1A-SOG data layer (research only).

Frozen official-source rows (vendored V1 acquisition, pinned by sha256), data-quality gates and the leak-free prediction-row builder.
Feature definitions are the fixed B2 inputs ported from nhl_sog_phase1a_data.build_prediction_rows (burned V1 corpus); declared changes:
(1) the cutoff is parameterised by the forecast horizon, (2) source games with no loaded player rows are excluded from a team's history
(no-op on complete historical data, required for live data), (3) features can be built for a game without reading its target rows."""
import bisect
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_player_outcome_v2"
DATA = OUT / "phase1a_data"
UTC = timezone.utc
SEASONS = list(range(2017, 2026))
TARGET_SEASONS = list(range(2018, 2026))
GAME_MAX_MINUTES = 210
HORIZONS = {"T24H": 1440, "T90": 90, "T30": 30}
MAX_SOG = 60
REST_CAP = 240.0
FEATURES = ["POS_F", "POS_D", "POS_UNKNOWN", "IS_HOME", "TEAM_REST_HOURS", "BACK_TO_BACK", "PLAYED_LAST1", "PLAY_RATE_TG3", "PLAY_RATE_TG10", "PLAY_DEN_TG3", "PLAY_DEN_TG10", "TEAM_GAMES_SINCE_APPEARANCE",
            "DAYS_SINCE_LAST_APPEARANCE", "N_CURRENT_SEASON_TEAM_GAMES_OBS", "N_CURRENT_SEASON_PLAYER_APPEARANCES", "SOG_MEAN_APP5", "SOG_MEAN_APP10", "SOG_SD_APP10", "SOG_PER60_APP10", "N_SKILL_APPEARANCES_10",
            "TOI_MEAN_CT_APP3", "TOI_MEAN_CT_APP10", "TOI_DELTA_CT_3_10", "PP_TOI_MEAN_CT_APP3", "PP_ALLOC_SHARE_MEAN_CT_APP3", "PP_ALLOC_SHARE_MEAN_CT_APP10", "PP_ALLOC_SHARE_DELTA_CT_3_10", "SHIFT_MEAN_CT_APP3",
            "SHIFT_MEAN_CT_APP10", "SHIFT_DELTA_CT_3_10", "N_ROLE_APPEARANCES_10", "TEAM_SOG_FOR_MEAN5", "OPP_SOG_ALLOWED_MEAN5"]
BINARY = {"POS_F", "POS_D", "POS_UNKNOWN", "IS_HOME", "BACK_TO_BACK", "PLAYED_LAST1"}
def _registered_banned_tokens():
    proto = json.loads((OUT / "protocol.json").read_text())
    return tuple(next(v for k, v in proto.items() if k.endswith("_firewall"))["banned_tokens"]) + ("spread", "price", "projection", "line")


BANNED_COLUMN_WORDS = _registered_banned_tokens()


def epoch(s):
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp())


def read_jsonl_gz(p):
    return [json.loads(l) for l in gzip.decompress(Path(p).read_bytes()).decode().splitlines() if l]


def load_frozen(data_dir=DATA):
    games, rows = {}, []
    for s in SEASONS:
        for g in read_jsonl_gz(Path(data_dir) / ("games_%d.jsonl.gz" % s)):
            games[g["game_id"]] = g
        rows += read_jsonl_gz(Path(data_dir) / ("skater_games_%d.jsonl.gz" % s))
    return games, rows


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def nanmean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else math.nan


def pos_class(code):
    return "F" if code in ("C", "L", "R", "F", "W") else "D" if code == "D" else "U"


def build_rows(games, rows, target_gids=None, target_seasons=TARGET_SEASONS, horizon_min=90, labels=True):
    """Candidate rows for every target team-game. Features use only rows of games with start <= target_start - (horizon + 210) min.
    labels=False never reads target rows (forward mode: target rows do not exist). Returns (table, coverage)."""
    back = (horizon_min + GAME_MAX_MINUTES) * 60
    gstart = {gid: epoch(g["game_start_utc"]) for gid, g in games.items()}
    team_games = defaultdict(list)
    for gid, g in games.items():
        for tid in (g["home_team_id"], g["away_team_id"]):
            team_games[tid].append((gstart[gid], gid))
    for t in team_games:
        team_games[t].sort()
    team_starts = {t: [x[0] for x in v] for t, v in team_games.items()}
    game_players = {}
    team_sog = defaultdict(int); team_pp = defaultdict(int)
    for r in rows:
        game_players.setdefault((r["game_id"], r["team_id"]), set()).add(r["player_id"])
        team_sog[(r["game_id"], r["team_id"])] += r["sog"]
        team_pp[(r["game_id"], r["team_id"])] += r["pp_toi_sec"] or 0
    sog_against = {}
    for gid, g in games.items():
        h, a = g["home_team_id"], g["away_team_id"]
        sog_against[(gid, h)] = team_sog[(gid, a)]; sog_against[(gid, a)] = team_sog[(gid, h)]
    apps = defaultdict(list)
    for r in rows:
        s = gstart[r["game_id"]]
        share = (r["pp_toi_sec"] / team_pp[(r["game_id"], r["team_id"])]) if (r["pp_toi_sec"] is not None and team_pp[(r["game_id"], r["team_id"])] > 0) else None
        apps[r["player_id"]].append((s, r["game_id"], r["team_id"], r["position"], r["sog"], r["toi_sec"], r["pp_toi_sec"], r["shifts"], share))
    for p in apps:
        apps[p].sort(key=lambda a: (a[0], a[1]))
    app_starts = {p: [a[0] for a in v] for p, v in apps.items()}
    by_game_team = defaultdict(list)
    if labels:
        for r in rows:
            by_game_team[(r["game_id"], r["team_id"])].append(r)
    cols = {k: [] for k in ("game_id", "team_id", "opp_id", "player_id", "season", "start", "cutoff", "played", "sog", "plays10", "den10", "hist10")}
    feats = {k: [] for k in FEATURES}
    cov = {s: {"team_games": 0, "candidate_rows": 0, "played_rows": 0, "actual_target_skaters": 0, "observable": 0, "unobservable": 0, "actual_SOG": 0, "observable_SOG": 0, "unobservable_diagnostics": defaultdict(int)} for s in target_seasons}
    order = sorted(target_gids, key=lambda g: (gstart[g], g)) if target_gids is not None else sorted(games, key=lambda g: (gstart[g], g))
    for gid in order:
        g = games[gid]
        season = int(str(gid)[:4])
        if target_gids is None and season not in target_seasons:
            continue
        s0 = gstart[gid]; limit = s0 - back
        for team, opp, home in ((g["home_team_id"], g["away_team_id"], 1), (g["away_team_id"], g["home_team_id"], 0)):
            ti = bisect.bisect_right(team_starts[team], limit)
            tg = [x for x in team_games[team][:ti] if (x[1], team) in game_players]          # declared change (2): games with loaded rows only
            last10 = [x[1] for x in tg[-10:]]
            cand = sorted({p for gg in last10 for p in game_players[(gg, team)]})
            pos_idx = team_games[team].index((s0, gid)) if (s0, gid) in team_games[team] else None
            prev_start = team_games[team][pos_idx - 1][0] if pos_idx else None
            rest = (s0 - prev_start) / 3600.0 if prev_start else math.nan
            tf5 = nanmean([team_sog[(x[1], team)] for x in tg[-5:]]) if tg else math.nan
            oi = bisect.bisect_right(team_starts[opp], limit)
            og = [x for x in team_games[opp][:oi] if (x[1], opp) in game_players]
            oa5 = nanmean([sog_against[(x[1], opp)] for x in og[-5:]]) if og else math.nan
            n_season_tg = sum(1 for x in tg if int(str(x[1])[:4]) == season)
            for pid in cand:
                pa = apps[pid]; k = bisect.bisect_right(app_starts[pid], limit)
                allowed = pa[:k]
                ct = [a for a in allowed if a[2] == team]
                last_ct = ct[-1]
                pos = pos_class(last_ct[3])
                plays = [1 if pid in game_players.get((gg, team), ()) else 0 for gg in last10]
                den10 = len(last10); den3 = len(last10[-3:])
                plays3 = sum(plays[-3:]); plays10 = sum(plays)
                since = 0
                for gg in reversed([x[1] for x in tg]):
                    if pid in game_players.get((gg, team), ()):
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
                     "TOI_MEAN_CT_APP3": toi3, "TOI_MEAN_CT_APP10": toi10, "TOI_DELTA_CT_3_10": d(toi3, toi10), "PP_TOI_MEAN_CT_APP3": pp3, "PP_ALLOC_SHARE_MEAN_CT_APP3": al3, "PP_ALLOC_SHARE_MEAN_CT_APP10": al10,
                     "PP_ALLOC_SHARE_DELTA_CT_3_10": d(al3, al10), "SHIFT_MEAN_CT_APP3": sh3, "SHIFT_MEAN_CT_APP10": sh10, "SHIFT_DELTA_CT_3_10": d(sh3, sh10), "N_ROLE_APPEARANCES_10": float(len(c10)),
                     "TEAM_SOG_FOR_MEAN5": tf5, "OPP_SOG_ALLOWED_MEAN5": oa5}
                for kf in FEATURES:
                    feats[kf].append(f[kf])
                cols["hist10"].append(sogs + [-1] * (10 - len(sogs)))
                cols["plays10"].append(plays10); cols["den10"].append(den10)
                for kk, vv in (("game_id", gid), ("team_id", team), ("opp_id", opp), ("player_id", pid), ("season", season), ("start", s0), ("cutoff", s0 - horizon_min * 60)):
                    cols[kk].append(vv)
                cols["played"].append(0); cols["sog"].append(0)
            if not labels:
                continue
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
    return (np.asarray(start_sec) - 345600) // 604800


def toi_classes(rows):
    out = {}
    for s in SEASONS:
        rs = [r for r in rows if r["season_start_year"] == s]
        full = [r for r in rs if all(r[k] is not None for k in ("toi_sec", "ev_toi_sec", "pp_toi_sec", "sh_toi_sec"))]
        incons = [r for r in full if abs(r["ev_toi_sec"] + r["pp_toi_sec"] + r["sh_toi_sec"] - r["toi_sec"]) > 1 and abs(r["ev_toi_sec"] + r["pp_toi_sec"] + r["sh_toi_sec"] + (r["ot_toi_sec"] or 0) - r["toi_sec"]) > 1]
        out[str(s)] = {"rows": len(rs), "all_components_present": len(full), "missing_components": len(rs) - len(full), "present_but_inconsistent": len(incons),
                       "inconsistent_fraction_of_fully_observed": round(len(incons) / len(full), 6) if full else None, "pass": (len(incons) / len(full) if full else 0) <= 0.002}
    return out


def quality_gates(games, rows, manifest):
    g = {}
    keys = [(r["game_id"], r["team_id"], r["player_id"]) for r in rows]
    g["unique_game_team_player_rows"] = len(keys) == len(set(keys))
    g["every_row_game_in_schedule"] = all(r["game_id"] in games for r in rows)
    g["every_scheduled_game_has_rows"] = len({r["game_id"] for r in rows}) == len(games)
    g["sog_nonnegative_integer"] = all(isinstance(r["sog"], int) and r["sog"] >= 0 for r in rows)
    g["sog_le_60"] = max(r["sog"] for r in rows) <= MAX_SOG
    g["toi_shift_nonnegative_where_present"] = all((r[k] is None or r[k] >= 0) for r in rows for k in ("toi_sec", "ev_toi_sec", "pp_toi_sec", "sh_toi_sec", "shifts"))
    g["player_id_numeric"] = all(isinstance(r["player_id"], int) for r in rows)
    g["no_betting_market_columns"] = not any(any(w in k.lower() for w in BANNED_COLUMN_WORDS) for k in rows[0].keys())
    g["ordered_by_utc_start"] = all(rows[i]["game_start_utc"] <= rows[i + 1]["game_start_utc"] for i in range(len(rows) - 1))
    g["positions_known_codes"] = {r["position"] for r in rows} <= {"C", "L", "R", "D", None}
    g["manifest_hashes_roundtrip"] = all(sha256_file(DATA / n) == f["sha256"] for n, f in manifest["files"].items())
    g["v1_acquisition_guard_failures_none"] = manifest["v1_acquisition"]["guard_failures"] == []
    return g
