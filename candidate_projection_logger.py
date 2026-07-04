"""
candidate_projection_logger.py

Prop Edge v8.17A Candidate Projection Logger

Purpose:
- Start building real training data for model upgrades.
- Logs official picks AND rejected/candidate picks.
- Grades candidate picks later against actual results.
- Does not change api.py.
- Does not change model predictions.
- Does not retrain anything.

Reads:
  /data/predictions/predictions_YYYY-MM-DD.json
  /data/predictions/hitter_candidates_YYYY-MM-DD.json

Writes:
  /data/candidate_logs/candidates_YYYY-MM-DD.json
  /data/candidate_logs/graded_candidates_YYYY-MM-DD.json
  /data/candidate_logs/candidate_record.json

Usage:
  python candidate_projection_logger.py --snapshot-today
  python candidate_projection_logger.py --grade-days 20
  python candidate_projection_logger.py --all
  python candidate_projection_logger.py --report
"""

import argparse
import datetime as dt
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import requests


VERSION = "8.17A-candidate-logger"

ET = ZoneInfo("America/New_York")
DATA_DIR = Path("/data")
PRED_DIR = DATA_DIR / "predictions"
LOG_DIR = DATA_DIR / "candidate_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MLB = "https://statsapi.mlb.com/api/v1"
MLB11 = "https://statsapi.mlb.com/api/v1.1"

S = requests.Session()
S.headers["User-Agent"] = f"prop-edge/{VERSION}"

PROP_STAT = {
    "batter_hits": ("hitting", "hits"),
    "pitcher_strikeouts": ("pitching", "strikeOuts"),
    "batter_total_bases": ("hitting", "totalBases"),
    "batter_rbis": ("hitting", "rbi"),
    "batter_runs": ("hitting", "runs"),
    "batter_home_runs": ("hitting", "homeRuns"),
}


def now_et():
    return dt.datetime.now(ET)


def today_et():
    return now_et().date()


def norm(s):
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = s.replace(".", " ")
    s = s.replace("-", " ")
    s = s.replace("'", " ")
    s = s.replace("’", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = "".join(c for c in s if c.isalpha() or c == " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def load_json(path, fallback=None):
    if fallback is None:
        fallback = []
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        print(f"Could not read {path}: {e}")
    return fallback


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def get(url, **params):
    for attempt in range(3):
        try:
            r = S.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"retry {url}: {e}")
            time.sleep(1.2 * (attempt + 1))
    return {}


def player_index(season):
    data = get(f"{MLB}/sports/1/players", season=season)
    idx = {}
    for p in data.get("people", []):
        name = norm(p.get("fullName", ""))
        if name:
            idx[name] = p.get("id")
    return idx


def final_game_pks(target_date):
    data = get(f"{MLB}/schedule", sportId=1, date=target_date)
    final = set()

    for day in data.get("dates", []):
        for g in day.get("games", []):
            state = g.get("status", {}).get("abstractGameState", "")
            detailed = g.get("status", {}).get("detailedState", "")
            if state == "Final" or detailed in ("Final", "Game Over", "Completed Early"):
                final.add(str(g.get("gamePk")))

    return final


def get_game_final_score(game_pk):
    data = get(f"{MLB11}/game/{game_pk}/feed/live")
    try:
        linescore = data.get("liveData", {}).get("linescore", {})
        teams = data.get("gameData", {}).get("teams", {})

        home_runs = linescore.get("teams", {}).get("home", {}).get("runs")
        away_runs = linescore.get("teams", {}).get("away", {}).get("runs")

        home_name = teams.get("home", {}).get("name", "")
        away_name = teams.get("away", {}).get("name", "")

        if home_runs is None or away_runs is None:
            return None

        return int(home_runs), int(away_runs), home_name, away_name
    except Exception:
        return None


def get_actual_stat(player_id, group, field, target_date, season):
    data = get(f"{MLB}/people/{player_id}/stats", stats="gameLog", group=group, season=season)

    try:
        for sp in reversed(data["stats"][0]["splits"]):
            if sp.get("date") == target_date:
                return float(sp["stat"].get(field, 0) or 0)
    except Exception:
        pass

    return None


def parse_pick_side(pick, prop_type=None):
    pick = str(pick or "").upper().strip()

    if prop_type == "moneyline":
        return "ML"
    if pick.startswith("OVER"):
        return "OVER"
    if pick.startswith("UNDER"):
        return "UNDER"
    if "+" in pick:
        return "THRESHOLD"

    return "UNKNOWN"


def parse_pick_line(pick):
    pick = str(pick or "").upper().strip()

    m = re.search(r"(?:OVER|UNDER)\s+(-?\d+(?:\.\d+)?)", pick)
    if m:
        return safe_float(m.group(1))

    m2 = re.search(r"^(\d+)\+", pick)
    if m2:
        return safe_float(m2.group(1))

    return None


def candidate_key(row):
    return "|".join([
        str(row.get("date", "")),
        str(row.get("game_id", "")),
        str(row.get("player_id") or norm(row.get("player", ""))),
        str(row.get("prop_type", "")),
        str(row.get("pick", "")),
    ])


def clean_candidate(row, date_str, source):
    r = dict(row)

    status = r.get("board_status")
    if not status:
        status = "official" if source == "official_board" else "candidate"

    r["date"] = r.get("date") or date_str
    r["candidate_logger_version"] = VERSION
    r["candidate_logged_at"] = now_et().isoformat()
    r["candidate_source"] = source
    r["candidate_status"] = status
    r["official_board"] = source == "official_board" or status == "official"
    r["candidate_key"] = candidate_key(r)

    r["pick_side"] = parse_pick_side(r.get("pick"), r.get("prop_type"))
    r["pick_line"] = parse_pick_line(r.get("pick"))

    if "projected" in r and r.get("projected") is not None:
        r["raw_projected"] = r.get("projected")

    if "model_prob" in r and r.get("model_prob") is not None:
        r["raw_model_prob"] = r.get("model_prob")

    return r


def snapshot_candidates(date_str=None):
    if date_str is None:
        date_str = today_et().isoformat()

    official_path = PRED_DIR / f"predictions_{date_str}.json"
    hitter_path = PRED_DIR / f"hitter_candidates_{date_str}.json"

    official = load_json(official_path, [])
    hitter_candidates = load_json(hitter_path, [])

    if not isinstance(official, list):
        official = []
    if not isinstance(hitter_candidates, list):
        hitter_candidates = []

    merged = {}

    for row in hitter_candidates:
        if not isinstance(row, dict):
            continue

        c = clean_candidate(row, date_str, "hitter_candidates")
        merged[c["candidate_key"]] = c

    for row in official:
        if not isinstance(row, dict):
            continue

        c = clean_candidate(row, date_str, "official_board")
        key = c["candidate_key"]

        if key in merged:
            old = merged[key]
            old["official_board"] = True
            old["candidate_status"] = "official"
            old["candidate_source"] = "hitter_candidates+official_board"
            old["official_api_version"] = c.get("api_version")
            old["official_model_prob"] = c.get("model_prob")
            old["official_projected"] = c.get("projected")
            merged[key] = old
        else:
            merged[key] = c

    candidates = list(merged.values())
    candidates.sort(
        key=lambda r: (
            str(r.get("game_id", "")),
            str(r.get("team", "")),
            str(r.get("prop_type", "")),
            str(r.get("player", "")),
        )
    )

    out_path = LOG_DIR / f"candidates_{date_str}.json"
    save_json(out_path, candidates)

    summary = summarize_candidate_snapshot(candidates)

    latest_path = LOG_DIR / "latest_candidate_snapshot.json"
    save_json(latest_path, {
        "version": VERSION,
        "date": date_str,
        "generated_at": now_et().isoformat(),
        "source_files": {
            "official": str(official_path),
            "hitter_candidates": str(hitter_path),
        },
        "summary": summary,
        "candidates": candidates,
    })

    print(f"Saved candidate snapshot: {out_path}")
    print(json.dumps(summary, indent=2))
    return candidates


def summarize_candidate_snapshot(rows):
    by_status = defaultdict(int)
    by_prop = defaultdict(int)
    by_prop_status = defaultdict(int)
    by_reject_reason = defaultdict(int)

    for r in rows:
        prop = r.get("prop_type", "unknown")
        status = r.get("candidate_status", "unknown")
        reject = r.get("reject_reason") or "none"

        by_status[status] += 1
        by_prop[prop] += 1
        by_prop_status[f"{prop}|{status}"] += 1
        by_reject_reason[reject] += 1

    return {
        "total_candidates": len(rows),
        "official_board_candidates": sum(1 for r in rows if r.get("official_board")),
        "by_status": dict(sorted(by_status.items())),
        "by_prop": dict(sorted(by_prop.items())),
        "by_prop_status": dict(sorted(by_prop_status.items())),
        "by_reject_reason": dict(sorted(by_reject_reason.items())),
    }


def grade_candidate(row, target_date, player_idx, final_games, score_cache):
    if not isinstance(row, dict):
        return None

    game_id = str(row.get("game_id", ""))
    if game_id not in final_games:
        return None

    prop = row.get("prop_type")
    pick = str(row.get("pick", ""))

    actual = None
    result = None
    season = int(target_date[:4])

    if prop == "moneyline":
        if game_id not in score_cache:
            score_cache[game_id] = get_game_final_score(game_id)

        score = score_cache.get(game_id)
        if not score:
            return None

        home_r, away_r, home_name, away_name = score
        picked_team = norm(row.get("team", ""))
        home_norm = norm(home_name)
        away_norm = norm(away_name)

        if picked_team == home_norm:
            actual = 1 if home_r > away_r else 0
            result = "hit" if home_r > away_r else "miss"
        elif picked_team == away_norm:
            actual = 1 if away_r > home_r else 0
            result = "hit" if away_r > home_r else "miss"
        else:
            return None

    elif prop in PROP_STAT:
        group, field = PROP_STAT[prop]

        player_id = row.get("player_id")
        if not player_id:
            player_id = player_idx.get(norm(row.get("player", "")))

        if not player_id:
            return None

        actual = get_actual_stat(player_id, group, field, target_date, season)
        if actual is None:
            return None

        side = parse_pick_side(pick, prop)
        line = parse_pick_line(pick)

        if side == "THRESHOLD":
            if line is None:
                return None
            result = "hit" if actual >= line else "miss"

        elif side == "OVER":
            if line is None:
                return None
            result = "hit" if actual > line else "miss"

        elif side == "UNDER":
            if line is None:
                return None
            result = "hit" if actual < line else "miss"

        else:
            return None

    elif prop == "total":
        if game_id not in score_cache:
            score_cache[game_id] = get_game_final_score(game_id)

        score = score_cache.get(game_id)
        if not score:
            return None

        home_r, away_r, _, _ = score
        actual = home_r + away_r

        side = parse_pick_side(pick, prop)
        line = parse_pick_line(pick)

        if line is None:
            return None

        if side == "OVER":
            result = "hit" if actual > line else "miss"
        elif side == "UNDER":
            result = "hit" if actual < line else "miss"
        else:
            return None

    else:
        return None

    out = dict(row)
    out["graded_at"] = now_et().isoformat()
    out["actual"] = actual
    out["result"] = result

    projected = safe_float(out.get("projected"))
    if projected is not None and actual is not None and prop != "moneyline":
        out["projection_error"] = round(projected - actual, 3)
        out["abs_projection_error"] = round(abs(projected - actual), 3)

    return out


def grade_candidates_for_date(date_str):
    cand_path = LOG_DIR / f"candidates_{date_str}.json"
    candidates = load_json(cand_path, [])

    if not candidates:
        print(f"No candidate snapshot found for {date_str}: {cand_path}")
        return []

    final_games = final_game_pks(date_str)
    if not final_games:
        print(f"No final games yet for {date_str}")
        return []

    player_idx = player_index(int(date_str[:4]))
    score_cache = {}

    graded = []
    for row in candidates:
        g = grade_candidate(row, date_str, player_idx, final_games, score_cache)
        if g:
            graded.append(g)
            time.sleep(0.05)

    out_path = LOG_DIR / f"graded_candidates_{date_str}.json"
    save_json(out_path, graded)

    print(f"Saved graded candidates: {out_path}")
    print(json.dumps(summarize_graded_rows(graded), indent=2))

    return graded


def update_candidate_record(graded_rows, regrade_dates=None):
    record_path = LOG_DIR / "candidate_record.json"
    regrade_dates = set(regrade_dates or [])

    record = load_json(record_path, {"results": []})
    if not isinstance(record, dict):
        record = {"results": []}

    existing = record.get("results", [])
    if not isinstance(existing, list):
        existing = []

    if regrade_dates:
        existing = [r for r in existing if r.get("date") not in regrade_dates]

    keyed = {}
    for r in existing:
        if isinstance(r, dict):
            keyed[candidate_key(r)] = r

    for r in graded_rows:
        if isinstance(r, dict):
            keyed[candidate_key(r)] = r

    results = list(keyed.values())
    results.sort(key=lambda r: (r.get("date", ""), r.get("prop_type", ""), r.get("player", "")), reverse=True)

    record = {
        "version": VERSION,
        "last_updated": now_et().isoformat(),
        "summary": summarize_graded_rows(results),
        "by_prop": group_summary(results, lambda r: r.get("prop_type")),
        "by_status": group_summary(results, lambda r: r.get("candidate_status")),
        "by_official_board": group_summary(results, lambda r: "official_board" if r.get("official_board") else "not_official"),
        "by_prop_status": group_summary(results, lambda r: f"{r.get('prop_type')}|{r.get('candidate_status')}"),
        "by_reject_reason": group_summary(results, lambda r: r.get("reject_reason") or "none"),
        "results": results,
    }

    save_json(record_path, record)
    return record


def grade_recent_days(days):
    all_graded = []
    regrade_dates = []

    for i in range(0, int(days) + 1):
        d = (today_et() - dt.timedelta(days=i)).isoformat()
        regrade_dates.append(d)
        graded = grade_candidates_for_date(d)
        all_graded.extend(graded)

    if all_graded:
        record = update_candidate_record(all_graded, regrade_dates=regrade_dates)
        print("Updated candidate record:")
        print(json.dumps(record.get("summary", {}), indent=2))
        return record

    print("No graded candidate rows found.")
    return None


def summarize_graded_rows(rows):
    total = len(rows)
    hits = sum(1 for r in rows if r.get("result") == "hit")
    misses = sum(1 for r in rows if r.get("result") == "miss")

    projected_vals = []
    actual_vals = []
    errors = []
    abs_errors = []

    for r in rows:
        prop = r.get("prop_type")
        if prop == "moneyline":
            continue

        projected = safe_float(r.get("projected"))
        actual = safe_float(r.get("actual"))

        if projected is not None and actual is not None:
            projected_vals.append(projected)
            actual_vals.append(actual)
            err = projected - actual
            errors.append(err)
            abs_errors.append(abs(err))

    avg_projected = sum(projected_vals) / len(projected_vals) if projected_vals else None
    avg_actual = sum(actual_vals) / len(actual_vals) if actual_vals else None
    bias = sum(errors) / len(errors) if errors else None
    avg_abs = sum(abs_errors) / len(abs_errors) if abs_errors else None

    if bias is None:
        bias_text = "not_available"
    elif bias > 0.05:
        bias_text = "over_projecting"
    elif bias < -0.05:
        bias_text = "under_projecting"
    else:
        bias_text = "roughly_balanced"

    return {
        "total": total,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total * 100, 1) if total else 0,
        "avg_projected": round(avg_projected, 3) if avg_projected is not None else None,
        "avg_actual": round(avg_actual, 3) if avg_actual is not None else None,
        "projection_bias": round(bias, 3) if bias is not None else None,
        "avg_abs_error": round(avg_abs, 3) if avg_abs is not None else None,
        "bias_interpretation": bias_text,
    }


def group_summary(rows, key_func):
    grouped = defaultdict(list)

    for row in rows:
        try:
            key = key_func(row)
        except Exception:
            key = "unknown"

        if key is None or key == "":
            key = "unknown"

        grouped[str(key)].append(row)

    out = {}
    for key in sorted(grouped.keys()):
        out[key] = summarize_graded_rows(grouped[key])

    return out


def report():
    record_path = LOG_DIR / "candidate_record.json"
    record = load_json(record_path, {"results": []})

    if not isinstance(record, dict):
        print("No candidate record yet.")
        return

    compact = {
        "version": record.get("version"),
        "last_updated": record.get("last_updated"),
        "summary": record.get("summary"),
        "by_prop": record.get("by_prop"),
        "by_status": record.get("by_status"),
        "by_official_board": record.get("by_official_board"),
        "by_prop_status": record.get("by_prop_status"),
        "by_reject_reason": record.get("by_reject_reason"),
    }

    print(json.dumps(compact, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-today", action="store_true")
    parser.add_argument("--snapshot-date", type=str, default=None)
    parser.add_argument("--grade-date", type=str, default=None)
    parser.add_argument("--grade-days", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    did = False

    if args.snapshot_today:
        snapshot_candidates(today_et().isoformat())
        did = True

    if args.snapshot_date:
        snapshot_candidates(args.snapshot_date)
        did = True

    if args.grade_date:
        graded = grade_candidates_for_date(args.grade_date)
        if graded:
            update_candidate_record(graded, regrade_dates=[args.grade_date])
        did = True

    if args.grade_days is not None:
        grade_recent_days(args.grade_days)
        did = True

    if args.all:
        snapshot_candidates(today_et().isoformat())
        grade_recent_days(20)
        did = True

    if args.report:
        report()
        did = True

    if not did:
        print("Nothing to do. Try:")
        print("  python candidate_projection_logger.py --snapshot-today")
        print("  python candidate_projection_logger.py --grade-days 20")
        print("  python candidate_projection_logger.py --all")
        print("  python candidate_projection_logger.py --report")


if __name__ == "__main__":
    main()
