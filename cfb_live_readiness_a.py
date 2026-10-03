#!/usr/bin/env python3
"""
CFB_LIVE_READINESS_A -- machine-readable readiness check for the CFB live board (existing system only: no model, feature or market is created here).

Reads the refreshed foundation db (games / player_games / current_roster / schedule_snapshot / data_freshness), the generated board
(docs/cfb_predictions.json) and the pregame ledger (docs/cfb_picks_log.jsonl), re-derives the serving-critical facts independently of the builder and
writes cfb_models/cfb_live_readiness_<season>_w<week>.json. READY is false if ANY hard requirement fails.

Hard requirements: active markets only (passing_yards / receiving_yards stay suspended), frozen artifact hashes + feature order, calibration built from completed prior data
only, no target-week leakage in served features, unique schedule, future-only board (strict now < kickoff, ESPN state pre), roster snapshot present, fresh schedule / roster
data, append-only pregame ledger covering every served pick, no sportsbook inputs. A candidate absent from the roster snapshot is reported, never hidden.

  python -u cfb_live_readiness_a.py --season 2026 --week 5            # check
  python -u cfb_live_readiness_a.py --write-manifest                  # (re)register the frozen artifact hashes (a deliberate, reviewed act)
"""
import argparse
import ast
import hashlib
import json
import sqlite3
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cfb_serving_builder_a as SB

REPO = Path(__file__).resolve().parent
MODELS = REPO / "cfb_models"
CTX_DIR = MODELS / "cfb_context_v2_work"
MANIFEST = MODELS / "cfb_live_artifact_manifest.json"
ACTIVE_MARKETS = ["rushing_yards", "passing_touchdowns", "anytime_touchdowns", "moneyline"]
SUSPENDED = ["passing_yards", "receiving_yards"]
MAX_SCHEDULE_AGE_H = 8.0
MAX_ROSTER_AGE_H = 30.0
ODDS_WORDS = ("odds", "spread", "vegas", "sportsbook", "bookmaker", "implied", "juice", "price", "vig")
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def utc(s):
    return SB.parse_kickoff(s)


def iso(d):
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def chk(ok, **detail):
    return {"ok": bool(ok), **detail}


# ------------------------------------------------------------------ artifacts
def expected_columns():
    import cfb_context_v2 as CX
    base = {"rushing_yards": SB.MARKETS["rushing_yards"]["features"], "passing_touchdowns": SB.MARKETS["passing_touchdowns"]["features"],
            "anytime_touchdowns": SB.ANYTIME_TD_FEATURES, "moneyline": SB.MONEYLINE_FEATURES}
    return base, {m: list(c) + [x for x in CX.CONTEXT_FEATURES if x not in c] for m, c in base.items()}


def artifact_paths():
    """Every file the live path can load for an active market: the context-v2 production models (the probabilities actually served) and
    the frozen walk-forward champions (used for the raw score / fallback)."""
    paths = {}
    paths["context:production.json"] = CTX_DIR / "production.json"
    old = {"rushing_yards": (SB.MARKETS["rushing_yards"]["model_dir"], "cfb_rushing_yards"), "passing_touchdowns": (SB.MARKETS["passing_touchdowns"]["model_dir"], "cfb_passing_touchdowns"),
           "anytime_touchdowns": (SB.ANYTIME_TD_MODEL_DIR, "cfb_anytime_touchdowns"), "moneyline": (SB.MONEYLINE_MODEL_DIR, "cfb_moneyline")}
    for m in ACTIVE_MARKETS:
        paths[f"context:{m}"] = CTX_DIR / f"cfb_{m}_context.json"
        paths[f"context:{m}:columns"] = CTX_DIR / f"cfb_{m}_context_columns.json"
        d, stem = old[m]
        paths[f"champion:{m}"] = d / f"{stem}.json"
        paths[f"champion:{m}:columns"] = d / f"{stem}_columns.json"
    return paths


def write_manifest():
    paths = artifact_paths()
    man = {"registered_at_utc": iso(datetime.now(timezone.utc)), "purpose": "frozen live-serving artifacts for the active CFB markets; readiness fails if any byte changes",
           "files": {k: {"path": str(p.relative_to(REPO)), "sha256": sha256_file(p), "bytes": p.stat().st_size} for k, p in sorted(paths.items())}}
    MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


def check_artifacts():
    base, ctx_cols = expected_columns()
    out = {"manifest_present": MANIFEST.exists(), "files": {}, "feature_order": {}}
    ok = MANIFEST.exists()
    if ok:
        man = json.loads(MANIFEST.read_text())
        for k, p in artifact_paths().items():
            rec = man["files"].get(k)
            good = bool(rec) and p.exists() and sha256_file(p) == rec["sha256"]
            out["files"][k] = good
            ok = ok and good
    for m in ACTIVE_MARKETS:
        try:
            cc = json.loads((CTX_DIR / f"cfb_{m}_context_columns.json").read_text())
            oc = json.loads(artifact_paths()[f"champion:{m}:columns"].read_text())
            out["feature_order"][m] = {"context_columns_match_code": cc == ctx_cols[m], "champion_columns_match_code": oc == base[m]}
            ok = ok and cc == ctx_cols[m] and oc == base[m]
        except Exception as e:                                                    # noqa
            out["feature_order"][m] = {"error": str(e)[:120]}; ok = False
    out["ok"] = bool(ok)
    return out


def check_calibration(season, week, board):
    prod = json.loads((CTX_DIR / "production.json").read_text())
    out = {"context_platt": {}, "champion_growing_pools": {}}
    ok = True
    for m in ACTIVE_MARKETS:
        c = prod.get(m)
        good = bool(c) and c["calibration_season"] < season and c["n_calibration"] > 0 and c["platt_a"] > 0
        out["context_platt"][m] = {"calibration_season": c and c["calibration_season"], "n_calibration": c and c["n_calibration"], "platt_a": c and c["platt_a"], "uses_only_completed_prior_season": bool(c and c["calibration_season"] < season)}
        ok = ok and good
    for m, meta in (board.get("markets") or {}).items():
        pool = meta.get("calibration_pool")
        if pool:
            good = pool["warmup_season"] < season
            out["champion_growing_pools"][m] = {**pool, "ok": good}
            ok = ok and good
    out["note"] = "served probabilities come from the context-v2 production models (Platt calibrated once on the completed 2025 season); the champion growing-pool Platt is computed but overridden when the context models load"
    out["ok"] = bool(ok)
    return out


def check_leakage(con, season, week, board):
    """Re-derive prior-games counts from player_games / games with week < target week and compare with what the board carries."""
    bad, n = [], 0
    for p in board["picks"]:
        if p.get("season") != season or p.get("week") != week:
            bad.append({"reason": "wrong_season_week", "pick": [p["market"], p.get("player"), p.get("team")]}); continue
        if p.get("player_id") and not p["market"].endswith("_early_season"):
            exp = con.execute("SELECT COUNT(*) FROM player_games WHERE player_id=? AND season=? AND week<?", (p["player_id"], season, week)).fetchone()[0]
            n += 1
            if p.get("games_played") != exp:
                bad.append({"reason": "games_played_mismatch", "pick": [p["market"], p["player"], p["team"]], "board": p.get("games_played"), "recount": exp})
        elif p["market"] == "moneyline":
            exp = con.execute("SELECT COUNT(*) FROM games WHERE season=? AND week<? AND home_points IS NOT NULL AND (home_team=? OR away_team=?)", (season, week, p["team"], p["team"])).fetchone()[0]
            n += 1
            if p.get("team_games_played") is not None and p["team_games_played"] != exp and p["market"] == "moneyline":
                bad.append({"reason": "team_games_mismatch", "pick": ["moneyline", p["team"]], "board": p.get("team_games_played"), "recount": exp})
    target_rows = con.execute("SELECT COUNT(*) FROM player_games WHERE season=? AND week>=?", (season, week)).fetchone()[0]
    return chk(not bad, picks_recounted=n, mismatches=bad[:10], n_mismatches=len(bad), target_or_later_week_player_rows_in_db=target_rows,
               note="features are recomputed by the builder from rows with week < target week only; the recount above must agree exactly")


# ------------------------------------------------------------------ schedule
def fetch_espn_day(d):
    url = f"{ESPN}?dates={d}&groups=80&limit=300"
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=ESPN_HEADERS), timeout=30).read().decode())


def check_schedule(con, season, week, now, live_compare=True):
    rows = con.execute("SELECT g.game_id, g.home_team, g.away_team, g.kickoff_utc, g.game_date, g.home_points, g.away_points, s.espn_state, s.espn_status, s.retrieved_at_utc "
                       "FROM games g LEFT JOIN schedule_snapshot s ON s.game_id = g.game_id WHERE g.season=? AND g.week=? ORDER BY g.kickoff_utc, g.game_id", (season, week)).fetchall()
    games, issues = [], []
    ids = Counter(r[0] for r in rows); pairs = Counter(frozenset((r[1], r[2])) for r in rows)
    for gid, h, a, ko, gd, hp, ap, st, stn, ret in rows:
        k = utc(ko)
        ok, why = SB.pregame_status(ko, now, st)
        games.append({"game_id": gid, "home": h, "away": a, "kickoff_utc": ko, "status": st, "espn_status": stn, "final": hp is not None, "schedule_retrieved_at_utc": ret, "future": ok, "not_future_reason": why, "source": "ESPN site API scoreboard (cfb_espn_live_foundation_a.py)"})
        if k is None:
            issues.append({"game_id": gid, "issue": "unparseable_or_missing_kickoff"})
        elif ko[:10] != gd and k.date().isoformat() != gd:
            issues.append({"game_id": gid, "issue": "kickoff_date_differs_from_game_date", "kickoff": ko, "game_date": gd})
        if ret is None:
            issues.append({"game_id": gid, "issue": "no_schedule_retrieval_timestamp"})
    dup = [g for g, c in ids.items() if c > 1]; dupp = [sorted(p) for p, c in pairs.items() if c > 1]
    live = {"compared": False}
    if live_compare:
        try:
            dates = sorted({g["kickoff_utc"][:10] for g in games if g["future"]} | {r[4] for r in rows if r[4]})
            espn = {}
            for d in dates:
                for e in fetch_espn_day(d.replace("-", "")).get("events") or []:
                    espn[e["id"]] = {"kickoff": e.get("date"), "state": ((e.get("status") or {}).get("type") or {}).get("state")}
            mism = []
            for g in games:
                if not g["future"]:
                    continue
                e = espn.get(g["game_id"])
                if e is None:
                    mism.append({"game_id": g["game_id"], "issue": "missing_from_live_espn"}); continue
                ek, dk = utc(e["kickoff"]), utc(g["kickoff_utc"])
                if ek is None or dk is None or abs((ek - dk).total_seconds()) > 0:
                    mism.append({"game_id": g["game_id"], "issue": "kickoff_changed_since_snapshot", "db": g["kickoff_utc"], "espn": e["kickoff"]})
                if e["state"] != "pre":
                    mism.append({"game_id": g["game_id"], "issue": "espn_state_not_pre_now", "state": e["state"]})
            live = {"compared": True, "espn_events_fetched": len(espn), "dates": dates, "mismatches": mism}
            issues += [{"live_compare": m} for m in mism]
        except Exception as e:                                                    # noqa
            live = {"compared": False, "error": f"{type(e).__name__}: {e}"[:160]}
            issues.append({"issue": "live_schedule_comparison_failed"})
    fut = [g for g in games if g["future"]]
    return chk(not dup and not dupp and not issues and bool(fut), n_week_games=len(games), future_games=len(fut), started_or_final_games=len(games) - len(fut), duplicate_game_ids=dup, duplicate_team_pairs=dupp, issues=issues[:20], n_issues=len(issues),
               earliest_future_kickoff_utc=min((g["kickoff_utc"] for g in fut), default=None), live_schedule_comparison=live), games


# ------------------------------------------------------------------ board / ledger / roster / freshness / odds
def check_board(board, games, now):
    by_pair = {frozenset((g["home"], g["away"])): g for g in games}
    gen = utc(board.get("generated_at_utc"))
    bad = []
    seen = Counter()
    for p in board["picks"]:
        g = by_pair.get(frozenset((p["team"], p["opponent"])))
        ko = utc(p.get("kickoff_utc"))
        if g is None or ko is None or not (gen < ko and now < ko) or not g["future"]:
            bad.append({"pick": [p["market"], p.get("player"), p["team"], p["opponent"]], "kickoff": p.get("kickoff_utc")})
        seen[(p["season"], p["week"], p["market"], p.get("player_id") or p["team"])] += 1
    dups = [list(k) for k, c in seen.items() if c > 1]
    return chk(not bad and not dups and gen is not None, n_picks=len(board["picks"]), picks_not_strictly_future=bad[:10], n_not_future=len(bad), duplicate_player_market_game=dups[:10], generated_at_utc=board.get("generated_at_utc"),
               checked_at_utc=iso(now), board_filter_present="board_filter" in board)


def check_ledger(ledger_path, board, games):
    rows = [json.loads(l) for l in Path(ledger_path).read_text().splitlines() if l.strip()]
    key = lambda r: (r.get("season"), r.get("week"), r.get("market"), r.get("player_id") or r.get("team"))
    cnt = Counter(key(r) for r in rows)
    dups = [list(k) for k, c in cnt.items() if c > 1]
    by_key = {key(r): r for r in rows}
    by_pair = {frozenset((g["home"], g["away"])): g for g in games}
    missing, post_ko, prob_diff = [], [], 0
    for p in board["picks"]:
        r = by_key.get(key(p))
        if r is None:
            missing.append([p["market"], p.get("player"), p["team"]]); continue
        ko = utc(by_pair[frozenset((p["team"], p["opponent"]))]["kickoff_utc"])
        la = utc(r["logged_at"])
        if la is None or ko is None or not la < ko:
            post_ko.append({"pick": [p["market"], p.get("player"), p["team"]], "logged_at": r["logged_at"], "kickoff": p.get("kickoff_utc")})
        if abs(r.get("model_prob", 0) - p.get("model_prob", 0)) > 1e-9:
            prob_diff += 1
    # historical entries logged after their game's kickoff (informational: evidence is never deleted)
    return chk(not dups and not missing and not post_ko, entries_total=len(rows), duplicate_keys=dups[:5], board_picks_missing_from_ledger=missing[:10], board_picks_logged_at_or_after_kickoff=post_ko[:10],
               board_picks_whose_current_prob_differs_from_original_logged_prob=prob_diff, append_only_note="the builder only appends; an existing key is never rewritten, so the ledger keeps the ORIGINAL independent probability and timestamp")


def check_roster(board, con, season):
    labels = Counter(p.get("roster_verification") for p in board["picks"])
    snap = con.execute("SELECT COUNT(*), COUNT(DISTINCT team) FROM current_roster WHERE season=?", (season,)).fetchone() if con.execute("SELECT name FROM sqlite_master WHERE name='current_roster'").fetchone() else (0, 0)
    not_on = [[p["market"], p["player"], p["team"]] for p in board["picks"] if p.get("roster_verification") == "NOT_ON_CURRENT_ROSTER_SNAPSHOT"]
    no_team = sorted({p["team"] for p in board["picks"] if p.get("roster_verification") == "NO_ROSTER_SNAPSHOT_FOR_TEAM"})
    unlabeled = sum(1 for p in board["picks"] if "roster_verification" not in p)
    return chk(snap[0] > 0 and unlabeled == 0 and not not_on, roster_snapshot_players=snap[0], roster_snapshot_teams=snap[1], label_counts=dict(labels), candidates_not_on_roster_snapshot=not_on[:15], n_not_on_roster=len(not_on),
               teams_without_roster_snapshot=no_team, unlabeled_picks=unlabeled,
               injury_inactive_knowledge="NONE: the roster snapshot lists names only; injuries / inactives are NOT represented anywhere in the CFB system and are not fabricated here")


def check_freshness(con, season, board, now):
    fr = dict(con.execute("SELECT key, value FROM data_freshness WHERE season=?", (season,)).fetchall()) if con.execute("SELECT name FROM sqlite_master WHERE name='data_freshness'").fetchone() else {}
    sc, rs = utc(fr.get("schedule_scan_completed_utc")), utc(fr.get("roster_snapshot_completed_utc"))
    age = lambda d: None if d is None else round((now - d).total_seconds() / 3600, 3)
    ok = bool(fr) and sc is not None and rs is not None and age(sc) <= MAX_SCHEDULE_AGE_H and age(rs) <= MAX_ROSTER_AGE_H
    last_final = con.execute("SELECT MAX(game_date) FROM player_games WHERE season=?", (season,)).fetchone()[0]
    prior = con.execute("SELECT MAX(game_date) FROM player_games WHERE season=?", (season - 1,)).fetchone()[0]
    return chk(ok, schedule_scan_completed_utc=fr.get("schedule_scan_completed_utc"), schedule_age_hours=age(sc), max_schedule_age_hours=MAX_SCHEDULE_AGE_H, roster_snapshot_completed_utc=fr.get("roster_snapshot_completed_utc"), roster_age_hours=age(rs),
               max_roster_age_hours=MAX_ROSTER_AGE_H, latest_2026_player_game_date=last_final, historical_player_data_last_game_date=prior, current_player_data_refreshed_utc=fr.get("schedule_scan_completed_utc"),
               calibration_pool=f"context Platt: completed {season - 1} season only; champion growing pool adds {season} weeks < target week (rows with week < target only)",
               cfbfastr_note="historical 2018-2025 data from cfbfastR-data (not real time); 2026 from ESPN box scores (same-day for COMPLETED games only)", board_generated_at_utc=board.get("generated_at_utc"))


def check_suspended_and_odds(board):
    names = {m.replace("_early_season", "") for m in {p["market"] for p in board["picks"]}}
    leaked = sorted(names & set(SUSPENDED)); unknown = sorted(names - set(ACTIVE_MARKETS))
    src = (REPO / "cfb_serving_builder_a.py").read_text()
    tree = ast.parse(src)
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "SUSPENDED_MARKETS" and isinstance(n.ctx, ast.Load)]
    suspended_cfg = sorted(SB.SUSPENDED_MARKETS)
    served_cfg = sorted(SB.MARKETS) + ["anytime_touchdowns", "moneyline"]
    base, ctx_cols = expected_columns()
    cols = {c for v in list(base.values()) + list(ctx_cols.values()) for c in v}
    odds_cols = sorted(c for c in cols if any(w in c.lower() for w in ODDS_WORDS))
    odds_fields = sorted({k for p in board["picks"] for k in p if any(w in k.lower() for w in ODDS_WORDS)})
    odds_imports = [w for w in ("ODDS_API_KEY", "the-odds-api", "THE_ODDS_API") if any(w in (REPO / f).read_text() for f in ("cfb_serving_builder_a.py", "cfb_context_v2.py"))]
    return (chk(not leaked and not unknown and not loads and suspended_cfg == sorted(SUSPENDED) and sorted(served_cfg) == sorted(ACTIVE_MARKETS), board_markets=sorted(names), suspended_markets_on_board=leaked, unknown_markets_on_board=unknown,
                suspended_config_keys=suspended_cfg, served_config_keys=sorted(served_cfg), code_paths_reading_SUSPENDED_MARKETS=len(loads)),
            chk(not odds_cols and not odds_fields and not odds_imports, odds_like_feature_columns=odds_cols, odds_like_pick_fields=odds_fields, odds_references_in_serving_code=odds_imports))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(SB.DB_DEFAULT))
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--board", default=str(SB.DOCS / "cfb_predictions.json"))
    ap.add_argument("--ledger", default=str(SB.PICKS_LOG_PATH))
    ap.add_argument("--out")
    ap.add_argument("--now")
    ap.add_argument("--no-live-schedule", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    a = ap.parse_args()
    if a.write_manifest:
        print(json.dumps(write_manifest()["files"], indent=1)); return 0
    now = utc(a.now) if a.now else datetime.now(timezone.utc)
    board = json.loads(Path(a.board).read_text())
    season, week = a.season or board.get("season"), a.week or board.get("week")
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    sched, games = check_schedule(con, season, week, now, live_compare=not a.no_live_schedule)
    suspended_chk, odds_chk = check_suspended_and_odds(board)
    checks = {"model_hash_checks": check_artifacts(), "schedule_check": sched, "started_game_filter_check": check_board(board, games, now), "roster_check": check_roster(board, con, season),
              "data_freshness": check_freshness(con, season, board, now), "calibration_check": check_calibration(season, week, board), "leakage_check": check_leakage(con, season, week, board),
              "ledger_check": check_ledger(a.ledger, board, games), "suspended_market_check": suspended_chk, "no_sportsbook_inputs_check": odds_chk}
    failing = sorted(k for k, v in checks.items() if not v["ok"])
    counts = Counter(p["market"] for p in board["picks"])
    out = {"READY": not failing, "checked_at_utc": iso(now), "season": season, "week": week, "board_generated_at_utc": board.get("generated_at_utc"),
           "future_games": sched["future_games"], "week_games": games, "active_markets": ACTIVE_MARKETS, "suspended_markets": {m: "SUSPENDED: failed re-validation after the completion/reception attribution correction (cfb_serving_builder_a.SUSPENDED_MARKETS); not rescued" for m in SUSPENDED},
           "board_prediction_counts_by_market": dict(counts), **checks, "failing_checks": failing,
           "evidence_labels": {"forward_evidence": "picks first logged in the pregame ledger strictly before kickoff (original timestamp + original probability)", "historical_validation": "2025 holdout / walk-forward reports, context-v2 report.json",
                               "not_forward_evidence": "any pick generated at/after kickoff (now excluded from board and ledger); games completed before the readiness patch are historical/graded only"}}
    path = Path(a.out) if a.out else MODELS / f"cfb_live_readiness_{season}_w{week:02d}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print(f"READY={out['READY']} failing={failing} future_games={out['future_games']} picks={dict(counts)} -> {path}")
    return 0 if out["READY"] else 2


if __name__ == "__main__":
    sys.exit(main())
