#!/usr/bin/env python3
"""
CFB_GRADE_RECORD_A

Grades every logged CFB pick (docs/cfb_picks_log.jsonl, appended to by
cfb_serving_builder_a.py's PICKS_LOG_PATH -- logged BEFORE the finished-
game filter drops a pick from the live board, see that file's docstring
for why the per-week archive alone can't be used for this) against real
outcomes in cfb_model.sqlite's player_games table, and writes
docs/cfb_record.json in the same summary/results shape MLB's record.json
uses (api.py's update_record()), adapted for CFB: segmented by market
instead of by_prop, no book-odds or bvp/confidence buckets since CFB
picks don't carry those fields (predictions-first, no odds).

A pick is gradable once player_games has a row for (player_id, season,
week) -- that row only exists once the real game has been played and
ingested, so no separate schedule-final check is needed here (unlike
MLB, which has to poll a live API and ask explicitly). Ungraded picks
(game not yet played/ingested) are silently skipped, not counted as a
miss.

Read-only against the DB and the ledger; only ever writes docs/cfb_record.json and docs/cfb_forward_status.json.

FORWARD-EVALUATION RULES (2026-10-03)
* The IMMUTABLE FIRST PREGAME LEDGER ENTRY is canonical: grading and probability scoring use its original logged_at, model_prob, side / pick and
  model_source -- never the (refreshable) live-board probability. If several ledger rows share a canonical prediction key the EARLIEST valid pregame
  one is used (never an average, never the latest) and the duplicates are reported as an audit issue.
* A ledger row is a valid pregame prediction only when logged_at < the game's kickoff (the row's own kickoff_utc, else the schedule's). Rows logged at /
  after kickoff, or whose kickoff cannot be verified, are excluded from forward evaluation (counted, never deleted).
* Canonical key = season | week | game_id | market | entity | line. entity = player_id for props, "GAME" for moneyline (one moneyline prediction per game; the
  predicted team is an attribute of the FIRST entry). The side / predicted team is deliberately NOT part of the key, so a refresh that flips a side can never
  become a second prediction. game_id is resolved from the schedule by (season, week, team pair) when the row has none; an ambiguous / unresolved game is
  ungraded and reported.
* moneyline / moneyline_early_season are graded at GAME level from the completed game truth already in the pipeline (ESPN final points in games):
  hit = predicted team won, miss = predicted team lost; a tie, a cancelled / postponed / forfeited game is EXCLUDED (never a win or a loss); an unfinished game
  stays ungraded. No sportsbook line is used anywhere.
* 2026 forward evidence is OBSERVATIONAL: it cannot trigger same-week rescue tuning (see cfb_models/CFB_FORWARD_EVALUATION_FREEZE.md).

Run
---
python -u cfb_grade_record_a.py
"""
import argparse
import json
import math
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

REPO = Path(__file__).resolve().parent
DB_DEFAULT = REPO / "cfb_models" / "cfb_model.sqlite"
DOCS = REPO / "docs"
LOG_DEFAULT = DOCS / "cfb_picks_log.jsonl"
OUT_DEFAULT = DOCS / "cfb_record.json"

# Actual-outcome column per base market (an "_early_season" pick is
# graded against the exact same real-world stat as its in-season
# counterpart -- only the model that produced the pick differs).
# anytime_touchdowns sums two raw columns, same as the in-season
# AnytimeTouchdownEngine/prior-season bootstrap both do.
MARKET_STAT_COLUMN = {
    "rushing_yards": "rushing_yards",
    "passing_touchdowns": "passing_touchdowns",
    "receiving_yards": "receiving_yards",
    "passing_yards": "passing_yards",
    "rushing_touchdowns": "rushing_touchdowns",
    "receiving_touchdowns": "receiving_touchdowns",
}


def actual_stat(market, row):
    base = market.replace("_early_season", "")
    if base == "anytime_touchdowns":
        return (row["rushing_touchdowns"] or 0) + (row["receiving_touchdowns"] or 0)
    col = MARKET_STAT_COLUMN.get(base)
    return row[col] if col else None


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


EVAL_SOURCE = "FIRST_PREGAME_LEDGER_ENTRY"
CANCELLED_STATUSES = {"STATUS_CANCELED", "STATUS_CANCELLED", "STATUS_POSTPONED", "STATUS_FORFEIT", "STATUS_ABANDONED", "STATUS_NO_CONTEST", "STATUS_SUSPENDED"}
# Legacy in-season ledger rows carry no model_source. Context-v2 serving went live with commit d3d46a7 (2026-09-27T16:05:38Z): rows logged at / after it are
# INFERRED context_v2, earlier rows INFERRED champion growing-pool Platt. New rows carry an explicit model_source (cfb_serving_builder_a.py).
CONTEXT_V2_CUTOVER_UTC = "2026-09-27T16:05:38+00:00"
FREEZE_NOTE = ("PRE-SLATE FREEZE: from the 2026-10-03 pre-slate infrastructure patch until the Week 5 slate is fully graded, NO CFB predictive model, calibration, feature, "
               "eligibility threshold or market status may be changed based on Week 5 outcomes. Operational bug fixes are allowed only if they do not use outcome knowledge.")
OBSERVATIONAL_NOTE = "2026 forward evidence is observational. It cannot trigger same-week rescue tuning and no promotion / rejection decision is made from it here."


def parse_ts(s):
    if not s:
        return None
    t = str(s).strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(t)
    except ValueError:
        return None
    return d.astimezone(timezone.utc) if d.tzinfo else None


def base_market(m):
    return str(m).replace("_early_season", "")


def is_moneyline(m):
    return base_market(m) == "moneyline"


def load_ledger(path):
    picks = []
    if not path.exists():
        return picks
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            picks.append(json.loads(line))
        except Exception:
            continue
    return picks


def grade(pick, actual):
    p = str(pick.get("pick", "")).upper().strip()
    m = re.search(r"(OVER|UNDER)\s+(-?\d+(?:\.\d+)?)", p)
    if not m:
        return None
    side, line = m.group(1), float(m.group(2))
    return "hit" if (actual > line if side == "OVER" else actual < line) else "miss"


# ------------------------------------------------------------------ games / identity
class Schedule:
    """Completed-game truth + schedule identity from games (+ schedule_snapshot status when present)."""

    def __init__(self, games):
        self.by_id = {str(g["game_id"]): g for g in games}
        self.by_pair = {}
        for g in games:
            self.by_pair.setdefault((g["season"], g["week"], frozenset((g["home_team"], g["away_team"]))), []).append(g)

    def resolve(self, entry):
        """-> (game | None, status). status: OK | UNRESOLVED_GAME | AMBIGUOUS_GAME."""
        gid = entry.get("game_id")
        if gid is not None and str(gid) in self.by_id:
            return self.by_id[str(gid)], "OK"
        cands = self.by_pair.get((entry.get("season"), entry.get("week"), frozenset((entry.get("team"), entry.get("opponent")))), [])
        if len(cands) == 1:
            return cands[0], "OK"
        return None, ("AMBIGUOUS_GAME" if cands else "UNRESOLVED_GAME")


def canonical_key(entry, game):
    """season|week|game_id|market|entity|line  (side / predicted team deliberately excluded -- see module docstring)."""
    gid = game["game_id"] if game else "UNRESOLVED:" + "~".join(sorted(str(x) for x in (entry.get("team"), entry.get("opponent"))))
    ent = "GAME" if is_moneyline(entry.get("market")) else str(entry.get("player_id"))
    line = "ML" if is_moneyline(entry.get("market")) else str(entry.get("line"))
    return "|".join([str(entry.get("season")), str(entry.get("week")), str(gid), str(entry.get("market")), ent, line])


def pregame_valid(entry, game):
    """(ok, kickoff_dt). Strict: logged_at < kickoff. The row's own kickoff_utc wins over the schedule's."""
    ko = parse_ts(entry.get("kickoff_utc")) or (parse_ts(game.get("kickoff_utc")) if game else None)
    la = parse_ts(entry.get("logged_at"))
    return (ko is not None and la is not None and la < ko), ko


def generation(entry):
    m = entry.get("market", "")
    if m.endswith("_early_season"):
        return "early_season_prior_season_informed"
    src = entry.get("model_source")
    if src == "context_v2":
        return "in_season_context_v2"
    if src == "champion_growing_platt":
        return "in_season_champion_growing_platt"
    la = parse_ts(entry.get("logged_at"))
    cut = parse_ts(CONTEXT_V2_CUTOVER_UTC)
    return "in_season_context_v2_inferred_by_log_time" if (la and la >= cut) else "in_season_champion_growing_platt_inferred_by_log_time"


def select_canonical(ledger, sched):
    """-> (canonical {key: record}, audit). record = {entry, game, kickoff, status}; the earliest VALID pregame row of each key."""
    groups = {}
    audit = {"ledger_entries_total": len(ledger), "rows_not_pregame_or_unverifiable": 0, "unresolved_or_ambiguous_game_rows": 0}
    for i, e in enumerate(ledger):
        game, gstatus = sched.resolve(e)
        if gstatus != "OK":
            audit["unresolved_or_ambiguous_game_rows"] += 1
        ok, ko = pregame_valid(e, game)
        groups.setdefault(canonical_key(e, game), []).append({"i": i, "entry": e, "game": game, "gstatus": gstatus, "valid": ok, "kickoff": ko})
    canonical, dup_keys, dup_extra, conflicts, only_invalid = {}, [], 0, [], []
    for k, rows in groups.items():
        if len(rows) > 1:
            dup_keys.append(k); dup_extra += len(rows) - 1
        valid = sorted((r for r in rows if r["valid"]), key=lambda r: (parse_ts(r["entry"]["logged_at"]), r["i"]))
        audit["rows_not_pregame_or_unverifiable"] += sum(1 for r in rows if not r["valid"])
        if not valid:
            only_invalid.append(k); continue
        first = valid[0]
        canonical[k] = first
        sides = {(r["entry"].get("pick"), r["entry"].get("team")) for r in rows}
        if len(sides) > 1:
            conflicts.append(k)
    audit.update({"duplicate_key_count": len(dup_keys), "duplicate_extra_rows": dup_extra, "duplicate_keys_sample": dup_keys[:5], "keys_with_conflicting_side_or_team": len(conflicts),
                  "keys_with_no_valid_pregame_row": len(only_invalid), "canonical_predictions": len(canonical),
                  "duplicate_policy": "earliest valid pregame row is used; later rows are never averaged or preferred"})
    return canonical, audit


# ------------------------------------------------------------------ grading
def grade_moneyline(entry, game, status_name):
    """-> (result, extra) with result in hit / miss / tie_excluded / cancelled_excluded / unfinished / invalid."""
    if game is None:
        return "unresolved", {}
    team = entry.get("team")
    if team not in (game["home_team"], game["away_team"]):
        return "invalid", {"reason": "predicted team is not a participant of the resolved game"}
    hp, ap = game.get("home_points"), game.get("away_points")
    if hp is None or ap is None:
        if status_name in CANCELLED_STATUSES:
            return "cancelled_excluded", {"espn_status": status_name}
        return "unfinished", {}
    if status_name in CANCELLED_STATUSES:
        return "cancelled_excluded", {"espn_status": status_name}
    if hp == ap:
        return "tie_excluded", {"home_points": hp, "away_points": ap}
    winner = game["home_team"] if hp > ap else game["away_team"]
    return ("hit" if winner == team else "miss"), {"home_points": hp, "away_points": ap, "winner": winner}


def p_success(entry):
    """Probability of the PICKED side, from the ORIGINAL ledger entry only."""
    p = entry.get("model_prob")
    return float(p) if isinstance(p, (int, float)) and 0.0 <= p <= 1.0 else None


def score_group(results):
    n = len(results)
    if not n:
        return {"n_graded": 0}
    hits = sum(1 for r in results if r["result"] == "hit")
    ps = [(r["model_prob"], 1.0 if r["result"] == "hit" else 0.0) for r in results if r.get("model_prob") is not None]
    eps = 1e-6
    brier = sum((p - y) ** 2 for p, y in ps) / len(ps) if ps else None
    ll = sum(-(y * math.log(min(max(p, eps), 1 - eps)) + (1 - y) * math.log(min(max(1 - p, eps), 1 - eps))) for p, y in ps) / len(ps) if ps else None
    return {"n_graded": n, "wins": hits, "losses": n - hits, "hit_rate": round(hits / n, 4), "n_probability_scored": len(ps), "mean_brier": None if brier is None else round(brier, 5),
            "mean_log_loss": None if ll is None else round(ll, 5), "mean_predicted_probability": None if not ps else round(sum(p for p, _ in ps) / len(ps), 4),
            "observed_success_rate": None if not ps else round(sum(y for _, y in ps) / len(ps), 4)}


def evaluate(ledger, games, player_rows, snapshot_status):
    """Pure function (unit-tested). games: list of game dicts; player_rows: {(player_id, game_id): row}; snapshot_status: {game_id: espn_status}."""
    sched = Schedule(games)
    canonical, audit = select_canonical(ledger, sched)
    results, ungraded, excluded = [], {}, {}
    for key, rec in sorted(canonical.items()):
        e, game = rec["entry"], rec["game"]
        base = {"canonical_key": key, "game_id": game["game_id"] if game else None, "season": e["season"], "week": e["week"], "market": e["market"], "player": e.get("player"), "player_id": e.get("player_id"),
                "team": e.get("team"), "opponent": e.get("opponent"), "pick": e.get("pick"), "line": e.get("line"), "model_prob": p_success(e), "original_model_prob": p_success(e), "model_source": e.get("model_source"),
                "generation": generation(e), "logged_at": e.get("logged_at"), "kickoff_utc": rec["kickoff"].strftime("%Y-%m-%dT%H:%M:%SZ") if rec["kickoff"] else None, "evaluation_probability_source": EVAL_SOURCE}
        if rec["gstatus"] != "OK":
            ungraded[rec["gstatus"]] = ungraded.get(rec["gstatus"], 0) + 1; continue
        if is_moneyline(e["market"]):
            res, extra = grade_moneyline(e, game, snapshot_status.get(str(game["game_id"])))
            if res in ("hit", "miss"):
                results.append({**base, "actual": extra.get("winner"), "result": res, "home_points": extra["home_points"], "away_points": extra["away_points"]})
            elif res.endswith("_excluded") or res == "invalid":
                excluded[res] = excluded.get(res, 0) + 1
            else:
                ungraded[res] = ungraded.get(res, 0) + 1
            continue
        row = player_rows.get((str(e.get("player_id")), str(game["game_id"])))
        if row is None:
            final = game.get("home_points") is not None
            k = "player_no_stat_line_after_final" if final else "unfinished"
            ungraded[k] = ungraded.get(k, 0) + 1; continue
        actual = actual_stat(e["market"], row)
        result = grade(e, actual) if actual is not None else None
        if result is None:
            excluded["ungradable_pick_string"] = excluded.get("ungradable_pick_string", 0) + 1; continue
        results.append({**base, "actual": actual, "result": result})
    return results, ungraded, excluded, audit


def aggregate(results):
    by_market, by_gen, by_gen_market = {}, {}, {}
    for r in results:
        by_market.setdefault(r["market"], []).append(r)
        by_gen.setdefault(r["generation"], []).append(r)
        by_gen_market.setdefault(f"{r['generation']}::{r['market']}", []).append(r)
    return ({k: score_group(v) for k, v in sorted(by_market.items())}, {k: score_group(v) for k, v in sorted(by_gen.items())}, {k: score_group(v) for k, v in sorted(by_gen_market.items())})


def load_games(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(games)")}
    ko = "kickoff_utc" if "kickoff_utc" in cols else "NULL AS kickoff_utc"
    games = [dict(r) for r in con.execute(f"SELECT game_id, season, week, {ko}, home_team, away_team, home_points, away_points FROM games")]
    snap = {}
    if con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schedule_snapshot'").fetchone():
        snap = {str(r[0]): r[1] for r in con.execute("SELECT game_id, espn_status FROM schedule_snapshot")}
    return games, snap


def load_player_rows(con, ledger):
    sw = {(p["season"], p["week"]) for p in ledger if p.get("player_id")}
    out = {}
    for season, week in sw:
        for r in con.execute("SELECT * FROM player_games WHERE season=? AND week=?", (season, week)):
            out[(str(r["player_id"]), str(r["game_id"]))] = r
    return out


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_DEFAULT))
    ap.add_argument("--log", default=str(LOG_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--status-out", default=str(DOCS / "cfb_forward_status.json"))
    ap.add_argument("--readiness", default=str(DOCS / "cfb_live_readiness.json"))
    ap.add_argument("--board", default=str(DOCS / "cfb_predictions.json"))
    args = ap.parse_args()

    print("CFB_GRADE_RECORD_A\n==================")
    ledger = load_ledger(Path(args.log))
    print(f"ledger: {len(ledger)} logged picks")
    if not ledger:
        Path(args.out).write_text(json.dumps({"summary": {"total": 0, "hits": 0, "misses": 0, "hit_rate": 0}, "by_market": {}, "results": [], "last_updated": now_utc()}, indent=2))
        print(f"no ledger entries yet -- wrote empty record to {args.out}")
        return 0

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    games, snap = load_games(con)
    player_rows = load_player_rows(con, ledger)
    con.close()

    results, ungraded, excluded, audit = evaluate(ledger, games, player_rows, snap)
    print(f"graded: {len(results)}  ungraded {ungraded}  excluded {excluded}  duplicates {audit['duplicate_key_count']}")
    by_market_detail, by_generation, by_generation_market = aggregate(results)
    total = len(results)
    hits = sum(1 for r in results if r["result"] == "hit")
    by_market = {k: {"hits": v["wins"], "total": v["n_graded"], "hit_rate": round(v["wins"] / v["n_graded"] * 100, 1)} for k, v in by_market_detail.items()}
    now = now_utc()
    record = {
        "summary": {"total": total, "hits": hits, "misses": total - hits, "hit_rate": round(hits / total * 100, 1) if total else 0},
        "by_market": by_market,
        "results": sorted(results, key=lambda r: (r["season"], r["week"]), reverse=True),
        "forward_evaluation": {"evaluation_probability_source": EVAL_SOURCE, "by_market": by_market_detail, "by_generation": by_generation, "by_generation_and_market": by_generation_market,
                               "generation_labels": {"early_season_prior_season_informed": "weeks 1-3 prior-season bootstrap models", "in_season_context_v2": "context-v2 serving (explicit model_source)",
                                                     "in_season_champion_growing_platt": "frozen champion + growing-pool Platt (explicit model_source)",
                                                     "in_season_context_v2_inferred_by_log_time": f"legacy rows with no model_source logged at/after {CONTEXT_V2_CUTOVER_UTC} (inferred)",
                                                     "in_season_champion_growing_platt_inferred_by_log_time": "legacy rows with no model_source logged before the context-v2 cutover (inferred)"},
                               "calibration_note": "Different model generations are NEVER pooled into one calibration claim: read by_generation / by_generation_and_market.", "ungraded": ungraded, "excluded": excluded, "audit": audit},
        "last_updated": now,
    }
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"\n{hits}/{total} ({record['summary']['hit_rate']}%) written to {args.out}")

    readiness, board = read_json(args.readiness), read_json(args.board)
    graded_keys = len(results)
    n_canon = audit["canonical_predictions"]
    status = {
        "generated_at_utc": now, "first_prediction_only": True, "evaluation_probability_source": EVAL_SOURCE,
        "ledger_entries_total": audit["ledger_entries_total"], "canonical_predictions": n_canon, "graded_predictions": graded_keys,
        "ungraded_predictions": n_canon - graded_keys - sum(excluded.values()), "excluded_predictions": excluded, "ungraded_reasons": ungraded,
        "duplicate_key_count": audit["duplicate_key_count"], "duplicate_extra_rows": audit["duplicate_extra_rows"], "keys_with_conflicting_side_or_team": audit["keys_with_conflicting_side_or_team"],
        "rows_excluded_not_pregame_or_unverifiable": audit["rows_not_pregame_or_unverifiable"], "unresolved_or_ambiguous_game_rows": audit["unresolved_or_ambiguous_game_rows"],
        "markets": by_market_detail, "by_generation": by_generation,
        "current_week_readiness": None if not readiness else {"READY": readiness.get("READY"), "checked_at_utc": readiness.get("checked_at_utc"), "season": readiness.get("season"), "week": readiness.get("week"),
                                                               "failing_checks": readiness.get("failing_checks"), "future_games": readiness.get("future_games")},
        "current_board_generated_at_utc": None if not board else board.get("generated_at_utc"),
        "pre_slate_freeze": FREEZE_NOTE, "note": OBSERVATIONAL_NOTE, "decision": "NONE: no promotion / rejection decision is derived from forward results here",
    }
    Path(args.status_out).write_text(json.dumps(status, indent=2))
    print(f"forward status -> {args.status_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
