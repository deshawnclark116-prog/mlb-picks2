#!/usr/bin/env python3
"""CFB as-of moneyline challenger: immutable pre-kickoff SHADOW forecasts.

Never serves official betting recommendations. A prospective record is accepted
only when (a) both the original incumbent and challenger are independently
observed before kickoff, (b) core stats are reconstructed using games from
weeks STRICTLY BEFORE the target week, and (c) BOTH teams can be scored.

Never train here; use the separately frozen, research-only 2019-25 models.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from cfb_asof_challenger import ShadowScorer
from cfb_serving_builder_a import MoneylineEngine, MIN_PRIOR_GAMES

SCHEMA = "CFB_ASOF_MONEYLINE_FIRST_PREGAME_SHADOW_V1"
NO_PROMOTION = "SHADOW_FROZEN_FIRST_PREGAME_NO_BET_NO_PROMOTION"


def utc(raw):
    if not isinstance(raw, str) or not raw:
        raise ValueError("missing UTC timestamp")
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timezone absent")
    return dt.astimezone(timezone.utc)


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite_probability(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x) and 0 <= x <= 1)


def canonical_incumbents(ledger):
    """Earliest original prediction for each game, not latest model refresh."""
    first = {}
    for offset, p in enumerate(ledger):
        if p.get("market") != "moneyline" or p.get("model_source") != "context_v2":
            continue
        try:
            observed = utc(p.get("logged_at"))
            kickoff = utc(p.get("kickoff_utc"))
        except (ValueError, TypeError):
            continue
        if observed >= kickoff or not _finite_probability(p.get("model_prob")):
            continue
        gid = str(p.get("game_id") or "")
        if not gid or p.get("season") is None or p.get("week") is None:
            continue
        key = (p["season"], p["week"], gid)
        old = first.get(key)
        if old is None or (observed, offset) < (old[0], old[1]):
            first[key] = (observed, offset, p)
    return {k: v[2] for k, v in first.items()}


def before_target_week_team_states(con, season, week):
    """Exact MoneylineEngine team-state formula, with whole target week held out.

    We deliberately CANNOT use MoneylineEngine.final_state directly: its
    already-finished SAME-WEEK games could leak into a later same-week match.
    """
    stats = {}
    rows = con.execute("""
        SELECT home_team, away_team, home_points, away_points
        FROM games
        WHERE season = ? AND week < ? AND home_points IS NOT NULL
          AND away_points IS NOT NULL
        ORDER BY week, game_date, game_id
    """, (season, week)).fetchall()
    for home, away, hp, ap in rows:
        for team, pf, pa in ((home, hp, ap), (away, ap, hp)):
            d = stats.setdefault(team, [0, 0., 0., 0])
            d[0] += int(pf > pa)
            d[1] += pf
            d[2] += pa
            d[3] += 1
    return {
        team: {"games_played": n, "win_rate": wins / n,
               "net_margin": (pf - pa) / n,
               "avg_points_for": pf / n,
               "avg_points_against": pa / n}
        for team, (wins, pf, pa, n) in stats.items() if n > 0
    }


def _snapshot_paths(root):
    required = ["research_manifest.json", "production.json",
                "cfb_moneyline_context.json",
                "cfb_moneyline_context_columns.json"]
    return [root / name for name in required]


def check_model_snapshot(root):
    root = Path(root)
    files = _snapshot_paths(root)
    for path in files:
        if not path.is_file():
            raise ValueError(f"missing frozen challenger artifact {path.name}")
    manifest = json.loads(files[0].read_text())
    if (manifest.get("status") != "CFB_ASOF_CHALLENGER_RESEARCH_ONLY_NO_PROMOTION"
            or manifest.get("predictions_published") is not False
            or manifest.get("blind_2026_forward_evidence_required") is not True):
        raise ValueError("unverified trained model snapshot")
    digest = hashlib.sha256(
        "".join(_file_hash(p) for p in files).encode("ascii")).hexdigest()
    return digest


def build_first_pregame_rows(con, board, history, scorer, *, captured_at,
                            model_digest, board_digest):
    captured = captured_at.astimezone(timezone.utc)
    season, week = board.get("season"), board.get("week")
    generated = utc(board["generated_at_utc"])
    if generated > captured:
        raise ValueError("board did not exist when shadow forecast recorded")
    if not isinstance(season, int) or not isinstance(week, int):
        raise ValueError("invalid board season/week")
    originals = canonical_incumbents(history)
    state = before_target_week_team_states(con, season, week)
    rows = []
    skipped = Counter()
    seen = set()
    for p in board.get("picks", []):
        if p.get("market") != "moneyline":
            continue
        if (p.get("season"), p.get("week")) != (season, week):
            raise ValueError("mixed board season or week")
        gid = str(p.get("game_id") or "")
        key = (season, week, gid)
        if not gid or key in seen:
            raise ValueError("ambiguous moneyline game in published board")
        seen.add(key)
        original = originals.get(key)
        if original is None:
            skipped["missing_immutable_incumbent_first_pregame"] += 1
            continue
        try:
            kickoff = utc(p.get("kickoff_utc"))
            origin_kickoff = utc(original.get("kickoff_utc"))
            original_at = utc(original.get("logged_at"))
        except (ValueError, TypeError):
            skipped["invalid_kickoff_or_original_clock"] += 1
            continue
        if (generated >= kickoff or original_at >= kickoff or
                captured >= kickoff or kickoff != origin_kickoff):
            skipped["not_provably_before_kickoff"] += 1
            continue
        if original_at > captured:
            skipped["original_not_yet_published"] += 1
            continue
        team, opponent = p.get("team"), p.get("opponent")
        if (not team or not opponent or team == opponent or
                set((team, opponent)) != set((original.get("team"),
                                             original.get("opponent")))):
            skipped["incumbent_pair_mismatch"] += 1
            continue
        if p.get("is_home") not in (0, 1) or p.get("is_neutral_site") not in (0, 1):
            skipped["unverified_home_neutral_context"] += 1
            continue
        home, away = ((team, opponent) if p["is_home"] == 1
                      else (opponent, team))
        hst, ast = state.get(home), state.get(away)
        if (not hst or not ast or
                min(hst["games_played"], ast["games_played"]) < MIN_PRIOR_GAMES):
            skipped["insufficient_strict_prior_week_games"] += 1
            continue
        if not _finite_probability(original.get("model_prob")):
            skipped["invalid_original_probability"] += 1
            continue
        neutral = bool(p["is_neutral_site"])
        hfeat = MoneylineEngine._feat(hst, ast, True, neutral)
        afeat = MoneylineEngine._feat(ast, hst, False, neutral)
        hp = scorer.probability("moneyline", hfeat, season, week,
                                home, away, None)["model_probability"]
        ap = scorer.probability("moneyline", afeat, season, week,
                                away, home, None)["model_probability"]
        if not all(_finite_probability(x) for x in (hp, ap)) or hp + ap <= 0:
            skipped["invalid_challenger_pair_probabilities"] += 1
            continue
        ph = hp / (hp + ap)
        pa = 1 - ph
        chosen = home if ph >= 0.5 else away
        original_team = original["team"]
        challenger_original_side = ph if original_team == home else pa
        rows.append({
            "schema": SCHEMA, "status": NO_PROMOTION,
            "season": season, "week": week, "game_id": gid,
            "team_home": home, "team_away": away,
            "kickoff_utc": kickoff.isoformat(),
            "board_generation_utc": generated.isoformat(),
            "first_incumbent_logged_at_utc": original_at.isoformat(),
            "first_challenger_logged_at_utc": captured.isoformat(),
            "incumbent_selected_team": original_team,
            "original_incumbent_first_probability": original["model_prob"],
            "challenger_home_pre_pair_probability": hp,
            "challenger_away_pre_pair_probability": ap,
            "challenger_home_final_probability": ph,
            "challenger_away_final_probability": pa,
            "challenger_probability_for_incumbent_side": challenger_original_side,
            "challenger_selected_team": chosen,
            "challenger_selected_confidence": max(ph, pa),
            "challenger_changed_side": chosen != original_team,
            "frozen_challenger_model_sha256": model_digest,
            "original_board_sha256": board_digest,
            "strict_prior_week_games_home": hst["games_played"],
            "strict_prior_week_games_away": ast["games_played"],
            "both_teams_scored": True,
            "sportsbook_line_verified": False,
            "starter_status_verified": False,
            "betting_edge_certified": False,
            "official_prediction": False,
        })
    return rows, dict(skipped)


def append_only(path, new_rows):
    """Refuse corruption, duplicates, and rewriting earlier forecast versions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        seen = set()
        for i, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt first-pregame shadow ledger line {i}") from exc
            if r.get("schema") != SCHEMA:
                raise ValueError("unsupported shadow ledger row")
            key = (r.get("season"), r.get("week"), str(r.get("game_id")))
            if key in seen:
                raise ValueError("duplicate immutable shadow identity")
            seen.add(key)
        added = 0
        for row in new_rows:
            key = (row["season"], row["week"], row["game_id"])
            if key in seen:
                continue
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            seen.add(key)
            added += 1
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return added, len(seen)


def capture(db, board_path, original_ledger, model_dir, shadow_ledger, *,
            captured_at=None):
    now = captured_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("naive capture time")
    board_path = Path(board_path)
    if not board_path.is_file():
        raise FileNotFoundError(board_path)
    model_sha = check_model_snapshot(model_dir)
    board = json.loads(board_path.read_text())
    history = [json.loads(line) for line in Path(original_ledger).read_text().splitlines()
               if line.strip()]
    con = sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)
    try:
        scorer = ShadowScorer(con, model_dir)
        rows, skips = build_first_pregame_rows(
            con, board, history, scorer, captured_at=now,
            model_digest=model_sha, board_digest=_file_hash(board_path))
    finally:
        con.close()
    added, total = append_only(shadow_ledger, rows)
    return {
        "schema": SCHEMA, "status": NO_PROMOTION,
        "eligible_new_game_forecasts": len(rows),
        "new_immutable_first_pregame": added, "all_frozen_shadow_games": total,
        "rejections": skips, "production_changed": False,
        "needs_future_blind_outcomes_before_promotion": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--board", default="docs/cfb_predictions.json")
    ap.add_argument("--original-ledger", default="docs/cfb_picks_log.jsonl")
    ap.add_argument("--shadow-ledger", default="docs/cfb_asof_shadow_moneyline_first_seen.jsonl")
    ap.add_argument("--status", default="docs/cfb_asof_shadow_moneyline_status.json")
    args = ap.parse_args()
    result = capture(args.db, args.board, args.original_ledger, args.model_dir,
                     args.shadow_ledger)
    Path(args.status).parent.mkdir(parents=True, exist_ok=True)
    Path(args.status).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
