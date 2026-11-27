#!/usr/bin/env python3
"""Immutable CBB pretip forecast receipts, optional actual sportsbook quotes and grading.

Research-only. Requires a *previously generated* score-model output; does not
adjust the source model, blend sportsbook prices into projections, or backfill
old games with newly reconstructed forecasts.

Each forecast row is captured BEFORE the original scheduled tipoff. It is
stored once per event per checkpoint. Subsequent jobs cannot replace that
receipt. If no verified provider odds quote exists, there is NO ATS / O-U
betting prediction, probability, or simulated ROI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCHEMA = "cbb_pretip_first_seen_v1"
BOOKMAKER = "fanduel"
QUOTED_MARKETS = ("h2h", "spreads", "totals")
ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds/"


def timestamp(s):
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("timestamp without timezone offset")
    return d.astimezone(timezone.utc)


def as_float(x):
    try:
        f = float(x)
    except (ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


def teamkey(s):
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def american_to_implied(american):
    x = as_float(american)
    if x is None or abs(x) < 100 or abs(x) > 100000:
        return None
    return round((100. / (x + 100.)) if x > 0 else (-x / (100. - x)), 9)


def american_profit(american):
    x = as_float(american)
    if x is None or abs(x) < 100:
        return None
    return (x / 100.) if x > 0 else (100. / (-x))


def canonical_event(prediction):
    required = ("game_id", "tip_utc", "home", "away", "home_team_id", "away_team_id",
                "projected_home_margin", "projected_total", "p_home_win", "p_away_win")
    if any(prediction.get(k) is None for k in required):
        raise ValueError("missing CBB game prediction field")
    if str(prediction["home_team_id"]) == str(prediction["away_team_id"]):
        raise ValueError("identical teams")
    return str(prediction["game_id"])


def validate_american_quote(value):
    p = american_to_implied(value)
    return int(value) if p is not None and as_float(value).is_integer() else None


def fetch_the_odds_api(api_key):
    """Single, optional API call; no request when the key is absent."""
    if not api_key:
        return [], {"source": "THE_ODDS_API_V4", "status": "NOT_CONFIGURED"}
    params = urllib.parse.urlencode({
        "apiKey": api_key, "regions": "us", "markets": ",".join(QUOTED_MARKETS),
        "bookmakers": BOOKMAKER, "oddsFormat": "american", "dateFormat": "iso",
    })
    try:
        req = urllib.request.Request(
            ODDS_URL + "?" + params,
            headers={"User-Agent": "cbb-forward-research/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=25) as response:
            events = json.loads(response.read().decode("utf-8"))
            remaining = response.headers.get("x-requests-remaining")
        if not isinstance(events, list):
            raise ValueError("unexpected provider response type")
        return events, {"source": "THE_ODDS_API_V4", "status": "OK",
                        "events_received": len(events),
                        "requests_remaining": remaining}
    except Exception as exc:
        # No secret, authorization URL or headers are exposed in the receipt.
        return [], {"source": "THE_ODDS_API_V4", "status": "FETCH_FAILED",
                    "exception_class": type(exc).__name__}


def market_quote(prediction, events, captured_at, bookmaker=BOOKMAKER):
    """Join by *both* exact normalized team names and near-identical start time.

    Never guess a college nickname or use fuzzy team string similarity: wrong
    opponent and campus-vs-neutral joins are worse than no available line.
    """
    tip = timestamp(prediction["tip_utc"])
    if captured_at >= tip:
        return None
    teams = {teamkey(prediction["home"]), teamkey(prediction["away"])}
    if len(teams) != 2:
        return None
    matches = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if {teamkey(event.get("home_team")), teamkey(event.get("away_team"))} != teams:
            continue
        if teamkey(event.get("home_team")) != teamkey(prediction["home"]):
            continue
        try:
            event_tip = timestamp(event["commence_time"])
        except (ValueError, KeyError, TypeError):
            continue
        if abs((tip - event_tip).total_seconds()) > 900:
            continue
        matches.append(event)
    if len(matches) != 1:  # ambiguous matches => withhold
        return None
    event = matches[0]
    books = [b for b in (event.get("bookmakers") or [])
             if isinstance(b, dict) and str(b.get("key")).lower() == bookmaker]
    if len(books) != 1:
        return None
    book = books[0]
    try:
        book_update = timestamp(book["last_update"])
    except (ValueError, KeyError, TypeError):
        return None
    if book_update > captured_at or book_update >= tip:
        return None
    # Stale quotes are not genuine current pretip betting information.
    if (captured_at - book_update).total_seconds() > 6 * 3600:
        return None
    out = {
        "provider": "THE_ODDS_API_V4", "bookmaker": bookmaker, "event_id": str(event["id"]),
        "source_event_start_utc": timestamp(event["commence_time"]).isoformat(),
        "bookmaker_last_update_utc": book_update.isoformat(),
        "capture_utc": captured_at.isoformat(), "status": "PARTIAL",
        "home_ml_american": None, "away_ml_american": None,
        "home_spread": None, "away_spread": None, "home_spread_american": None,
        "away_spread_american": None, "total_line": None,
        "over_american": None, "under_american": None,
    }
    home = teamkey(prediction["home"])
    away = teamkey(prediction["away"])
    valid = set()
    for market in (book.get("markets") or []):
        if not isinstance(market, dict):
            continue
        name = market.get("key")
        outcomes = market.get("outcomes")
        if name not in QUOTED_MARKETS or name in valid:
            continue
        if not isinstance(outcomes, list) or len(outcomes) != 2:
            continue
        if name in {"h2h", "spreads"}:
            sides = {teamkey(o.get("name")): o for o in outcomes}
            if set(sides) != {home, away}:
                continue
            hp, ap = (validate_american_quote(sides[k].get("price")) for k in (home, away))
            if hp is None or ap is None:
                continue
            if name == "h2h":
                out["home_ml_american"], out["away_ml_american"] = hp, ap
            else:
                hs, aus = (as_float(sides[k].get("point")) for k in (home, away))
                if hs is None or aus is None or abs(hs + aus) > 1e-6:
                    continue
                out["home_spread"], out["away_spread"] = hs, aus
                out["home_spread_american"], out["away_spread_american"] = hp, ap
            valid.add(name)
        else:
            sides = {str(o.get("name", "")).lower(): o for o in outcomes}
            if set(sides) != {"over", "under"}:
                continue
            op, up = (validate_american_quote(sides[k].get("price")) for k in ("over", "under"))
            over, under = (as_float(sides[k].get("point")) for k in ("over", "under"))
            if None in (op, up, over, under) or abs(over - under) > 1e-6 or not 70 <= over <= 260:
                continue
            out["total_line"] = over
            out["over_american"], out["under_american"] = op, up
            valid.add(name)
    if not valid:
        return None
    out["status"] = "MATCHED_" + "_".join(sorted(valid)).upper()
    return out


def stdnorm(x):
    return .5 * (1. + math.erf(x / math.sqrt(2.)))


def discrete_two_sided(mean, sd, line):
    """Approximate integer scoreboard support by rounding a latent normal.

    Returns P(> line), P(< line), P(= line). Whole-number spreads/totals
    can push; fractional half-point lines cannot. Probabilities sum to one.
    """
    threshold = float(line)
    if threshold.is_integer():
        high = threshold + .5
        low = threshold - .5
        above = 1 - stdnorm((high - mean) / sd)
        below = stdnorm((low - mean) / sd)
        push = max(0., 1 - above - below)
    else:
        boundary = math.floor(threshold) + .5
        above = 1 - stdnorm((boundary - mean) / sd)
        below, push = 1 - above, 0.
    return above, below, push


def probs_with_market(prediction, quote, margin_sd, total_sd):
    """Every market from the original non-bookmaker score distribution."""
    if margin_sd <= 0 or total_sd <= 0:
        raise ValueError("invalid residual scale")
    mean_margin = float(prediction["projected_home_margin"])
    mean_total = float(prediction["projected_total"])
    hwin = stdnorm(mean_margin / margin_sd)
    out = {
        "home_win": round(hwin, 6), "away_win": round(1 - hwin, 6),
        "home_cover": None, "away_cover": None, "spread_push": None,
        "over": None, "under": None, "total_push": None,
        "home_win_no_vig_market": None, "home_cover_no_vig_market": None,
        "over_no_vig_market": None,
    }
    if not quote:
        return out
    hodd, aodd = quote["home_ml_american"], quote["away_ml_american"]
    if hodd is not None and aodd is not None:
        a, b = american_to_implied(hodd), american_to_implied(aodd)
        if a and b:
            out["home_win_no_vig_market"] = round(a / (a + b), 6)
    hs, aps = quote["home_spread"], quote["away_spread"]
    if hs is not None and aps is not None:
        hc, ac, pu = discrete_two_sided(mean_margin, margin_sd, -float(hs))
        out["home_cover"] = round(hc, 6)
        out["away_cover"] = round(ac, 6)
        out["spread_push"] = round(pu, 6)
        hp = american_to_implied(quote["home_spread_american"])
        ap = american_to_implied(quote["away_spread_american"])
        if hp and ap:
            out["home_cover_no_vig_market"] = round(hp / (hp + ap), 6)
    line = quote["total_line"]
    if line is not None:
        over, under, pu = discrete_two_sided(mean_total, total_sd, float(line))
        out["over"] = round(over, 6)
        out["under"] = round(under, 6)
        out["total_push"] = round(pu, 6)
        op = american_to_implied(quote["over_american"])
        up = american_to_implied(quote["under_american"])
        if op and up:
            out["over_no_vig_market"] = round(op / (op + up), 6)
    return out


def first_seen_receipts(forecast, odds_events=None, captured_at=None, checkpoint="T24"):
    captured_at = captured_at or datetime.now(timezone.utc)
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must contain a timezone")
    captured_at = captured_at.astimezone(timezone.utc)
    if checkpoint not in {"T24", "T90"}:
        raise ValueError("unknown checkpoint")
    if forecast.get("status") != "RESEARCH_ONLY_NO_PRETIP_FIRST_SEEN_ARCHIVE":
        raise ValueError("unrecognized research forecast")
    model_at = timestamp(forecast["generated_utc"])
    if model_at > captured_at + timedelta(minutes=2):
        raise ValueError("model timestamp is future relative to capture")
    report = forecast["report"]
    ms, ts = report["residual_margin_sd_from_2025"], report["residual_total_sd_from_2025"]
    records = []
    seen = set()
    for p in forecast["game_predictions"]:
        gid = canonical_event(p)
        tip = timestamp(p["tip_utc"])
        until = (tip - captured_at).total_seconds() / 60.
        # Near-game captures only: no seven-days-out snapshot later mis-sold as
        # a verified pre-tip forecast.
        if checkpoint == "T24" and not 90 < until <= 24 * 60:
            continue
        if checkpoint == "T90" and not 0 < until <= 90:
            continue
        if gid in seen:
            raise ValueError("duplicate game id in a single forecast output")
        seen.add(gid)
        if model_at >= tip:
            continue
        quote = market_quote(p, odds_events or [], captured_at)
        pms=float(p.get("estimated_margin_sd",ms))
        pts=float(p.get("estimated_total_sd",ts))
        if forecast.get("primary_model") == "CBB_HYBRID_V1" and (
            p.get("model_version") != "CBB_HYBRID_V1" or
            "estimated_margin_sd" not in p or "estimated_total_sd" not in p):
            raise ValueError("missing champion forecast provenance")
        probs = probs_with_market(p, quote, pms, pts)
        if p.get("model_version") == "CBB_HYBRID_V1":
            # The board's probability is authoritative; do not recalculate with
            # rounded margins or the retired competitor's scale.
            probs["home_win"]=float(p["p_home_win"])
            probs["away_win"]=float(p["p_away_win"])
        row = {
            "schema": SCHEMA,
            "model_version": p.get("model_version",forecast["schema"]),
            "model_generated_utc": model_at.isoformat(),
            "first_seen_utc": captured_at.isoformat(),
            "checkpoint": checkpoint,
            "game_id": gid, "tip_utc": tip.isoformat(),
            "home": p["home"], "away": p["away"],
            "home_team_id": str(p["home_team_id"]), "away_team_id": str(p["away_team_id"]),
            "home_prior_games": p["home_prior_games"], "away_prior_games": p["away_prior_games"],
            "neutral_site": bool(p.get("neutral_site")),
            "full_slate_shadow_challengers": p.get("full_slate_shadow_challengers", {}),
            "selection_2025_only": report.get("full_slate_challenger", {}).get("selection_2025_only", {}),
            "confirmed_lineup": False,
            "projected_home_score": p["projected_home_score"],
            "projected_away_score": p["projected_away_score"],
            "projected_home_margin": p["projected_home_margin"],
            "projected_total": p["projected_total"],
            "estimated_margin_sd": pms,
            "estimated_total_sd": pts,
            "game_intelligence": p.get("game_intelligence"),
            "market": quote,
            "probabilities": probs,
            "no_market_value_claim_if_quote_missing": quote is None,
            "research_only": True,
        }
        row["receipt_sha256"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        records.append(row)
    return records


def file_path(root, row):
    t = timestamp(row["tip_utc"])
    safe_id = re.sub(r"[^0-9a-zA-Z_-]", "", str(row["game_id"]))
    if not safe_id:
        raise ValueError("unsafe game id")
    return Path(root) / row["checkpoint"] / str(t.year) / f"{t.month:02d}" / f"{safe_id}.json"


def write_immutable(root, records):
    created, preexisting, conflicts = [], [], []
    for r in records:
        path = file_path(root, r)
        if path.exists():
            old = json.loads(path.read_text())
            key_fields = ("game_id", "checkpoint", "tip_utc", "home_team_id", "away_team_id")
            if any(old.get(k) != r.get(k) for k in key_fields):
                conflicts.append(str(path))
            else:
                preexisting.append(str(path))
            # Always leave the existing first-seen record untouched.
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create: never overwrite even under accidental duplicate.
        try:
            with path.open("x", encoding="utf-8") as f:
                json.dump(r, f, indent=2, sort_keys=True)
                f.write("\n")
        except FileExistsError:
            preexisting.append(str(path))
            continue
        created.append(str(path))
    if conflicts:
        raise ValueError("HISTORICAL_RECEIPT_IDENTITY_CONFLICT: " + ",".join(conflicts[:5]))
    return {"new": len(created), "unchanged_existing": len(preexisting),
            "created_paths": created}


def grade_receipt(row, final):
    """Score one historically frozen receipt versus official ESPN identity/final."""
    if not final.get("completed"):
        return {"status": "PENDING"}
    if any(str(final[k]) != str(row[k]) for k in ("game_id", "home_team_id", "away_team_id")):
        return {"status": "IDENTITY_MISMATCH"}
    home, away = as_float(final.get("home_score")), as_float(final.get("away_score"))
    if home is None or away is None or home < 0 or away < 0:
        return {"status": "INVALID_FINAL"}
    margin, total = home - away, home + away
    if margin == 0:
        return {"status": "INVALID_TIE"}
    pmargin, ptotal = float(row["projected_home_margin"]), float(row["projected_total"])
    pred = row["probabilities"]
    brier = (float(pred["home_win"]) - float(home > away)) ** 2
    result = {
        "status": "GRADED", "home_score": home, "away_score": away,
        "actual_margin": margin, "actual_total": total,
        "moneyline_home_correct": (float(pred["home_win"]) >= .5) == (home > away),
        "moneyline_brier": round(brier, 8),
        "absolute_margin_error": round(abs(pmargin - margin), 4),
        "absolute_total_error": round(abs(ptotal - total), 4),
        "markets": {},
    }
    result["shadow_model_grades"] = {}
    for name,p in row.get("full_slate_shadow_challengers", {}).items():
        prob=float(p["p_home_win"])
        result["shadow_model_grades"][name]={
            "moneyline_home_correct": (prob>=.5)==(home>away),
            "moneyline_brier": round((prob-float(home>away))**2,8),
            "absolute_margin_error": round(abs(float(p["projected_home_margin"])-margin),4),
            "absolute_total_error": round(abs(float(p["projected_total"])-total),4),
        }
    quote = row.get("market")
    if not quote:
        return result
    trials = [
        ("home_ml", float(home > away), pred["home_win"], quote["home_ml_american"]),
        ("away_ml", float(away > home), pred["away_win"], quote["away_ml_american"]),
    ]
    if quote.get("home_spread") is not None:
        adjusted = margin + float(quote["home_spread"])
        trials.extend([
            ("home_spread", None if adjusted == 0 else float(adjusted > 0),
             pred["home_cover"], quote["home_spread_american"]),
            ("away_spread", None if adjusted == 0 else float(adjusted < 0),
             pred["away_cover"], quote["away_spread_american"]),
        ])
    if quote.get("total_line") is not None:
        delta = total - float(quote["total_line"])
        trials.extend([
            ("over", None if delta == 0 else float(delta > 0),
             pred["over"], quote["over_american"]),
            ("under", None if delta == 0 else float(delta < 0),
             pred["under"], quote["under_american"]),
        ])
    for market, won, probability, price in trials:
        profit = american_profit(price)
        if profit is None or probability is None:
            continue
        push_prob = (pred.get("spread_push") if market.endswith("_spread")
                     else pred.get("total_push") if market in ("over", "under") else 0.)
        push_prob = as_float(push_prob) or 0.
        loss_prob = max(0., 1. - float(probability) - push_prob)
        result["markets"][market] = {
            "won": None if won is None else bool(won),
            "push": won is None,
            "p_model": float(probability),
            "p_push_model": push_prob,
            "american_odds": price,
            "expected_roi_per_unit": round(float(probability) * profit - loss_prob, 5),
            "realized_roi_per_unit": (0. if won is None else (profit if won else -1.)),
        }
    return result


def finals_from_schedule(schedule):
    """Narrow official final source: same ESPN game id and team identities."""
    completed = {}
    for row in schedule.to_dict("records"):
        if not row.get("game_id"):
            continue
        if not str(row.get("status_type_completed", "")).lower() in ("true", "1"):
            continue
        gid = str(row["game_id"])
        completed[gid] = {
            "completed": True, "game_id": gid,
            "home_team_id": str(row["home_id"]), "away_team_id": str(row["away_id"]),
            "home_score": row.get("home_score"), "away_score": row.get("away_score"),
        }
    return completed


def grade_frozen_folder(root, official_schedule, output_root):
    import pandas as pd
    tab = pd.read_parquet(official_schedule) if official_schedule.endswith(".parquet") else pd.read_csv(official_schedule)
    finals = finals_from_schedule(tab)
    stats = {"read": 0, "new_grades": 0, "already_graded": 0,
             "pending": 0, "identity_mismatch": 0, "invalid_finals": 0}
    for p in sorted(Path(root).glob("T*/**/*.json")):
        if "grade" in p.parts:
            continue
        row = json.loads(p.read_text())
        if row.get("schema") != SCHEMA:
            continue
        stats["read"] += 1
        gradefile = Path(output_root) / row["checkpoint"] / p.relative_to(Path(root) / row["checkpoint"])
        if gradefile.exists():
            stats["already_graded"] += 1
            continue
        result = grade_receipt(row, finals.get(str(row["game_id"]), {"completed": False}))
        if result["status"] == "PENDING":
            stats["pending"] += 1
            continue
        if result["status"] != "GRADED":
            stats["identity_mismatch" if result["status"] == "IDENTITY_MISMATCH" else "invalid_finals"] += 1
            continue
        gradefile.parent.mkdir(parents=True, exist_ok=True)
        with gradefile.open("x") as f:
            json.dump({
                "schema": "cbb_first_seen_grade_v1",
                "source_receipt_sha256": row["receipt_sha256"],
                "source_receipt": str(p), "graded_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": "ESPN_HOOPR_OFFICIAL_COMPLETED_SCHEDULE",
                "grade": result,
            }, f, indent=2, sort_keys=True)
            f.write("\n")
        stats["new_grades"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-odds", help="One shared optional provider request per job")
    fetch.add_argument("--out", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--forecast", required=True)
    freeze.add_argument("--out-root", required=True)
    freeze.add_argument("--odds-json", help="Actual The Odds API response fixture, never odds inferred from scores")
    freeze.add_argument("--checkpoint", choices=["T24", "T90"], required=True)
    freeze.add_argument("--capture-utc")
    grade = sub.add_parser("grade")
    grade.add_argument("--receipts", required=True)
    grade.add_argument("--official-schedule", required=True)
    grade.add_argument("--out-root", required=True)
    a = ap.parse_args()
    if a.command == "grade":
        print(json.dumps(grade_frozen_folder(a.receipts, a.official_schedule, a.out_root)))
        return
    if a.command == "fetch-odds":
        odds, status = fetch_the_odds_api(os.getenv("THE_ODDS_API_KEY"))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(odds, indent=2) + "\n")
        print(json.dumps({"odds_source": status, "saved_events": len(odds)}, sort_keys=True))
        return
    forecast = json.loads(Path(a.forecast).read_text())
    captured_at = timestamp(a.capture_utc) if a.capture_utc else datetime.now(timezone.utc)
    if a.odds_json:
        events = json.loads(Path(a.odds_json).read_text())
        status = {"status": "ACTUAL_ODDS_FILE", "events_received": len(events)}
    else:
        events, status = fetch_the_odds_api(os.getenv("THE_ODDS_API_KEY"))
    records = first_seen_receipts(forecast, events, captured_at, checkpoint=a.checkpoint)
    write = write_immutable(a.out_root, records)
    print(json.dumps({
        "checkpoint": a.checkpoint,
        "capture_utc": captured_at.isoformat(),
        "candidate_games": len(records),
        "odds_source": status, "markets_verified": sum(r["market"] is not None for r in records),
        "new_receipts": write["new"], "unchanged": write["unchanged_existing"],
        "created_paths": write["created_paths"][:10],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
