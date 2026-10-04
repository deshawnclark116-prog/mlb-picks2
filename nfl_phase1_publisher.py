"""
NFL_PHASE1_PUBLISHER -- deterministic, READ-ONLY presentation publisher for the NFL new-engine page.

Reads the checked-out `nfl-shadow-state` branch (dispatch ledger, status.json, forecast batches) and writes ONE file: docs/nfl_phase1_shadow.json.

Safety properties (enforced by tests/test_nfl_new_engine_frontend.py):
  * never imports the store / scheduler / simulator modules (nothing here can create or alter a forecast);
  * opens every state file for reading only; the state tree is hashed-identical before and after;
  * a forecast is published ONLY when ALL hold: it is in a finalized batch whose footer + sha256 verify; horizon in {T24, T90};
    the dispatch ledger's LAST state for (game_id, horizon, cutoff) is DONE / PARTIAL_V2_MISSING (never PLANNED / MISSED_REAL_CUTOFF / FAILED);
    the record is not an ad-hoc / time-travel / backfilled one (batch header without adhoc_label, input_snapshots.time_travel false, cutoff <= kickoff);
  * horizons are never combined: T24 and T90 live in separate arrays;
  * the sportsbook adapter is downstream display only: lines are joined into a separate `lines` section, never into / from the forecast record, and the file works with none.
Determinism: the output is a pure function of (state tree, odds cache, week override); the reference clock is state/status.json last_invocation_utc, never the wall clock.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "nfl-new-engine-shadow-v1"
HORIZONS = ("T24", "T90")
GOOD_STATES = ("DONE", "PARTIAL_V2_MISSING")
OUTCOMES = ["rush_yds", "rec_yds", "rec", "pass_yds", "pass_td", "int", "rush_td", "rec_td", "atd", "tackles", "sacks", "def_int"]
ROW_FIELDS = ["game_id", "player", "team", "opp", "pos", "outcome", "p_active", "exp_opp", "role_share", "mean", "median", "sd", "p10", "p25", "p75", "p90", "p0", "p_ge1", "unc", "unc_reasons", "id"]
LINE_MARKETS = {"rushing_yards": "rush_yds", "receiving_yards": "rec_yds"}
R11_STATUS = "UNRESOLVED (BLOCKER): operational N=100,000 draws; forecasts are shadow evidence, not freeze evidence"
HEARTBEAT_KEYS = ("last_shadow_run_utc", "last_provider_retrieval_utc", "reference_clock_utc")


class PublisherError(RuntimeError):
    pass


# ---------------------------------------------------------------- read-only state access
def _read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def read_batch_verified(path):
    """Verify footer count + sha256 exactly as the store does, with plain read-only I/O. Returns (header, records) or None for a batch that does not verify."""
    raw = Path(path).read_bytes()
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines = lines[:-1]
    if len(lines) < 2:
        return None
    try:
        footer = json.loads(lines[-1]); header = json.loads(lines[0])
        if not footer.get("_footer"):
            return None
        body = b"\n".join(lines[:-1]) + b"\n"
        if hashlib.sha256(body).hexdigest() != footer["sha256"]:
            return None
        recs = [json.loads(x) for x in lines[1:-1]]
        return (header, recs) if len(recs) == footer["n"] else None
    except Exception:                                                                # noqa
        return None


def ledger_last_state(state):
    last = {}
    for r in _read_jsonl(Path(state) / "dispatch_ledger.jsonl"):
        last[r["key"]] = r                                                                    # append-only file order: the last row wins
    return last


def week_of(game_id):
    s, w = game_id.split("_")[:2]
    return int(s), int(w)


def pick_week(last, ref, override=None):
    """Current week = week of the earliest still-upcoming planned decision key (kickoff after the reference clock); otherwise the latest week in the ledger."""
    if override:
        return tuple(override)
    ups = sorted((r["kickoff"], r["game_id"]) for r in last.values() if r["state"] == "PLANNED" and r["kickoff"] > ref)
    if ups:
        return week_of(ups[0][1])
    ws = sorted({week_of(r["game_id"]) for r in last.values()})
    return ws[-1] if ws else None


# ---------------------------------------------------------------- forecast selection
def genuine_records(state, last, week):
    """-> {horizon: {game_id: [records]}} of genuine immutable forecasts for `week` (and the set of ids, to prove uniqueness)."""
    out = {h: {} for h in HORIZONS}
    seen = {}
    for f in sorted((Path(state) / "forecasts" / "batches").glob("*.jsonl")):
        got = read_batch_verified(f)
        if got is None:
            continue
        header, recs = got
        if header.get("adhoc_label") or str(header.get("horizon", "")).startswith("AD_HOC"):
            continue
        for r in recs:
            hz = r.get("horizon")
            if hz not in HORIZONS or "outcome" not in r or r.get("season") is None or (r["season"], r["week"]) != week:
                continue
            key = f"{r['game_id']}|{hz}|{r['cutoff']}"
            led = last.get(key)
            if not led or led["state"] not in GOOD_STATES:
                continue
            snaps = r.get("input_snapshots") or {}
            if (isinstance(snaps, dict) and snaps.get("time_travel")) or r["cutoff"] > r["kickoff"]:
                continue
            if r["id"] in seen:
                if seen[r["id"]] != json.dumps(r, sort_keys=True):
                    raise PublisherError(f"conflicting bytes for forecast id {r['id']}")
                continue                                                                      # verified duplicate: counted once
            seen[r["id"]] = json.dumps(r, sort_keys=True)
            out[hz].setdefault(r["game_id"], []).append(r)
    return out


def _r(x, n=4):
    return None if x is None else round(float(x), n)


def row_of(r):
    q = r.get("quantiles") or {}
    rs = r.get("role_state") or {}
    unc = r.get("uncertainty") or {}
    share = rs.get("propensity_share")
    return [r["game_id"], r["player_name"], r["team"], r["opponent"], r["position"], r["outcome"], _r(r["p_active"]), _r(r.get("expected_opportunities")), _r(share), _r(r["mean"]), _r(r["median"]), _r(r["sd"]),
            _r(q.get("p10")), _r(q.get("p25")), _r(q.get("p75")), _r(q.get("p90")), _r(r["p_zero"]), _r(r.get("event_probability_ge1")), _r(unc.get("score")), list(unc.get("reasons") or []), r["id"]]


# ---------------------------------------------------------------- optional downstream sportsbook adapter (display only)
def p_over_under(q99, line):
    """Approximate P(X > line), P(X <= line) from the stored 99-point quantile grid (levels 0.01..0.99), linear interpolation; resolution ~1 percentage point."""
    q = np.asarray(q99, float); lv = np.arange(1, 100) / 100.0
    if line < q[0]:
        f = 0.005
    elif line >= q[-1]:
        f = 0.995
    else:
        f = float(np.interp(line, q, lv))
    return round(1 - f, 3), round(f, 3)


def line_adapter(odds_path, recs_by_hz):
    """-> list of {horizon, game_id, player, outcome, line, book, over_price, under_price, p_over, p_under}. Joined by normalized player name + outcome; ambiguous names are skipped."""
    p = Path(odds_path) if odds_path else None
    if not p or not p.exists():
        return [], None
    try:
        d = json.loads(p.read_text())
    except Exception:                                                                        # noqa
        return [], None
    out = []
    for hz in HORIZONS:
        idx = {}
        for gid, recs in recs_by_hz[hz].items():
            for r in recs:
                if r["outcome"] in LINE_MARKETS.values():
                    idx.setdefault((r["player_name"].strip().lower(), r["outcome"]), []).append(r)
        for ln in d.get("lines", []):
            oc = LINE_MARKETS.get(ln.get("market"))
            c = idx.get((str(ln.get("player_norm", "")).strip().lower(), oc)) if oc else None
            if not c or len(c) != 1 or ln.get("line") is None:
                continue
            r = c[0]
            po, pu = p_over_under(r["quantile_grid_99"], float(ln["line"]))
            out.append({"horizon": hz, "game_id": r["game_id"], "player": r["player_name"], "outcome": oc, "line": ln["line"], "book": ln.get("book"), "over_price": ln.get("over_price"), "under_price": ln.get("under_price"), "p_over": po,
                        "p_under": pu})
    out.sort(key=lambda x: (x["horizon"], x["game_id"], x["player"], x["outcome"], str(x["book"]), x["line"]))
    return out, {"generated_at_utc": d.get("generated_at_utc"), "date_et": d.get("date_et"), "note": "downstream display only; never an input to the forecast"}


# ---------------------------------------------------------------- build
def build(state, odds_path=None, week_override=None):
    state = Path(state)
    status = json.loads((state / "status.json").read_text()) if (state / "status.json").exists() else {}
    ref = status.get("last_invocation_utc") or ""
    last = ledger_last_state(state)
    week = pick_week(last, ref, week_override)
    keys = [r for r in last.values() if week and week_of(r["game_id"]) == week]
    recs = genuine_records(state, last, week) if week else {h: {} for h in HORIZONS}
    forecasts, games, counts = {}, {}, {}
    for hz in HORIZONS:
        rows = sorted((row_of(r) for g in recs[hz].values() for r in g), key=lambda x: (x[0], x[2], x[1], OUTCOMES.index(x[5]) if x[5] in OUTCOMES else 99, x[-1]))
        forecasts[hz] = rows
        ks = {r["game_id"]: r for r in keys if r["horizon"] == hz and r["state"] in GOOD_STATES}
        for gid in recs[hz]:
            r0 = recs[hz][gid][0]
            games.setdefault(gid, {"game_id": gid, "away": None, "home": None, "kickoff_utc": r0["kickoff"], "horizons": {}})
            games[gid]["horizons"][hz] = {"cutoff": r0["cutoff"], "model_version": r0["model_version"], "n_records": len(recs[hz][gid]), "n_draws": (r0.get("simulation") or {}).get("n_draws")}
        counts[hz] = {"valid_games": len(recs[hz]), "missed": sum(r["state"] == "MISSED_REAL_CUTOFF" for r in keys if r["horizon"] == hz), "failed": sum(r["state"] == "FAILED" for r in keys if r["horizon"] == hz),
                      "planned": sum(r["state"] == "PLANNED" for r in keys if r["horizon"] == hz), "ledger_done_keys": len(ks)}
    for gid, g in games.items():                                                                           # home/away: the first record's team/opponent pair; resolved from the game id (season_week_AWAY_HOME)
        parts = gid.split("_")
        g["away"], g["home"] = (parts[2], parts[3]) if len(parts) >= 4 else (None, None)
    versions = sorted({m["model_version"] for g in games.values() for m in g["horizons"].values()})
    nxt = sorted((r["cutoff"], r["key"], r["kickoff"]) for r in last.values() if r["state"] == "PLANNED" and r["cutoff"] > ref)
    lines, odds_meta = line_adapter(odds_path, recs)
    default = "T90" if counts["T90"]["valid_games"] else "T24" if counts["T24"]["valid_games"] else None
    return {
        "schema": SCHEMA,
        "labels": ["NEW NFL OUTCOME ENGINE", "SHADOW / RESEARCH", "NO SPORTSBOOK INPUTS", "NOT YET PROMOTED TO PRODUCTION"],
        "empty_message": "No valid New Engine forecast has been captured for this slate yet.",
        "status": {"season": week[0] if week else None, "week": week[1] if week else None, "model_versions": versions, "r11": R11_STATUS, "last_shadow_run_utc": status.get("last_invocation_utc"),
                   "last_provider_retrieval_utc": status.get("last_successful_provider_retrieval_utc"), "reference_clock_utc": ref or None, "readiness_ready": (status.get("readiness") or {}).get("READY"),
                   "counts": counts, "next_cutoff": ({"cutoff": nxt[0][0], "key": nxt[0][1], "kickoff": nxt[0][2]} if nxt else None), "scheduler_version": status.get("scheduler_version")},
        "default_horizon": default,
        "row_fields": ROW_FIELDS, "outcomes": OUTCOMES,
        "games": [games[g] for g in sorted(games)],
        "forecasts": forecasts,
        "lines": lines, "lines_meta": odds_meta,
    }


def stable_view(doc):
    d = json.loads(json.dumps(doc))
    for k in HEARTBEAT_KEYS:
        d["status"].pop(k, None)
    return d


def dumps(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def publish(state, out, odds_path=None, week_override=None, min_refresh_min=60):
    """Write `out` only if the forecast-bearing content changed, or the heartbeat is older than min_refresh_min (keeps git history quiet). Returns 'written' | 'unchanged'."""
    doc = build(state, odds_path, week_override)
    out = Path(out)
    if out.exists():
        try:
            old = json.loads(out.read_text())
            if stable_view(old) == stable_view(doc):
                from datetime import datetime
                a, b = old["status"].get("reference_clock_utc"), doc["status"].get("reference_clock_utc")
                if a and b and (datetime.fromisoformat(b.replace("Z", "+00:00")) - datetime.fromisoformat(a.replace("Z", "+00:00"))).total_seconds() < min_refresh_min * 60:
                    return "unchanged"
        except Exception:                                                                    # noqa
            pass
    out.write_text(dumps(doc))
    return "written"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True); ap.add_argument("--out", default="docs/nfl_phase1_shadow.json"); ap.add_argument("--odds", default="docs/nfl_odds_cache.json")
    ap.add_argument("--min-refresh-min", type=int, default=60)
    a = ap.parse_args()
    print(publish(a.state, a.out, a.odds, None, a.min_refresh_min))
