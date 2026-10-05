#!/usr/bin/env python3
"""NFL V2 Phase 1B: opportunity-first direct projection candidate.

Research-only. No Monte Carlo. No sportsbook inputs.

This candidate exists to answer the Week 4 forensic finding that many V1 misses
started with workload. It therefore selects the opportunity layer separately
from the efficiency layer:

  1. tune team-volume + role-share logic on burned 2024 to minimize player
     opportunity error;
  2. freeze that opportunity configuration;
  3. tune efficiency on the same burned 2024 period;
  4. evaluate both layers on untouched-for-this-candidate 2025 burned
     validation data;
  5. report 2026 Weeks 1-4 only as already-burned diagnostics.

The direct point forecast is still:

    projected player opportunities x projected efficiency = point projection

The point of this phase is not to declare a champion. It is to stop two wrong
components from receiving credit merely because their final-stat errors cancel.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import nfl_v2_phase1a_direct as p1a

OUTCOMES = p1a.OUTCOMES
SPEC = p1a.SPEC

_POS_INDEX = defaultdict(list)
_ALL_TEAM = []


def build_extra_indexes(players, team_totals):
    _POS_INDEX.clear()
    _ALL_TEAM.clear()
    for r in players:
        if r.get("position"):
            _POS_INDEX[r["position"]].append(r)
    for pos in _POS_INDEX:
        _POS_INDEX[pos].sort(key=lambda r: (r["season"], r["week"], r["player_id"]))
    for (s, w, team), st in team_totals.items():
        _ALL_TEAM.append(((s, w), team, st))
    _ALL_TEAM.sort(key=lambda x: (x[0], x[1]))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def ewma(xs, decay):
    """Chronological values, latest observation has weight 1."""
    if not xs:
        return None
    if decay >= 0.999999:
        return mean(xs)
    n = len(xs)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    den = sum(weights)
    return sum(x * w for x, w in zip(xs, weights)) / den if den else None


def prior_league_team_values(target, col, window):
    vals = []
    for key, _team, st in _ALL_TEAM:
        if key >= target:
            break
        vals.append(st.get(col, 0.0))
    return vals[-window:]


def role_projection(shares, decay, shift_gain):
    if not shares:
        return None
    base = ewma(shares, decay)
    if len(shares) < 2 or shift_gain <= 0:
        return clamp(base, 0.0, 1.0)
    prior = ewma(shares[:-1], decay)
    shift = shares[-1] - prior
    return clamp(base + shift_gain * shift, 0.0, 1.0)


def opportunity_receipt(players, team_totals, team_opp, r, outcome, cfg):
    sp = SPEC[outcome]
    target = (r["season"], r["week"])
    tm, opp = r["team"], r["opponent"]

    own = p1a.prior_team_values(
        team_totals, team_opp, target, tm, sp["opp"], cfg["team_window"], False
    )
    allowed = p1a.prior_team_values(
        team_totals, team_opp, target, opp, sp["opp"], cfg["team_window"], True
    )
    league = prior_league_team_values(target, sp["opp"], cfg["league_window"])
    shares = p1a.player_share_history(
        players, team_totals, target, r["player_id"], tm, sp["opp"], cfg["share_window"]
    )

    if len(own) < 2 or len(allowed) < 2 or len(league) < 16 or len(shares) < 2:
        return None

    own_v = ewma(own, cfg["team_decay"])
    allowed_v = ewma(allowed, cfg["team_decay"])
    league_v = mean(league)
    role = role_projection(shares, cfg["role_decay"], cfg["shift_gain"])

    ow = cfg["offense_weight"]
    dw = cfg["defense_weight"]
    lw = 1.0 - ow - dw
    team_proj = ow * own_v + dw * allowed_v + lw * league_v
    player_proj = team_proj * role

    return {
        "offense_recent_team_opportunities": own_v,
        "opponent_recent_allowed_opportunities": allowed_v,
        "league_recent_team_opportunities": league_v,
        "offense_weight": ow,
        "defense_weight": dw,
        "league_weight": lw,
        "projected_role_share": role,
        "last_role_share": shares[-1],
        "role_share_history_n": len(shares),
        "team_opportunity_projection": team_proj,
        "player_opportunity_projection": player_proj,
    }


def position_efficiency_prior(players, target, position, value_col, opp_col, max_rows=600):
    if not position:
        return None
    rows = _POS_INDEX.get(position, ())
    eligible = [r for r in rows if (r["season"], r["week"]) < target and r.get(opp_col, 0.0) > 0]
    eligible = eligible[-max_rows:]
    den = sum(r[opp_col] for r in eligible)
    return None if den <= 0 else sum(r[value_col] for r in eligible) / den


def league_efficiency_prior(target, value_col, opp_col, max_team_games=256):
    vals = []
    for key, _team, st in _ALL_TEAM:
        if key >= target:
            break
        den = st.get(opp_col, 0.0)
        if den > 0:
            vals.append((st.get(value_col, 0.0), den))
    vals = vals[-max_team_games:]
    den = sum(x[1] for x in vals)
    return None if den <= 0 else sum(x[0] for x in vals) / den


def player_eff_with_shrink(players, target, r, outcome, cfg):
    sp = SPEC[outcome]
    hist = p1a.prior_player_rows(
        players, target, r["player_id"], None, cfg["eff_window"]
    )
    hist = [x for x in hist if x.get(sp["opp"], 0.0) > 0]
    den = sum(x[sp["opp"]] for x in hist)
    num = sum(x[sp["value"]] for x in hist)
    pos = position_efficiency_prior(
        players, target, r.get("position"), sp["value"], sp["opp"]
    )
    if den <= 0 or pos is None:
        return None
    k = cfg["player_eff_shrink_opps"]
    return (num + k * pos) / (den + k)


def defense_eff_with_shrink(team_totals, team_opp, target, defense, outcome, cfg):
    sp = SPEC[outcome]
    vals = []
    s0, w0 = target
    for (s, w, offense), st in team_totals.items():
        if (s, w) >= (s0, w0):
            continue
        if team_opp.get((s, w, offense)) != defense:
            continue
        den = st.get(sp["opp"], 0.0)
        if den > 0:
            vals.append(((s, w), st.get(sp["value"], 0.0), den))
    vals.sort()
    vals = vals[-cfg["def_eff_window"]:]
    den = sum(x[2] for x in vals)
    num = sum(x[1] for x in vals)
    league = league_efficiency_prior(target, sp["value"], sp["opp"])
    if den <= 0 or league is None:
        return None
    k = cfg["def_eff_shrink_opps"]
    return (num + k * league) / (den + k)


def efficiency_receipt(players, team_totals, team_opp, r, outcome, cfg):
    target = (r["season"], r["week"])
    pe = player_eff_with_shrink(players, target, r, outcome, cfg)
    de = defense_eff_with_shrink(
        team_totals, team_opp, target, r["opponent"], outcome, cfg
    )
    if pe is None or de is None:
        return None
    dw = cfg["matchup_eff_weight"]
    final = (1.0 - dw) * pe + dw * de
    return {
        "player_shrunk_efficiency": pe,
        "opponent_shrunk_allowed_efficiency": de,
        "matchup_eff_weight": dw,
        "final_efficiency_projection": final,
    }


def opportunity_configs():
    for td, ow, dw, rd, sg in itertools.product(
        (0.55, 0.75, 1.0),
        (0.60, 0.75),
        (0.10, 0.20),
        (0.50, 0.75, 1.0),
        (0.0, 0.25, 0.50),
    ):
        if ow + dw >= 1.0:
            continue
        yield {
            "team_window": 8,
            "league_window": 96,
            "share_window": 5,
            "team_decay": td,
            "offense_weight": ow,
            "defense_weight": dw,
            "role_decay": rd,
            "shift_gain": sg,
        }


def efficiency_configs(outcome):
    shrink = {
        "rush_yds": (20.0, 50.0),
        "rec_yds": (20.0, 50.0),
        "rec": (20.0, 50.0),
        "pass_yds": (75.0, 150.0),
    }[outcome]
    dshrink = {
        "rush_yds": (80.0, 160.0),
        "rec_yds": (80.0, 160.0),
        "rec": (80.0, 160.0),
        "pass_yds": (200.0, 400.0),
    }[outcome]
    for ew, pk, dk, mw in itertools.product(
        (5, 8), shrink, dshrink, (0.0, 0.15, 0.30)
    ):
        yield {
            "eff_window": ew,
            "player_eff_shrink_opps": pk,
            "def_eff_window": 8,
            "def_eff_shrink_opps": dk,
            "matchup_eff_weight": mw,
        }


def target_rows(players, season):
    return [r for r in players if r["season"] == season and 1 <= r["week"] <= 18]


def evaluate_opportunity(players, team_totals, team_opp, rows, outcome, cfg):
    errs = []
    role_errs = []
    team_errors = {}
    receipts = []
    sp = SPEC[outcome]
    for r in rows:
        rec = opportunity_receipt(players, team_totals, team_opp, r, outcome, cfg)
        if rec is None or rec["player_opportunity_projection"] < sp["min_pred_opp"]:
            continue
        actual_opp = r[sp["opp"]]
        errs.append(abs(rec["player_opportunity_projection"] - actual_opp))
        actual_team = team_totals.get((r["season"], r["week"], r["team"]), {}).get(sp["opp"], 0.0)
        if actual_team > 0:
            actual_share = actual_opp / actual_team
            role_errs.append(abs(rec["projected_role_share"] - actual_share))
            tk = (r["season"], r["week"], r["team"])
            team_errors[tk] = abs(rec["team_opportunity_projection"] - actual_team)
        receipts.append((r, rec, actual_opp))
    return {
        "n": len(errs),
        "player_opportunity_mae": mean(errs),
        "role_share_mae": mean(role_errs),
        "team_opportunity_mae": mean(team_errors.values()),
        "receipts": receipts,
    }


def evaluate_full(players, team_totals, team_opp, rows, outcome, opp_cfg, eff_cfg):
    sp = SPEC[outcome]
    errs, signed, opp_errs, role_errs = [], [], [], []
    team_errors = {}
    receipts = []
    for r in rows:
        o = opportunity_receipt(players, team_totals, team_opp, r, outcome, opp_cfg)
        if o is None or o["player_opportunity_projection"] < sp["min_pred_opp"]:
            continue
        e = efficiency_receipt(players, team_totals, team_opp, r, outcome, eff_cfg)
        if e is None:
            continue
        point = o["player_opportunity_projection"] * e["final_efficiency_projection"]
        y = r[sp["value"]]
        ao = r[sp["opp"]]
        errs.append(abs(point - y))
        signed.append(point - y)
        opp_errs.append(abs(o["player_opportunity_projection"] - ao))
        actual_team = team_totals.get((r["season"], r["week"], r["team"]), {}).get(sp["opp"], 0.0)
        if actual_team > 0:
            role_errs.append(abs(o["projected_role_share"] - ao / actual_team))
            tk = (r["season"], r["week"], r["team"])
            team_errors[tk] = abs(o["team_opportunity_projection"] - actual_team)
        receipts.append((r, {**o, **e, "point_projection": point}, y))
    return {
        "n": len(errs),
        "mae": mean(errs),
        "bias": mean(signed),
        "player_opportunity_mae": mean(opp_errs),
        "role_share_mae": mean(role_errs),
        "team_opportunity_mae": mean(team_errors.values()),
        "receipts": receipts,
    }


def choose_opportunity_on_2024(players, team_totals, team_opp, outcome):
    rows = target_rows(players, 2024)
    minimum = {"rush_yds": 220, "rec_yds": 500, "rec": 500, "pass_yds": 180}[outcome]
    best = None
    for cfg in opportunity_configs():
        ev = evaluate_opportunity(players, team_totals, team_opp, rows, outcome, cfg)
        if ev["n"] < minimum or ev["player_opportunity_mae"] is None:
            continue
        key = (
            ev["player_opportunity_mae"],
            ev["role_share_mae"] if ev["role_share_mae"] is not None else 1e9,
            ev["team_opportunity_mae"] if ev["team_opportunity_mae"] is not None else 1e9,
            json.dumps(cfg, sort_keys=True),
        )
        if best is None or key < best[0]:
            best = (key, cfg, ev)
    if best is None:
        raise RuntimeError(f"No eligible 2024 opportunity config for {outcome}")
    return best[1], best[2]


def choose_efficiency_on_2024(players, team_totals, team_opp, outcome, opp_cfg):
    rows = target_rows(players, 2024)
    minimum = {"rush_yds": 220, "rec_yds": 500, "rec": 500, "pass_yds": 180}[outcome]
    best = None
    for cfg in efficiency_configs(outcome):
        ev = evaluate_full(players, team_totals, team_opp, rows, outcome, opp_cfg, cfg)
        if ev["n"] < minimum or ev["mae"] is None:
            continue
        key = (ev["mae"], abs(ev["bias"] or 0.0), json.dumps(cfg, sort_keys=True))
        if best is None or key < best[0]:
            best = (key, cfg, ev)
    if best is None:
        raise RuntimeError(f"No eligible 2024 efficiency config for {outcome}")
    return best[1], best[2]


def summarize(ev, outcome):
    if not ev or not ev.get("n"):
        return {"n": 0}
    tols = (5, 10, 15, 20, 25) if outcome.endswith("_yds") else (0, 1, 2)
    out = {
        "n": ev["n"],
        "mae": ev.get("mae"),
        "bias": ev.get("bias"),
        "player_opportunity_mae": ev.get("player_opportunity_mae"),
        "role_share_mae": ev.get("role_share_mae"),
        "team_opportunity_mae": ev.get("team_opportunity_mae"),
    }
    if ev.get("receipts") and "point_projection" in ev["receipts"][0][1]:
        es = [abs(p["point_projection"] - y) for _r, p, y in ev["receipts"]]
        out["within"] = {str(t): sum(e <= t for e in es) / len(es) for t in tols}
    return out


def compare_week4_to_v1(players, team_totals, team_opp, audit, outcome, opp_cfg, eff_cfg):
    clean = {
        (r["player_id"], r["outcome"]): r
        for r in audit["rows"]
        if r.get("clean_meaningful") and r["outcome"] == outcome
    }
    candidates = [
        r for r in players
        if r["season"] == 2026 and r["week"] == 4 and (r["player_id"], outcome) in clean
    ]
    rows = []
    for r in candidates:
        o = opportunity_receipt(players, team_totals, team_opp, r, outcome, opp_cfg)
        e = efficiency_receipt(players, team_totals, team_opp, r, outcome, eff_cfg)
        if o is None or e is None:
            continue
        point = o["player_opportunity_projection"] * e["final_efficiency_projection"]
        a = clean[(r["player_id"], outcome)]
        y = float(a["actual"])
        v1 = float(a["median"])
        rows.append({
            "player_id": r["player_id"],
            "player": a["player"],
            "team": r["team"],
            "opponent": r["opponent"],
            "outcome": outcome,
            "actual": y,
            "v1_median": v1,
            "v1_abs_error": abs(v1 - y),
            "v2_phase1b_point": point,
            "v2_phase1b_abs_error": abs(point - y),
            "actual_opportunities": r[SPEC[outcome]["opp"]],
            "receipt": {**o, **e, "point_projection": point},
        })
    if not rows:
        return {"n": 0, "rows": []}
    v1e = [x["v1_abs_error"] for x in rows]
    v2e = [x["v2_phase1b_abs_error"] for x in rows]
    return {
        "n": len(rows),
        "v1_mae_same_rows": mean(v1e),
        "v2_phase1b_mae": mean(v2e),
        "v2_minus_v1_mae": mean(v2e) - mean(v1e),
        "v2_row_wins": sum(a < b for a, b in zip(v2e, v1e)),
        "v1_row_wins": sum(b < a for a, b in zip(v2e, v1e)),
        "rows": sorted(rows, key=lambda x: -x["v1_abs_error"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="append", required=True)
    ap.add_argument("--week4-audit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    players, team_totals, team_opp = p1a.load_stats(a.stats)
    p1a.build_indexes(players, team_totals, team_opp)
    build_extra_indexes(players, team_totals)
    audit = json.loads(Path(a.week4_audit).read_text())

    report = {
        "schema": "nfl-v2-phase1b-opportunity-first-v1",
        "status": "BURNED_RESEARCH_ONLY",
        "sportsbook_inputs_used": False,
        "monte_carlo_used": False,
        "selection_period": "2024 regular season only",
        "burned_validation_period": "2025 regular season; not used for candidate selection",
        "diagnostic_period": "2026 weeks 1-4 burned",
        "selection_policy": "opportunity config minimizes player opportunity MAE first; efficiency config is selected only after opportunity config is frozen",
        "formula": "projected team opportunity x projected player role share x projected efficiency",
        "outcomes": {},
    }

    for oc in OUTCOMES:
        opp_cfg, opp_dev = choose_opportunity_on_2024(players, team_totals, team_opp, oc)
        eff_cfg, full_dev = choose_efficiency_on_2024(
            players, team_totals, team_opp, oc, opp_cfg
        )
        val_2025 = evaluate_full(
            players, team_totals, team_opp, target_rows(players, 2025), oc, opp_cfg, eff_cfg
        )
        diag_rows = [
            r for r in players
            if r["season"] == 2026 and 1 <= r["week"] <= 4
        ]
        diag = evaluate_full(
            players, team_totals, team_opp, diag_rows, oc, opp_cfg, eff_cfg
        )
        week4 = compare_week4_to_v1(
            players, team_totals, team_opp, audit, oc, opp_cfg, eff_cfg
        )
        report["outcomes"][oc] = {
            "selected_opportunity_config": opp_cfg,
            "selected_efficiency_config": eff_cfg,
            "selection_2024_opportunity": {
                "n": opp_dev["n"],
                "player_opportunity_mae": opp_dev["player_opportunity_mae"],
                "role_share_mae": opp_dev["role_share_mae"],
                "team_opportunity_mae": opp_dev["team_opportunity_mae"],
            },
            "selection_2024_full": summarize(full_dev, oc),
            "burned_validation_2025": summarize(val_2025, oc),
            "diagnostic_2026_wk1_4": summarize(diag, oc),
            "week4_exact_v1_clean_comparison": week4,
        }

    Path(a.out).write_text(json.dumps(report, indent=2, sort_keys=True))
    compact = {
        oc: {
            "opp_cfg": v["selected_opportunity_config"],
            "eff_cfg": v["selected_efficiency_config"],
            "val2025": v["burned_validation_2025"],
            "diag2026": v["diagnostic_2026_wk1_4"],
            "week4": {
                k: x
                for k, x in v["week4_exact_v1_clean_comparison"].items()
                if k != "rows"
            },
        }
        for oc, v in report["outcomes"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
