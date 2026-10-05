#!/usr/bin/env python3
"""NFL V2 Phase 1C-T: team opportunity / game-environment research head.

This phase predicts *team* opportunities before any player allocation:
- team pass attempts
- team targets
- team carries

It deliberately does NOT predict player yards, receptions, TDs, or betting
probabilities. The question is narrower: can we predict the amount and type of
offensive opportunity a team will create better than simple history and the
Phase 1B team-volume layer?

Football inputs are historical-only:
- official prior team opportunities from nflverse weekly stats;
- prior play-by-play dropbacks, rush plays, total offensive plays;
- neutral-situation pass/run tendency;
- opponent allowed play/opportunity history;
- current known home/away and rest context from the schedule.

Selection: burned 2024 only, with 2023 supplying prior history.
Validation: 2025, never used for selection.
Diagnostics: 2026 Weeks 1-4, already burned.
No Monte Carlo. No sportsbook inputs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import nfl_v2_phase1a_direct as p1a

OPPORTUNITIES = ("attempts", "targets", "carries")
PB_COMPONENT = {
    "attempts": "dropbacks",
    "targets": "dropbacks",
    "carries": "rush_plays",
}
NEUTRAL_COMPONENT = {
    "attempts": "neutral_pass_rate",
    "targets": "neutral_pass_rate",
    "carries": "neutral_rush_rate",
}

_TEAM_WEEK = defaultdict(list)
_ALLOWED_WEEK = defaultdict(list)
_LEAGUE_WEEK = []
_PBP_TEAM_WEEK = defaultdict(list)
_PBP_ALLOWED_WEEK = defaultdict(list)
_PBP_LEAGUE_WEEK = []
_SCHEDULE = {}


def fnum(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def flag(v):
    x = fnum(v)
    return 0.0 if x is None else x


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def ewma(xs, decay):
    if not xs:
        return None
    if decay >= 0.999999:
        return mean(xs)
    n = len(xs)
    ws = [decay ** (n - 1 - i) for i in range(n)]
    den = sum(ws)
    return sum(x * w for x, w in zip(xs, ws)) / den if den else None


def open_csv(path):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt", newline="", encoding="utf-8")
    return open(p, newline="", encoding="utf-8")


def load_pbp(paths):
    """Aggregate prior-game football structure from play-by-play.

    Plays are defined as dropbacks + designed/recorded rush attempts. The
    neutral split uses Q1-Q3 plays with pre-play score differential within 7.
    """
    agg = defaultdict(lambda: defaultdict(float))
    opp = {}
    for path in paths:
        with open_csv(path) as f:
            for r in csv.DictReader(f):
                try:
                    s = int(float(r.get("season") or 0))
                    w = int(float(r.get("week") or 0))
                except (TypeError, ValueError):
                    continue
                if s <= 0 or w <= 0:
                    continue
                gt = str(r.get("season_type") or r.get("game_type") or "REG")
                if gt != "REG":
                    continue
                offense = r.get("posteam")
                defense = r.get("defteam")
                if not offense or not defense:
                    continue

                drop = flag(r.get("qb_dropback"))
                rush = flag(r.get("rush_attempt"))
                # Keep only football opportunity plays. Penalty/no-play rows
                # generally have both flags at zero under nflfastR semantics.
                if drop <= 0 and rush <= 0:
                    continue

                key = (s, w, offense)
                opp[key] = defense
                st = agg[key]
                st["dropbacks"] += drop
                st["rush_plays"] += rush
                # A scramble can be both a historical dropback and a rushing
                # attempt. It is still only one offensive play.
                play = 1.0 if (drop > 0 or rush > 0) else 0.0
                st["plays"] += play

                qtr = fnum(r.get("qtr"))
                sd = fnum(r.get("score_differential"))
                if qtr is not None and qtr <= 3 and sd is not None and abs(sd) <= 7:
                    st["neutral_dropbacks"] += drop
                    st["neutral_rush_plays"] += rush
                    st["neutral_plays"] += play

    out = {}
    for key, st in agg.items():
        n = st.get("neutral_plays", 0.0)
        st["neutral_pass_rate"] = st.get("neutral_dropbacks", 0.0) / n if n > 0 else None
        st["neutral_rush_rate"] = st.get("neutral_rush_plays", 0.0) / n if n > 0 else None
        out[key] = dict(st)
    return out, opp


def load_schedule(path):
    games = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt = str(r.get("game_type") or r.get("season_type") or "REG")
            if gt != "REG":
                continue
            try:
                s = int(float(r.get("season") or 0))
                w = int(float(r.get("week") or 0))
            except (TypeError, ValueError):
                continue
            home, away = r.get("home_team"), r.get("away_team")
            raw = r.get("gameday") or r.get("game_date") or r.get("date")
            if not home or not away or not raw:
                continue
            try:
                d = date.fromisoformat(str(raw)[:10])
            except ValueError:
                continue
            games.append((d, s, w, home, away))

    games.sort()
    last_date = {}
    schedule = {}
    for d, s, w, home, away in games:
        hr = None if home not in last_date else (d - last_date[home]).days
        ar = None if away not in last_date else (d - last_date[away]).days
        # Offseason gaps are not "extra rest" in the weekly context sense.
        if hr is not None and hr > 21:
            hr = None
        if ar is not None and ar > 21:
            ar = None
        schedule[(s, w, home)] = {
            "home": 1.0,
            "rest_days": hr,
            "opponent_rest_days": ar,
        }
        schedule[(s, w, away)] = {
            "home": 0.0,
            "rest_days": ar,
            "opponent_rest_days": hr,
        }
        last_date[home] = d
        last_date[away] = d
    return schedule


def build_indexes(team_totals, team_opp, pbp, pbp_opp, schedule):
    _TEAM_WEEK.clear()
    _ALLOWED_WEEK.clear()
    _LEAGUE_WEEK.clear()
    _PBP_TEAM_WEEK.clear()
    _PBP_ALLOWED_WEEK.clear()
    _PBP_LEAGUE_WEEK.clear()
    _SCHEDULE.clear()
    _SCHEDULE.update(schedule)

    for (s, w, team), st in team_totals.items():
        key = (s, w)
        _TEAM_WEEK[team].append((key, st))
        defense = team_opp.get((s, w, team))
        if defense:
            _ALLOWED_WEEK[defense].append((key, st))
        _LEAGUE_WEEK.append((key, team, st))

    for (s, w, team), st in pbp.items():
        key = (s, w)
        _PBP_TEAM_WEEK[team].append((key, st))
        defense = pbp_opp.get((s, w, team))
        if defense:
            _PBP_ALLOWED_WEEK[defense].append((key, st))
        _PBP_LEAGUE_WEEK.append((key, team, st))

    for d in (_TEAM_WEEK, _ALLOWED_WEEK, _PBP_TEAM_WEEK, _PBP_ALLOWED_WEEK):
        for k in d:
            d[k].sort(key=lambda x: x[0])
    _LEAGUE_WEEK.sort(key=lambda x: (x[0], x[1]))
    _PBP_LEAGUE_WEEK.sort(key=lambda x: (x[0], x[1]))


def prior_values(index, key, target, field, window):
    vals = []
    for tw, st in index.get(key, ()):
        if tw >= target:
            break
        v = st.get(field)
        if v is not None:
            vals.append(float(v))
    return vals[-window:]


def prior_league_values(rows, target, field, window):
    vals = []
    for tw, _team, st in rows:
        if tw >= target:
            break
        v = st.get(field)
        if v is not None:
            vals.append(float(v))
    return vals[-window:]


def recent_ratio(team, target, numerator, denominator, window, allowed=False):
    idx = _PBP_ALLOWED_WEEK if allowed else _PBP_TEAM_WEEK
    vals = []
    for tw, st in idx.get(team, ()):
        if tw >= target:
            break
        den = st.get(denominator, 0.0) or 0.0
        num = st.get(numerator, 0.0) or 0.0
        if den > 0:
            vals.append(num / den)
    return vals[-window:]


def recent_official_conversion(team, target, opportunity, pb_component, window, allowed=False):
    """Official opportunity divided by PBP component on matching prior games."""
    official_idx = _ALLOWED_WEEK if allowed else _TEAM_WEEK
    pbp_idx = _PBP_ALLOWED_WEEK if allowed else _PBP_TEAM_WEEK
    omap = {tw: st for tw, st in official_idx.get(team, ()) if tw < target}
    pmap = {tw: st for tw, st in pbp_idx.get(team, ()) if tw < target}
    vals = []
    for tw in sorted(set(omap) & set(pmap)):
        den = pmap[tw].get(pb_component, 0.0) or 0.0
        if den > 0:
            vals.append((omap[tw].get(opportunity, 0.0) or 0.0) / den)
    return vals[-window:]


def feature_receipt(team, opponent, season, week, opportunity, cfg):
    target = (season, week)
    component = PB_COMPONENT[opportunity]
    neutral_field = NEUTRAL_COMPONENT[opportunity]

    own_direct = prior_values(_TEAM_WEEK, team, target, opportunity, cfg["window"])
    opp_direct = prior_values(_ALLOWED_WEEK, opponent, target, opportunity, cfg["window"])
    lg_direct = prior_league_values(_LEAGUE_WEEK, target, opportunity, cfg["league_window"])

    own_comp = prior_values(_PBP_TEAM_WEEK, team, target, component, cfg["window"])
    opp_comp = prior_values(_PBP_ALLOWED_WEEK, opponent, target, component, cfg["window"])
    lg_comp = prior_league_values(_PBP_LEAGUE_WEEK, target, component, cfg["league_window"])

    own_plays = prior_values(_PBP_TEAM_WEEK, team, target, "plays", cfg["window"])
    opp_plays = prior_values(_PBP_ALLOWED_WEEK, opponent, target, "plays", cfg["window"])
    lg_plays = prior_league_values(_PBP_LEAGUE_WEEK, target, "plays", cfg["league_window"])

    own_neutral = prior_values(_PBP_TEAM_WEEK, team, target, neutral_field, cfg["window"])
    opp_neutral = prior_values(_PBP_ALLOWED_WEEK, opponent, target, neutral_field, cfg["window"])
    lg_neutral = prior_league_values(_PBP_LEAGUE_WEEK, target, neutral_field, cfg["league_window"])

    own_conv = recent_official_conversion(
        team, target, opportunity, component, cfg["window"], False
    )
    opp_conv = recent_official_conversion(
        opponent, target, opportunity, component, cfg["window"], True
    )

    required = (
        own_direct, opp_direct, own_comp, opp_comp, own_plays, opp_plays,
        own_neutral, opp_neutral, own_conv, opp_conv
    )
    if any(len(x) < 2 for x in required):
        return None
    if len(lg_direct) < 16 or len(lg_comp) < 16 or len(lg_plays) < 16 or len(lg_neutral) < 16:
        return None

    d = cfg["decay"]
    ow, dw = cfg["offense_weight"], cfg["defense_weight"]
    lw = 1.0 - ow - dw

    direct = (
        ow * ewma(own_direct, d)
        + dw * ewma(opp_direct, d)
        + lw * mean(lg_direct)
    )

    component_proj = (
        ow * ewma(own_comp, d)
        + dw * ewma(opp_comp, d)
        + lw * mean(lg_comp)
    )

    plays_proj = (
        ow * ewma(own_plays, d)
        + dw * ewma(opp_plays, d)
        + lw * mean(lg_plays)
    )
    neutral_rate = (
        ow * ewma(own_neutral, d)
        + dw * ewma(opp_neutral, d)
        + lw * mean(lg_neutral)
    )
    neutral_component_proj = plays_proj * neutral_rate
    nw = cfg["neutral_weight"]
    component_proj = (1.0 - nw) * component_proj + nw * neutral_component_proj

    conversion = (
        cfg["conversion_offense_weight"] * ewma(own_conv, d)
        + (1.0 - cfg["conversion_offense_weight"]) * ewma(opp_conv, d)
    )
    structured = component_proj * conversion

    sw = cfg["structure_weight"]
    point = (1.0 - sw) * direct + sw * structured

    context = _SCHEDULE.get((season, week, team), {})
    home = context.get("home")
    rest = context.get("rest_days")
    opp_rest = context.get("opponent_rest_days")
    if home is not None:
        point += cfg["home_adjust"] * (1.0 if home >= 0.5 else -1.0)
    rest_diff = None
    if rest is not None and opp_rest is not None:
        rest_diff = max(-4.0, min(4.0, float(rest - opp_rest)))
        point += cfg["rest_adjust"] * rest_diff

    return {
        "offense_recent_direct_opportunity": ewma(own_direct, d),
        "opponent_recent_allowed_direct_opportunity": ewma(opp_direct, d),
        "league_direct_opportunity": mean(lg_direct),
        "direct_blend": direct,
        "offense_recent_pbp_component": ewma(own_comp, d),
        "opponent_recent_allowed_pbp_component": ewma(opp_comp, d),
        "league_pbp_component": mean(lg_comp),
        "projected_total_plays": plays_proj,
        "neutral_tendency_rate": neutral_rate,
        "projected_pbp_component": component_proj,
        "opportunity_per_component_conversion": conversion,
        "structured_projection": structured,
        "structure_weight": sw,
        "home": home,
        "rest_days": rest,
        "opponent_rest_days": opp_rest,
        "rest_difference_capped": rest_diff,
        "team_opportunity_projection": max(0.0, point),
    }


def configs():
    for window, decay, ow, dw, sw, nw, hc, rc in itertools.product(
        (5, 8),
        (0.65, 0.85),
        (0.55, 0.70),
        (0.10, 0.20),
        (0.50, 0.75, 1.00),
        (0.00, 0.25),
        (0.00, 0.50),
        (0.00, 0.15),
    ):
        if ow + dw >= 1.0:
            continue
        yield {
            "window": window,
            "league_window": 128,
            "decay": decay,
            "offense_weight": ow,
            "defense_weight": dw,
            "structure_weight": sw,
            "neutral_weight": nw,
            "conversion_offense_weight": 0.75,
            "home_adjust": hc,
            "rest_adjust": rc,
        }


def target_team_games(team_totals, team_opp, season, max_week=18):
    out = []
    for (s, w, team), st in sorted(team_totals.items()):
        if s == season and 1 <= w <= max_week:
            opponent = team_opp.get((s, w, team))
            if opponent:
                out.append((s, w, team, opponent, st))
    return out


def evaluate(rows, opportunity, cfg):
    errors, signed, receipts = [], [], []
    for s, w, team, opp, st in rows:
        rec = feature_receipt(team, opp, s, w, opportunity, cfg)
        if rec is None:
            continue
        actual = float(st.get(opportunity, 0.0) or 0.0)
        pred = rec["team_opportunity_projection"]
        errors.append(abs(pred - actual))
        signed.append(pred - actual)
        receipts.append({
            "season": s, "week": w, "team": team, "opponent": opp,
            "opportunity": opportunity, "actual": actual, "prediction": pred,
            "abs_error": abs(pred - actual), "receipt": rec,
        })
    return {
        "n": len(errors),
        "mae": mean(errors),
        "bias": mean(signed),
        "rows": receipts,
    }


def simple_baselines(rows, opportunity, window):
    errors, signed = [], []
    for s, w, team, _opp, st in rows:
        hist = prior_values(_TEAM_WEEK, team, (s, w), opportunity, window)
        if len(hist) < 2:
            continue
        pred = mean(hist)
        actual = float(st.get(opportunity, 0.0) or 0.0)
        errors.append(abs(pred - actual))
        signed.append(pred - actual)
    return {"n": len(errors), "mae": mean(errors), "bias": mean(signed)}


def select_2024(rows, opportunity):
    best = None
    for cfg in configs():
        ev = evaluate(rows, opportunity, cfg)
        if ev["n"] < 400 or ev["mae"] is None:
            continue
        key = (ev["mae"], abs(ev["bias"] or 0.0), json.dumps(cfg, sort_keys=True))
        if best is None or key < best[0]:
            best = (key, cfg, ev)
    if best is None:
        raise RuntimeError(f"No eligible Phase 1C-T config for {opportunity}")
    return best[1], best[2]


def compact_eval(ev):
    return {"n": ev["n"], "mae": ev["mae"], "bias": ev["bias"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="append", required=True)
    ap.add_argument("--pbp", action="append", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--phase1b-results", required=False)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    players, team_totals, team_opp = p1a.load_stats(a.stats)
    del players
    pbp, pbp_opp = load_pbp(a.pbp)
    schedule = load_schedule(a.games)
    build_indexes(team_totals, team_opp, pbp, pbp_opp, schedule)

    phase1b = None
    if a.phase1b_results and Path(a.phase1b_results).exists():
        phase1b = json.loads(Path(a.phase1b_results).read_text())

    report = {
        "schema": "nfl-v2-phase1c-team-environment-v1",
        "status": "BURNED_RESEARCH_ONLY",
        "sportsbook_inputs_used": False,
        "monte_carlo_used": False,
        "selection_period": "2024 regular season only",
        "validation_period": "2025 regular season; never used for selection",
        "diagnostic_period": "2026 weeks 1-4 burned",
        "football_scope": "team opportunity only; no player role or efficiency prediction",
        "opportunities": {},
    }

    rows24 = target_team_games(team_totals, team_opp, 2024)
    rows25 = target_team_games(team_totals, team_opp, 2025)
    rows26 = target_team_games(team_totals, team_opp, 2026, 4)

    for opp in OPPORTUNITIES:
        cfg, dev = select_2024(rows24, opp)
        val = evaluate(rows25, opp, cfg)
        diag = evaluate(rows26, opp, cfg)
        baselines = {
            "recent3_mean": simple_baselines(rows25, opp, 3),
            "recent5_mean": simple_baselines(rows25, opp, 5),
            "recent8_mean": simple_baselines(rows25, opp, 8),
        }
        p1b_2025 = None
        p1b_2026 = None
        if phase1b is not None:
            # Phase1B stores the same team opportunity metric under both
            # receiving outcomes; map team target volume through rec_yds.
            oc = {"attempts": "pass_yds", "targets": "rec_yds", "carries": "rush_yds"}[opp]
            prior = phase1b.get("outcomes", {}).get(oc, {})
            p1b_2025 = prior.get("burned_validation_2025", {}).get("team_opportunity_mae")
            p1b_2026 = prior.get("diagnostic_2026_wk1_4", {}).get("team_opportunity_mae")

        report["opportunities"][opp] = {
            "selected_config": cfg,
            "selection_2024": compact_eval(dev),
            "validation_2025": compact_eval(val),
            "diagnostic_2026_wk1_4": compact_eval(diag),
            "transparent_2025_baselines": baselines,
            "phase1b_team_opportunity_mae_2025": p1b_2025,
            "phase1b_team_opportunity_mae_2026_wk1_4": p1b_2026,
            "largest_2025_misses": sorted(
                val["rows"], key=lambda x: -x["abs_error"]
            )[:20],
        }

    Path(a.out).write_text(json.dumps(report, indent=2, sort_keys=True))
    compact = {}
    for opp, v in report["opportunities"].items():
        compact[opp] = {
            "cfg": v["selected_config"],
            "dev2024": v["selection_2024"],
            "val2025": v["validation_2025"],
            "diag2026": v["diagnostic_2026_wk1_4"],
            "baselines2025": v["transparent_2025_baselines"],
            "phase1b_2025": v["phase1b_team_opportunity_mae_2025"],
            "phase1b_2026": v["phase1b_team_opportunity_mae_2026_wk1_4"],
        }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())