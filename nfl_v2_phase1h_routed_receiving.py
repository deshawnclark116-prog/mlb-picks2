#!/usr/bin/env python3
"""Phase1H-R: research-only routed catch / completed-air / YAC corrections.

Run `--stage develop` before `--stage confirm`. Confirmation reads the frozen
development specification, never selects a penalty, feature or combination.
No production imports, sportsbook inputs, Monte Carlo, or final-yard fitting.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

import nfl_v2_phase1a_direct as p1a
import nfl_v2_phase1b_opportunity as p1b
import nfl_v2_phase1d_role_allocation as p1d
import nfl_v2_phase1e_integrated as p1e
import nfl_v2_phase1g_receiving_mechanics as p1g
from nfl_v2_phase1h_sources import records, local_name

ROOT = Path(__file__).resolve().parent
ART = ROOT / "nfl_models/nfl_player_outcome_v2"
PROTOCOL = ART / "phase1h_routed_receiving_protocol.json"
ROUTES = json.loads(PROTOCOL.read_text())["families"]
GCFG = json.loads((ART / "phase1g_receiving_mechanics_snapshot.json").read_text())["selected_config"]


def number(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def boolean(x):
    if str(x).upper() in {"1", "TRUE"}:
        return 1.0
    if str(x).upper() in {"0", "FALSE"}:
        return 0.0
    return None


def play_key(row, play_field):
    v = number(row.get(play_field))
    return (row.get("nflverse_game_id") or row.get("game_id"), int(v) if v is not None else None)


def average(items):
    return float(np.mean(items)) if len(items) else None


class Mechanics:
    """Prior-only event store. V4's join keys/label parsing, no V4 model/features."""

    def __init__(self, directory, positions):
        self.player, self.defpos, self.pos, self.defdrop, self.drop = [defaultdict(list) for _ in range(5)]
        self.labels = defaultdict(list)
        self.game_dates = {}
        self.quality = defaultdict(int)
        self._prior_cache = {}
        self._league_cache = {}
        for season in (2023, 2024, 2025, 2026):
            part, ftn = {}, {}
            path = Path(directory) / f"participation_{season}.csv"
            if path.exists():
                for r in records(path):
                    part[play_key(r, "play_id")] = (
                        r.get("defense_man_zone_type"), boolean(r.get("was_pressure")))
            path = Path(directory) / f"ftn_{season}.csv"
            if path.exists():
                for r in records(path):
                    n = number(r.get("n_blitzers"))
                    ftn[play_key(r, "nflverse_play_id")] = None if n is None else float(n > 0)
            for r in records(Path(directory) / f"pbp_{season}.csv.gz"):
                if r.get("season_type") != "REG" or r.get("two_point_attempt") == "1" or r.get("no_play") == "1":
                    continue
                week = int(float(r["week"]))
                if season == 2026 and week > 4:
                    continue  # Future clean evidence is not even ingested.
                gid, pid = r["game_id"], r.get("receiver_player_id")
                date = r.get("game_date")
                if not date:
                    raise ValueError("PBP game_date required for cutoff check")
                key = (season, week)
                self.game_dates[(season, week, r.get("posteam"))] = date
                scheme, pressure = part.get(play_key(r, "play_id"), (None, None))
                blitz = ftn.get(play_key(r, "play_id"))
                common = {"key": key, "date": date, "game_id": gid, "pressure": pressure,
                          "blitz": blitz, "scheme": scheme}
                if r.get("qb_dropback") == "1":
                    self.defdrop[r["defteam"]].append(common)
                    self.drop["league"].append(common)
                    self.quality[f"{season}_dropbacks"] += 1
                    self.quality[f"{season}_joined_pressure_dropbacks"] += pressure is not None
                    self.quality[f"{season}_joined_blitz_dropbacks"] += blitz is not None
                pos = p1g.GROUPS.get(positions.get((season, pid)))
                if not pid or not pos or r.get("play_type") != "pass":
                    continue
                catch = boolean(r.get("complete_pass")) or 0.0
                air, yac = number(r.get("air_yards")), number(r.get("yards_after_catch"))
                yards = (number(r.get("yards_gained")) or 0.0) if catch else 0.0
                complete_air = yards - yac if catch and yac is not None else (0.0 if not catch else None)
                e = {**common, "targets": 1., "receptions": catch, "rec_yards": yards,
                     "completed_air": complete_air, "yac": yac if catch else 0., "air_yards": air}
                self.player[pid].append(e)
                self.defpos[(r["defteam"], pos)].append(e)
                self.pos[pos].append(e)
                self.labels[(season, week, pid)].append(e)
        self.keys = {}
        for idx in (self.player, self.defpos, self.pos, self.defdrop, self.drop):
            for k, vals in idx.items():
                vals.sort(key=lambda e: (e["key"], e["date"], e["game_id"]))
                self.keys[(id(idx), k)] = [e["key"] for e in vals]

    def prior(self, idx, identity, target, date, games=None, events=None):
        vals = idx.get(identity, ())
        end = bisect_left(self.keys.get((id(idx), identity), []), target)
        if events is not None:
            vals = vals[max(0, end - events):end]
        else:
            vals = vals[:end]
        vals = [e for e in vals if e["date"] < date]
        if games:
            seen, chosen = set(), []
            for e in reversed(vals):
                if e["game_id"] not in seen and len(seen) >= games:
                    break
                seen.add(e["game_id"])
                chosen.append(e)
            vals = list(reversed(chosen))
        return vals

    def position(self, target, date, pos):
        key = (target, date, pos)
        if key not in self._prior_cache:
            events = self.prior(self.pos, pos, target, date, events=1500)
            self._prior_cache[key] = (events, profile(events))
        return self._prior_cache[key]

    def receipt(self, r):
        target, pid = (r["season"], r["week"]), r["player_id"]
        pos = p1g.GROUPS.get(r.get("position"))
        date = self.game_dates.get((*target, r["team"]))
        if pos is None or date is None:
            return None
        events, pp = self.position(target, date, pos)
        if not events:
            return None
        pe = self.prior(self.player, pid, target, date, games=8)
        de = self.prior(self.defpos, (r["opponent"], pos), target, date, games=8)
        bp, dp = profile(pe, pp, 25), profile(de, pp, 80)
        # Scheme/pressure missing in 2026: legal older games remain, exposed as stale.
        sd = self.prior(self.defdrop, r["opponent"], target, date, games=8)
        known_sd = self.prior(self.defdrop, r["opponent"], target, date)
        known_pe = self.prior(self.player, pid, target, date)
        press = [e for e in known_sd if e["pressure"] is not None][-500:]
        mz = [e for e in known_sd if e["scheme"] in {"MAN_COVERAGE", "ZONE_COVERAGE"}][-500:]
        ps = [e for e in known_pe if e["pressure"] is not None][-150:]
        ms = [e for e in known_pe if e["scheme"] in {"MAN_COVERAGE", "ZONE_COVERAGE"}][-150:]
        league_key = (target, date)
        if league_key not in self._league_cache:
            ld = self.prior(self.drop, "league", target, date, events=6000)
            league_press = [e for e in self.prior(self.drop, "league", target, date) if e["pressure"] is not None][-6000:]
            self._league_cache[league_key] = (
                average([e["pressure"] for e in league_press]),
                average([e["blitz"] for e in ld if e["blitz"] is not None]))
        lp, lb = self._league_cache[league_key]
        pressure_rate = shrunk_rate([e["pressure"] for e in press], lp, 80)
        blitz_rate = shrunk_rate([e["blitz"] for e in sd if e["blitz"] is not None], lb, 80)
        player_pressure = average([e["pressure"] for e in ps])
        press_gap = profile([e for e in ps if e["pressure"] == 1], bp, 25)["catch"] - profile([e for e in ps if e["pressure"] == 0], bp, 25)["catch"]
        man = shrunk_rate([float(e["scheme"] == "MAN_COVERAGE") for e in mz], .35, 80)
        player_man = average([float(e["scheme"] == "MAN_COVERAGE") for e in ms])
        mp = profile([e for e in ms if e["scheme"] == "MAN_COVERAGE"], bp, 25)
        zp = profile([e for e in ms if e["scheme"] == "ZONE_COVERAGE"], bp, 25)
        scheme_delta = (man - player_man) if ms and mz else 0.
        features = {
            "player_adot_minus_position": bp["adot"] - pp["adot"],
            "player_deep_share_x_def_deep_air_delta": bp["deep"] * (dp["deep_air"] - pp["deep_air"]),
            "player_air20_share_x_def_air20_delta": bp["air20"] * (dp["air20"] - pp["air20"]),
            "def_pressure_minus_league": pressure_rate - lp if press and lp is not None else 0.,
            "def_blitz_minus_league": blitz_rate - lb if lb is not None and any(e["blitz"] is not None for e in sd) else 0.,
            "pressure_exposure_x_player_catch_split": (pressure_rate - player_pressure) * press_gap if press and ps else 0.,
            "scheme_catch_edge": scheme_delta * (mp["catch"] - zp["catch"]),
            "scheme_air_edge": scheme_delta * (mp["air"] - zp["air"]),
            "def_position_yac_delta": dp["yac"] - pp["yac"],
            "player_yac_x_def_yac10_delta": bp["yac"] * (dp["yac10"] - pp["yac10"]),
        }
        provenance = lambda es: {"n": len(es), "last_season_week": max((e["key"] for e in es), default=None),
                                  "last_game_date": max((e["date"] for e in es), default=None)}
        return {"baseline_catch_probability": bp["catch"], "baseline_air_component": bp["air"],
                "baseline_yac_component": bp["yac"], "features": features,
                "source_status": {"man_zone": "STALE_PRIOR_SEASON_FALLBACK" if mz and max(e["key"][0] for e in mz) < r["season"] else ("PRIOR_COMPLETED_GAMES" if mz else "UNAVAILABLE_NEUTRAL"),
                                  "pressure": "STALE_PRIOR_SEASON_FALLBACK" if press and max(e["key"][0] for e in press) < r["season"] else ("PRIOR_COMPLETED_GAMES" if press else "UNAVAILABLE_NEUTRAL"),
                                  "defensive_absences": "BLOCKED_DATA"},
                "history": {"player": provenance(pe), "opponent_position": provenance(de),
                            "scheme": provenance(mz), "pressure": provenance(press)},
                "receiver_depth_profile": {k: bp[k] for k in ("adot", "deep", "air20", "reception20", "yac10")}}


def shrunk_rate(values, prior, k):
    if prior is None:
        return None
    return (sum(values) + k * prior) / (len(values) + k)


def profile(events, prior=None, k=0):
    """Target/catch denominators are distinct; missing charting is never zero."""
    prior = prior or {"catch": .65, "air": 7., "yac": 5., "adot": 8.,
                      "deep": .15, "deep_air": 25., "air20": .08, "yac10": .15, "reception20": .15}
    catches = [e for e in events if e["receptions"] and e["completed_air"] is not None and e["yac"] is not None]
    depth = [e for e in events if e["air_yards"] is not None]
    deep_catches = [e for e in catches if e["air_yards"] is not None and e["air_yards"] >= 20]
    kc = k * prior["catch"]
    def sm(vals, name, strength):
        return (sum(vals) + strength * prior[name]) / (len(vals) + strength) if len(vals) + strength else prior[name]
    return {"catch": sm([e["receptions"] for e in events], "catch", k),
            "air": sm([e["completed_air"] for e in catches], "air", kc),
            "yac": sm([e["yac"] for e in catches], "yac", kc),
            "adot": sm([e["air_yards"] for e in depth], "adot", k),
            "deep": sm([float(e["air_yards"] >= 20) for e in depth], "deep", k),
            "deep_air": sm([e["completed_air"] for e in deep_catches], "deep_air", kc),
            "air20": sm([float(e["completed_air"] >= 20) for e in catches], "air20", kc),
            "yac10": sm([float(e["yac"] >= 10) for e in catches], "yac10", kc),
            "reception20": sm([float(e["rec_yards"] >= 20) for e in catches], "reception20", kc)}


def logit(p):
    p = np.clip(p, .02, .98)
    return np.log(p / (1 - p))


def assemble(directory):
    players, team, opp = p1a.load_stats([Path(directory) / local_name("stats", s) for s in (2023, 2024, 2025, 2026)])
    # Filter clean-forward rows before building any index.
    players = [r for r in players if r["season"] != 2026 or r["week"] <= 4]
    team = {k: v for k, v in team.items() if k[0] != 2026 or k[1] <= 4}
    opp = {k: v for k, v in opp.items() if k[0] != 2026 or k[1] <= 4}
    p1a.build_indexes(players, team, opp)
    p1b.build_extra_indexes(players, team, opp)
    p1d.build_context(players, team)
    p1d.load_rosters([Path(directory) / local_name("roster", s) for s in (2024, 2025, 2026)])
    pos = {(r["season"], r["player_id"]): r["position"] for r in players}
    mech = Mechanics(directory, pos)
    p1g.load_pbp([Path(directory) / local_name("pbp", s) for s in (2023, 2024, 2025, 2026)], pos)
    b = json.loads((ART / "phase1b_opportunity_snapshot.json").read_text())
    d = json.loads((ART / "phase1d_role_allocation_snapshot.json").read_text())
    rows = []
    quality = dict(mech.quality)
    for r in p1b.fixed_meaningful_rows(players, [r for r in players if r["season"] >= 2024], "rec_yds"):
        q = mech.receipt(r)
        f = p1b.efficiency_receipt(players, team, opp, r, "rec_yds", b["outcomes"]["rec_yds"]["selected_efficiency_config"])
        g = p1g.efficiency_receipt(r, GCFG)
        h = p1a.player_eff_history(players, (r["season"], r["week"]), r["player_id"], "receiving_yards", "targets", 8)
        if q is None or f is None or g is None or h is None:
            quality["missing_common_prediction_rows"] = quality.get("missing_common_prediction_rows", 0) + 1
            continue
        labels = mech.labels.get((r["season"], r["week"], r["player_id"]), [])
        reconciled = len(labels) == r["targets"] and abs(sum(e["rec_yards"] for e in labels) - r["receiving_yards"]) < .01 and sum(e["receptions"] for e in labels) == r["receptions"] and all(e["completed_air"] is not None and e["yac"] is not None for e in labels)
        quality[f"{r['season']}_component_reconciled"] = quality.get(f"{r['season']}_component_reconciled", 0) + int(reconciled)
        quality[f"{r['season']}_component_unreconciled"] = quality.get(f"{r['season']}_component_unreconciled", 0) + int(not reconciled)
        base = p1e.receipt(players, team, opp, r, "rec_yds", b, d)
        rows.append({**r, **q, "game_id": f"{r['season']}_{r['week']:02d}_" + "_".join(sorted((r["team"], r["opponent"]))),
                     "component_labels_reconciled": reconciled,
                     "actual_completed_air": sum(e["completed_air"] or 0 for e in labels) if reconciled else None,
                     "actual_yac": sum(e["yac"] or 0 for e in labels) if reconciled else None,
                     "predicted_targets": base["player_opportunity_projection"] if base else None,
                     "phase1f_ypt": f["final_efficiency_projection"], "history_ypt": h,
                     "phase1g_catch": g["projected_catch_rate"], "phase1g_air": g["projected_air_per_catch"],
                     "phase1g_yac": g["projected_yac_per_catch"], "phase1g_ypt": g["projected_yards_per_target"]})
    return rows, quality


def fit(rows, family, penalty):
    if any(r["season"] != 2024 for r in rows):
        raise ValueError("Fitting allowed only on 2024")
    fitted = {}
    for component, cols in ROUTES[family].items():
        if component not in {"catch", "air", "yac"}:
            continue
        usable = [r for r in rows if r["targets"] > 0 and (component == "catch" or (r["receptions"] > 0 and r["component_labels_reconciled"]))]
        x = np.array([[r["features"][c] for c in cols] for r in usable], dtype=float)
        w = np.array([r["targets"] if component == "catch" else r["receptions"] for r in usable])
        scale = np.sqrt(np.average(x*x, weights=w, axis=0))
        scale = np.where(scale > 1e-9, scale, 1.)
        x = x / scale
        baseline = np.array([r[{"catch": "baseline_catch_probability", "air": "baseline_air_component", "yac": "baseline_yac_component"}[component]] for r in usable])
        if component == "catch":
            y = np.array([r["receptions"] / r["targets"] for r in usable])
            offset = logit(baseline)
            beta = np.zeros(len(cols))
            for _ in range(40):
                p = 1 / (1 + np.exp(-np.clip(offset + x @ beta, -30, 30)))
                grad = x.T @ (w * (p-y)) + penalty * beta
                hess = (x.T * (w*p*(1-p))) @ x + penalty * np.eye(len(cols))
                step = np.linalg.solve(hess, grad)
                beta -= step
                if np.max(np.abs(step)) < 1e-10:
                    break
        else:
            y = np.array([r["actual_completed_air" if component == "air" else "actual_yac"] / r["receptions"] for r in usable]) - baseline
            beta = np.linalg.solve((x.T*w)@x + penalty*np.eye(len(cols)), x.T@(w*y))
        fitted[component] = {"columns": cols, "scale": scale.tolist(), "coefficients": beta.tolist(), "training_n": len(usable)}
    return {"families": [family], "penalty": penalty, "models": {family: fitted}}


def project(r, spec=None):
    adjustments = {"catch": 0., "air": 0., "yac": 0.}
    family_adjustments = {}
    if spec:
        for family, models in spec["models"].items():
            fa = {}
            for component, m in models.items():
                value = sum(r["features"][c] / s * b for c, s, b in zip(m["columns"], m["scale"], m["coefficients"]))
                adjustments[component] += value
                fa[component] = value
            family_adjustments[family] = fa
    catch = float(1 / (1 + np.exp(-(logit(r["baseline_catch_probability"]) + np.clip(adjustments["catch"], -.5, .5))))) if adjustments["catch"] else r["baseline_catch_probability"]
    air = float(np.clip(r["baseline_air_component"] + np.clip(adjustments["air"], -4, 4), -5, 35))
    yac = float(np.clip(r["baseline_yac_component"] + np.clip(adjustments["yac"], -3, 3), 0, 25))
    ypt = catch * (air + yac)
    return {"catch": catch, "air": air, "yac": yac, "ypt": ypt,
            "matchup_catch_adjustment": catch-r["baseline_catch_probability"],
            "matchup_air_adjustment": air-r["baseline_air_component"],
            "matchup_yac_adjustment": yac-r["baseline_yac_component"],
            "family_raw_adjustments": family_adjustments,
            "final_yards_per_target": ypt,
            "final_direct_receiving_yard_projection": None if r["predicted_targets"] is None else r["predicted_targets"]*ypt}


def predictions(rows, spec=None, comparator=None):
    ans = []
    for r in rows:
        q = project(r, spec)
        if comparator == "phase1g":
            q.update(catch=r["phase1g_catch"], air=r["phase1g_air"], yac=r["phase1g_yac"], ypt=r["phase1g_ypt"])
        elif comparator in {"phase1f", "history"}:
            q.update(ypt=r[f"{comparator}_ypt"], catch=None, air=None, yac=None)
        ans.append(q)
    return ans


def losses(rows, preds):
    values = defaultdict(list)
    for r, q in zip(rows, preds):
        t, c, y = r["targets"], r["receptions"], r["receiving_yards"]
        if t <= 0:
            continue
        err = abs(t*q["ypt"]-y)
        values["oracle_target_rec_yds_mae"].append(err)
        values["yards_per_target_mae"].append(abs(q["ypt"]-y/t))
        values["catastrophic_efficiency_miss_rate"].append(float(err > 40))
        values["predicted_mean_yards_per_target"].append(q["ypt"])
        values["observed_mean_yards_per_target"].append(y/t)
        if q["catch"] is not None:
            values["catch_rate_mae"].append(abs(q["catch"]-c/t))
            values["oracle_target_rec_mae"].append(abs(t*q["catch"]-c))
        if q["air"] is not None and r["component_labels_reconciled"]:
            values["completed_air_per_target_mae"].append(abs(q["catch"]*q["air"]-r["actual_completed_air"]/t))
            values["yac_per_target_mae"].append(abs(q["catch"]*q["yac"]-r["actual_yac"]/t))
            if c > 0:
                values["conditional_air_per_catch_mae"].append(abs(q["air"]-r["actual_completed_air"]/c))
                values["conditional_yac_per_catch_mae"].append(abs(q["yac"]-r["actual_yac"]/c))
    return {"n": len(values["oracle_target_rec_yds_mae"]), **{k: average(v) for k, v in values.items()},
            "metric_n": {k: len(v) for k, v in values.items()}}


def bootstrap(rows, a, b):
    """Paired calendar-week moving blocks, each sampled game keeps all players."""
    sums, counts = defaultdict(float), defaultdict(int)
    for r, qa, qb in zip(rows, a, b):
        if r["targets"] <= 0:
            continue
        key = (r["season"], r["week"], r["game_id"])
        sums[key] += abs(r["targets"]*qa["ypt"]-r["receiving_yards"]) - abs(r["targets"]*qb["ypt"]-r["receiving_yards"])
        counts[key] += 1
    keys = sorted(sums)
    if not keys:
        return {"delta_mae": None, "ci95": [None, None]}
    seasons = sorted({k[0] for k in keys})
    blocks = {}
    for s in seasons:
        weeks = list(range(min(k[1] for k in keys if k[0] == s), max(k[1] for k in keys if k[0] == s)+1))
        blocks[s] = [[k for k in keys if k[0] == s and k[1] in {w, weeks[(i+1) % len(weeks)]}] for i, w in enumerate(weeks)]
    rng, draws = np.random.default_rng(164), []
    for _ in range(2000):
        sample = []
        for s in seasons:
            bs = blocks[s]
            for ix in rng.integers(0, len(bs), size=(len(bs)+1)//2):
                sample.extend(bs[ix])
        n = sum(counts[k] for k in sample)
        if n:
            draws.append(sum(sums[k] for k in sample)/n)
    return {"delta_mae": sum(sums.values())/sum(counts.values()),
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "resamples": len(draws), "seed": 164, "unit": "2-week moving blocks; whole games"}


def component_metrics(families):
    return sorted({{"catch": "catch_rate_mae", "air": "conditional_air_per_catch_mae", "yac": "conditional_yac_per_catch_mae"}[c]
                   for f in families for c in ROUTES[f] if c in {"catch", "air", "yac"}})


def gate(metrics, base, components, boot=None):
    checks = {"oracle_practical_gain": metrics["oracle_target_rec_yds_mae"] <= .995*base["oracle_target_rec_yds_mae"],
              "ypt_not_worse": metrics["yards_per_target_mae"] <= base["yards_per_target_mae"]}
    checks.update({k: metrics.get(k) is not None and base.get(k) is not None and metrics[k] <= .995*base[k] for k in components})
    if boot is not None:
        checks["blocked_bootstrap"] = boot["ci95"][1] is not None and boot["ci95"][1] < 0
    return {"passes": all(checks.values()), "checks": checks}


def develop(rows, quality, audit):
    train = [r for r in rows if r["season"] == 2024 and r["week"] <= 8]
    dev = [r for r in rows if r["season"] == 2024 and r["week"] >= 9]
    basep = predictions(dev)
    basem = losses(dev, basep)
    comps = {n: losses(dev, predictions(dev, comparator=n)) for n in ("phase1f", "phase1g", "history")}
    all_specs, results, earned = {}, {}, {}
    for family in ROUTES:
        trials = []
        for penalty in (10., 100., 1000.):
            spec = fit(train, family, penalty)
            metrics = losses(dev, predictions(dev, spec))
            trials.append((metrics["oracle_target_rec_yds_mae"], -penalty, spec, metrics))
        _, _, spec, m = min(trials, key=lambda x: x[:2])
        boot = bootstrap(dev, predictions(dev, spec), basep)
        g = gate(m, basem, component_metrics([family]), boot)
        all_specs[family] = spec
        results[family] = {"metrics": m, "penalty": spec["penalty"], "bootstrap_vs_mechanical": boot,
                           "gate": g, "status": "SURVIVES" if g["passes"] else "REJECTED",
                           "penalty_trials": [{"penalty": t[2]["penalty"], "metrics": t[3]} for t in trials]}
        if g["passes"]:
            earned[family] = spec
    for size in range(2, len(earned)+1):
        for families in itertools.combinations(sorted(earned), size):
            name = "+".join(families)
            spec = {"families": list(families), "models": {f: earned[f]["models"][f] for f in families}, "penalties": {f: earned[f]["penalty"] for f in families}}
            m = losses(dev, predictions(dev, spec))
            boot = bootstrap(dev, predictions(dev, spec), basep)
            g = gate(m, basem, component_metrics(families), boot)
            all_specs[name] = spec
            results[name] = {"metrics": m, "bootstrap_vs_mechanical": boot, "gate": g,
                             "status": "SURVIVES" if g["passes"] else "REJECTED"}
    eligible = []
    for name, v in results.items():
        v["incumbent_gate"] = gate(v["metrics"], comps["phase1f"], [])
        v["history_gate"] = gate(v["metrics"], comps["history"], [])
        if v["status"] == "SURVIVES" and v["incumbent_gate"]["passes"] and v["history_gate"]["passes"]:
            eligible.append(name)
    selected = min(eligible, key=lambda n: (results[n]["metrics"]["oracle_target_rec_yds_mae"], len(all_specs[n]["families"]), n)) if eligible else None
    # Refit every frozen family specification for transparent fixed validation ablations;
    # rejection is locked, these cannot enter selection again.
    refit_rows = [r for r in rows if r["season"] == 2024]
    refit = {f: fit(refit_rows, f, all_specs[f]["penalty"]) for f in ROUTES}
    if selected and "+" in selected:
        families = all_specs[selected]["families"]
        refit[selected] = {"families": families, "models": {f: refit[f]["models"][f] for f in families}, "penalties": all_specs[selected]["penalties"]}
    return {"schema": "nfl-v2-phase1h-development-lock-v1", "selected": selected,
            "fit_n": len(train), "selection_n": len(dev), "baseline": basem, "comparators": comps,
            "candidates": results, "frozen_refit_specs": refit, "data_quality": quality,
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "audit_sha256": hashlib.sha256(Path(audit).read_bytes()).hexdigest(),
            "selection_rule": "2024 W1-8 fit / W9-18 select, then refit all 2024. No 2025/2026 performance accessed."}


def slices(rows, preds):
    selections = {pos: [i for i, r in enumerate(rows) if p1g.GROUPS.get(r["position"]) == pos] for pos in ("WR", "TE", "RB")}
    selections.update(early=[i for i,r in enumerate(rows) if r["week"] <= 4], established=[i for i,r in enumerate(rows) if r["week"] > 4])
    return {name: losses([rows[i] for i in ids], [preds[i] for i in ids]) for name, ids in selections.items()}


def full_metrics(rows, preds):
    err, signed = [], []
    for r, q in zip(rows, preds):
        if r["predicted_targets"] is not None:
            e = r["predicted_targets"]*q["ypt"]-r["receiving_yards"]
            err.append(abs(e)); signed.append(e)
    return {"n": len(err), "mae": average(err), "within_10": average([e<=10 for e in err]),
            "within_20": average([e<=20 for e in err]), "misses_over_40": average([e>40 for e in err]), "signed_bias": average(signed)}


def confirm(rows, lock):
    if lock["protocol_sha256"] != hashlib.sha256(PROTOCOL.read_bytes()).hexdigest():
        raise ValueError("Protocol changed after development freeze")
    selected = lock["selected"]
    out = {"schema": "nfl-v2-phase1h-results-v1", "selected": selected, "periods": {},
           "sportsbook_inputs_used": False, "monte_carlo_used": False,
           "validation_rule": "Frozen development ablations evaluated once; cannot rescue rejected families.",
           "development": {k: v for k,v in lock.items() if k != "frozen_refit_specs"}}
    receipt_rows = []
    for label, season in (("validation_2025", 2025), ("diagnostic_2026_wk1_4", 2026)):
        rs = [r for r in rows if r["season"] == season]
        allp = {n: predictions(rs, comparator=n) for n in ("phase1f", "phase1g", "history")}
        allp["mechanical"] = predictions(rs)
        for n, spec in lock["frozen_refit_specs"].items():
            allp[n] = predictions(rs, spec)
        ms = {n: losses(rs,p) for n,p in allp.items()}
        bs = {n: {ref: bootstrap(rs,p,allp[ref]) for ref in ("mechanical", "phase1f")} for n,p in allp.items() if n in lock["frozen_refit_specs"]}
        statuses = {}
        for n in lock["frozen_refit_specs"]:
            families = lock["frozen_refit_specs"][n]["families"]
            component_gate = gate(ms[n], ms["mechanical"], component_metrics(families), bs[n]["mechanical"])
            incumbent_gate = gate(ms[n], ms["phase1f"], [], bs[n]["phase1f"])
            history_gate = gate(ms[n], ms["history"], [])
            passes = component_gate["passes"] and incumbent_gate["passes"] and history_gate["passes"]
            statuses[n] = {"component_gate": component_gate, "incumbent_gate": incumbent_gate, "history_gate": history_gate,
                           "status": "SURVIVES" if n == selected and passes else "REJECTED",
                           "frozen_development_status": lock["candidates"][n]["status"]}
        out["periods"][label] = {"metrics": ms, "bootstrap": bs, "decisions": statuses,
                                  "slices": {n: slices(rs,p) for n,p in allp.items()}}
        for n, spec in lock["frozen_refit_specs"].items():
            for r,q in zip(rs,allp[n]):
                receipt_rows.append({"architecture": n, "period": label, **r, **q})
    val = out["periods"]["validation_2025"]
    passed = selected is not None and val["decisions"][selected]["status"] == "SURVIVES"
    out["verdict"] = "SURVIVES_RESEARCH_ONLY_PENDING_CLEAN_FORWARD" if passed else "REJECTED_EFFICIENCY_REPLACEMENT"
    out["feature_family_decisions"] = {f: ("SURVIVES" if passed and f in lock["frozen_refit_specs"][selected]["families"] else "REJECTED") for f in ROUTES}
    out["feature_family_decisions"]["defensive_personnel"] = "BLOCKED_DATA"
    out["feature_family_decisions"]["current_2026_man_zone_true_pressure"] = "BLOCKED_DATA"
    out["full_projection_evaluation"] = {"status": "RUN_AFTER_EFFICIENCY_PASS" if passed else "NOT_RUN_EFFICIENCY_GATE_FAILED"}
    if passed:
        for label, season in (("validation_2025", 2025), ("diagnostic_2026_wk1_4", 2026)):
            rs = [r for r in rows if r["season"] == season]
            out["full_projection_evaluation"][label] = full_metrics(rs,predictions(rs,lock["frozen_refit_specs"][selected]))
    return out, receipt_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stage", choices=("develop", "confirm"), required=True)
    ap.add_argument("--audit", default=str(ART / "phase1h_source_coverage.json"))
    ap.add_argument("--lock", default=str(ART / "phase1h_development_lock.json"))
    ap.add_argument("--out", default=str(ART / "phase1h_routed_receiving_results.json"))
    ap.add_argument("--receipts", default=str(ART / "phase1h_receipts.jsonl.gz"))
    a = ap.parse_args()
    if not Path(a.audit).is_file():
        raise ValueError("Source coverage audit required before any fitting")
    audit = json.loads(Path(a.audit).read_text())
    for season, sources in audit["seasons"].items():
        for kind, meta in sources.items():
            path = Path(a.data_dir) / local_name(kind, int(season))
            if meta["status"] == "NOT_PUBLISHED":
                if path.exists():
                    raise ValueError(f"Unaudited provider source appeared: {kind}/{season}")
            elif not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
                raise ValueError(f"Source differs from frozen coverage audit: {kind}/{season}")
    rows, quality = assemble(a.data_dir)
    if a.stage == "develop":
        result = develop(rows, quality, a.audit)
        Path(a.lock).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k not in {"frozen_refit_specs", "candidates"}},indent=2))
        print({n:(v["status"],v["metrics"]["oracle_target_rec_yds_mae"]) for n,v in result["candidates"].items()})
    else:
        lock = json.loads(Path(a.lock).read_text())
        if lock["audit_sha256"] != hashlib.sha256(Path(a.audit).read_bytes()).hexdigest():
            raise ValueError("Coverage snapshot changed after development")
        result, receipts = confirm(rows, lock)
        Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        # Reproducible compressed research ledger (not a scientific forecast store).
        import gzip
        with open(a.receipts, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
                for r in receipts:
                    f.write((json.dumps(r,sort_keys=True,separators=(",", ":"))+"\n").encode())
        print(json.dumps({"verdict":result["verdict"],"selected":result["selected"],"families":result["feature_family_decisions"], "metrics":{k:v["metrics"] for k,v in result["periods"].items()}},indent=2))


if __name__ == "__main__":
    main()
