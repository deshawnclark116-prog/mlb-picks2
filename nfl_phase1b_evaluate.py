"""
NFL_PHASE1B_EVALUATE  (Phase 1B, development / burned data; shadow; nothing is frozen)

  python -u nfl_phase1b_evaluate.py --data-dir /tmp/nflcsv --stage-cache stage.pkl [--p1a-cache p1a.pkl] [--out-dir DIR]

1. consumes the Phase 1A opportunity samples (selected configuration; no recomputation of carries / targets / attempts)
2. builds as-of per-player-game efficiency records (realized players + forecast-only candidates)
3. component studies: rushing (two-process vs single), receiving chain, passing chain, TD hazards, defensive events
4. scheme / interaction / archetype / persistence / data-source audit
5. DEVELOPMENT end-to-end game simulation, compared with the historical last-8 empirical baseline and with simpler efficiency variants
"""
import argparse
import json
import os
import pickle
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

import nfl_context_v4 as v4
import nfl_phase1_common as C
import nfl_phase1_data as P1
import nfl_phase1_defense_events as DE
import nfl_phase1_efficiency as F
import nfl_phase1_event_models as EM
import nfl_phase1_opportunity as O
import nfl_phase1_passing_efficiency as PA
import nfl_phase1_receiving_efficiency as RC
import nfl_phase1_rushing_efficiency as RU
import nfl_phase1b_data as B

OUT = Path(__file__).resolve().parent / "nfl_models" / "nfl_player_outcome_phase1b"
N_DRAWS = 200
SEASONS = [2022, 2023, 2024, 2025, 2026]


# ------------------------------------------------------------------ Phase 1A samples (consumed as-is)
def get_phase1a(data_dir, stage_cache, cache_path):
    if cache_path and Path(cache_path).exists():
        return pickle.load(open(cache_path, "rb"))
    import nfl_phase1_evaluate as E
    rep, sel, rej, out = E.run(data_dir, None, write=False, stage_cache=stage_cache)
    opp = out["opp_rep"]
    samples, extras = {}, defaultdict(set)
    for name in ("carry", "target", "qb_att", "rz_carry", "rz_target", "def_snap"):
        y, S, meta = opp["_cache"][name]["selected"]
        samples[name] = {"y": y.astype(float), "S": S.astype(np.int16), "ids": [(m[0], m[1], m[2], m[3]) for m in meta]}
        for (s, w, team, gid, _) in meta:
            extras[(s, w)].add((gid, team, name))
    pack = {"samples": samples, "extras": {k: sorted(v) for k, v in extras.items()}, "selected_config": {k: v["selected_config"] for k, v in opp["types"].items()}}
    if cache_path:
        pickle.dump(pack, open(cache_path, "wb"), protocol=4)
    return pack


# ------------------------------------------------------------------ actuals
def load_actuals(data_dir):
    A = {}
    for s in SEASONS:
        for r in __import__("csv").DictReader(open(Path(data_dir) / f"stats_player_week_{s}.csv", newline="", encoding="utf-8")):
            if r.get("season_type", "REG") != "REG":
                continue
            g = lambda k: P1.fnum(r.get(k)) or 0.0
            A[(s, int(r["week"]), r["player_id"])] = {
                "rush_yds": g("rushing_yards"), "rec_yds": g("receiving_yards"), "rec": g("receptions"), "pass_yds": g("passing_yards"),
                "pass_td": g("passing_tds"), "int": g("passing_interceptions"), "rush_td": g("rushing_tds"), "rec_td": g("receiving_tds"),
                "tackles": g("def_tackles_solo") + g("def_tackle_assists"), "sacks": g("def_sacks"), "def_int": g("def_interceptions"),
                "pos": r["position"]}
    return A


# ------------------------------------------------------------------ simulation helpers
def rng_for(key, tag):
    return np.random.default_rng(zlib.crc32(repr((key, tag)).encode()))


def draw_bins(p, u):
    return np.minimum(np.searchsorted(np.cumsum(p), u), len(p) - 1)


def sum_rush(p, vals, ns, rng):
    tot = int(ns.sum())
    if tot == 0:
        return np.zeros(len(ns))
    v = vals[draw_bins(p, rng.random(tot))]
    return np.bincount(np.repeat(np.arange(len(ns)), ns), weights=v, minlength=len(ns))


def chain(P_air, c_b, P_yac, ns, rng):
    """Per-target / per-attempt chain: air bin -> catch | bucket -> YAC | bucket. Returns (yards, completions) per draw."""
    tot = int(ns.sum())
    if tot == 0:
        return np.zeros(len(ns)), np.zeros(len(ns))
    u1, u2, u3 = rng.random(tot), rng.random(tot), rng.random(tot)
    a = draw_bins(P_air, u1)
    bk = RC.AIR_BUCKET[a]
    catch = u2 < c_b[bk]
    y = np.zeros(tot, int)
    for b in range(4):
        m = bk == b
        if m.any():
            y[m] = draw_bins(P_yac[b], u3[m])
    yards = np.where(catch, RC.AIR_V[a] + RC.YAC_V[y], 0.0)
    idx = np.repeat(np.arange(len(ns)), ns)
    return np.bincount(idx, weights=yards, minlength=len(ns)), np.bincount(idx, weights=catch.astype(float), minlength=len(ns))


class Pack:
    """Efficiency predictions of one variant, indexed by record position."""

    def __init__(self, rush, rush_td, air_r, catch_r, yac_r, rec_td, air_p, comp_p, yac_p, pass_td, pass_int, rate):
        self.rush, self.rush_td, self.air_r, self.catch_r, self.yac_r, self.rec_td = rush, rush_td, air_r, catch_r, yac_r, rec_td
        self.air_p, self.comp_p, self.yac_p, self.pass_td, self.pass_int, self.rate = air_p, comp_p, yac_p, pass_td, pass_int, rate


def make_pack(kind, comps):
    """kind in {'selected','B0','simple'}; comps = dict of component aux dicts."""
    pick = lambda aux: (aux["sel"] if kind == "selected" else aux["P0"] if kind == "B0" else aux["P_pos"])
    rush = comps["rush"]["P_sel"] if kind == "selected" else comps["rush"]["P0"] if kind == "B0" else comps["rush"]["P_pos"]
    rt, ct = comps["td"]["rush_td"], comps["td"]["rec_td"]
    n_r, n_c = len(rush), len(ct["P0"]) // 2
    rtd = pick(rt)[:, 1]; ctd = pick(ct)[:, 1]
    n_rec = len(comps["rec"]["air"]["P0"])
    catch = pick(comps["rec"]["catch"])[:, 1]; yac = pick(comps["rec"]["yac"])
    n_p = len(comps["pass"]["air"]["P0"])
    pcatch = pick(comps["pass"]["completion"])[:, 1]; pyac = pick(comps["pass"]["yac"])
    rate = {t: (comps["def"][t]["_rate"] if kind == "selected" else comps["def"][t]["_rate_by_level"]["B0"] if kind == "B0" else comps["def"][t]["_rate_pos"])
            for t in DE.TARGETS}
    return Pack(rush, (rtd[:n_r], rtd[n_r:]), pick(comps["rec"]["air"]), catch.reshape(4, n_rec).T, yac.reshape(4, n_rec, -1).transpose(1, 0, 2),
                (ctd[:n_rec], ctd[n_rec:]), pick(comps["pass"]["air"]), pcatch.reshape(4, n_p).T, pyac.reshape(4, n_p, -1).transpose(1, 0, 2),
                pick(comps["pass"]["pass_td"])[:, 1], pick(comps["pass"]["interception"])[:, 1], rate)


def simulate(kind, pack, P1A, idx, D):
    """DEVELOPMENT game simulation: Phase 1A opportunity draws x Phase 1B per-opportunity distributions. Returns dict stat -> (ids, S)."""
    out = {}
    smp = P1A["samples"]
    rz_lookup = {}
    for nm in ("rz_carry", "rz_target"):
        rz_lookup[nm] = {i: k for k, i in enumerate(smp[nm]["ids"])}
    rec = {k: [] for k in ("rush_yds", "rush_td", "rec_yds", "rec", "rec_td", "pass_yds", "pass_td", "int", "tackles", "sacks", "def_int", "atd")}
    keys = {k: [] for k in rec}
    missing = defaultdict(int)
    # --- rushing (RB, QB) with rushing TDs
    carry_S = {}
    for j, (s, w, team, gid) in enumerate(smp["carry"]["ids"]):
        ns = smp["carry"]["S"][j].astype(int)
        ri = idx["rush"].get((s, w, gid))
        if ri is None:
            missing["rush"] += 1
            continue
        rng = rng_for((s, w, gid), "rush")
        yds = sum_rush(pack.rush[ri], RU.BIN_VALS, ns, rng)
        rzj = rz_lookup["rz_carry"].get((s, w, team, gid))
        n_rz = np.minimum(smp["rz_carry"]["S"][rzj].astype(int), ns) if rzj is not None else np.zeros_like(ns)
        td = rng.binomial(n_rz, pack.rush_td[0][ri]) + rng.binomial(ns - n_rz, pack.rush_td[1][ri])
        rec["rush_yds"].append(yds); rec["rush_td"].append(td)
        keys["rush_yds"].append((s, w, team, gid)); keys["rush_td"].append((s, w, team, gid))
        carry_S[(s, w, gid)] = td
    # --- receiving (RB, WR, TE)
    rec_td_map = {}
    for j, (s, w, team, gid) in enumerate(smp["target"]["ids"]):
        ns = smp["target"]["S"][j].astype(int)
        ri = idx["rec"].get((s, w, gid))
        if ri is None:
            missing["rec"] += 1
            continue
        rng = rng_for((s, w, gid), "rec")
        yds, cat = chain(pack.air_r[ri], pack.catch_r[ri], pack.yac_r[ri], ns, rng)
        rzj = rz_lookup["rz_target"].get((s, w, team, gid))
        n_rz = np.minimum(smp["rz_target"]["S"][rzj].astype(int), ns) if rzj is not None else np.zeros_like(ns)
        td = rng.binomial(n_rz, pack.rec_td[0][ri]) + rng.binomial(ns - n_rz, pack.rec_td[1][ri])
        for k, v in (("rec_yds", yds), ("rec", cat), ("rec_td", td)):
            rec[k].append(v); keys[k].append((s, w, team, gid))
        rec_td_map[(s, w, gid)] = td
    # --- passing (QB)
    for j, (s, w, team, gid) in enumerate(smp["qb_att"]["ids"]):
        ns = smp["qb_att"]["S"][j].astype(int)
        ri = idx["pass"].get((s, w, gid))
        if ri is None:
            missing["pass"] += 1
            continue
        rng = rng_for((s, w, gid), "pass")
        yds, cat = chain(pack.air_p[ri], pack.comp_p[ri], pack.yac_p[ri], ns, rng)
        td = rng.binomial(ns, pack.pass_td[ri]); it = rng.binomial(ns, pack.pass_int[ri])
        for k, v in (("pass_yds", yds), ("pass_td", td), ("int", it)):
            rec[k].append(v); keys[k].append((s, w, team, gid))
    # --- anytime TD (rush + receiving TD >= 1) for the union of rushing / receiving candidates
    union = {}
    for (s, w, gid), a in carry_S.items():
        union[(s, w, gid)] = a
    for (s, w, gid), a in rec_td_map.items():
        union[(s, w, gid)] = union.get((s, w, gid), 0) + a
    team_of = {(s, w, gid): team for src in ("carry", "target") for (s, w, team, gid) in smp[src]["ids"]}
    for k, v in union.items():
        rec["atd"].append(v); keys["atd"].append((k[0], k[1], team_of[k], k[2]))
    # --- defence
    for j, (s, w, team, gid) in enumerate(smp["def_snap"]["ids"]):
        ns = smp["def_snap"]["S"][j].astype(float)
        di = idx["def"].get((s, w, gid))
        if di is None:
            missing["def"] += 1
            continue
        rng = rng_for((s, w, gid), "def")
        for t, nm in (("tackles", "tackles"), ("sacks", "sacks"), ("interceptions", "def_int")):
            lam = pack.rate[t][di] * ns
            rec[nm].append(rng.poisson(lam)); keys[nm].append((s, w, team, gid))
    res = {k: (keys[k], np.array(rec[k], float)) for k in rec if rec[k]}
    res["_missing"] = dict(missing)
    return res


# ------------------------------------------------------------------ historical last-8 empirical baseline
def hist_baseline(D, ACT, keys, stat, N=N_DRAWS):
    """Empirical distribution of the player's stat over his team's previous 8 games (missing game rows = 0) resampled to N draws."""
    S = np.zeros((len(keys), N))
    tg = {}
    for i, (s, w, team, gid) in enumerate(keys):
        if team not in tg:
            tg[team] = [(ss, ww) for (_, ss, ww) in D.team_games[team]]
        lst = tg[team]
        k = lst.index((s, w)) if (s, w) in lst else None
        prev = lst[max(0, (k or 0) - 8):(k or 0)]
        vals = np.array([ACT.get((ss, ww, gid), {}).get(stat, 0.0) for ss, ww in prev]) if prev else np.zeros(1)
        rng = rng_for((s, w, gid), ("hist", stat))
        S[i] = vals[rng.integers(0, len(vals), N)]
    return S


def score_end_to_end(name, keys, S, ACT, stat, D, base_S, other_S):
    y = np.array([ACT.get((s, w, gid), {}).get(stat, 0.0) for (s, w, team, gid) in keys])
    if name == "atd":
        y = np.array([float(ACT.get((s, w, gid), {}).get("rush_td", 0.0) + ACT.get((s, w, gid), {}).get("rec_td", 0.0) > 0) for (s, w, team, gid) in keys])
        S = (S >= 1).astype(float)
        base_S = (base_S >= 1).astype(float); other_S = {k: (v >= 1).astype(float) for k, v in other_S.items()}
    sw = np.array([(k[0], k[1]) for k in keys])
    blocks = np.array([f"{a}-{b}" for a, b in sw])
    cr = O.crps_rows(S, y); crb = O.crps_rows(base_S, y)
    out = {"n": int(len(y)), "mean_actual": round(float(y.mean()), 4)}
    med = np.median(S, 1)
    q = {p: np.quantile(S, p, axis=1) for p in (0.1, 0.9)}
    out.update({"crps": round(float(cr.mean()), 4), "mae_median": C.cont(med, y)["mae"], "rmse_mean": C.cont(S.mean(1), y)["rmse"], "bias_mean": C.cont(S.mean(1), y)["bias"],
                "cov80": C.coverage(q[0.1], q[0.9], y), "baseline_last8_crps": round(float(crb.mean()), 4)})
    imp, p = C.block_boot(cr, crb, blocks)
    out["vs_last8_baseline"] = {"crps_improvement": imp, "p_not_better": p}
    for k, v in other_S.items():
        cro = O.crps_rows(v, y)
        imp, p = C.block_boot(cr, cro, blocks)
        out[f"vs_{k}"] = {"crps_improvement_of_selected": imp, "p_not_better": p, "other_crps": round(float(cro.mean()), 4)}
    for tag, m in (("2025", sw[:, 0] == 2025), ("2026_wk1_3", sw[:, 0] == 2026)):
        if m.sum():
            out[tag] = {"n": int(m.sum()), "crps": round(float(cr[m].mean()), 4), "baseline_last8_crps": round(float(crb[m].mean()), 4),
                        "mae_median": C.cont(med[m], y[m])["mae"]}
    if name == "atd":
        pm = S.mean(1)
        pc = np.clip(pm, 1e-4, 1 - 1e-4)
        out["logloss"] = round(float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()), 4)
        pb = np.clip(base_S.mean(1), 1e-2, 1 - 1e-2)
        out["baseline_last8_logloss(clipped .01)"] = round(float(-(y * np.log(pb) + (1 - y) * np.log(1 - pb)).mean()), 4)
        out["brier"] = round(float(((pm - y) ** 2).mean()), 4); out["mean_forecast"] = round(float(pm.mean()), 4)
    return out


def recalibrate_def_snaps(S, pit, seed_tag="defpit"):
    """Apply the validation-fit PIT map (Phase 1A hardening item C) to def_snap samples and redraw N samples per row (stratified, permuted)."""
    import nfl_phase1_hardening as H
    cap = 140
    Hm = H.pmf_from_samples(S.astype(np.int64), cap)
    Hc = H.apply_pit_map(Hm, np.array(pit["xs"]), np.array(pit["G"]))
    Fc = np.cumsum(Hc, 1); Fc[:, -1] = 1.0
    n, N = S.shape
    out = np.zeros((n, N), np.int16)
    for i in range(n):
        rng = np.random.default_rng(zlib.crc32(repr((i, seed_tag)).encode()))
        u = (rng.permutation(N) + 0.5) / N
        out[i] = np.minimum(np.searchsorted(Fc[i], u, side="left"), cap)
    return out


def archetype_tests(Rs, comps, rep):
    out = {}
    R = Rs["rush"]; P0 = comps["rush"]["P0"]
    traits = np.column_stack([P0[:, RU.BIN_VALS >= 15].sum(1), P0[:, RU.BIN_VALS < 0].sum(1), P0 @ RU.BIN_VALS])
    fams = [f for f in F.NESTED[rep["rushing"]["single_process"]["selected_level"]] if f != "interaction"]
    out["rush_single_process"], _ = EM.archetype_test(R, R.cnt["y"], RU.BIN_VALS, P0, RU.PHI_SINGLE, traits, fams, None, "rush")
    R = Rs["rec"]; P0 = comps["rec"]["air"]["P0"]
    traits = np.column_stack([P0[:, RC.AIR_V >= 20].sum(1), P0 @ RC.AIR_V, P0[:, RC.AIR_V < 0].sum(1)])
    fams = [f for f in F.NESTED[rep["receiving"]["air_yards"]["nested"]["selected_level"]] if f != "interaction"]
    out["receiving_air_yards"], preds = EM.archetype_test(R, R.cnt["air"], RC.AIR_V, P0, RC.PHI_AIR, traits, fams, None, "rec_air")
    out["rule"] = ("k-means (k=3,5) on as-of trait vectors fit on TRAIN rows only; archetype dummies appended to the selected feature level; adopted only if the validation log score "
                   "improves by >= 1e-4 nats AND the development log score improves with p < 0.10 and >= 1e-3 nats/opportunity")
    out["_air_preds"] = preds
    return out


def _round(o):
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_round(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def data_source_audit(D, inj):
    """OL / DL / LB / DB personnel data-source audit: only pregame, timestamp-safe sources are used."""
    tw = defaultdict(lambda: {"ol": 0, "dl": 0, "lb": 0, "db": 0})
    for (s, w, gid), (st, pr, tm) in D.inj.items():
        pos = (D.players.get(gid) or {}).get("pos")
        u = "ol" if pos in B.OL_POS else "dl" if pos in B.DL_POS else "lb" if pos in B.LB_POS else "db" if pos in B.DB_POS else None
        if u:
            tw[(s, w, tm)][u] += 1
    tot = sum(1 for (s, w, t) in D.game if s in (2023, 2024, 2025) or (s == 2026 and w <= 3))
    cover = {u: round(sum(1 for (s, w, t) in D.game if (s in (2023, 2024, 2025) or (s == 2026 and w <= 3)) and tw.get((s, w, t), {}).get(u, 0) > 0) / tot, 3) for u in ("ol", "dl", "lb", "db")}
    ol_out = sum(1 for k, v in inj.items() if v["ol_out"] > 0 and (k[0] in (2023, 2024, 2025) or (k[0] == 2026 and k[1] <= 3)))
    return {
        "used": {"unit_availability": "injury-report Out/Doubtful (+0.5 x Questionable) counts by unit from the historical final-weekly-report proxy (A2); "
                                      "positions from the static players table. No starter identities are reconstructed.",
                 "team_weeks_with_any_report_row_by_unit_2023_to_2026wk3": cover, "team_weeks_with_ol_out": ol_out, "team_weeks": tot},
        "not_used_and_why": {
            "participation_offense_players / defense_players": "realized same-game participation (post-game); forbidden as a pregame input; also unpublished for 2026",
            "snap_counts": "realized same-game snaps; usable only as history through prior weeks",
            "weekly_roster status": "game-day snapshot (A3); T-90m only, and cannot identify offensive-line starters",
            "depth_charts": "timestamped snapshots exist for 2025+ (dt <= cutoff) and could give OL starters, but the tables carry no 2023-24 snapshots, so any OL-starter feature "
                            "could only be fit on the 2025 development weeks (leaves nothing out-of-time); NOT used",
            "ftn box / blitz (2026)": "FTN charting continues through 2026 and feeds the scheme profile through prior weeks only (via nfl_context_v4.def_profile)",
            "man/zone, coverage, pressure (2026)": "participation is unpublished for 2026; def_profile falls back to the prior season for coverage-type rates (stale but as-of legal)"},
        "limitation": "OL quality (pass-block win rate, run-block grades) is not available from a timestamp-safe pregame source in this repository; the personnel family "
                      "measures only reported absences."}


def _best_row(o):
    n = o["nested"]
    return n["selected_level"]


def write_artifacts(rep, od):
    """Split the report into the required review artifacts."""
    W = lambda name, obj: (od / name).write_text(json.dumps(obj, indent=1, default=float))
    sel, rej, scheme, inter, tails, pers = {}, {}, {}, {}, {}, {}
    def comp_entries():
        yield "rushing", rep["rushing"]
        for k, v in rep["receiving"].items():
            if isinstance(v, dict) and "nested" in v:
                yield f"receiving/{k}", v
        for k, v in rep["passing"].items():
            if isinstance(v, dict) and "nested" in v:
                yield f"passing/{k}", v
        for k, v in rep["td_events"].items():
            yield f"td/{k}", v
        for k, v in rep["defense_events"].items():
            yield f"defense/{k}", v
    for name, o in comp_entries():
        if name == "rushing":
            sel[name] = {"structure": o["selected"]["structure"], "level": o["selected"]["level"], "hierarchy": o["hier_tuning"], "rule": o["selected"]["rule"]}
            rej[name] = {"single_process_levels": {l: e["development"]["combined"] for l, e in o["single_process"]["levels"].items()},
                         "two_process": {T: {"selected_level": x["selected_level"], "valid_logscore_by_level": x["valid_logscore_by_level"]} for T, x in o["two_process"].items()}}
            scheme[name] = {"single_process_leave_one_family_out_from_B4": o["single_process"]["leave_one_family_out_from_B4"],
                            "B2_to_B3_scheme_added": o["single_process"]["levels"]["B3"]["development"].get("vs_ref")}
            inter[name] = {"B4": o["single_process"]["levels"]["B4"]["development"]["combined"], "B5": o["single_process"]["levels"]["B5"]["development"]["combined"],
                           "B5_vs_B0": o["single_process"]["levels"]["B5"]["development"].get("vs_ref")}
            tails[name] = {l: e.get("tail_calibration") for l, e in o["single_process"]["levels"].items()}
            pers[name] = o["player_persistence_B0_vs_position"]
        elif name.startswith("defense/"):
            n = o["nested"]
            sel[name] = {"level": n["selected_level"], "hierarchy": o["hier_tuning"], "distribution": o["distribution"], "dispersion_r": o["dispersion_r"]}
            rej[name] = {"levels": {l: e["development"] for l, e in n["levels"].items()}}
            inter[name] = {"B4": n["levels"]["B4"]["development"], "B5": n["levels"]["B5"]["development"]}
            scheme[name] = {"B2": n["levels"]["B2"]["development"].get("combined"), "B3": n["levels"]["B3"]["development"].get("combined")}
            tails[name] = {"threshold": o["tail_threshold"], "calibration": o["tail_calibration"]}
            pers[name] = {"baselines": o["baselines"]}
        else:
            n = o["nested"]
            sel[name] = {"level": n["selected_level"], "hierarchy": o["hier_tuning"]}
            rej[name] = {"levels": {l: e["development"].get("combined") for l, e in n["levels"].items()}}
            scheme[name] = {"leave_one_family_out_from_B4": n.get("leave_one_family_out_from_B4")}
            inter[name] = {"B4": n["levels"]["B4"]["development"].get("combined"), "B5": n["levels"]["B5"]["development"].get("combined"),
                           "B5_vs_B0": n["levels"]["B5"]["development"].get("vs_ref")}
            tails[name] = {"tail_calibration_B_selected": n["levels"][n["selected_level"]].get("tail_calibration"), "reliability": o.get("reliability")}
            pers[name] = o.get("player_persistence_B0_vs_position")
    W("selected_architecture.json", sel); W("rejected_candidates.json", rej); W("scheme_ablation.json", scheme)
    W("interaction_ablation.json", inter); W("tail_calibration.json", tails); W("persistence_diagnostics.json", pers)
    W("data_source_audit.json", rep["data_source_audit"]); W("archetype_tests.json", rep["archetypes"])
    W("end_to_end_development.json", rep["end_to_end_development"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stage-cache", default=None)
    ap.add_argument("--p1a-cache", default=None)
    ap.add_argument("--rec-cache", default=None, help="pickle of Phase 1B records (development speed-up only)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--hardening-json", default=None, help="phase1a_hardening_selected.json (per-type calibration decisions and PIT maps)")
    a = ap.parse_args()
    t0 = time.time()
    P1A = get_phase1a(a.data_dir, a.stage_cache, a.p1a_cache)
    print(f"phase1a samples ({time.time() - t0:.0f}s)", flush=True)
    D = P1.Data(a.data_dir)
    T = B.PlayTallies(a.data_dir, SEASONS)
    pos = {(s, g): r["pos"] for (s, w, g), r in D.roster.items()}
    PD = v4.PlayData(a.data_dir, SEASONS, pos)
    inj = B.build_inj_index(D)
    extras = defaultdict(list)
    for (s, w), lst in P1A["extras"].items():
        for gid, team, nm in lst:
            if nm != "def_snap":
                extras[(s, w)].append((gid, team))
    Rs = B.build_records(D, T, PD, inj, extras=extras)
    print(f"records {[(k, v.n) for k, v in Rs.items()]} ({time.time() - t0:.0f}s)", flush=True)
    comps = {}
    rep = {"phase": "1B (shadow)", "label": C.DEV_LABEL, "protocol_version": "1.1"}
    rush_rep, comps["rush"] = RU.run(Rs["rush"])
    rec_rep, aux_rec = RC.run(Rs["rec"]); comps["rec"] = aux_rec
    pass_rep, aux_pass = PA.run(Rs["pass"]); comps["pass"] = aux_pass
    td_rep, aux_td = EM.run_td(Rs["rush"], Rs["rec"]); comps["td"] = aux_td
    print(f"component studies done ({time.time() - t0:.0f}s)", flush=True)
    # defence
    drows = DE.load_defense_rows(D, a.data_dir)
    dex = defaultdict(list)
    for (s, w), lst in P1A["extras"].items():
        for gid, team, nm in lst:
            if nm == "def_snap":
                dex[(s, w)].append((gid, team))
    drows = DE.add_extras(D, drows, dex)
    drecs = DE.build(D, T, PD, inj, drows)
    def_rep = DE.run(drecs)
    comps["def"] = def_rep
    print(f"defence done ({time.time() - t0:.0f}s)", flush=True)
    rep["rushing"] = rush_rep; rep["receiving"] = rec_rep; rep["passing"] = pass_rep; rep["td_events"] = td_rep
    rep["defense_events"] = {t: {k: v for k, v in o.items() if not k.startswith("_")} for t, o in def_rep.items()}
    # ---- end-to-end DEVELOPMENT assembly
    ACT = load_actuals(a.data_dir)
    idx = {"rush": {k: i for i, k in enumerate(Rs["rush"].key)}, "rec": {k: i for i, k in enumerate(Rs["rec"].key)},
           "pass": {k: i for i, k in enumerate(Rs["pass"].key)}, "def": {(r["s"], r["w"], r["gid"]): i for i, r in enumerate(drecs)}}
    # position-only defence rate (for the 'simple' variant)
    for t in DE.TARGETS:
        y = np.array([r[t] for r in drecs], float); gi = def_rep[t]["hier_tuning"]["gamma_idx"]; kq = def_rep[t]["hier_tuning"]["kappa_pos_snaps"]
        def_rep[t]["_rate_pos"] = np.array([DE._rate_base(r["base"], t, gi, 1e12, kq) for r in drecs])
    rep["archetypes"] = archetype_tests(Rs, comps, rep)     # component-level adoption decision; the downstream assembly check is reported separately
    variants = {k: make_pack(k, comps) for k in ("selected", "B0", "simple")}
    sims = {k: simulate(k, v, P1A, idx, D) for k, v in variants.items()}
    e2e = {"label": "DEVELOPMENT end-to-end (2025 + 2026 wk1-3; burned, not holdout)", "missing_efficiency_records": sims["selected"]["_missing"], "stats": {}}
    STAT = {"rush_yds": "rush_yds", "rush_td": "rush_td", "rec_yds": "rec_yds", "rec": "rec", "rec_td": "rec_td", "pass_yds": "pass_yds", "pass_td": "pass_td", "int": "int",
            "tackles": "tackles", "sacks": "sacks", "def_int": "def_int", "atd": "atd"}
    for name, stat in STAT.items():
        if name not in sims["selected"]:
            continue
        keys, S = sims["selected"][name]
        base_S = hist_baseline(D, ACT, keys, stat if name != "atd" else "rush_td")
        if name == "atd":
            b2 = hist_baseline(D, ACT, keys, "rec_td"); base_S = base_S + b2
        others = {"phase1a_plus_simple_efficiency": sims["simple"][name][1], "phase1a_plus_hierarchical_B0_efficiency": sims["B0"][name][1]}
        e2e["stats"][name] = score_end_to_end(name, keys, S, ACT, stat, D, base_S, others)
        if name == "rush_yds":
            qk = [i for i, k in enumerate(keys) if (ACT.get((k[0], k[1], k[3])) or {}).get("pos") == "QB"]
            if qk:
                kq = [keys[i] for i in qk]
                e2e["stats"]["qb_rush_yds"] = score_end_to_end("qb_rush_yds", kq, S[qk], ACT, "rush_yds", D, base_S[qk], {k: v[qk] for k, v in others.items()})
    k_adopt = rep["archetypes"]["receiving_air_yards"]["adopted_k"]
    if k_adopt is not None:
        keep = comps["rec"]["air"]["sel"]
        comps["rec"]["air"]["sel"] = rep["archetypes"]["_air_preds"][k_adopt]
        sim_arch = simulate("selected", make_pack("selected", comps), P1A, idx, D)
        comps["rec"]["air"]["sel"] = keep
        e2e["receiving_with_air_archetypes"] = {"note": f"air-yard pmf refit with k={k_adopt} archetype dummies (adopted at component level); all else unchanged", "stats": {}}
        for name in ("rec_yds", "rec"):
            keys, S = sim_arch[name]
            base_S = hist_baseline(D, ACT, keys, STAT[name])
            e2e["receiving_with_air_archetypes"]["stats"][name] = score_end_to_end(name, keys, S, ACT, STAT[name], D, base_S, {"without_archetypes": sims["selected"][name][1]})
    if a.hardening_json and Path(a.hardening_json).exists():
        hj = json.loads(Path(a.hardening_json).read_text())
        pit = hj.get("def_snap", {}).get("pit_map")
        if pit:
            P1A_cal = dict(P1A); smp = dict(P1A["samples"]); d = dict(smp["def_snap"]); d["S"] = recalibrate_def_snaps(d["S"], pit); smp["def_snap"] = d; P1A_cal["samples"] = smp
            sim_cal = simulate("selected", variants["selected"], P1A_cal, idx, D)
            e2e["defense_with_recalibrated_snaps"] = {"note": "def_snap samples replaced by the validation-fit PIT recalibration (Phase 1A hardening item C); other inputs unchanged", "stats": {}}
            for name in ("tackles", "sacks", "def_int"):
                keys, S = sim_cal[name]
                base_S = hist_baseline(D, ACT, keys, STAT[name])
                r = score_end_to_end(name, keys, S, ACT, STAT[name], D, base_S, {"raw_phase1a_def_snap": sims["selected"][name][1]})
                e2e["defense_with_recalibrated_snaps"]["stats"][name] = r
    e2e["comparators_not_reproduced"] = {"incumbent_production_projections": "no distributional incumbent output exists in the development pipeline (the production champions emit point projections)",
                                          "phase_0B_decomposed": "the Phase 0B oracle decomposition is an error-budget analysis on realized opportunities, not a pregame forecaster; it is not a comparator for a full pregame distribution"}
    rep["end_to_end_development"] = e2e
    rep["data_source_audit"] = data_source_audit(D, inj)
    rep["fit_audit"] = C.FIT_AUDIT
    if not a.no_write:
        od = Path(a.out_dir) if a.out_dir else OUT
        od.mkdir(parents=True, exist_ok=True)
        (od / "report.json").write_text(json.dumps(_round(rep), indent=1, default=float))
        write_artifacts(_round(rep), od)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)
    return rep


if __name__ == "__main__":
    main()
