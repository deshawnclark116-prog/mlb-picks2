"""
CFB_PHASE1B_EXPERIMENTS -- development experiments of the Phase 1B core engine on BURNED development 2019-2024 (D1-D4). Each experiment writes <scratch>/exp_<name>.json (full record incl. rejected variants); the driver `register` step copies decisions into the
repository artifacts. 2025 is never loaded (harness.assert_dev_seasons + load seasons <= 2024).
  python cfb_phase1b_experiments.py NAME [NAME...]    names: o2 qb receptions completion rush_yards pass_yards rec_yards hazards volume points
"""
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import cfb_phase1_common as C
import cfb_phase1_data as D
import cfb_phase1_efficiency as EF
import cfb_phase1_opportunity as OP
import cfb_phase1_role_state as RS
import cfb_phase1_team_environment as TE
import cfb_phase1b_components as K
import cfb_phase1b_harness as H

SCRATCH = Path("/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/cfb1")
DEV = list(range(2018, 2025))


def load():
    tg, pg = D.load_frozen(seasons=DEV)
    p = SCRATCH / "rows_v2_dev.pkl"
    if p.exists():
        rows = pickle.load(open(p, "rb"))
    else:
        cand, _ = D.build_candidates(tg, pg, list(range(2019, 2025)))
        rows = RS.build_feature_table(tg, pg, cand, list(range(2019, 2025))); pickle.dump(rows, open(p, "wb"), protocol=4)
    rows = K.attach_team_labels([r for r in rows if r["target_valid"]], pg, tg)
    team = TE.build_team_table_v2(tg, pg, list(range(2019, 2025)))
    return tg, pg, rows, team


def save(name, res):
    (SCRATCH / f"exp_{name}.json").write_text(json.dumps(res, indent=1, default=float))
    print(name, "selected", res["selected"], "best", res["best_by_primary_loss"], {v: {k: round(x, 4) for k, x in m.items()} for v, m in res["mean_over_folds"].items()})
    for h in res["promotion_history"]:
        print("  ", h["challenger"], "vs", h["incumbent"], "promoted" if h["promoted"] else "no", {g: x["pass"] for g, x in h["gates"].items()}, "rel", round(h["gates"]["G1"]["rel_gain"], 4),
              {n: round(s["relative_worse"], 3) for n, s in h["slices"].items() if s.get("eligible") and not s["pass"]})


# ------------------------------------------------------------------ rows helpers
def sub(rows, f):
    return [r for r in rows if f(r)]


def run_o2(rows):
    return H.run_experiment("O2_carries", rows, lambda v: [r["y_carries"] for r in v], [("O1_B0", K.o1_variant("base")), ("O1_C1", K.o1_variant("logit")), ("O2a", K.o2_variant(False)), ("O2b_activity", K.o2_variant(True))], K.player_slices)


def run_o2_families(rows):
    """Family-level comparison (primary rushers QB / RB / FB vs gadget WR / TE / other), each against O1 B0 on ITS OWN rows (development amendment 2)."""
    out = {}
    for fam, pred in (("primary", lambda r: r["position"] in K.PRIMARY), ("gadget", lambda r: r["position"] not in K.PRIMARY)):
        fr = sub(rows, pred)
        def wrap(fp):
            return lambda tr, va: fp(tr, va)
        # fit on ALL rows (the allocation needs the whole team) but score only the family rows
        def mk(fp):
            def f(tr, va):
                return fp(tr, va)
            return f
        vs = [("O1_B0", K.o1_variant("base")), ("O1_C1", K.o1_variant("logit")), ("O2a", K.o2_variant(False)), ("O2b_activity", K.o2_variant(True))]
        def fam_variant(fp):
            def f(tr, va):
                full = fp(tr, va)
                idx = np.array([pred(r) for r in va])
                return full[idx]
            return f
        res = H.run_experiment(f"O2_carries_{fam}", [r for r in rows], lambda v, pred=pred: [r["y_carries"] for r in v if pred(r)], [(n, fam_variant(fp)) for n, fp in vs],
                               lambda va, tr, pred=pred: {k: m[np.array([pred(r) for r in va])] for k, m in K.player_slices(va, tr).items()})
        out[fam] = res
    return out


def run_qb(rows):
    q = sub(rows, lambda r: r["POS_QB"] == 1 and r["n_att"] > 0)
    names = K.ROLE_NAMES
    vs = [("Q0_base", K.alloc_variant("base", "P_ATT_SHARE_L5", "P_ATT_SHARE_L5", "y_pass_att", "n_att", K_ATT, lambda r: 0, names=names)),
          ("Q1_logit", K.alloc_variant("logit", "P_ATT_SHARE_L5", "P_ATT_SHARE_L5", "y_pass_att", "n_att", K_ATT, lambda r: 0, names=names)),
          ("Q2_logit_activity", K.alloc_variant("logit", "P_ATT_SHARE_L5", "P_ATT_SHARE_ACTIVE_L5", "y_pass_att", "n_att", K_ATT, lambda r: 0, activity=True, act_y=lambda r: int(r["y_pass_att"] >= 1), names=names))]
    return H.run_experiment("QB_attempt_share", q, lambda v: [r["y_pass_att"] for r in v], vs, K.player_slices)


K_ATT = K.K_ATT


def run_receptions(rows):
    q = sub(rows, lambda r: r["n_rec"] > 0)
    names = K.ROLE_NAMES
    grp = lambda r: K.pos_group(r)
    vs = [("R0_base", K.alloc_variant("base", "P_REC_SHARE_L5", "P_REC_SHARE_L5", "y_receptions", "n_rec", K.K_REC, grp, names=names)),
          ("R1_logit", K.alloc_variant("logit", "P_REC_SHARE_L5", "P_REC_SHARE_L5", "y_receptions", "n_rec", K.K_REC, grp, names=names)),
          ("R2_logit_activity", K.alloc_variant("logit", "P_REC_SHARE_L5", "P_REC_SHARE_ACTIVE_L5", "y_receptions", "n_rec", K.K_REC, grp, activity=True, act_y=lambda r: int(r["y_receptions"] >= 1), names=names))]
    return H.run_experiment("receiver_reception_share", q, lambda v: [r["y_receptions"] for r in v], vs, K.player_slices)


def run_completion(rows):
    q = sub(rows, lambda r: r["POS_QB"] == 1 and r["y_pass_att"] >= 1)
    mk = lambda kind, ctx: (lambda tr, va: K.RateComponent(kind, "P_COMP_L20", "P_ATT_L20", "y_completions", "y_pass_att", 80.0, lambda r: 0, ctx).fit(sub(tr, lambda r: r["POS_QB"] == 1)).pmf(va, K.K_ATT))
    ctx = ["T_COMP_PCT_B0", "O_COMP_PCT_ALLOWED_B0", "RATING_GAP", "IS_HOME", "T_PASS_RATE_B0", "T_YPCOMP_B0"]
    vs = [("K0_league_shrunk", mk("base", [])), ("K1_context_logit", mk("logit", ctx))]
    return H.run_experiment("completion_rate", q, lambda v: [r["y_completions"] for r in v], vs, K.player_slices)


def run_hazards(rows):
    out = {}
    grp = lambda r: K.pos_group(r)
    spec = {"rush_td": ("P_RUSH_TD_L20", "P_CARRIES_L20", "y_rush_td", "y_carries", 100.0, ["T_TD_B0", "O_TD_ALLOWED_B0", "T_RZ_RATE_B0", "RATING_GAP", "T_YPC_B0"], lambda r: True),
            "rec_td": ("P_REC_TD_L20", "P_REC_L20", "y_rec_td", "y_receptions", 60.0, ["T_TD_B0", "O_TD_ALLOWED_B0", "T_RZ_RATE_B0", "RATING_GAP", "T_YPCOMP_B0"], lambda r: True),
            "pass_td": ("P_PASS_TD_L20", "P_COMP_L20", "y_pass_td", "y_completions", 100.0, ["T_TD_B0", "O_TD_ALLOWED_B0", "T_RZ_RATE_B0", "RATING_GAP", "T_YPCOMP_B0"], lambda r: r["POS_QB"] == 1),
            "interception": ("P_INT_L20", "P_ATT_L20", "y_int", "y_pass_att", 300.0, ["T_INT_RATE_B0", "O_INT_FORCED_B0", "RATING_GAP", "T_PASS_RATE_B0"], lambda r: r["POS_QB"] == 1)}
    for nm, (ev, op, yk, nk, k0, ctx, filt) in spec.items():
        q = sub(rows, lambda r: r[nk] >= 1 and filt(r))
        mk = lambda kind, ctx_: (lambda tr, va: K.RateComponent(kind, ev, op, yk, nk, k0, grp, ctx_).fit(sub(tr, lambda r: filt(r))).pmf(va, 40))
        out[nm] = H.run_experiment(f"hazard_{nm}", q, lambda v, yk=yk: [r[yk] for r in v], [(f"{nm}_H0_shrunk_rate", mk("base", [])), (f"{nm}_H1_context_logit", mk("logit", ctx))], K.player_slices)
    return out


# ------------------------------------------------------------------ efficiency compounds (per carry / per completion / per reception)
def pos_map(pg):
    c = defaultdict(Counter)
    for r in pg:
        c[r["player_id"]][r["position"]] += 1
    return {p: v.most_common(1)[0][0] for p, v in c.items()}


def grp_of(pos):
    return "QB" if pos == "QB" else "RB" if pos in ("RB", "FB") else "WR" if pos == "WR" else "TE" if pos == "TE" else "OTH"


def eff_experiment(name, ev_rows, key_fn, rows, row_key, c_key, y_key, z_col, pm, team_z, filt):
    q = sub(rows, filt)
    index = EF.EventIndex(ev_rows, key_fn)
    zs = np.array([r[z_col] for r in rows if np.isfinite(r[z_col])]); zmu, zsd = float(zs.mean()), float(zs.std())
    group_of_key = lambda k: grp_of(pm.get(k, "?"))
    def build(tr, va, kappa, tilt):
        tmax = max(r["season"] for r in tr) * 100 + 99
        pos_pmf, league = EF.pooled_pmf(index, None, group_of_key, tmax)
        base_va = np.stack([pos_pmf.get(grp_of(r["position"]) if name != "pass_yards" else "QB", league) for r in va])
        widx = [C.week_index(r["season"], r["week"]) for r in va]
        keys = [r[row_key] for r in va]
        if kappa is None:
            p = base_va
        else:
            p, _ = EF.shrink_rows(index, keys, widx, base_va, kappa)
        if tilt:
            ev_t = [e for e in ev_rows if C.week_index(e["season"], e["week"]) <= tmax and (e["game_id"], e["team"]) in team_z]
            if len(ev_t) > 20000:
                ev_t = ev_t[::max(1, len(ev_t) // 60000)]
            bp = np.stack([pos_pmf.get(group_of_key(key_fn(e)), league) for e in ev_t]); z = np.array([(team_z[(e["game_id"], e["team"])] - zmu) / zsd for e in ev_t]); yi = np.array([EF.clip_idx(e["yards"]) for e in ev_t])
            theta = EF.fit_tilt(bp, z, yi)
            zv = np.array([(r[z_col] - zmu) / zsd if np.isfinite(r[z_col]) else 0.0 for r in va])
            p = EF.tilt_rows(p, zv, theta)
        return EF.CompoundPmf(p, np.array([r[c_key] for r in va]))
    vs = [("E0_position_pmf", lambda tr, va: build(tr, va, None, False)), ("E1_player_k60", lambda tr, va: build(tr, va, 60.0, False)), ("E1_player_k150", lambda tr, va: build(tr, va, 150.0, False)), ("E2_player_k60_opp_tilt", lambda tr, va: build(tr, va, 60.0, True))]
    return H.run_experiment(name, q, lambda v: [r[y_key] for r in v], vs, K.player_slices)


def run_efficiency(rows, team, which, tg, pg):
    pm = pos_map(pg)
    ev = D.load_events(list(range(2018, 2025)))
    team_rows = {(r["game_id"], r["team"]): r for r in team}
    if which == "rush_yards":
        tz = {k: r["O_YPC_ALLOWED_B0"] for k, r in team_rows.items()}
        return eff_experiment("rush_yards", ev["R"], lambda e: e["player_id"], rows, "player_id", "y_carries", "y_rush_yards", "O_YPC_ALLOWED_B0", pm, tz, lambda r: r["y_carries"] >= 1)
    comp = [e for e in ev["P"] if e["kind"] == "C"]
    tz = {k: r["O_YPCOMP_ALLOWED_B0"] for k, r in team_rows.items()}
    if which == "pass_yards":
        return eff_experiment("pass_yards", comp, lambda e: e["qb_id"], rows, "player_id", "y_completions", "y_pass_yards", "O_YPCOMP_ALLOWED_B0", pm, tz, lambda r: r["POS_QB"] == 1 and r["y_completions"] >= 1)
    return eff_experiment("rec_yards", [e for e in comp if e["receiver_id"]], lambda e: e["receiver_id"], rows, "player_id", "y_receptions", "y_rec_yards", "O_YPCOMP_ALLOWED_B0", pm, tz, lambda r: r["y_receptions"] >= 1)


def team_slices(va, tr):
    a = TE.table_arrays(va, TE.TEAM_FEATURES_V2)
    return {"fbs_vs_fbs": a["O_IS_FCS"] == 0, "fbs_vs_fcs": a["O_IS_FCS"] == 1, "home": a["IS_HOME"] == 1, "away": (a["IS_HOME"] == 0) & (a["IS_NEUTRAL"] == 0), "neutral": a["IS_NEUTRAL"] == 1, "early(<=3 games)": a["T_N_SEASON"] <= 3, "established": a["T_N_SEASON"] > 3,
            "gap<100": a["ABS_RATING_GAP"] < 100, "gap100-250": (a["ABS_RATING_GAP"] >= 100) & (a["ABS_RATING_GAP"] <= 250), "gap>250": a["ABS_RATING_GAP"] > 250, "low_coverage": (a["T_N_SEASON"] <= 3) | (a["O_RATING_N"] <= 3), "high_coverage": (a["T_N_SEASON"] > 3) & (a["O_RATING_N"] > 3)}


def run_volume(team):
    out = {}
    for tgt, yk in (("R", "y_rush"), ("A", "y_pass"), ("N", "y_plays")):
        out[tgt] = H.run_experiment(f"volume_{tgt}", team, lambda v, yk=yk: [r[yk] for r in v], K.volume_variants(tgt), team_slices, min_rows=H.TH["min_team"])
    return out


def run_points(team):
    def mk(kind):
        def fp(tr, va):
            ta, vt = TE.table_arrays(tr, TE.TEAM_FEATURES_V2), TE.table_arrays(va, TE.TEAM_FEATURES_V2); y = np.array([r["y_points"] for r in tr], float)
            m = TE.B0Count("T_PTS_B0").fit(ta, y) if kind == "b0" else TE.C1Count(TE.TEAM_FEATURES_V2).fit(ta, y)
            pm, _ = C.nb2_pmf_matrix(m.mean(vt), m.alpha(), 130)
            return pm
        return fp
    return H.run_experiment("team_points", team, lambda v: [r["y_points"] for r in v], [("S0_B0_blend", mk("b0")), ("S1_C1_state_glm", mk("c1"))], team_slices, min_rows=H.TH["min_team"])


if __name__ == "__main__":
    names = sys.argv[1:]
    tg, pg, rows, team = load()
    t0 = time.time()
    for nm in names:
        if nm == "o2":
            save("o2", run_o2(rows)); [save(f"o2_{k}", v) for k, v in run_o2_families(rows).items()]
        elif nm == "qb":
            save("qb", run_qb(rows))
        elif nm == "receptions":
            save("receptions", run_receptions(rows))
        elif nm == "completion":
            save("completion", run_completion(rows))
        elif nm == "hazards":
            for k, v in run_hazards(rows).items():
                save(f"hazard_{k}", v)
        elif nm in ("rush_yards", "pass_yards", "rec_yards"):
            save(nm, run_efficiency(rows, team, nm, tg, pg))
        elif nm == "volume":
            for k, v in run_volume(team).items():
                save(f"volume_{k}", v)
        elif nm == "points":
            save("points", run_points(team))
        print(nm, "done", round(time.time() - t0), "s", flush=True)


# ------------------------------------------------------------------ integrated development evaluation (D1-D4)
def integrated_dev(choices, M=1000, folds=None, limit_games=None):
    import cfb_phase1_forecast as FC
    import cfb_phase1_integrated as IT
    tg, pg, rows, team = load()
    pm = pos_map(pg)
    ev = D.load_events(DEV)
    team_by_key = {(r["game_id"], r["team"]): r for r in team}
    out = {}
    for fid, tr_s, va_s in (folds or C.DEV_FOLDS):
        tr_rows, va_rows = H.sel(rows, tr_s), H.sel(rows, [va_s]); tr_team, va_team = H.sel(team, tr_s), H.sel(team, [va_s])
        if limit_games:
            gids = sorted({t["game_id"] for t in va_team})[::limit_games]; gs = set(gids); va_team = [t for t in va_team if t["game_id"] in gs]; keys = {(t["game_id"], t["team"]) for t in va_team}; va_rows = [r for r in va_rows if (r["game_id"], r["team"]) in keys]
        t0 = time.time()
        import hashlib
        ck = SCRATCH / f"engine_{fid}_{hashlib.sha256(json.dumps({k: v for k, v in sorted(choices.items()) if k != 'reweight_points'}, sort_keys=True).encode()).hexdigest()[:10]}.pkl"
        if ck.exists():
            eng = pickle.load(open(ck, 'rb'))
        else:
            eng = FC.CoreEngine(choices).fit(tr_rows, tr_team, ev, pm, team_by_key); pickle.dump(eng, open(ck, 'wb'), protocol=4)
        eng.ch = {**eng.ch, **choices}
        t1 = time.time()
        res = IT.evaluate_integrated(eng, tr_rows, va_rows, va_team, M, seed0=1000 * va_s, model_id=f"core_{fid}", int_valid=lambda r: r["season"] in (2018, 2019, 2020, 2024))
        res["seconds"] = {"fit": round(t1 - t0), "simulate_score": round(time.time() - t1)}; res["engine_artifact"] = eng.artifact()
        out[fid] = res
        print(fid, "fit", round(t1 - t0), "s sim+score", round(time.time() - t1), "s identities", res["identity_failures"], flush=True)
    return out


# ------------------------------------------------------------------ specs of the FROZEN selections for the single-split integrated confirmation (same builders as development)
def eff_parts(name, ev_rows, key_fn, rows, row_key, c_key, filt, pm):
    q = sub(rows, filt)
    index = EF.EventIndex(ev_rows, key_fn)
    group_of_key = lambda k: grp_of(pm.get(k, "?"))
    def build(tr, va, kappa):
        tmax = max(r["season"] for r in tr) * 100 + 99
        pos_pmf, league = EF.pooled_pmf(index, None, group_of_key, tmax)
        base_va = np.stack([pos_pmf.get(grp_of(r["position"]) if name != "pass_yards" else "QB", league) for r in va])
        p = base_va if kappa is None else EF.shrink_rows(index, [r[row_key] for r in va], [C.week_index(r["season"], r["week"]) for r in va], base_va, kappa)[0]
        return EF.CompoundPmf(p, np.array([r[c_key] for r in va]))
    return q, build


def confirm_specs(rows, team, tg, pg, ev, choices):
    """-> list of spec dicts {name, rows, y_fn, variants, baseline, selected, kind, slice_fn, min_rows}; baselines are the simple Phase 1 constructions, 'selected' the frozen choice."""
    pm = pos_map(pg); specs = []
    for tgt, yk in (("R", "y_rush"), ("A", "y_pass"), ("N", "y_plays")):
        vv = K.volume_variants(tgt)
        names = dict(vv)
        specs.append({"name": f"volume_{tgt}", "rows": team, "y_fn": lambda v, yk=yk: [r[yk] for r in v], "variants": [("IND_T1B0_T2B0", names["IND_T1B0_T2B0"]), ("IND_T1C1_T2B0", names["IND_T1C1_T2B0"]), ("V2_N_C1_pi_logit", names["V2_N_C1_pi_logit"])],
                      "baseline": "IND_T1B0_T2B0", "selected": "V2_N_C1_pi_logit", "kind": "count", "slice_fn": team_slices, "min_rows": H.TH["min_team"]})
    specs.append({"name": "carries_O2", "rows": rows, "y_fn": lambda v: [r["y_carries"] for r in v], "variants": [("O1_B0", K.o1_variant("base")), ("O1_C1", K.o1_variant("logit")), ("O2b_activity", K.o2_variant(choices["carry_activity"]))], "baseline": "O1_B0", "selected": "O2b_activity", "kind": "count", "slice_fn": K.player_slices, "min_rows": H.TH["min_player"]})
    q = sub(rows, lambda r: r["POS_QB"] == 1 and r["n_att"] > 0)
    specs.append({"name": "qb_attempt_share", "rows": q, "y_fn": lambda v: [r["y_pass_att"] for r in v], "variants": [("Q0_base", K.alloc_variant("base", "P_ATT_SHARE_L5", "P_ATT_SHARE_L5", "y_pass_att", "n_att", K.K_ATT, K.g_zero, names=K.ROLE_NAMES)), ("Q1_logit", K.alloc_variant("logit", "P_ATT_SHARE_L5", "P_ATT_SHARE_L5", "y_pass_att", "n_att", K.K_ATT, K.g_zero, names=K.ROLE_NAMES))], "baseline": "Q0_base", "selected": "Q1_logit", "kind": "count", "slice_fn": K.player_slices, "min_rows": H.TH["min_player"]})
    q = sub(rows, lambda r: r["n_rec"] > 0)
    specs.append({"name": "receiver_reception_share", "rows": q, "y_fn": lambda v: [r["y_receptions"] for r in v], "variants": [("R0_base", K.alloc_variant("base", "P_REC_SHARE_L5", "P_REC_SHARE_L5", "y_receptions", "n_rec", K.K_REC, K.pos_group, names=K.ROLE_NAMES)), ("R1_logit", K.alloc_variant("logit", "P_REC_SHARE_L5", "P_REC_SHARE_L5", "y_receptions", "n_rec", K.K_REC, K.pos_group, names=K.ROLE_NAMES))], "baseline": "R0_base", "selected": "R1_logit", "kind": "count", "slice_fn": K.player_slices, "min_rows": H.TH["min_player"]})
    q = sub(rows, lambda r: r["POS_QB"] == 1 and r["y_pass_att"] >= 1)
    mk = lambda kind, ctx: (lambda tr, va: K.RateComponent(kind, "P_COMP_L20", "P_ATT_L20", "y_completions", "y_pass_att", 80.0, K.g_zero, ctx).fit(sub(tr, lambda r: r["POS_QB"] == 1)).pmf(va, K.K_ATT))
    specs.append({"name": "completion_rate", "rows": q, "y_fn": lambda v: [r["y_completions"] for r in v], "variants": [("K0_league_shrunk", mk("base", [])), ("K1_context_logit", mk("logit", FC_CTX["completion"]))], "baseline": "K0_league_shrunk", "selected": "K1_context_logit", "kind": "count", "slice_fn": K.player_slices, "min_rows": H.TH["min_player"]})
    hz = {"rush_td": ("P_RUSH_TD_L20", "P_CARRIES_L20", "y_rush_td", "y_carries", 100.0, "rush_td", lambda r: True), "rec_td": ("P_REC_TD_L20", "P_REC_L20", "y_rec_td", "y_receptions", 60.0, "rec_td", lambda r: True), "pass_td": ("P_PASS_TD_L20", "P_COMP_L20", "y_pass_td", "y_completions", 100.0, "rush_td", lambda r: r["POS_QB"] == 1)}
    ctxmap = {"rush_td": FC_CTX["rush_td"], "rec_td": FC_CTX["rec_td"], "pass_td": FC_CTX["rec_td"]}
    for nm, (ev_, op, yk, nk, k0, _c, filt) in hz.items():
        q = sub(rows, lambda r, nk=nk, filt=filt: r[nk] >= 1 and filt(r))
        mk2 = lambda kind, ctx_, ev_=ev_, op=op, yk=yk, nk=nk, k0=k0, filt=filt: (lambda tr, va: K.RateComponent(kind, ev_, op, yk, nk, k0, K.pos_group, ctx_).fit(sub(tr, lambda r: filt(r))).pmf(va, 40))
        specs.append({"name": f"hazard_{nm}", "rows": q, "y_fn": lambda v, yk=yk: [r[yk] for r in v], "variants": [("H0", mk2("base", [])), ("H1", mk2("logit", ctxmap[nm]))], "baseline": "H0", "selected": "H1", "kind": "count", "slice_fn": K.player_slices, "min_rows": H.TH["min_player"]})
    team_rows = {(r["game_id"], r["team"]): r for r in team}
    q, b = eff_parts("rush_yards", ev["R"], lambda e: e["player_id"], rows, "player_id", "y_carries", lambda r: r["y_carries"] >= 1, pm)
    specs.append({"name": "rush_yards_per_carry_compound", "rows": q, "y_fn": lambda v: [r["y_rush_yards"] for r in v], "variants": [("E0", lambda tr, va: b(tr, va, None)), ("E1_k150", lambda tr, va: b(tr, va, choices["rush_kappa"]))], "baseline": "E0", "selected": "E1_k150", "kind": "count", "slice_fn": K.player_slices, "min_rows": H.TH["min_player"]})
    comp = [e for e in ev["P"] if e["kind"] == "C"]
    q, b = eff_parts("rec_yards", [e for e in comp if e["receiver_id"]], lambda e: e["receiver_id"], rows, "player_id", "y_receptions", lambda r: r["y_receptions"] >= 1, pm)
    specs.append({"name": "rec_yards_per_reception_compound", "rows": q, "y_fn": lambda v: [r["y_rec_yards"] for r in v], "variants": [("E0", lambda tr, va: b(tr, va, None)), ("E1_k60", lambda tr, va: b(tr, va, choices["rec_kappa"]))], "baseline": "E0", "selected": "E1_k60", "kind": "count", "slice_fn": K.player_slices, "min_rows": H.TH["min_player"]})
    q, b = eff_parts("pass_yards", comp, lambda e: e["qb_id"], rows, "player_id", "y_completions", lambda r: r["POS_QB"] == 1 and r["y_completions"] >= 1, pm)
    specs.append({"name": "pass_yards_per_completion_compound", "rows": q, "y_fn": lambda v: [r["y_pass_yards"] for r in v], "variants": [("E0", lambda tr, va: b(tr, va, None)), ("E1_k150", lambda tr, va: b(tr, va, 150.0))], "baseline": "E0", "selected": "E1_k150", "kind": "count", "slice_fn": K.player_slices, "min_rows": H.TH["min_player"]})
    def pts(kind):
        def fp(tr, va):
            ta, vt = TE.table_arrays(tr, TE.TEAM_FEATURES_V2), TE.table_arrays(va, TE.TEAM_FEATURES_V2); y = np.array([r["y_points"] for r in tr], float)
            m = TE.B0Count("T_PTS_B0").fit(ta, y) if kind == "b0" else TE.C1Count(TE.TEAM_FEATURES_V2).fit(ta, y)
            return C.nb2_pmf_matrix(m.mean(vt), m.alpha(), 130)[0]
        return fp
    specs.append({"name": "team_points_direct", "rows": team, "y_fn": lambda v: [r["y_points"] for r in v], "variants": [("S0_B0_blend", pts("b0")), ("S1_C1_state_glm", pts("c1"))], "baseline": "S0_B0_blend", "selected": "S1_C1_state_glm", "kind": "count", "slice_fn": team_slices, "min_rows": H.TH["min_team"]})
    return specs


import cfb_phase1_forecast as _FC
FC_CTX = _FC.CTX
