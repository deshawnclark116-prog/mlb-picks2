"""
NFL_PHASE1D_P1A  (Phase 1D, shadow research)  -- Phase 1A as deterministic, serializable artifacts + a live loader

The Phase 1A research pipeline (team environment -> availability -> role state -> propensity -> opportunity) fits and predicts in one pass over a
frozen train/validation split. For live use each learned object has to be (a) refit by a deterministic algorithm from data available at the forecast
cutoff and (b) written to bytes, hash-verified and loaded again without refitting. This module does both WITHOUT changing the architecture:

  * the frozen selection (which team-environment candidate, availability model, role-state feature families, Dirichlet concentrations, Kalman / HMM /
    change-point / EWMA hyper-parameters) is read from `phase1a_frozen_selection.json`, extracted once from the accepted Phase 1A depth-universe run;
  * `fit_artifacts(units, windows)` refits ONLY fitted quantities (GLM / XGBoost / logistic coefficients, NB dispersions, status lookup tables, group priors,
    HMM estimated on the training window, outside-bucket weights, share noise sd) on the training window;
  * `predict_units(artifacts, units)` applies serialized artifacts to as-of units (the loaded-model path is the only path forecasts use);
  * `Artifacts.save/load` write each learned object to its own file with a sha256 manifest.

Windows: RESEARCH = the Phase 1A split (train 2023 + 2024 wk1-12, valid 2024 wk13-18). Walk-forward = every completed week before the target week from
2023 on, the last 25% of those weeks being the early-stopping / validation window (same algorithm every week).
"""
import contextlib
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

import nfl_phase1_availability as A
import nfl_phase1_common as C
import nfl_phase1_opportunity as O
import nfl_phase1_role_state as R
import nfl_phase1_team_environment as TE

REPO = Path(__file__).resolve().parent
FROZEN = REPO / "nfl_models" / "nfl_player_outcome_phase1d" / "phase1a_frozen_selection.json"
SIM_TYPES = ("carry", "target", "qb_att", "rz_carry", "rz_target", "def_snap")
ARTIFACT_VERSION = "phase1a-artifacts-1"


# ------------------------------------------------------------------ frozen selection
def extract_frozen_selection(stage_pkl, pack_pkl, out=FROZEN):
    """One-time extraction of the accepted Phase 1A selection (development-selected; now frozen)."""
    import pickle
    st = pickle.load(open(stage_pkl, "rb")); pk = pickle.load(open(pack_pkl, "rb"))
    import nfl_phase1_evaluate as E
    fam = E.families_kept(st["role_rep"])
    sel = {
        "note": "Phase 1A selection extracted from the accepted depth-universe run; development-selected on burned data and FROZEN. Walk-forward refits may only "
                "update fitted coefficients / empirical estimates, never anything in this file.",
        "team_environment": {k: v["selected"] for k, v in st["fitted"].items()},
        "availability": {f"{side}_{tag}": st["av_rep"][side][tag]["selected"] for side in ("offense", "defense") for tag in ("T24", "T90")},
        "role_families_kept": {t: fam[t] for t in SIM_TYPES},
        "role_hyper": {t: {k: st["role_rep"]["types"][t]["params"][k] for k in ("ewma_half_life", "kalman_q", "kalman_r", "bocpd_hazard")} for t in SIM_TYPES},
        "dirichlet_alpha": {n: pk["meta"][n]["alpha"] for n in SIM_TYPES},
        "compositional": {n: pk["meta"][n]["compositional"] for n in SIM_TYPES},
        "opportunity_selected_config": {n: pk["meta"][n]["selected_config"] for n in SIM_TYPES},
        "xgboost_params": C.XGB,
        "universe": "depth-chart-extended candidate universe (Phase 1C adoption; development-selected, frozen)",
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(sel, indent=1, sort_keys=True))
    return sel


def load_frozen(path=FROZEN):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ windows
class Windows:
    def __init__(self, train, valid, label, detail=None):
        self.train, self.valid, self.label, self.detail = train, valid, label, detail or {}


RESEARCH = Windows(lambda s, w: s == 2023 or (s == 2024 and w <= 12), lambda s, w: s == 2024 and w > 12,
                   "research: train 2023 + 2024 wk1-12, validation 2024 wk13-18")


def walk_forward_windows(unit_keys_completed, target_sw):
    """Deterministic walk-forward split: every completed (season, week) strictly before the target week, from 2023 on; the last 25% of those
    weeks (by count) validate."""
    weeks = sorted({(s, w) for (s, w, _) in unit_keys_completed if s >= 2023 and (s, w) < tuple(target_sw)})
    if len(weeks) < 8:
        raise ValueError(f"walk-forward refit needs >= 8 completed weeks from 2023 on before {target_sw}; have {len(weeks)}")
    n_val = max(1, int(round(len(weeks) * 0.25)))
    tr, va = set(weeks[:-n_val]), set(weeks[-n_val:])
    return Windows(lambda s, w: (s, w) in tr, lambda s, w: (s, w) in va, f"walk-forward before {tuple(target_sw)}",
                   {"train_weeks": [list(x) for x in sorted(tr)], "valid_weeks": [list(x) for x in sorted(va)], "last_training_week": list(weeks[-1])})


@contextlib.contextmanager
def use_windows(win, audit_cutoff_week=None):
    """Point the shared C.TRAIN / C.VALID at `win` for the duration of a fit. The research audit (nothing after the last burned week) is replaced by an
    audit that no fit row lies at or after the target week."""
    old = (C.TRAIN, C.VALID, C.audit_fit)

    def audit(label, tr, va=()):
        if audit_cutoff_week is not None:
            for r in list(tr) + list(va):
                if (r["s"], r["w"]) >= tuple(audit_cutoff_week):
                    raise AssertionError(f"fit '{label}' contains row {(r['s'], r['w'])} at/after the target week {audit_cutoff_week}")
    C.TRAIN, C.VALID, C.audit_fit = win.train, win.valid, audit
    try:
        yield
    finally:
        C.TRAIN, C.VALID, C.audit_fit = old


# ------------------------------------------------------------------ xgboost helpers
def xgb_raw(b):
    return bytes(b.save_raw("ubj"))


def xgb_load(raw):
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(bytearray(raw))
    return b


def xgb_predict(b, rounds, X, cols):
    import xgboost as xgb
    return b.predict(xgb.DMatrix(X, feature_names=cols), iteration_range=(0, int(rounds)))


def jnp(a):
    return np.asarray(a, float).tolist()


# ------------------------------------------------------------------ artifacts
class Artifacts:
    """Everything Phase 1A learned. `blobs` = {filename: bytes} (what is written to disk / hashed); `obj` = the parsed structures used for prediction."""

    def __init__(self, obj, blobs, meta):
        self.obj, self.blobs, self.meta = obj, blobs, meta

    def manifest(self):
        files = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(self.blobs.items())}
        m = {"artifact_version": ARTIFACT_VERSION, "files": files, "meta": self.meta}
        m["bundle_sha256"] = hashlib.sha256(json.dumps({"files": files, "meta": self.meta}, sort_keys=True, default=str).encode()).hexdigest()
        return m

    def save(self, d):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        for k, v in self.blobs.items():
            (d / k).write_bytes(v)
        man = self.manifest()
        (d / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True, default=str))
        return man

    @staticmethod
    def load(d):
        d = Path(d)
        man = json.loads((d / "manifest.json").read_text())
        blobs = {}
        for k, h in man["files"].items():
            b = (d / k).read_bytes()
            if hashlib.sha256(b).hexdigest() != h:
                raise RuntimeError(f"artifact {k} failed hash verification")
            blobs[k] = b
        a = Artifacts(_parse(blobs), blobs, man["meta"])
        if a.manifest()["bundle_sha256"] != man["bundle_sha256"]:
            raise RuntimeError("artifact bundle hash mismatch")
        return a

    def roundtrip(self):
        return Artifacts(_parse(self.blobs), self.blobs, self.meta)


def _parse(blobs):
    """blobs -> prediction objects (boosters are deserialized here; the ONLY route to a model is through bytes)."""
    o = {"te": json.loads(blobs["te.json"]), "av": json.loads(blobs["availability.json"]), "role": json.loads(blobs["role.json"]),
         "prop": json.loads(blobs["propensity.json"]), "opp": json.loads(blobs["opportunity.json"])}
    o["xgb"] = {k: xgb_load(v) for k, v in blobs.items() if k.endswith(".ubj")}
    return o


# ------------------------------------------------------------------ team environment
def fit_te(units, sel):
    qb = TE.qb_out_flags(units)
    rows = TE.team_rows(units, qb)
    lg = {k: rows[0][f"lg_{k}"] for k in TE.TARGETS}
    tr = [r for r in rows if C.TRAIN(r["s"], r["w"])]; va = [r for r in rows if C.VALID(r["s"], r["w"])]
    out, blobs = {"lg": lg, "targets": {}}, {}
    for k in TE.TARGETS:
        ok = lambda rr: [r for r in rr if r[f"b0_{k}"] is not None]
        tr_, va_ = ok(tr), ok(va)
        y = lambda rr: np.array([r[f"y_{k}"] for r in rr])
        b0 = lambda rr: np.array([r[f"b0_{k}"] for r in rr])
        name = sel[k]
        f = {"selected": name}
        if name == "B0_blend":
            pred = b0
        elif name == "C1_opp_adjust":
            adj = lambda rr: np.array([C.safe(r[f"opp_{k}"], r[f"lg_{k}"]) - r[f"lg_{k}"] for r in rr])
            a = float(np.dot(adj(tr_), y(tr_) - b0(tr_)) / max(np.dot(adj(tr_), adj(tr_)), 1e-9))
            f["a"] = a
            pred = lambda rr: b0(rr) + a * adj(rr)
        elif name == "C2_poisson_glm":
            beta = TE.fit_glm(TE._feat_glm(tr_, k), y(tr_))
            f["beta"] = jnp(beta)
            pred = lambda rr: np.exp(TE._feat_glm(rr, k) @ beta)
        else:
            cols = TE.XGB_COLS(k)
            C.audit_fit(f"team_{k}_xgb", tr_, va_)
            bst = C.fit_xgb(C.matrix(tr_, cols), y(tr_), C.matrix(va_, cols), y(va_), cols, "count:poisson")
            f["rounds"] = int(bst.best_iteration + 1)
            blobs[f"te_{k}.ubj"] = xgb_raw(bst)
            pred = lambda rr: C.xgb_pred(bst, C.matrix(rr, cols), cols)
        f["k"] = TE.nb_k(np.concatenate([pred(tr_), pred(va_)]), np.concatenate([y(tr_), y(va_)]))
        C.audit_fit(f"team_{k}_glm", tr_)
        out["targets"][k] = f
    return out, blobs


def predict_te(art, units):
    o = art.obj["te"]
    qb = TE.qb_out_flags(units)
    rows = TE.team_rows(units, qb, lg=o["lg"])
    pred = {}
    for k in TE.TARGETS:
        f = o["targets"][k]
        ok = [r for r in rows if r[f"b0_{k}"] is not None]
        if f["selected"] == "B0_blend":
            mu = np.array([r[f"b0_{k}"] for r in ok])
        elif f["selected"] == "C1_opp_adjust":
            mu = np.array([r[f"b0_{k}"] + f["a"] * (C.safe(r[f"opp_{k}"], r[f"lg_{k}"]) - r[f"lg_{k}"]) for r in ok])
        elif f["selected"] == "C2_poisson_glm":
            mu = np.exp(TE._feat_glm(ok, k) @ np.array(f["beta"]))
        else:
            cols = TE.XGB_COLS(k)
            mu = xgb_predict(art.obj["xgb"][f"te_{k}.ubj"], f["rounds"], C.matrix(ok, cols), cols)
        for r, m in zip(ok, mu):
            pred[(r["s"], r["w"], r["team"], k)] = (max(float(m), 0.1), f["k"])
        for r in rows:
            pred.setdefault((r["s"], r["w"], r["team"], k), (r[f"lg_{k}"], f["k"]))
    return pred


# ------------------------------------------------------------------ availability
def fit_availability(off, dfn, sel):
    out, blobs = {}, {}
    for side, rows, cols24 in (("offense", off, A.OFF_T24), ("defense", dfn, A.DEF_T24)):
        tr = [r for r in rows if C.TRAIN(r["s"], r["w"])]; va = [r for r in rows if C.VALID(r["s"], r["w"])]
        y = lambda rr: np.array([r["y"] for r in rr], float)
        for tag, cols, t90 in (("T24", cols24, False), ("T90", cols24 + A.T90_EXTRA, True)):
            C.audit_fit(f"availability_{side}_{tag}", tr, va)
            tab, base, _ = A.status_table(tr + va, y(tr + va), t90=t90)
            name = sel[f"{side}_{tag}"]
            m = {"selected": name, "cols": cols, "tab": {k: float(v) for k, v in tab.items()}, "base": float(base), "t90": t90}
            if name == "C1_logistic":
                lg = A.fit_logit(C.matrix(tr + va, cols), y(tr + va))
                m["logit"] = {k: jnp(v) for k, v in lg.items()}
            elif name == "C2_xgb":
                b = C.fit_xgb(C.matrix(tr, cols), y(tr), C.matrix(va, cols), y(va), cols, "binary:logistic")
                m["rounds"] = int(b.best_iteration + 1)
                blobs[f"avail_{side}_{tag}.ubj"] = xgb_raw(b)
            out[f"{side}_{tag}"] = m
    return out, blobs


def predict_availability(art, off, dfn):
    o = art.obj["av"]
    for side, rows in (("offense", off), ("defense", dfn)):
        for tag in ("T24", "T90"):
            m = o[f"{side}_{tag}"]
            cols = m["cols"]
            if m["selected"] == "B0_status_lookup":
                key = (lambda r: r["_t90status"]) if m["t90"] else (lambda r: r["_status"])
                p = np.array([m["tab"].get(key(r), m["base"]) for r in rows])
            elif m["selected"] == "C1_logistic":
                lg = {k: np.array(v) for k, v in m["logit"].items()}
                p = A.pred_logit(lg, C.matrix(rows, cols))
            else:
                p = xgb_predict(art.obj["xgb"][f"avail_{side}_{tag}.ubj"], m["rounds"], C.matrix(rows, cols), cols)
            for r, x in zip(rows, p):
                r["_ref"]["p_active_" + tag] = float(x)
    return o


def lookup_p(art, ref):
    """Status-lookup P(active) used as `pact_lookup` (T-24h table)."""
    o = art.obj["av"]
    m = o["defense_T24"] if "pfr" in ref else o["offense_T24"]
    return m["tab"].get(A._status(ref.get("inj")), m["base"])


# ------------------------------------------------------------------ role state + propensity
def _apply_filters(rows, prm, prior, fb, hmm):
    for r in rows:
        r["prior"] = prior.get(r["grp"], fb)
        r["R0"] = R.r_last8(r["v"], r["prior"])[0]
        r["R1"] = R.r_ewma(r["v"], r["prior"], prm["ewma_half_life"])[0]
        r["R2"], r["R2v"] = R.r_kalman(r["v"], r["prior"], prm["kalman_q"], prm["kalman_r"])
        r["R3"], r["R3v"] = R.r_bocpd(r["v"][-12:], r["prior"], prm["bocpd_hazard"], prm["kalman_r"])
        r["R4"], r["R4v"] = hmm.predict(r["v"][-12:])


def _hmm_from(d):
    h = R.HMM.__new__(R.HMM)
    h.mu = np.array(d["mu"]); h.K = len(h.mu); h.T = np.array(d["T"]); h.sd = float(d["sd"]); h.pi = np.array(d["pi"])
    return h


def fit_role_propensity(units, frozen):
    role, prop, blobs, rows_by_type = {}, {}, {}, {}
    for t in SIM_TYPES:
        rows = R.type_rows(units, t)
        prm = frozen["role_hyper"][t]
        tr = [r for r in rows if r["y"] is not None and (C.TRAIN(r["s"], r["w"]) or C.VALID(r["s"], r["w"]))]
        pb = defaultdict(list)
        for r in tr:
            pb[r["grp"]].append(r["y"])
        prior = {g: float(np.mean(v)) for g, v in pb.items()}
        fb = float(np.mean([x["y"] for x in tr]))
        hmm = R.HMM([r["v"][-12:] for r in tr])
        _apply_filters(rows, prm, prior, fb, hmm)
        info = O.fit_propensity(rows, t, frozen["role_families_kept"][t])
        b = info.pop("_booster")
        blobs[f"prop_{t}.ubj"] = xgb_raw(b)
        role[t] = {"prior": prior, "fallback": fb, "hyper": prm, "hmm": {"mu": jnp(hmm.mu), "T": jnp(hmm.T), "sd": float(hmm.sd), "pi": jnp(hmm.pi)}}
        prop[t] = {"cols": info["cols"], "rounds": info["rounds"], "families": info["families"], "n_train": info["n_train"], "n_valid": info["n_valid"]}
        rows_by_type[t] = rows
    return role, prop, blobs, rows_by_type


def predict_role_propensity(art, units):
    rows_by_type = {}
    for t in SIM_TYPES:
        rows = R.type_rows(units, t)
        r_ = art.obj["role"][t]
        _apply_filters(rows, r_["hyper"], r_["prior"], r_["fallback"], _hmm_from(r_["hmm"]))
        p_ = art.obj["prop"][t]
        for r in rows:
            r["LF"] = R.learned_features(r, t)
            O.add_pos(r)
        if rows:
            X = C.matrix([r["LF"] for r in rows], p_["cols"])
            p = np.clip(xgb_predict(art.obj["xgb"][f"prop_{t}.ubj"], p_["rounds"], X, p_["cols"]), 0.0, 1.0)
            for r, x in zip(rows, p):
                r["P1"] = float(x)
        rows_by_type[t] = rows
    return rows_by_type


# ------------------------------------------------------------------ opportunity constants
def fit_opportunity(rows_by_type, units, frozen):
    opp = {}
    by_unit = {t: defaultdict(list) for t in SIM_TYPES}
    for t in SIM_TYPES:
        for r in rows_by_type[t]:
            by_unit[t][r["unit"]["key"]].append(r)
    for name in SIM_TYPES:
        rtype, tkey, akey, poss, comp = O.ALLOC[name]
        other_vals = []
        if comp:
            for u in units:
                k = u["key"]
                if not C.TRAIN(k[0], k[1]):
                    continue
                rs = by_unit[rtype].get(k)
                if not rs:
                    continue
                act = np.array([r["ref"]["actual"][akey] if r["ref"]["actual"][akey] is not None else np.nan for r in rs], float)
                T_act = u["team_actual"][tkey]
                if T_act:
                    other_vals.append(max(0.0, 1.0 - np.nansum(act) / T_act))
        res = [r["y"] - r["P1"] for u in units if C.VALID(u["key"][0], u["key"][1]) for r in by_unit[rtype].get(u["key"], []) if r["y"] is not None]
        opp[name] = {"alpha": frozen["dirichlet_alpha"][name], "other": float(np.mean(other_vals)) if (comp and other_vals) else (0.03 if comp else 0.0),
                     "share_sd": float(np.std(res)) if res else 0.12, "compositional": bool(comp)}
    return opp


def prior_usage_of(ref):
    """As-of usage summary the protocol's eligibility rules read (mean of the last 3 game rows). Logged with every forecast so eligibility is decidable
    from the forecast record alone, never from the outcome."""
    h = ref["hist"][-3:]
    mean = lambda k: float(np.mean([g[k] for g in h])) if h else 0.0
    if "pfr" in ref:       # defender
        return {"pos": ref.get("grp"), "career_rows": len(ref["hist"]), "def_snap_share_l3": mean("pct")}
    return {"pos": ref.get("pos"), "career_rows": int(ref.get("n_hist", len(ref["hist"]))), "carries_l3": mean("car"), "targets_l3": mean("tgt"), "attempts_l3": mean("att")}


# ------------------------------------------------------------------ fit / predict entry points
def fit_artifacts(units, windows, frozen=None, code=None, extra_meta=None, audit_week=None):
    """Refit every learned Phase 1A object on `windows` from `units` (as-of units built from data available at the cutoff)."""
    frozen = frozen or load_frozen()
    with use_windows(windows, audit_week):
        off, dfn = A.build_rows(units)
        te, te_blobs = fit_te(units, frozen["team_environment"])
        av, av_blobs = fit_availability(off, dfn, frozen["availability"])
        art0 = Artifacts({"av": av, "xgb": {k: xgb_load(v) for k, v in av_blobs.items()}}, {}, {})
        predict_availability(art0, off, dfn)                # in-sample availability predictions feed the role features exactly as in the research run
        role, prop, prop_blobs, rows_by_type = fit_role_propensity(units, frozen)
        opp = fit_opportunity(rows_by_type, units, frozen)
    blobs = {"te.json": json.dumps(te, sort_keys=True).encode(), "availability.json": json.dumps(av, sort_keys=True).encode(),
             "role.json": json.dumps(role, sort_keys=True).encode(), "propensity.json": json.dumps(prop, sort_keys=True).encode(),
             "opportunity.json": json.dumps(opp, sort_keys=True).encode(), "frozen_selection.json": json.dumps(frozen, sort_keys=True).encode(),
             **te_blobs, **av_blobs, **prop_blobs}
    tr_rows = sum(1 for u in units if C.TRAIN(u["key"][0], u["key"][1]))
    meta = {"windows": windows.label, "window_detail": windows.detail, "train_team_games": tr_rows, "code": code or {}, **(extra_meta or {})}
    return Artifacts(_parse(blobs), blobs, meta)


def predict_units(art, units, want=None):
    """Apply serialized artifacts to as-of `units` (only those in `want` = set of (s, w, team) when given). Returns the Phase 1A pack structure the
    Phase 1C simulator consumes: {"games": {(s,w,team): {...}}, "meta": {...}, "extras": {...}}."""
    us = [u for u in units if want is None or u["key"] in want]
    off, dfn = A.build_rows(us)
    predict_availability(art, off, dfn)
    te_pred = predict_te(art, us)
    rows_by_type = predict_role_propensity(art, us)
    games, extras = {}, defaultdict(set)
    by_unit = {t: defaultdict(list) for t in SIM_TYPES}
    for t in SIM_TYPES:
        for r in rows_by_type[t]:
            by_unit[t][r["unit"]["key"]].append(r)
    opp = art.obj["opp"]
    for u in us:
        k = u["key"]
        g = {"key": k, "kick": u["kick"], "types": {}}
        for name in SIM_TYPES:
            rtype, tkey, akey, poss, comp = O.ALLOC[name]
            rs = by_unit[rtype].get(k)
            if not rs:
                continue
            mu, kk = te_pred[(k[0], k[1], k[2], tkey)]
            g["types"][name] = {"ids": [r["id"] for r in rs], "pos": [r["ref"].get("pos") or r["ref"].get("grp") for r in rs],
                                "P1": np.array([r["P1"] for r in rs], float), "P0": np.array([r["R0"] for r in rs], float),
                                "pact24": np.array([r["ref"]["p_active_T24"] for r in rs], float), "pact90": np.array([r["ref"]["p_active_T90"] for r in rs], float),
                                "pact_lookup": np.array([lookup_p(art, r["ref"]) for r in rs], float), "y": None, "T_act": None,
                                "mu": float(mu), "k": float(kk), "share_sd": opp[name]["share_sd"],
                                "prior_usage": [prior_usage_of(r["ref"]) for r in rs]}
            for gid in g["types"][name]["ids"]:
                extras[(k[0], k[1])].add((gid, k[2], name))
        games[k] = g
    meta = {name: {"alpha": opp[name]["alpha"], "other": opp[name]["other"], "k0": None, "compositional": opp[name]["compositional"],
                   "selected_config": art.obj["role"] and None, "share_sd": opp[name]["share_sd"]} for name in SIM_TYPES}
    frozen_cfg = json.loads(art.blobs["frozen_selection.json"])["opportunity_selected_config"]
    for name in SIM_TYPES:
        meta[name]["selected_config"] = frozen_cfg[name]
    return {"games": games, "meta": meta, "extras": {k: sorted(v) for k, v in extras.items()}}
