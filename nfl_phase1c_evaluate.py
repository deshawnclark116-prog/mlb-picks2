"""
NFL_PHASE1C_EVALUATE  (Phase 1C, DEVELOPMENT = 2025 + 2026 wk1-3, burned; never a holdout)

Runs the joint game simulator over every development game and scores player-level distributions.
Helpers here are imported by the later study scripts (state, universe, calibration, convergence, curves, uncertainty).
"""
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import nfl_phase1_common as C
import nfl_phase1_data as P1
import nfl_phase1_opportunity as O
import nfl_phase1b_evaluate as EV
import nfl_phase1c_fit as FT
import nfl_phase1c_sim as SM

REPO = Path(__file__).resolve().parent
OUT = REPO / "nfl_models" / "nfl_player_outcome_phase1c"
STATS = ["rush_yds", "rush_td", "rec_yds", "rec", "rec_td", "pass_yds", "pass_td", "int", "tackles", "sacks", "def_int", "atd"]
ACT_KEY = {"rush_yds": "rush_yds", "rush_td": "rush_td", "rec_yds": "rec_yds", "rec": "rec", "rec_td": "rec_td", "pass_yds": "pass_yds", "pass_td": "pass_td",
           "int": "int", "tackles": "tackles", "sacks": "sacks", "def_int": "def_int", "rush_att": None, "targets": None}
SIM_NAME = {"rush_yds": "rush_yds", "rush_td": "rush_td", "rec_yds": "rec_yds", "rec": "rec", "rec_td": "rec_td", "pass_yds": "pass_yds", "pass_td": "pass_td",
            "int": "int", "tackles": "tackles", "sacks": "sacks", "def_int": "def_int", "atd": "atd"}


def load_config(adj_files):
    """Adjudicated component versions -> fit_all config."""
    res = {}
    for f in adj_files:
        res.update(json.load(open(f)))
    cfg = {}
    for name, e in res.items():
        lvl = e["adjudication"]["selected"]
        struct = "single"
        if name == "rush" and "structure_adjudication" in e:
            sel = e["structure_adjudication"]["selected"]
            struct = "single" if not sel.startswith("two") else sel.split("_")[0].replace("two", "two")
            struct = "single" if not sel.startswith("two") else "two" + sel[3:].split("_")[0]
        cfg[name] = {"level": lvl, "structure": struct}
    return cfg


def game_pairs(pack, D):
    seen, out = set(), []
    for (s, w, tm) in sorted(pack["games"]):
        opp = D.game[(s, w, tm)]["opp"]
        k = (s, w, tuple(sorted((tm, opp))))
        if k in seen:
            continue
        seen.add(k); out.append((s, w, tm, opp))
    return out


class Ctx:
    """Everything the study scripts need, built once."""

    def __init__(self, data_dir, scratch, config, fit_end=202418, variants=("adjudicated", "B0", "simple"), pack_file="p1a_inputs.pkl", records_file="records_full.pkl",
                 samples_file="p1a_samples.pkl"):
        S = Path(scratch)
        rec = pickle.load(open(S / records_file, "rb"))
        import nfl_phase1c_adjudicate as AJ
        self.Rs, self.drecs = AJ.mask_warmup(rec["Rs"], rec["drecs"])
        self.pack = pickle.load(open(S / pack_file, "rb"))
        self.C = SM.Const(json.load(open(OUT / "constants.json")))
        self.idx = {"rush": {k: i for i, k in enumerate(self.Rs["rush"].key)}, "rec": {k: i for i, k in enumerate(self.Rs["rec"].key)},
                    "pass": {k: i for i, k in enumerate(self.Rs["pass"].key)}, "def": {(r["s"], r["w"], r["gid"]): i for i, r in enumerate(self.drecs)}}
        self.D = P1.Data(data_dir)
        self.ACT = EV.load_actuals(data_dir)
        self.config = config
        self.eff, self.defaults = {}, {}
        for v in variants:
            self.eff[v] = FT.fit_all(self.Rs, self.drecs, config, v, fit_end)
            d = SM.defaults_from(self.eff[v], self.Rs)
            d["def_rate"] = {t: float(np.median(self.eff[v]["def_rate"][t][self.eff[v]["def_rate"][t] > 0])) for t in self.eff[v]["def_rate"]}
            self.defaults[v] = d
        self.pairs = game_pairs(self.pack, self.D)
        self.p1a_samples = pickle.load(open(S / samples_file, "rb")) if samples_file else None

    def run_joint(self, variant, N, seed, horizon, script_fn=None, qb_adjust=True, pairs=None, want_def=True):
        acc = defaultdict(lambda: ([], []))
        logs = []
        for (s, w, A, B) in (pairs or self.pairs):
            script = script_fn(s, w, A, B, N, seed) if script_fn else None
            res, gs = SM.run_game(self.pack, A, B, s, w, self.eff[variant], self.idx, self.defaults[variant], self.C, N, seed, horizon, script, qb_adjust, want_def)
            for name, (keys, S) in SM.collect(res, gs, s, w).items():
                acc[name][0].extend(keys); acc[name][1].append(S.astype(np.float32))
            for tm, r in res.items():
                if r.get("off") is not None:
                    logs.append({"key": (s, w, tm), **SM.summarize_accounting(r["off"], N)})
        return {k: (v[0], np.vstack(v[1])) for k, v in acc.items()}, logs

    def independent(self, variant):
        """Phase 1B-style independent assembly with the SAME component fits (200 draws from the Phase 1A samples)."""
        e = self.eff[variant]
        pack = EV.Pack(e["rush"], e["rush_td"], e["air"], e["catch"], e["yac"], e["rec_td"], e["air_p"], e["comp_p"], e["yac_p"], e["pass_td"], e["qb_int"], e["def_rate"])
        sim = EV.simulate(variant, pack, self.p1a_samples, self.idx, self.D)
        out = {}
        for name in STATS:
            if name in sim:
                out[name] = (sim[name][0], sim[name][1].astype(np.float32))
        return out

    def actual(self, name, keys):
        if name == "atd":
            return np.array([float((self.ACT.get((s, w, g), {}).get("rush_td", 0.0) + self.ACT.get((s, w, g), {}).get("rec_td", 0.0)) > 0) for (s, w, t, g) in keys])
        return np.array([self.ACT.get((s, w, g), {}).get(ACT_KEY[name], 0.0) for (s, w, t, g) in keys])


def crps(S, y):
    return O.crps_rows(np.asarray(S, np.float64), y)


def score_named(ctx, name, keys, S, y=None):
    y = ctx.actual(name, keys) if y is None else y
    Sx = (S >= 1).astype(np.float64) if name == "atd" else S.astype(np.float64)
    med = np.median(Sx, 1); mean = Sx.mean(1)
    q = lambda p: np.quantile(Sx, p, axis=1)
    return {"n": int(len(y)), "crps": float(crps(Sx, y).mean()), "mae_median": float(np.abs(med - y).mean()), "median_ae": float(np.median(np.abs(med - y))),
            "rmse_mean": float(np.sqrt(((mean - y) ** 2).mean())), "bias_mean": float((mean - y).mean()),
            "cov80": float(((y >= q(0.1)) & (y <= q(0.9))).mean()), "cov50": float(((y >= q(0.25)) & (y <= q(0.75))).mean()),
            "cov90": float(((y >= q(0.05)) & (y <= q(0.95))).mean()), "width80": float((q(0.9) - q(0.1)).mean())}


def paired(ctx, name, keys, Sa, Sb):
    """CRPS improvement of a over b (positive = a better) with week-block bootstrap."""
    y = ctx.actual(name, keys)
    A = (Sa >= 1).astype(np.float64) if name == "atd" else Sa.astype(np.float64)
    Bm = (Sb >= 1).astype(np.float64) if name == "atd" else Sb.astype(np.float64)
    blocks = np.array([f"{k[0]}-{k[1]}" for k in keys])
    imp, p = C.block_boot(crps(A, y), crps(Bm, y), blocks)
    return {"crps_improvement": imp, "p_not_better": p}


def align(keysA, SA, keysB, SB):
    ib = {k: i for i, k in enumerate(keysB)}
    ia = [i for i, k in enumerate(keysA) if k in ib]
    ok = [keysA[i] for i in ia]
    return ok, SA[ia], SB[[ib[k] for k in ok]]
