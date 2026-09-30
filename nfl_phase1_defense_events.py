"""
NFL_PHASE1_DEFENSE_EVENTS  (Phase 1B, shadow research)

Defensive player events per SNAP: solo+assist tackles, sacks, interceptions.
  rate_i = hierarchical gamma-Poisson (league -> position group -> player, decayed as-of snaps and events)
  tilt   = log-linear Poisson adjustment with nested matchup families (own defensive scheme, opponent offense, personnel)
  count | snaps ~ Poisson or negative binomial (dispersion tuned on VALIDATION)
Exposure (snaps) is Phase 1A's defense-snap forecast in the end-to-end assembly; here it is the realized snap count so that the
efficiency layer is scored on its own (opportunity error removed), exactly like the offensive components.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

import nfl_phase1_common as C
import nfl_phase1_data as P1
import nfl_phase1b_data as B

TARGETS = {"tackles": ("def_tackles_solo", "def_tackle_assists"), "sacks": ("def_sacks",), "interceptions": ("def_interceptions",)}
POSG = {"DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB", "CB": "CB", "DB": "CB",
        "S": "S", "SS": "S", "FS": "S"}
GAMMA = 0.985
KAPPA_GRID = (60.0, 150.0, 400.0, 1000.0, 3000.0, 1e12)      # in snaps


def load_defense_rows(D, data_dir):
    stats = {}
    for s in P1.SEASONS:
        p = Path(data_dir) / f"stats_player_week_{s}.csv"
        for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
            if r.get("season_type", "REG") != "REG":
                continue
            g = lambda k: P1.fnum(r.get(k)) or 0.0
            stats[(s, int(r["week"]), r["player_id"])] = {"pos": r["position"], "tackles": g("def_tackles_solo") + g("def_tackle_assists"),
                                                          "sacks": g("def_sacks"), "interceptions": g("def_interceptions")}
    rows = []
    for (s, w, team), dd in D.dsnap.items():
        for pfr, x in dd.items():
            gid = x["gsis"]
            if not gid:
                continue
            st = stats.get((s, w, gid), {})
            pos = st.get("pos") or (D.players.get(gid) or {}).get("pos") or x["grp"]
            pg = POSG.get(pos) or {"DL": "DL", "LB": "LB", "DB": "S"}.get(x["grp"], "S")
            game = D.game.get((s, w, team))
            if game is None:
                continue
            rows.append({"s": s, "w": w, "team": team, "opp": game["opp"], "gid": gid, "pg": pg, "snaps": x["snaps"], "home": float(game["home"]),
                         "tackles": st.get("tackles", 0.0), "sacks": st.get("sacks", 0.0), "interceptions": st.get("interceptions", 0.0)})
    rows.sort(key=lambda r: (r["s"], r["w"], r["team"], r["gid"]))
    return rows


def add_extras(D, rows, extras):
    """Forecast-only defenders (Phase 1A def_snap candidates that may not have played): zero-exposure rows flagged extra."""
    have = {(r["s"], r["w"], r["gid"]) for r in rows}
    out = list(rows)
    for (s, w), lst in extras.items():
        for gid, team in lst:
            if (s, w, gid) in have or (s, w, team) not in D.game:
                continue
            pos = (D.roster.get((s, w, gid)) or {}).get("pos") or (D.players.get(gid) or {}).get("pos") or "LB"
            game = D.game[(s, w, team)]
            out.append({"s": s, "w": w, "team": team, "opp": game["opp"], "gid": gid, "pg": POSG.get(pos, "LB"), "snaps": 0.0, "home": float(game["home"]),
                        "tackles": 0.0, "sacks": 0.0, "interceptions": 0.0, "extra": True})
    out.sort(key=lambda r: (r["s"], r["w"], r["team"], r["gid"]))
    return out


def build(D, T, PD, inj, rows):
    """As-of features (weeks strictly before) for every defensive player-game."""
    wi = lambda s, w: B.week_index(s, w)
    stores = {t: {lvl: B.Store(2) for lvl in ("p", "pos", "lg")} for t in TARGETS}     # dims: snaps, events
    tdef = B.Store(8); tpass = B.Store(8); trush = B.Store(8)
    by_week = defaultdict(list)
    for r in rows:
        by_week[(r["s"], r["w"])].append(r)
    keys_by_week = defaultdict(list)
    for k in T.pass_team:
        keys_by_week[(k[-3], k[-2])].append(("pass", k))
    for k in T.rush_team:
        keys_by_week[(k[-3], k[-2])].append(("rush", k))
    out = []
    cache = {}
    def prof(s, w, team):
        k = (s, w, team)
        if k not in cache:
            cache[k] = PD.def_profile(s, w, team)
        return cache[k]
    for (s, w) in sorted(by_week):
        i = wi(s, w)
        for r in by_week[(s, w)]:
            rec = dict(r)
            rec["base"] = {t: {lvl: stores[t][lvl].get({"p": r["gid"], "pos": r["pg"], "lg": "lg"}[lvl], i) for lvl in ("p", "pos", "lg")} for t in TARGETS}
            # own defense context (team), opponent offense (opp), own scheme, personnel (all as-of; injury = final-report proxy A2)
            lgp = tpass.get("lg", i)[1]
            def rate(x, lg, ni, di, k):
                return B.shrunk_rate(x[ni], x[di], lg[ni] / max(lg[di], 1.0), k)
            rec["fam"] = {
                "team": [r["home"], rate(tpass.get(("D", r["team"]), i)[1], lgp, 1, 0, 300.0), rate(trush.get(("D", r["team"]), i)[1], trush.get("lg", i)[1], 1, 0, 150.0)],
                "opp": [rate(tpass.get(("O", r["opp"]), i)[1], lgp, 1, 0, 300.0),                       # sacks allowed per dropback
                        rate(tpass.get(("O", r["opp"]), i)[1], lgp, 4, 2, 300.0),                       # INT per attempt
                        tpass.get(("O", r["opp"]), i)[1][0] / max(tpass.get(("O", r["opp"]), i)[1][0] + trush.get(("O", r["opp"]), i)[1][0], 1.0)],  # pass share
                "scheme": [prof(s, w, r["team"]).get(k) if prof(s, w, r["team"]).get(k) is not None else np.nan
                           for k in ("def_blitz_rate", "def_pressure_rate", "def_stacked_rate", "def_man_rate")],
                "personnel": [(inj.get((s, w, r["team"])) or {}).get("dl_out", 0) + 0.5 * (inj.get((s, w, r["team"])) or {}).get("dl_q", 0),
                              (inj.get((s, w, r["team"])) or {}).get("lb_out", 0) + 0.5 * (inj.get((s, w, r["team"])) or {}).get("lb_q", 0),
                              (inj.get((s, w, r["team"])) or {}).get("db_out", 0) + 0.5 * (inj.get((s, w, r["team"])) or {}).get("db_q", 0),
                              (inj.get((s, w, r["opp"])) or {}).get("ol_out", 0) + 0.5 * (inj.get((s, w, r["opp"])) or {}).get("ol_q", 0)]}
            out.append(rec)
        for r in by_week[(s, w)]:
            if r.get("extra"):
                continue
            for t, cols in TARGETS.items():
                v = np.array([r["snaps"], r[t]])
                for lvl, key in (("p", r["gid"]), ("pos", r["pg"]), ("lg", "lg")):
                    stores[t][lvl].add(key, v, i)
        for typ, k in keys_by_week.get((s, w), []):
            side = "O" if len(k) == 3 else "D"; team = k[-1]
            tgt = tpass if typ == "pass" else trush
            v = (T.pass_team if typ == "pass" else T.rush_team)[k]
            tgt.add((side, team), v, i)
            if side == "O":
                tgt.add("lg", v, i)
    return out


def _rate_base(rec_base, t, gi, kp, kq, lg_floor=1e-4):
    """Gamma-Poisson shrinkage in snaps units; returns per-snap rate."""
    lg = rec_base[t]["lg"][gi]; pos = rec_base[t]["pos"][gi]; pl = rec_base[t]["p"][gi]
    r_lg = max(lg[1] / max(lg[0], 1.0), lg_floor)
    r_pos = (pos[1] + kq * r_lg) / (pos[0] + kq)
    return (pl[1] + kp * r_pos) / (pl[0] + kp)


class PoissonTilt:
    def __init__(self, l2):
        self.l2 = l2

    def fit(self, y, expo_rate, X):
        self.mu = X.mean(0) if X.shape[1] else None; self.sd = X.std(0) + 1e-9 if X.shape[1] else None
        Z = (X - self.mu) / self.sd
        Z = np.column_stack([np.ones(len(Z)), Z])
        d = Z.shape[1]; pen = np.r_[0.0, np.ones(d - 1)]

        def f(b):
            eta = Z @ b; mu = expo_rate * np.exp(eta)
            return ((mu - y * eta).sum() + 0.5 * self.l2 * (pen * b ** 2).sum()) / len(y), (Z.T @ (mu - y) + self.l2 * pen * b) / len(y)
        self.b = minimize(f, np.zeros(d), jac=True, method="L-BFGS-B").x
        return self

    def predict(self, expo_rate, X):
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return expo_rate * np.exp(Z @ self.b)


def nb_logpmf(y, mu, r):
    if r is None:
        return y * np.log(np.maximum(mu, 1e-12)) - mu - gammaln(y + 1)
    return gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(np.maximum(mu, 1e-12) / (r + mu))


def run(recs):
    """recs: list from build()."""
    s_ = np.array([r["s"] for r in recs]); w_ = np.array([r["w"] for r in recs])
    tr = np.array([C.TRAIN(a, b) for a, b in zip(s_, w_)]); va = np.array([C.VALID(a, b) for a, b in zip(s_, w_)]); dv = np.array([C.DEV(a, b) for a, b in zip(s_, w_)])
    snaps = np.array([r["snaps"] for r in recs], float)
    act = np.array([not r.get("extra", False) for r in recs], bool)
    tr, va, dv = tr & act, va & act, dv & act
    C.audit_fit("phase1b_defense_events", [{"s": a, "w": b} for a, b in zip(s_[tr | va], w_[tr | va])])
    fam = {f: np.array([r["fam"][f] for r in recs], float) for f in ("team", "opp", "scheme", "personnel")}
    res = {}
    for t in TARGETS:
        y = np.array([r[t] for r in recs], float)
        best = None
        for gi in (0, 1, 2):
            for kp in KAPPA_GRID:
                for kq in (300.0, 1500.0):
                    lam = np.array([_rate_base(r["base"], t, gi, kp, kq) for r in recs]) * snaps
                    ll = nb_logpmf(y[va], lam[va], None).sum() / len(y[va])
                    if best is None or -ll < best[0]:
                        best = (-ll, gi, kp, kq)
        _, gi, kp, kq = best
        rate0 = np.array([_rate_base(r["base"], t, gi, kp, kq) for r in recs])
        lg_rate = np.array([_rate_base(r["base"], t, gi, 1e12, 1e12) for r in recs])
        pos_rate = np.array([_rate_base(r["base"], t, gi, 1e12, kq) for r in recs])
        naive = np.array([_rate_base(r["base"], t, gi, 60.0, kq) for r in recs])
        tuned = {"gamma_idx": gi, "kappa_player_snaps": kp, "kappa_pos_snaps": kq, "valid_negloglik": best[0]}
        # extra: dispersion by VALID
        def nll_vec(mu, r): return -nb_logpmf(y, mu, r)
        mu0 = rate0 * snaps
        rgrid = [None, 50.0, 20.0, 10.0, 5.0, 2.5] if t == "tackles" else [None, 20.0, 5.0, 2.0]
        rbest = min(rgrid, key=lambda r: (-nb_logpmf(y[va], mu0[va], r)).sum())
        ref_nl = nll_vec(lg_rate * snaps, rbest)
        def period(nl, ref=None):
            o = {}
            for tag, m in (("2025", dv & (s_ == 2025)), ("2026_wk1_3", dv & (s_ == 2026)), ("combined", dv)):
                if m.sum():
                    o[tag] = {"n_records": int(m.sum()), "neg_logpmf": float(nl[m].mean()), "mean_pred": None}
            if ref is not None:
                blocks = np.array([f"{a}-{b}" for a, b in zip(s_[dv], w_[dv])])
                imp, p = C.block_boot(nl[dv], ref[dv], blocks); o["vs_league_rate"] = {"improvement_per_record": imp, "p_not_better": p}
            return o
        lvl_rep, preds, vls, rates = {}, {}, {}, {}
        # nested families (own scheme -> team/opp/scheme/personnel)
        NEST = {"B0": [], "B1": ["team"], "B2": ["team", "opp"], "B3": ["team", "opp", "scheme"], "B4": ["team", "opp", "scheme", "personnel"], "B5": ["team", "opp", "scheme", "personnel", "inter"]}
        z = (np.log(np.maximum(rate0, 1e-6)) - np.log(np.maximum(rate0[tr], 1e-6)).mean())
        for lvl, fl in NEST.items():
            blocks_ = [fam[f] for f in fl if f != "inter"]
            if "inter" in fl:
                blocks_.append(np.column_stack([z * fam["opp"][:, 0], z * np.nan_to_num(fam["scheme"][:, 0], nan=np.nanmean(fam["scheme"][tr, 0]))]))
            if not blocks_:
                mu = rate0 * snaps; preds[lvl] = mu; rates[lvl] = rate0.copy(); vls[lvl] = float((-nb_logpmf(y[va], mu[va], rbest)).mean())
                lvl_rep[lvl] = {"families": fl, "development": period(nll_vec(mu, rbest), ref_nl)}; continue
            X = np.column_stack(blocks_); m_ = np.nanmean(X[tr | va], 0); ix = np.where(np.isnan(X)); X[ix] = m_[ix[1]]
            best_l = None
            for l2 in (100.0, 1000.0, 10000.0):
                m = PoissonTilt(l2).fit(y[tr], (rate0 * snaps)[tr], X[tr])
                v = float((-nb_logpmf(y[va], m.predict((rate0 * snaps)[va], X[va]), rbest)).mean())
                if best_l is None or v < best_l[0]:
                    best_l = (v, l2)
            m = PoissonTilt(best_l[1]).fit(y[tr | va], (rate0 * snaps)[tr | va], X[tr | va])
            mu = m.predict(rate0 * snaps, X); preds[lvl] = mu; rates[lvl] = m.predict(rate0, X); vls[lvl] = best_l[0]
            lvl_rep[lvl] = {"families": fl, "l2": best_l[1], "development": period(nll_vec(mu, rbest), ref_nl)}
        kept = "B0"
        for lvl in list(NEST)[1:]:
            if vls[kept] - vls[lvl] >= 1e-4:
                kept = lvl
        # tail calibration: P(y>=1) (sacks, interceptions) or P(y>=6) (tackles)
        thr = 6 if t == "tackles" else 1
        def tail_p(mu, r):
            k = np.arange(0, thr)
            pm = np.exp(np.stack([nb_logpmf(np.full(len(mu), kk, float), mu, r) for kk in k], 1)).sum(1)
            return 1 - pm
        tailc = {}
        for nm, mu in (("league_rate", lg_rate * snaps), ("hierarchical_B0", preds["B0"]), ("selected", preds[kept])):
            pt = tail_p(mu[dv], rbest)
            tailc[nm] = {"pred": round(float(pt.mean()), 5), "obs": round(float((y[dv] >= thr).mean()), 5)}
        res[t] = {"hier_tuning": tuned, "dispersion_r": rbest, "distribution": "poisson" if rbest is None else "negative_binomial",
                  "baselines": {"league_rate": period(ref_nl), "position_rate": period(nll_vec(pos_rate * snaps, rbest), ref_nl),
                                "naive_player_specific_kappa60": period(nll_vec(naive * snaps, rbest), ref_nl), "hierarchical_B0": lvl_rep["B0"]["development"]},
                  "nested": {"levels": lvl_rep, "selected_level": kept, "valid_negloglik_by_level": vls}, "tail_threshold": thr, "tail_calibration": tailc,
                  "mean_rate_per_snap_dev": float(y[dv].sum() / snaps[dv].sum())}
        res[t]["_rate"] = rates[kept]; res[t]["_rate_by_level"] = rates
    return res
