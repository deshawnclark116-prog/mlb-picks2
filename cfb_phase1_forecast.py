"""
CFB_PHASE1_FORECAST -- the core engine: fits every selected component on a training window and builds per-team-game simulator specs from AS-OF features only (CFB Outcome Engine v1, RESEARCH / SHADOW ONLY).
`choices` records the development-selected variant of each component (frozen in cfb_phase1_freeze_candidate.json). Team volume, carries (O2 role families), QB attempt share, completion, receptions, per-event yardage pmfs, TD / INT hazards and the team-points residual
are all training-window fits; nothing reads the target game's labels, 2025 (before the freeze) or sportsbook data.
"""
import math
from collections import defaultdict

import numpy as np

import cfb_phase1_common as C
import cfb_phase1_efficiency as EF
import cfb_phase1_opportunity as OP
import cfb_phase1_team_environment as TE
import cfb_phase1b_components as K

INT_VALID_SEASONS = (2018, 2019, 2020, 2024)       # source defect: interception attribution collapses in 2021-2023 (rate 0.6-1.2% vs 2.2-2.7%); those seasons are excluded from INT fitting / scoring
DEFAULT_CHOICES = {"carry_activity": True, "qb_activity": False, "recv_kind": "logit", "recv_activity": False, "rush_kappa": 150.0, "rush_tilt": False, "rec_kappa": 60.0, "rec_tilt": False,
                   "haz_rush": "logit", "haz_rec": "logit", "haz_int": "base", "completion": "logit", "vol_n": "c1", "vol_pi": "logit", "resid_shift": 2}
CTX = {"rush_td": ["T_TD_B0", "O_TD_ALLOWED_B0", "T_RZ_RATE_B0", "RATING_GAP", "T_YPC_B0"], "rec_td": ["T_TD_B0", "O_TD_ALLOWED_B0", "T_RZ_RATE_B0", "RATING_GAP", "T_YPCOMP_B0"], "int": ["T_INT_RATE_B0", "O_INT_FORCED_B0", "RATING_GAP", "T_PASS_RATE_B0"],
       "completion": ["T_COMP_PCT_B0", "O_COMP_PCT_ALLOWED_B0", "RATING_GAP", "IS_HOME", "T_PASS_RATE_B0", "T_YPCOMP_B0"]}


def key_player(e):
    return e["player_id"]


def key_receiver(e):
    return e["receiver_id"]


def grp_of(pos):
    return "QB" if pos == "QB" else "RB" if pos in ("RB", "FB") else "WR" if pos == "WR" else "TE" if pos == "TE" else "OTH"


class CoreEngine:
    def __init__(self, choices=None):
        self.ch = {**DEFAULT_CHOICES, **(choices or {})}

    # ------------------------------------------------------------------ fit
    def fit(self, rows, team, events, pos_map, team_rows_by_key):
        """rows: candidate player rows of the TRAINING seasons (target_valid, with team labels attached); team: training team rows (TEAM_FEATURES_V2 + labels); events: {'R','P'} event lists up to the training end."""
        ch = self.ch
        self.train_max = max(r["season"] for r in rows) * 100 + 99
        self.vol = K.V2Volume(ch["vol_n"], ch["vol_pi"]).fit(team)
        self.o2 = K.O2Carries(activity=ch["carry_activity"]).fit(rows)
        qb_rows = [r for r in rows if r["POS_QB"] == 1 and r["n_att"] > 0]
        self.qb = OP.ShareAlloc("logit", "P_ATT_SHARE_L5", "P_ATT_SHARE_ACTIVE_L5" if ch["qb_activity"] else "P_ATT_SHARE_L5", K.ROLE_NAMES, "y_pass_att", "n_att", K.g_zero, 1.0, use_activity=ch["qb_activity"], activity_y=K.act_att).fit(qb_rows)
        rec_rows = [r for r in rows if r["n_rec"] > 0 and r["POS_QB"] != 1]
        self.rec = OP.ShareAlloc(ch["recv_kind"], "P_REC_SHARE_L5", "P_REC_SHARE_ACTIVE_L5" if ch["recv_activity"] else "P_REC_SHARE_L5", K.ROLE_NAMES, "y_receptions", "n_rec", K.pos_group, 1.0, use_activity=ch["recv_activity"], activity_y=K.act_rec).fit(rec_rows)
        qbs = [r for r in rows if r["POS_QB"] == 1]
        self.comp = K.RateComponent(ch["completion"], "P_COMP_L20", "P_ATT_L20", "y_completions", "y_pass_att", 80.0, K.g_zero, CTX["completion"] if ch["completion"] == "logit" else []).fit(qbs)
        self.h_rush = K.RateComponent(ch["haz_rush"], "P_RUSH_TD_L20", "P_CARRIES_L20", "y_rush_td", "y_carries", 100.0, K.pos_group, CTX["rush_td"] if ch["haz_rush"] == "logit" else []).fit(rows)
        self.h_rec = K.RateComponent(ch["haz_rec"], "P_REC_TD_L20", "P_REC_L20", "y_rec_td", "y_receptions", 60.0, K.pos_group, CTX["rec_td"] if ch["haz_rec"] == "logit" else []).fit(rows)
        self.h_int = K.RateComponent("base", "P_INT_L20", "P_ATT_L20", "y_int", "y_pass_att", 1e9, K.g_zero, []).fit([r for r in qbs if r["season"] in INT_VALID_SEASONS])   # league-constant rate (histories spanning 2021-2023 are corrupted)
        # league OTHER-bucket parameters from training team-games
        tg = defaultdict(lambda: {"prim": 0.0, "n_prim": 0.0, "qb": 0.0, "n_att": 0.0, "rec": 0.0, "n_rec": 0.0, "car": 0.0, "rtd": 0.0, "rectd": 0.0, "recn": 0.0})
        for r in rows:
            t = tg[(r["game_id"], r["team"])]; t["n_prim"] = r["n_prim"]; t["n_att"] = r["n_att"]; t["n_rec"] = r["n_rec"]
            if r["position"] in K.PRIMARY:
                t["prim"] += r["y_carries"]
            if r["POS_QB"] == 1:
                t["qb"] += r["y_pass_att"]
            if r["POS_QB"] != 1:
                t["rec"] += r["y_receptions"]
        a = np.array([[t["prim"], t["n_prim"], t["qb"], t["n_att"], t["rec"], t["n_rec"]] for t in tg.values()])
        self.s_other = {"prim": float(np.clip(1 - a[:, 0].sum() / max(a[:, 1].sum(), 1), 0.005, 0.5)), "qb": float(np.clip(1 - a[:, 2].sum() / max(a[:, 3].sum(), 1), 0.005, 0.5)), "rec": float(np.clip(1 - a[:, 4].sum() / max(a[:, 5].sum(), 1), 0.005, 0.5))}
        self.p_other = {"rush_td": self.h_rush.pall, "rec_td": self.h_rec.pall, "comp": self.comp.pall}
        # per-event yardage pmfs (training-window pooled pmfs; as-of player counts at prediction time)
        grp = lambda k: grp_of(pos_map.get(k, "?"))
        self.idx_r = EF.EventIndex(events["R"], key_player)
        comp_ev = [e for e in events["P"] if e["kind"] == "C" and e["receiver_id"]]
        self.idx_c = EF.EventIndex(comp_ev, key_receiver)
        self.pmf_r, self.league_r = EF.pooled_pmf(self.idx_r, None, grp, self.train_max)
        self.pmf_c, self.league_c = EF.pooled_pmf(self.idx_c, None, grp, self.train_max)
        zs_r = np.array([r["O_YPC_ALLOWED_B0"] for r in team]); zs_c = np.array([r["O_YPCOMP_ALLOWED_B0"] for r in team])
        self.zr, self.zc = (float(zs_r.mean()), float(zs_r.std())), (float(zs_c.mean()), float(zs_c.std()))
        self.theta_r = self.theta_c = 0.0
        if ch["rush_tilt"] or ch["rec_tilt"]:
            self._fit_tilts(events, comp_ev, team_rows_by_key, grp)
        # team-points residual: points - 7 * offensive TDs (+ shift)
        tr = [r for r in team if np.isfinite(r["y_td"])]
        y = np.array([max(r["y_points"] - 7 * r["y_td"] + ch["resid_shift"], 0) for r in tr], float)
        tab = TE.table_arrays(tr, TE.TEAM_FEATURES_V2)
        self.resid = TE.C1Count(TE.TEAM_FEATURES_V2).fit(tab, y)
        self.points_model = TE.C1Count(TE.TEAM_FEATURES_V2).fit(TE.table_arrays(team, TE.TEAM_FEATURES_V2), np.array([r["y_points"] for r in team], float))   # validated direct team-points component (S1)
        self.pm = pos_map
        self._train_team = team
        return self

    def _fit_tilts(self, events, comp_ev, team_rows_by_key, grp):
        import random
        rng = random.Random(1)
        def tilt_for(evs, key_fn, pos_pmf, league, zcol, mu_sd):
            evs = [e for e in evs if C.week_index(e["season"], e["week"]) <= self.train_max and (e["game_id"], e["team"]) in team_rows_by_key]
            if len(evs) > 60000:
                evs = rng.sample(evs, 60000)
            bp = np.stack([pos_pmf.get(grp(key_fn(e)), league) for e in evs]); z = np.array([(team_rows_by_key[(e["game_id"], e["team"])][zcol] - mu_sd[0]) / mu_sd[1] for e in evs]); yi = np.array([EF.clip_idx(e["yards"]) for e in evs])
            return EF.fit_tilt(bp, z, yi)
        if self.ch["rush_tilt"]:
            self.theta_r = tilt_for(events["R"], lambda e: e["player_id"], self.pmf_r, self.league_r, "O_YPC_ALLOWED_B0", self.zr)
        if self.ch["rec_tilt"]:
            self.theta_c = tilt_for(comp_ev, lambda e: e["receiver_id"], self.pmf_c, self.league_c, "O_YPCOMP_ALLOWED_B0", self.zc)

    # ------------------------------------------------------------------ yard pmfs (as-of)
    def rush_pmfs(self, rows):
        base = np.stack([self.pmf_r.get(grp_of(r["position"]), self.league_r) for r in rows])
        widx = [C.week_index(r["season"], r["week"]) for r in rows]
        p = base if self.ch["rush_kappa"] is None else EF.shrink_rows(self.idx_r, [r["player_id"] for r in rows], widx, base, self.ch["rush_kappa"])[0]
        if self.ch["rush_tilt"]:
            p = EF.tilt_rows(p, np.array([(r["O_YPC_ALLOWED_B0"] - self.zr[0]) / self.zr[1] if np.isfinite(r["O_YPC_ALLOWED_B0"]) else 0.0 for r in rows]), self.theta_r)
        return p

    def rec_pmfs(self, rows):
        base = np.stack([self.pmf_c.get(grp_of(r["position"]), self.league_c) for r in rows])
        widx = [C.week_index(r["season"], r["week"]) for r in rows]
        p = base if self.ch["rec_kappa"] is None else EF.shrink_rows(self.idx_c, [r["player_id"] for r in rows], widx, base, self.ch["rec_kappa"])[0]
        if self.ch["rec_tilt"]:
            p = EF.tilt_rows(p, np.array([(r["O_YPCOMP_ALLOWED_B0"] - self.zc[0]) / self.zc[1] if np.isfinite(r["O_YPCOMP_ALLOWED_B0"]) else 0.0 for r in rows]), self.theta_c)
        return p

    # ------------------------------------------------------------------ spec for one team-game
    def team_spec(self, team_row, cand):
        """team_row: team-game feature row; cand: candidate rows of that team-game (index = position in `cand`)."""
        tab = TE.table_arrays([team_row], TE.TEAM_FEATURES_V2)
        v = self.vol
        pi = float(np.clip(v._pi([team_row], tab)[0], 1e-4, 1 - 1e-4))
        spec = {"vol": {"mu": float(v.nm.mean(tab)[0]), "alpha": v.nm.alpha(), "pi": pi, "kappa": v.kappa, "sack": v.s}}
        idx_prim = [i for i, r in enumerate(cand) if r["position"] in K.PRIMARY]; idx_gad = [i for i, r in enumerate(cand) if r["position"] not in K.PRIMARY]
        prim = [cand[i] for i in idx_prim]
        carry = {"rho": self.o2.rho, "kg": self.o2.kg["kappa"], "primary": [], "primary_other": {"kappa": float(self.o2.share.kappa.get(1, {"kappa": 5.0})["kappa"]), "s": self.s_other["prim"]}, "gadget": [], "gadget_other": 0.0}
        if prim:
            s = self.o2.share.share(prim); a = self.o2.share.activity(prim) if self.ch["carry_activity"] else [None] * len(prim)
            for j, i in enumerate(idx_prim):
                k = 0 if cand[i]["POS_QB"] == 1 else 1
                carry["primary"].append({"idx": i, "kappa": float(self.o2.share.kappa.get(k, {"kappa": 5.0})["kappa"]), "s": float(s[j]), "a": None if a[j] is None else float(a[j])})
        if idx_gad:
            gad = [cand[i] for i in idx_gad]
            alpha, W = self.o2._group_alpha(gad, self.o2.lam)
            carry["gadget"] = [{"idx": i, "alpha": float(alpha[j])} for j, i in enumerate(idx_gad)]
        carry["gadget_other"] = float(math.exp(np.clip(self.o2.lam[-1], -12, 8)))
        spec["carry"] = carry
        idx_qb = [i for i, r in enumerate(cand) if r["POS_QB"] == 1]; qbr = [cand[i] for i in idx_qb]
        spec["qb"] = []
        if qbr:
            s = self.qb.share(qbr); p = self.comp._p(qbr); ip = self.h_int._p(qbr)
            for j, i in enumerate(idx_qb):
                spec["qb"].append({"idx": i, "s": float(s[j]), "kappa": float(self.qb.kappa.get(0, {"kappa": 5.0})["kappa"]), "p": float(p[j]), "kc": self.comp.kappa, "int_p": float(ip[j])})
        spec["qb_other"] = {"s": self.s_other["qb"], "kappa": float(self.qb.kappa.get(0, {"kappa": 5.0})["kappa"]), "p": self.p_other["comp"], "kc": self.comp.kappa}
        idx_rc = [i for i, r in enumerate(cand) if r["POS_QB"] != 1]; rcr = [cand[i] for i in idx_rc]
        spec["recv"] = []
        if rcr:
            s = self.rec.share(rcr); act = self.rec.activity(rcr) if self.ch["recv_activity"] else [None] * len(rcr)
            for j, i in enumerate(idx_rc):
                g = K.pos_group(cand[i])
                spec["recv"].append({"idx": i, "s": float(s[j]), "kappa": float(self.rec.kappa.get(g, {"kappa": 5.0})["kappa"]), "a": None if act[j] is None else float(act[j])})
        spec["recv_other"] = {"s": self.s_other["rec"], "kappa": float(self.rec.kappa.get(2, {"kappa": 5.0})["kappa"])}
        carriers = [cand[x["idx"]] for x in carry["primary"]] + [cand[x["idx"]] for x in carry["gadget"]]
        carr_idx = [x["idx"] for x in carry["primary"]] + [x["idx"] for x in carry["gadget"]]
        rp = self.rush_pmfs(carriers) if carriers else []
        cp = self.rec_pmfs(rcr) if rcr else []
        rush_td_p = self.h_rush._p(carriers) if carriers else []
        rec_td_p = self.h_rec._p(rcr) if rcr else []
        spec["pmf"] = {"rush": {i: rp[j] for j, i in enumerate(carr_idx)}, "rec": {i: cp[j] for j, i in enumerate(idx_rc)}, "rush_other": self.league_r, "rec_other": self.league_c}
        spec["td"] = {"rush": {i: float(rush_td_p[j]) for j, i in enumerate(carr_idx)}, "rec": {i: float(rec_td_p[j]) for j, i in enumerate(idx_rc)}, "kappa_rush": self.h_rush.kappa, "kappa_rec": self.h_rec.kappa, "rush_other_p": self.p_other["rush_td"], "rec_other_p": self.p_other["rec_td"], "kappa_int": self.h_int.kappa}
        spec["points_q"] = {"mu": float(self.points_model.mean(tab)[0]), "alpha": self.points_model.alpha()} if self.ch.get("reweight_points", True) else None
        spec["resid"] = {"mu": float(self.resid.mean(tab)[0]), "alpha": self.resid.alpha(), "shift": self.ch["resid_shift"]}
        return spec

    def artifact(self):
        return {"choices": self.ch, "volume": self.vol.artifact(), "carries": self.o2.artifact(), "qb": self.qb.artifact(), "receptions": self.rec.artifact(), "completion": self.comp.artifact(), "hazards": {"rush_td": self.h_rush.artifact(), "rec_td": self.h_rec.artifact(), "int": self.h_int.artifact()},
                "other_shares": self.s_other, "tilt": {"rush": self.theta_r, "rec": self.theta_c}, "residual": {"nb": self.resid.nb, "shift": self.ch["resid_shift"]}}
