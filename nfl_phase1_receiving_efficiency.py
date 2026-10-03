"""
NFL_PHASE1_RECEIVING_EFFICIENCY  (Phase 1B, shadow research)

Receiving chain PER TARGET:  air yards ~ pmf ; catch | air bucket ~ hazard ; YAC | catch, air bucket ~ pmf ;
receiving yards = air + YAC on a catch, 0 on an incompletion (yac_def = receiving_yards - air_yards).
Air yards, catch and YAC each use the hierarchical league -> position -> player pmf/rate with a tilt over as-of matchup families.
"""
import numpy as np

import nfl_phase1_efficiency as F
import nfl_phase1_event_models as EM
import nfl_phase1b_data as B

AIR_V = np.array([(lo + hi) / 2.0 for lo, hi in B.AIR_BINS]); AIR_V[0] = -7.0
YAC_V = np.array([(lo + hi) / 2.0 for lo, hi in B.YAC_BINS]); YAC_V[0] = -3.0
AIR_BUCKET = np.array([B.air_bucket(v) for v in AIR_V])
PHI_AIR = np.column_stack([(AIR_V < 0).astype(float), (AIR_V >= 20).astype(float), np.clip(AIR_V, -10, 40) / 10.0])
PHI_YAC = np.column_stack([(YAC_V >= 15).astype(float), (YAC_V <= 0).astype(float), np.clip(YAC_V, -5, 40) / 10.0])
AIR_GROUPS = {"behind_los": np.where(AIR_V < 0)[0], "short_0-9": np.where((AIR_V >= 0) & (AIR_V < 10))[0],
              "intermediate_10-19": np.where((AIR_V >= 10) & (AIR_V < 20))[0], "deep_20+": np.where(AIR_V >= 20)[0]}
YAC_GROUPS = {"yac<=0": np.where(YAC_V <= 0)[0], "yac_1-5": np.where((YAC_V >= 1) & (YAC_V <= 5))[0],
              "yac_6-14": np.where((YAC_V >= 6) & (YAC_V < 15))[0], "yac_15+": np.where(YAC_V >= 15)[0]}
# (air bin a, yac bin y, receiving-yards group g): g = 1 (<10), 2 (10-19), 3 (20+)
_G = np.zeros((len(AIR_V), len(YAC_V), 4))
for _a, av in enumerate(AIR_V):
    for _y, yv in enumerate(YAC_V):
        t = av + yv
        _G[_a, _y, 1 if t < 10 else 2 if t < 20 else 3] = 1.0


def chain_group_probs(P_air, c_bucket, P_yac):
    """Per-target group probabilities [n,4] = (incompletion, <10, 10-19, 20+ receiving yards) from the three chain components.
    P_air [n,K_A]; c_bucket [n,4] catch probability by air bucket; P_yac [n,4,K_Y]."""
    n = len(P_air)
    out = np.zeros((n, 4))
    comp_total = np.zeros(n)
    for b in range(4):
        m = (AIR_BUCKET == b)
        A = P_air * m[None, :] * c_bucket[:, [b]]                      # P(air=a, catch)
        comp_total += A.sum(1)
        out[:, 1:] += np.einsum("na,ny,ayg->ng", A, P_yac[:, b, :], _G[:, :, 1:], optimize=True)
    out[:, 0] = 1.0 - comp_total
    return out


def _slice_base(R, key, b, K):
    return R.base[key][:, :, b * K:(b + 1) * K]


def run(R):
    tr, va, dv = F.masks(R)
    res = {}
    # ---------------- air yards
    res["air_yards"], aux_air = EM.run_pmf(R, R.cnt["air"], ("air_pl", "air_pp", "air_lg"), AIR_V, PHI_AIR, AIR_GROUPS, "rec_air",
                                           lambda P0: P0[:, AIR_V >= 20].sum(1), opp_i=1, scheme_i=(6,))
    # ---------------- catch | air bucket (stacked over buckets)
    S = EM.Stacked(R, 4)
    cnt2, bp = [], {"pl": [], "pp": [], "lg": []}
    for b in range(4):
        tgt, cat = R.cnt["cb"][:, 2 * b], R.cnt["cb"][:, 2 * b + 1]
        cnt2.append(np.column_stack([tgt - cat, cat]))
        for lvl in ("pl", "pp", "lg"):
            a = R.base[f"cb_{lvl}"]
            bp[lvl].append(np.stack([a[:, :, 2 * b] - a[:, :, 2 * b + 1], a[:, :, 2 * b + 1]], axis=2))
    cnt2 = np.concatenate(cnt2); bp = {k: np.concatenate(v) for k, v in bp.items()}
    res["catch"], aux_c = EM.run_binary(S, cnt2, bp, opp_i=0, scheme_i=(0, 7), name="rec_catch")
    # ---------------- YAC | catch, air bucket (stacked)
    Sy = EM.Stacked(R, 4)
    cy = np.concatenate([R.cnt["yac"][:, b * B.K_Y:(b + 1) * B.K_Y] for b in range(4)])
    Sy.base = {f"yac_{lvl}": np.concatenate([R.base[f"yac_{lvl}"][:, :, b * B.K_Y:(b + 1) * B.K_Y] for b in range(4)]) for lvl in ("pl", "pp", "lg")}
    res["yac"], aux_y = EM.run_pmf(Sy, cy, ("yac_pl", "yac_pp", "yac_lg"), YAC_V, PHI_YAC, YAC_GROUPS, "rec_yac",
                                   lambda P0: P0[:, YAC_V >= 15].sum(1), opp_i=2, scheme_i=(0,))
    # ---------------- composed per-target chain vs observed groups
    n = R.n
    def compose(air, cat, yac):
        cb = np.column_stack([cat[b * n:(b + 1) * n, 1] for b in range(4)])
        py = np.stack([yac[b * n:(b + 1) * n] for b in range(4)], axis=1)
        return chain_group_probs(air, cb, py)
    grp_cnt = R.cnt["grp"]
    lvls = {"B0_hierarchical": (aux_air["preds"]["B0"], aux_c["preds"]["B0"], aux_y["preds"]["B0"]),
            "selected": (aux_air["sel"], aux_c["sel"], aux_y["sel"]),
            "B5_all": (aux_air["preds"]["B5"], aux_c["preds"]["B5"], aux_y["preds"]["B5"])}
    chain = {}
    sc_ref = None
    for nm, (a_, c_, y_) in lvls.items():
        Pg = np.maximum(compose(a_, c_, y_), 1e-9); Pg /= Pg.sum(1, keepdims=True)
        sc = F.score_rows(Pg, grp_cnt, np.arange(4.0))
        if nm.startswith("B0"):
            sc_ref = sc
        chain[nm] = {"development": F.period_summary(sc, R, dv, None if nm.startswith("B0") else sc_ref),
                     "group_calibration": F.group_calibration(Pg, grp_cnt, {"incompletion": np.array([0]), "short_<10": np.array([1]),
                                                                            "intermediate_10-19": np.array([2]), "explosive_20+": np.array([3])}, dv)}
    res["composed_target_outcome"] = chain
    # league lookup baseline for the composed chain (as-of league pmfs / rates only)
    lg = (F.hier_base(R, "air_pl", "air_pp", "air_lg", 0, 1e12, 1e12), F.hier_base(S, "pl", "pp", "lg", 0, 1e12, 1e12),
          F.hier_base(Sy, "yac_pl", "yac_pp", "yac_lg", 0, 1e12, 1e12))
    Pl = np.maximum(compose(*lg), 1e-9); Pl /= Pl.sum(1, keepdims=True)
    sc_lg = F.score_rows(Pl, grp_cnt, np.arange(4.0))
    chain["league_lookup"] = {"development": F.period_summary(sc_lg, R, dv)}
    for nm in ("B0_hierarchical", "selected", "B5_all"):
        Pg = np.maximum(compose(*lvls[nm]), 1e-9); Pg /= Pg.sum(1, keepdims=True)
        chain[nm]["vs_league_lookup"] = F.period_summary(F.score_rows(Pg, grp_cnt, np.arange(4.0)), R, dv, sc_lg).get("vs_ref")
    return res, {"air": aux_air, "catch": aux_c, "yac": aux_y}
