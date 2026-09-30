"""
NFL_PHASE1_PASSING_EFFICIENCY  (Phase 1B, shadow research)

Passing chain per DROPBACK: sack hazard; per ATTEMPT: air yards pmf, completion | air bucket, YAC | completion, INT hazard,
pass-TD hazard. Same engine, nested families and shrinkage as receiving; QB-level counts are the player layer.
"""
import numpy as np

import nfl_phase1_efficiency as F
import nfl_phase1_event_models as EM
import nfl_phase1_receiving_efficiency as RC
import nfl_phase1b_data as B


def _binary(R, num_col, den_col, name, opp_i, scheme_i):
    S = EM.Stacked(R, 1)
    pt = R.cnt["pt"]
    cnt2 = np.column_stack([pt[:, den_col] - pt[:, num_col], pt[:, num_col]])
    bp = {}
    for lvl in ("pl", "pp", "lg"):
        a = R.base[f"pt_{lvl}"]
        bp[lvl] = np.stack([a[:, :, den_col] - a[:, :, num_col], a[:, :, num_col]], axis=2)
    return EM.run_binary(S, cnt2, bp, opp_i, scheme_i, name)


def run(R):
    tr, va, dv = F.masks(R)
    res, aux = {}, {}
    res["sack"], aux["sack"] = _binary(R, 1, 0, "pass_sack", 3, (7, 6))
    res["interception"], aux["interception"] = _binary(R, 4, 2, "pass_int", 4, (7,))
    res["pass_td"], aux["pass_td"] = _binary(R, 5, 2, "pass_td", 5, (6,))
    res["air_yards"], aux["air"] = EM.run_pmf(R, R.cnt["air"], ("air_pl", "air_pp", "air_lg"), RC.AIR_V, RC.PHI_AIR, RC.AIR_GROUPS, "pass_air",
                                              lambda P0: P0[:, RC.AIR_V >= 20].sum(1), opp_i=1, scheme_i=(6,))
    S = EM.Stacked(R, 4)
    cnt2, bp = [], {"pl": [], "pp": [], "lg": []}
    for b in range(4):
        tgt, cat = R.cnt["cb"][:, 2 * b], R.cnt["cb"][:, 2 * b + 1]
        cnt2.append(np.column_stack([tgt - cat, cat]))
        for lvl in ("pl", "pp", "lg"):
            a = R.base[f"cb_{lvl}"]
            bp[lvl].append(np.stack([a[:, :, 2 * b] - a[:, :, 2 * b + 1], a[:, :, 2 * b + 1]], axis=2))
    cnt2 = np.concatenate(cnt2); bp = {k: np.concatenate(v) for k, v in bp.items()}
    res["completion"], aux["completion"] = EM.run_binary(S, cnt2, bp, opp_i=0, scheme_i=(0, 7), name="pass_comp")
    Sy = EM.Stacked(R, 4)
    cy = np.concatenate([R.cnt["yac"][:, b * B.K_Y:(b + 1) * B.K_Y] for b in range(4)])
    Sy.base = {f"yac_{lvl}": np.concatenate([R.base[f"yac_{lvl}"][:, :, b * B.K_Y:(b + 1) * B.K_Y] for b in range(4)]) for lvl in ("pl", "pp", "lg")}
    res["yac"], aux["yac"] = EM.run_pmf(Sy, cy, ("yac_pl", "yac_pp", "yac_lg"), RC.YAC_V, RC.PHI_YAC, RC.YAC_GROUPS, "pass_yac",
                                        lambda P0: P0[:, RC.YAC_V >= 15].sum(1), opp_i=2, scheme_i=(0,))
    n = R.n
    def compose(air, cat, yac):
        cb = np.column_stack([cat[b * n:(b + 1) * n, 1] for b in range(4)])
        py = np.stack([yac[b * n:(b + 1) * n] for b in range(4)], axis=1)
        return RC.chain_group_probs(air, cb, py)
    grp = R.cnt["grp"]
    chain, sc_ref = {}, None
    lg = (F.hier_base(R, "air_pl", "air_pp", "air_lg", 0, 1e12, 1e12), F.hier_base(S, "pl", "pp", "lg", 0, 1e12, 1e12),
          F.hier_base(Sy, "yac_pl", "yac_pp", "yac_lg", 0, 1e12, 1e12))
    Pl = np.maximum(compose(*lg), 1e-9); Pl /= Pl.sum(1, keepdims=True)
    sc_lg = F.score_rows(Pl, grp, np.arange(4.0))
    chain["league_lookup"] = {"development": F.period_summary(sc_lg, R, dv)}
    for nm, a_, c_, y_ in (("B0_hierarchical", aux["air"]["preds"]["B0"], aux["completion"]["preds"]["B0"], aux["yac"]["preds"]["B0"]),
                           ("selected", aux["air"]["sel"], aux["completion"]["sel"], aux["yac"]["sel"])):
        Pg = np.maximum(compose(a_, c_, y_), 1e-9); Pg /= Pg.sum(1, keepdims=True)
        sc = F.score_rows(Pg, grp, np.arange(4.0))
        chain[nm] = {"development": F.period_summary(sc, R, dv, sc_lg),
                     "group_calibration": F.group_calibration(Pg, grp, {"incompletion": np.array([0]), "short_<10": np.array([1]),
                                                                        "intermediate_10-19": np.array([2]), "explosive_completion_20+": np.array([3])}, dv)}
    res["composed_attempt_outcome"] = chain
    return res, aux
