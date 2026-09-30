"""
NFL_PHASE1C_INPUTS  (Phase 1C, shadow research)

Extract the Phase 1A per-team-game inputs the joint simulator consumes (propensities, availability at T24 and T90, team-total NB parameters,
Dirichlet alpha, outside-bucket weights). DEVELOPMENT weeks only (2025 + 2026 wk1-3): Phase 1A propensity models were fit through 2024, so these
rows are out-of-sample for them. The forward runner reads the same structure from live inputs (see nfl_phase1_forecast.py).
"""
import pickle
import sys
from pathlib import Path

import numpy as np

import nfl_phase1_common as C
import nfl_phase1_opportunity as O

TYPES = ("carry", "target", "qb_att", "rz_carry", "rz_target", "def_snap")


def extract(data_dir, stage_cache, out_pkl, depth_universe=False):
    import nfl_phase1_evaluate as E
    rep, sel, rej, out = E.run(data_dir, None, write=False, stage_cache=stage_cache, depth_universe=depth_universe)
    opp, frames_all = out["opp_rep"], out["frames_all"]
    games = {}
    meta = {}
    for name in TYPES:
        rt = opp["types"][name]
        akey = O.ALLOC[name][2]
        comp = rt["compositional"]
        other = O.other_weight(frames_all[name], akey) if comp else 0.0
        meta[name] = {"alpha": rt["dirichlet_alpha"], "other": other, "k0": rt["baseline_nb_k"], "compositional": comp,
                      "selected_config": rt["selected_config"], "share_sd": rt.get("share_noise_sd")}
        for f in frames_all[name]:
            if not C.DEV(f["s"], f["w"]):
                continue
            g = games.setdefault(f["key"], {"key": f["key"], "kick": f["kick"], "types": {}})
            rows = f["rows"]
            g["types"][name] = {
                "ids": [r["id"] for r in rows], "pos": [r["ref"].get("pos") or r["ref"].get("grp") for r in rows],
                "P1": np.array(f["P1"], float), "P0": np.array(f["P0"], float),
                "pact24": np.array([r["ref"]["p_active_T24"] for r in rows], float), "pact90": np.array([r["ref"]["p_active_T90"] for r in rows], float),
                "pact_lookup": np.array(f["pact_lookup"], float), "y": np.array(f["y"], float), "T_act": f["T_act"],
                "mu": float(f["mu_sel"]), "k": float(f["k_sel"]), "share_sd": f.get("share_sd")}
    extras = {}
    for k, g in games.items():
        for name, t in g["types"].items():
            for gid in t["ids"]:
                extras.setdefault((k[0], k[1]), set()).add((gid, k[2], name))
    extras = {k: sorted(v) for k, v in extras.items()}
    pack = {"games": games, "meta": meta, "extras": extras, "selected_config_note": "Phase 1A selected configuration; extraction does not recompute any allocation"}
    pickle.dump(pack, open(out_pkl, "wb"), protocol=4)
    return pack


if __name__ == "__main__":
    p = extract(sys.argv[1], sys.argv[2], sys.argv[3], depth_universe=len(sys.argv) > 4 and sys.argv[4] == "depth")
    print(len(p["games"]), {k: (v["alpha"], round(v["other"], 4)) for k, v in p["meta"].items()})
