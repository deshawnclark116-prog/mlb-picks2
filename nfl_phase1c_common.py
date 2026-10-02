"""
NFL_PHASE1C_COMMON  (Phase 1C, shadow research)

Shared loaders for Phase 1C steps. Caches (development speed-up only, never a source of truth) live under a scratch directory.
"""
import pickle
import time
from collections import defaultdict
from pathlib import Path

import nfl_context_v4 as v4
import nfl_phase1_data as P1
import nfl_phase1_defense_events as DE
import nfl_phase1b_data as B

SEASONS = [2022, 2023, 2024, 2025, 2026]


def tk(s, w):
    return s * 100 + w


def load_records(data_dir, cache=None, p1a_pack=None):
    """Phase 1B as-of records (offense + defence). Extras = Phase 1A candidates that may not have realized opportunities."""
    if cache and Path(cache).exists():
        return pickle.load(open(cache, "rb"))
    t0 = time.time()
    D = P1.Data(data_dir)
    T = B.PlayTallies(data_dir, SEASONS)
    pos = {(s, g): r["pos"] for (s, w, g), r in D.roster.items()}
    PD = v4.PlayData(data_dir, SEASONS, pos)
    inj = B.build_inj_index(D)
    extras, dex = defaultdict(list), defaultdict(list)
    for (s, w), lst in (p1a_pack or {"extras": {}})["extras"].items():
        for gid, team, nm in lst:
            (dex if nm == "def_snap" else extras)[(s, w)].append((gid, team))
    Rs = B.build_records(D, T, PD, inj, extras=extras)
    drows = DE.add_extras(D, DE.load_defense_rows(D, data_dir), dex)
    drecs = DE.build(D, T, PD, inj, drows)
    out = {"Rs": Rs, "drecs": drecs, "inj": dict(inj)}
    if cache:
        pickle.dump(out, open(cache, "wb"), protocol=4)
    print(f"records built ({time.time() - t0:.0f}s)", flush=True)
    return out
