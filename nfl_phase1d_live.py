"""
NFL_PHASE1D_LIVE  (Phase 1D, shadow research)  -- LiveLoader: Phase 1A/1B inputs regenerated ONLY from verified immutable snapshots

Given one snapshot set (materialized from the content-addressed store) and the target games, `prepare` rebuilds, with no burned-week stage-output shortcut:

  1. the as-of data (Data) and the chronological replay units, target games included as never-absorbed future units;
  2. Phase 1A artifacts: loaded from bytes (fixed research artifacts) or REFIT by the walk-forward algorithm from completed weeks before the target week;
  3. the Phase 1A opportunity inputs of every target team-game (`nfl_phase1d_p1a.predict_units`, serialized-model path only);
  4. Phase 1B as-of records for the target players (forecast-only candidates) and the Phase 1B efficiency arrays, refit on completed weeks
     (walk-forward, hyper-parameters frozen from the freeze fit);
  5. the forecast Bundle (model_version includes the weekly-fit identity, artifact bundle hash, code hashes).

Everything is deterministic given the snapshot bytes and the code.
"""
import hashlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import nfl_context_v4 as v4
import nfl_phase1_data as P1
import nfl_phase1_defense_events as DE
import nfl_phase1_forecast as FC
import nfl_phase1b_data as B
import nfl_phase1c_adjudicate as AJ
import nfl_phase1c_fit as FT
import nfl_phase1c_sim as SM
import nfl_phase1d_p1a as P

SEASONS = P1.SEASONS
REPO = Path(__file__).resolve().parent
P1B_HYPER = REPO / "nfl_models" / "nfl_player_outcome_phase1d" / "phase1b_frozen_hyper.json"


def week_fit_end(s, w):
    """Phase 1B expanding-window end key for a walk-forward refit: the last regular-season week before the target week."""
    return s * 100 + (w - 1) if w > 1 else (s - 1) * 100 + 18


def load_p1b_hyper(path=P1B_HYPER):
    return json.loads(Path(path).read_text())["hyper"]


class LiveLoader:
    """Serves Phase 1A inputs regenerated from snapshots (interface of BurnedWeekLoader.game_inputs)."""

    def __init__(self, pack):
        self.pack = pack

    def game_inputs(self, s, w, team):
        return self.pack["games"].get((s, w, team))


class Prepared:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class DView:
    """The few Data lookups the record builder needs (names / positions), restricted to the players of the target week."""

    def __init__(self, D, pack, s, w):
        ids = {gid for g in pack["games"].values() for t in g["types"].values() for gid in t["ids"]}
        self.roster = {(s, ww, gid): D.roster[(s, ww, gid)] for ww in range(max(1, w - 3), w + 1) for gid in ids if (s, ww, gid) in D.roster}
        self.players = {gid: D.players[gid] for gid in ids if gid in D.players}
        self.game = {k: v for k, v in D.game.items() if k[0] == s and k[1] == w}


def names_live(D, pack, s, w):
    out = {}
    for g in pack["games"].values():
        for t in g["types"].values():
            for gid in t["ids"]:
                for back in range(0, 4):
                    r = D.roster.get((s, w - back, gid)) if w - back >= 1 else None
                    if r:
                        out[(s, w, gid)] = r.get("name", "")
                        break
                else:
                    out[(s, w, gid)] = ""
    return out


def records_for(D, asof_dir, extras, dex):
    T = B.PlayTallies(asof_dir, SEASONS)
    pos = {(s, g): r["pos"] for (s, w, g), r in D.roster.items()}
    PD = v4.PlayData(asof_dir, SEASONS, pos)
    inj = B.build_inj_index(D)
    Rs = B.build_records(D, T, PD, inj, extras=extras)
    drows = DE.add_extras(D, DE.load_defense_rows(D, asof_dir), dex)
    drecs = DE.build(D, T, PD, inj, drows)
    return Rs, drecs, inj


def prepare(asof_dir, targets, art_provider, cfg, consts, calibration, n_draws, p1b_hyper, target_sw, cache_dir=None, cache_key=None, log=print):
    """targets: iterable of (season, week, team) keys of the games to forecast. art_provider(D, U, target_sw) -> Artifacts."""
    t0 = time.time()
    cache = Path(cache_dir) / f"prepared_{cache_key}.pkl" if cache_dir and cache_key else None
    if cache and cache.exists():
        log(f"    prepared cache hit {cache.name}")
        return pickle.load(open(cache, "rb"))
    targets = sorted(set(targets))
    D = P1.Data(asof_dir)
    U = P1.build(D, depth_universe=True, targets=targets)
    log(f"    data+units {time.time() - t0:.0f}s ({len(U)} units)")
    art = art_provider(D, U, target_sw)
    log(f"    artifacts {art.manifest()['bundle_sha256'][:12]} ({time.time() - t0:.0f}s)")
    pack = P.predict_units(art, U, want=set(targets))
    log(f"    phase1A inputs for {len(pack['games'])} team-games ({time.time() - t0:.0f}s)")
    extras, dex = defaultdict(list), defaultdict(list)
    for (s, w), lst in pack["extras"].items():
        for gid, team, nm in lst:
            (dex if nm == "def_snap" else extras)[(s, w)].append((gid, team))
    Rs, drecs, inj = records_for(D, asof_dir, extras, dex)
    Rs, drecs = AJ.mask_warmup(Rs, drecs)
    log(f"    phase1B records ({time.time() - t0:.0f}s)")
    fit_end = week_fit_end(*target_sw)
    eff = FT.fit_all(Rs, drecs, cfg, "adjudicated", fit_end, frozen=p1b_hyper)
    d = SM.defaults_from(eff, Rs)
    d["def_rate"] = {t: float(np.median(eff["def_rate"][t][eff["def_rate"][t] > 0])) for t in eff["def_rate"]}
    idx = {"rush": {k: i for i, k in enumerate(Rs["rush"].key)}, "rec": {k: i for i, k in enumerate(Rs["rec"].key)},
           "pass": {k: i for i, k in enumerate(Rs["pass"].key)}, "def": {(r["s"], r["w"], r["gid"]): i for i, r in enumerate(drecs)}}
    bundle = FC.Bundle(cfg, consts, calibration or {}, n_draws, eff, d, idx,
                       extra={"fit_end": fit_end, "variant": "adjudicated", "target_week": list(target_sw), "artifact_bundle_sha256": art.manifest()["bundle_sha256"],
                              "phase1b_hyper_sha256": hashlib.sha256(json.dumps(p1b_hyper, sort_keys=True, default=float).encode()).hexdigest()})
    log(f"    phase1B fit + bundle {bundle.model_version} ({time.time() - t0:.0f}s)")
    prep = Prepared(D=DView(D, pack, *target_sw), pack=pack, art_manifest=art.manifest(), bundle=bundle, names=names_live(D, pack, *target_sw), fit_end=fit_end, seconds=time.time() - t0,
                    training_summary={"phase1a_windows": art.meta.get("windows"), "phase1b_fit_end": fit_end})
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(prep, open(cache, "wb"), protocol=4)
    return prep
