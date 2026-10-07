#!/usr/bin/env python3
"""Builds the NHL V2 Phase1A-SOG engine lock: fits the fixed B2 per horizon on targets 2018-2025 (T24H, T90, T30), freezes comparator constants and base rates,
and pins code / model / protocol hashes. Run once; the lock is then committed and pushed BEFORE any forward forecast exists."""
import json
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nhl_v2_phase1a_sog_compare as C
import nhl_v2_phase1a_sog_data as D
import nhl_v2_phase1a_sog_forward as F
import nhl_v2_phase1a_sog_model as M
from nhl_v2_phase1a_sog_quality import meaningful_mask
from nhl_v2_phase1a_sog_repro import fit_base_rates

OUT = D.OUT
MODELS = OUT / "phase1a_sog_models"
PROTOCOL_FILES = ("phase1a_sog_protocol.json", "phase1a_sog_protocol_amendment_1.json", "phase1a_sog_forward_protocol.json", "phase1a_sog_data_manifest.json", "phase1a_sog_burned_reproduction.json", "phase1a_v1_migration_audit.json")


def main():
    games, rows = D.load_frozen()
    MODELS.mkdir(parents=True, exist_ok=True)
    arts, tabs = {}, {}
    for h, minutes in D.HORIZONS.items():
        cache = Path("/tmp/nhl_v2_tab%d.pkl" % minutes)
        if h == "T90" and Path("/tmp/nhl_v2_tab90.pkl").exists():
            cache = Path("/tmp/nhl_v2_tab90.pkl")
        if cache.exists():
            tab, _ = pickle.load(open(cache, "rb"))
        else:
            tab, cov = D.build_rows(games, rows, horizon_min=minutes)
            pickle.dump((tab, cov), open(cache, "wb"), protocol=4)
        tabs[h] = tab
        art = M.fit_b2(tab, h)
        arts[h] = art
        p = MODELS / ("engine_%s.json" % h)
        p.write_text(json.dumps(art, indent=1, sort_keys=True) + "\n")
    tab90 = tabs["T90"]
    cf = pickle.load(open("/tmp/nhl_v2_cf90.pkl", "rb"))
    const = C.fit_constants(cf, tab90, rows)
    mm = meaningful_mask(tab90)
    base_rates = {"FULL": fit_base_rates(tab90, np.ones(len(tab90["sog"]), bool)), "MEANINGFUL": fit_base_rates(tab90, mm)}
    now = datetime.now(timezone.utc)
    lock = {"artifact": "phase1a_sog_engine_lock", "lock_id": "nhl-v2-sog-engine-lock-1", "engine_version": M.ENGINE_VERSION, "architecture": "B2_FIXED",
            "locked_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "eligible_from_cutoff_utc": (now + timedelta(minutes=45)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "eligibility_note": "only horizon cutoffs at/after eligible_from_cutoff_utc can produce valid forecasts, and only after this lock commit is pushed",
            "horizons": D.HORIZONS, "fit": {"targets": "2018-2025", "per_horizon": {h: {"n_train_rows": a["n_train_rows"], "nb2_alpha": a["nb2"]["alpha"], "poisson_alpha": a["poisson_alpha"], "converged": a["converged"], "model_sha256": a["sha256"]} for h, a in arts.items()}},
            "model_sha256": {("engine_%s.json" % h): F.sha_file(MODELS / ("engine_%s.json" % h)) for h in arts},
            "code_sha256": {f: F.sha_file(D.REPO / f) for f in F.LOCK_CODE_FILES}, "protocol_sha256": {f: F.sha_file(OUT / f) for f in PROTOCOL_FILES},
            "comparator_constants": const, "base_rates": base_rates,
            "meaningful_rule": "candidate AND PLAY_DEN_TG10>=5 AND PLAY_RATE_TG10>=0.6 AND TEAM_GAMES_SINCE_APPEARANCE<=1 AND TOI_MEAN_CT_APP3>=480 (prior information only)",
            "burned_reproduction_status": json.loads((OUT / "phase1a_sog_burned_reproduction.json").read_text())["reproduction_vs_v1"]["status"],
            "no_parameter_refit_during_forward_window": True, "availability_used_in_forecast": False, "no_betting_market_inputs": True, "no_simulation": True,
            "table_sha256": {h: D.table_hash(t) for h, t in tabs.items()}}
    F.LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    print("lock written", lock["locked_at_utc"], lock["eligible_from_cutoff_utc"], {h: (a["nb2"]["alpha"], a["converged"]) for h, a in arts.items()})


if __name__ == "__main__":
    main()
