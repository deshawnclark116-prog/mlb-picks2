"""
NFL_PHASE1E_V2 (Phase 1E, shadow) -- frozen v2 incumbent comparator + pre-kickoff append-only forward logger.

Sportsbook inputs (spread / total) are inputs of the v2 RECIPE and are read here only, from a separate raw-schedule snapshot (never from the Phase 1 snapshot, which has the
market columns removed). Nothing here feeds the Phase 1 engine.

Frozen comparator (`v2_frozen_comparator`, the Amendment G scientific comparator): the nfl_yardage_v2 feature set (v2.FEATURES) and v2.PARAMS, XGBoost reg:absoluteerror, trained on
rows 2023 + 2024 wk1-12 (v3.Replayer features, history from 2022), early-stopped on 2024 wk13-18; residual distribution = empirical residuals on 2024 wk13-18 (added to the point
forecast, floored at 0). The booster is truncated to its best iteration and saved with sha256; it is never refit after the freeze.
`v2_live_production`: the production artifacts nfl_models/nfl_yardage_v2_work/*.json (trained through 2025), point only, operational context -- never used by a gate.

  python nfl_phase1e_v2.py build-frozen --data-dir DIR        (one-off: trains + writes nfl_models/nfl_player_outcome_phase1e/v2_frozen/*, manifest)
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

import nfl_phase1_common as C
import nfl_phase1_forecast as FC
import nfl_phase1_store as ST
import nfl_yardage_v2 as v2
import nfl_yardage_v3 as v3

REPO = Path(__file__).resolve().parent
P1E = REPO / "nfl_models" / "nfl_player_outcome_phase1e"
FROZEN = P1E / "v2_frozen"
COMPARATOR_VERSION = "v2-frozen-comparator-1"
SEASONS = [2022, 2023, 2024, 2025, 2026]
MARKETS = {"rush_yds": "rushing_yards", "rec_yds": "receiving_yards"}              # Phase 1 outcome -> v2 market
GRID99 = [round(x, 2) for x in np.arange(0.01, 0.995, 0.01)]
CODE_FILES = ["nfl_yardage_v2.py", "nfl_yardage_v3.py", "nfl_phase1e_v2.py"]


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def code_hash():
    return sha_bytes(json.dumps({f: sha_file(REPO / f) for f in CODE_FILES}, sort_keys=True).encode())


# ------------------------------------------------------------------------------------------------ training (one-off)
def train_frozen(data_dir, out=FROZEN):
    import xgboost as xgb
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    d3 = v3.Data(data_dir, SEASONS)
    man = {}
    for oc, market in MARKETS.items():
        cfg = v3.MARKETS[market]
        rows = v3.Replayer(d3, market).replay()
        feat = v2.FEATURES
        X = np.array([[r[1].get(c) if r[1].get(c) is not None else np.nan for c in feat] for r in rows], dtype=np.float32)
        y = np.array([r[2] for r in rows], float)
        sw = np.array([(r[0]["s"], r[0]["w"]) for r in rows])
        tr = np.array([C.TRAIN(s, w) for s, w in sw]); va = np.array([C.VALID(s, w) for s, w in sw])
        C.audit_fit("v2_frozen_comparator", [{"s": s, "w": w} for s, w in sw[tr | va]])
        b = xgb.train(dict(v2.PARAMS), xgb.DMatrix(X[tr], label=y[tr], feature_names=feat), 2000, evals=[(xgb.DMatrix(X[va], label=y[va], feature_names=feat), "va")],
                      early_stopping_rounds=60, verbose_eval=False)
        best = b.best_iteration + 1
        b = b[:best]
        path = out / f"v2_frozen_{oc}.json"
        b.save_model(str(path))
        pred = b.predict(xgb.DMatrix(X, feature_names=feat))
        resid = (y - pred)[va]
        rp = out / f"v2_frozen_{oc}_residuals.json"
        rp.write_text(json.dumps({"calibration_period": "2024 wk13-18", "n": int(len(resid)), "residuals": [float(x) for x in resid]}))
        man[oc] = {"artifact_file": str(path.relative_to(REPO)), "artifact_sha256": sha_file(path), "residual_file": str(rp.relative_to(REPO)), "residual_sha256": sha_file(rp),
                   "n_train_rows": int(tr.sum()), "n_valid_rows": int(va.sum()), "best_iteration_plus_1": int(best), "market": market, "eligibility": {"positions": list(cfg["positions"]), "min_last3_vol": cfg["min_last3_vol"], "vol": cfg["vol"]},
                   "n_rows_total_replay": len(rows)}
    return man


# ------------------------------------------------------------------------------------------------ frozen model objects
class Frozen:
    def __init__(self, frozen_dir=FROZEN):
        import xgboost as xgb
        self.dir = Path(frozen_dir)
        self.boost, self.resid, self.hashes, self.prod, self.prod_hash = {}, {}, {}, {}, {}
        for oc, market in MARKETS.items():
            b = xgb.Booster(); b.load_model(str(self.dir / f"v2_frozen_{oc}.json"))
            self.boost[oc] = b
            rj = json.loads((self.dir / f"v2_frozen_{oc}_residuals.json").read_text())
            self.resid[oc] = np.array(rj["residuals"], float)
            self.hashes[oc] = {"artifact_sha256": sha_file(self.dir / f"v2_frozen_{oc}.json"), "residual_sha256": sha_file(self.dir / f"v2_frozen_{oc}_residuals.json")}
            pp = v2.MODEL_DIR / f"nfl_{market}_v2.json"
            if pp.exists():
                pb = xgb.Booster(); pb.load_model(str(pp))
                self.prod[oc] = pb; self.prod_hash[oc] = sha_file(pp)

    def predict(self, oc, X):
        import xgboost as xgb
        return self.boost[oc].predict(xgb.DMatrix(X, feature_names=v2.FEATURES))

    def predict_prod(self, oc, X):
        import xgboost as xgb
        if oc not in self.prod:
            return None
        return self.prod[oc].predict(xgb.DMatrix(X, feature_names=v2.FEATURES))

    def quantiles(self, oc, point):
        q = np.quantile(self.resid[oc], GRID99)
        return np.maximum(point[:, None] + q[None, :], 0.0)


# ------------------------------------------------------------------------------------------------ features as of a directory
def candidate_rows(data_dir, season, week, game_filter=None, until=None):
    """{oc: [(meta, features dict)]}: every eligible player of the teams playing in (season, week) (optionally only `game_filter` game ids), with v2 features computed from the
    stats / schedule files in `data_dir` ONLY through completed weeks before (season, week). Returns (rows, d3). Participation is not used (v2 rows are conditional on it; scoring
    is conditional on play)."""
    d3 = v3.Data(data_dir, SEASONS)
    gmeta = {}
    import csv
    for r in csv.DictReader(open(Path(data_dir) / "games.csv", newline="", encoding="utf-8")):
        if r["game_type"] == "REG" and r["season"] == str(season) and r["week"] == str(week) and (game_filter is None or r["game_id"] in game_filter):
            gmeta[r["home_team"]] = (r["game_id"], r["away_team"], True)
            gmeta[r["away_team"]] = (r["game_id"], r["home_team"], False)
    out = {}
    for oc, market in MARKETS.items():
        rep = v3.Replayer(d3, market)
        rep.replay(until=(season, week))                      # absorbs every completed week strictly before the target week (and nothing else)
        rows = []
        for pid, ph in rep.hist.items():
            if not ph or ph[-1]["team"] not in gmeta:
                continue
            team = ph[-1]["team"]
            gid, opp, home = gmeta[team]
            ctx = d3.games.get((season, week, team))
            if ctx is None:
                continue
            ok = rep.eligible(pid)
            f = rep.player_features(pid, season, week, team, ctx, {}) if ok else None
            rows.append(({"player_id": pid, "name": ph[-1]["name"], "team": team, "opponent": opp, "game_id": gid, "position": ph[-1]["pos"], "eligible": ok,
                          "abstain_reason": None if ok else ("min_history_or_position_or_min_last3_volume")}, f))
        out[oc] = sorted(rows, key=lambda x: (x[0]["game_id"], x[0]["player_id"]))
    return out, d3


def v2_record_id(game_id, player_id, outcome, horizon, cutoff_iso, comparator_version=COMPARATOR_VERSION):
    return ST.make_id("v2", comparator_version, game_id, player_id, outcome, horizon, cutoff_iso)


def build_records(frozen, data_dir, season, week, game_ids, horizon, kickoffs, cutoffs, provenance):
    """Deterministic records (no wall-clock inside the record: the generation time goes in the batch header) for all candidate players of `game_ids`."""
    rows, d3 = candidate_rows(data_dir, season, week, set(game_ids))
    recs = []
    ch = code_hash()
    for oc in MARKETS:
        elig = [(m, f) for m, f in rows[oc] if m["eligible"]]
        if elig:
            X = np.array([[f.get(c) if f.get(c) is not None else np.nan for c in v2.FEATURES] for _, f in elig], dtype=np.float32)
            pt = frozen.predict(oc, X).astype(np.float64)
            pp = frozen.predict_prod(oc, X)
            Q = frozen.quantiles(oc, pt)
        for i, (m, f) in enumerate(elig):
            cut = cutoffs[m["game_id"]]; kick = kickoffs[m["game_id"]]
            recs.append({"id": v2_record_id(m["game_id"], m["player_id"], oc, horizon, cut), "kind": "v2_forward_forecast", "comparator_version": COMPARATOR_VERSION,
                         "season": season, "week": week, "game_id": m["game_id"], "player_id": m["player_id"], "player_name": m["name"], "team": m["team"], "opponent": m["opponent"],
                         "position": m["position"], "outcome": oc, "horizon": horizon, "cutoff": cut, "kickoff": kick, "as_of_retrieval_ts": provenance["retrieval_ts"],
                         "v2_frozen_comparator": {"point": float(pt[i]), "point_floored": float(max(pt[i], 0.0)), "quantile_grid_levels": GRID99, "quantiles": [float(x) for x in Q[i]],
                                                  "artifact_sha256": frozen.hashes[oc]["artifact_sha256"], "residual_distribution_sha256": frozen.hashes[oc]["residual_sha256"]},
                         "v2_live_production": ({"point": float(pp[i]), "artifact_sha256": frozen.prod_hash[oc], "label": "operational context only; not used by any gate"} if pp is not None else None),
                         "features": {c: (None if f.get(c) is None else float(f[c])) for c in v2.FEATURES},
                         "code_sha256": ch, "input_hashes": provenance["input_hashes"], "eligibility": "eligible", "abstain_reason": None})
        for m, f in rows[oc]:
            if not m["eligible"]:
                cut = cutoffs[m["game_id"]]; kick = kickoffs[m["game_id"]]
                recs.append({"id": v2_record_id(m["game_id"], m["player_id"], oc, horizon, cut), "kind": "v2_forward_abstention", "comparator_version": COMPARATOR_VERSION, "season": season, "week": week,
                             "game_id": m["game_id"], "player_id": m["player_id"], "player_name": m["name"], "team": m["team"], "opponent": m["opponent"], "position": m["position"],
                             "outcome": oc, "horizon": horizon, "cutoff": cut, "kickoff": kick, "as_of_retrieval_ts": provenance["retrieval_ts"], "v2_frozen_comparator": None, "v2_live_production": None,
                             "code_sha256": ch, "input_hashes": provenance["input_hashes"], "eligibility": "ineligible", "abstain_reason": m["abstain_reason"]})
    return recs


class LateV2(RuntimeError):
    pass


def log_group(root, frozen, data_dir, season, week, game_ids, horizon, kickoffs, cutoffs, provenance, now_fn, run_id="v2-live"):
    """Append the v2 records of one (kickoff group, horizon) to the append-only v2 store. Refuses at / after kickoff. Idempotent: same id + same bytes = verified duplicate; same id +
    different bytes = HardError (nfl_phase1_store)."""
    st = ST.Store(root, "v2forecasts")
    recs = build_records(frozen, data_dir, season, week, game_ids, horizon, kickoffs, cutoffs, provenance)
    out = {}
    for gid in sorted(set(game_ids)):
        from datetime import datetime
        if now_fn() >= _parse(kickoffs[gid]):
            raise LateV2(f"v2 forecast for {gid} generated after kickoff {kickoffs[gid]}")
        mine = [r for r in recs if r["game_id"] == gid]
        hdr = {"generated_at": now_fn().isoformat(), "run_id": run_id, "horizon": horizon, "game_id": gid, "comparator_version": COMPARATOR_VERSION, "code_sha256": code_hash()}
        out[gid] = {**st.append_batch(f"{season}_wk{week:02d}_{horizon}_{gid}", hdr, mine), "n_records": len(mine), "n_eligible": sum(r["eligibility"] == "eligible" for r in mine)}
    return out


def _parse(iso_s):
    from datetime import datetime, timezone
    return datetime.strptime(iso_s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


# ------------------------------------------------------------------------------------------------ deterministic Phase 1 <-> v2 matching
def match(phase1_records, v2_records):
    """Pairs on (game_id, player_id, outcome, horizon, cutoff). Returns (pairs, phase1_only, v2_only) all sorted by key; independent of input order. A pair is
    {'phase1_forecast': record, 'v2_frozen_comparator': ..., 'v2_live_production': ...}; v2 abstentions are reported separately with their reason."""
    k = lambda r: (r["game_id"], r["player_id"], r["outcome"], r["horizon"], r["cutoff"])
    p = {k(r): r for r in phase1_records if r["outcome"] in MARKETS}
    v = {k(r): r for r in v2_records}
    pairs, p_only, v_only = [], [], []
    for key in sorted(set(p) | set(v)):
        if key in p and key in v and v[key]["kind"] == "v2_forward_forecast":
            pairs.append({"key": key, "phase1_forecast": p[key], "v2_frozen_comparator": v[key]["v2_frozen_comparator"], "v2_live_production": v[key]["v2_live_production"], "v2_record_id": v[key]["id"]})
        elif key in p:
            p_only.append({"key": key, "reason": ("v2_abstained: " + v[key]["abstain_reason"]) if key in v else "player_absent_from_v2_candidate_set"})
        else:
            v_only.append({"key": key, "reason": "not_in_phase1_forecast_set"})
    return pairs, p_only, v_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-frozen"])
    ap.add_argument("--data-dir", default="/tmp/nflcsv")
    a = ap.parse_args()
    man = train_frozen(a.data_dir)
    print(json.dumps(man, indent=1))
    (FROZEN / "training_manifest.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
