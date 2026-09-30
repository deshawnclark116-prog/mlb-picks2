"""
NFL_PHASE1_FORECAST  (Phase 1C, shadow research)  -- the shadow forecast runner

  1. determine the forecast cutoff (T24 = kickoff - 24h, T90 = kickoff - 90m)
  2. load ONLY immutable snapshots retrieved at or before the cutoff and re-verify their sha256
  3. bind the frozen-compatible model bundle (architecture, constants, calibration, code hashes -> model_version)
  4. simulate the game (nfl_phase1c_sim) and derive every player distribution from the simulated events
  5. write append-only, idempotent forecast records (nfl_phase1_store)

Failure policy (nothing is invented): a game with a missing / corrupt / late required snapshot, or without Phase 1A inputs, is NOT forecast; the run
log names the reason. Records already stored are never touched. Same inputs + same model bytes => identical records (verified duplicate no-op);
same forecast id with different bytes => HARD ERROR.

Phase 1A inputs (opportunity propensities, availability, team-total parameters) come from a pluggable loader. `BurnedWeekLoader` serves burned development
weeks from the Phase 1A stage outputs (models fit through 2024 only). `LiveLoader` (regenerating Phase 1A features + models for a future week from the
snapshots) is NOT implemented: it is a freeze blocker recorded in freeze_candidate_validation.md.
"""
import csv
import hashlib
import io
import json
import subprocess
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import nfl_phase1_snapshots as SN
import nfl_phase1_store as ST
import nfl_phase1c_metrics as MT
import nfl_phase1c_sim as SM

REPO = Path(__file__).resolve().parent
UTC = timezone.utc
PROTOCOL_VERSION = "1.1+phase1c-candidate"
SIM_VERSION = "phase1c-sim-1"
REQUIRED = {"T24": ("injuries",), "T90": ("injuries", "weekly_rosters")}
OUTCOMES = {  # outcome -> (opportunity stat, event probability threshold or None)
    "rush_yds": ("rush_att", None), "rush_td": ("rush_att", 1), "rec_yds": ("targets", None), "rec": ("targets", None), "rec_td": ("targets", 1),
    "pass_yds": ("pass_att", None), "pass_td": ("pass_att", 1), "int": ("pass_att", 1), "atd": ("rush_att", 1),
    "tackles": ("def_snaps", None), "sacks": ("def_snaps", 1), "def_int": ("def_snaps", 1)}
CODE_FILES = ["nfl_phase1c_sim.py", "nfl_phase1c_fit.py", "nfl_phase1c_script.py", "nfl_phase1c_metrics.py", "nfl_phase1_forecast.py", "nfl_phase1_store.py",
              "nfl_phase1_efficiency.py", "nfl_phase1_event_models.py", "nfl_phase1_rushing_efficiency.py", "nfl_phase1_receiving_efficiency.py",
              "nfl_phase1_passing_efficiency.py", "nfl_phase1_defense_events.py", "nfl_phase1b_data.py"]


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def file_hashes(files=CODE_FILES):
    return {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in files}


class Bundle:
    """Frozen-compatible model bundle. model_version = hash of (architecture config, constants, calibration, simulation settings, code hashes)."""

    def __init__(self, config, constants, calibration, n_draws, eff, defaults, idx, S_qb_adjust=True, extra=None):
        self.config, self.constants, self.calibration, self.n_draws = config, constants, calibration, n_draws
        self.eff, self.defaults, self.idx, self.qb_adjust = eff, defaults, idx, S_qb_adjust
        self.C = SM.Const(constants)
        self.code_hashes = file_hashes()
        spec = {"config": config, "constants": constants, "calibration": calibration, "n_draws": n_draws, "sim_version": SIM_VERSION, "qb_adjust": S_qb_adjust,
                "code": self.code_hashes, "extra": extra or {}}
        self.spec = spec
        self.model_version = "p1c-" + hashlib.sha256(json.dumps(spec, sort_keys=True, default=float).encode()).hexdigest()[:16]
        self.calibration_version = "cal-" + hashlib.sha256(json.dumps(calibration, sort_keys=True, default=float).encode()).hexdigest()[:12]


def forecast_id(model_version, game_id, player_id, outcome, horizon, cutoff_iso):
    return ST.make_id(model_version, game_id, player_id, outcome, horizon, cutoff_iso)


class BurnedWeekLoader:
    """Serves Phase 1A inputs for burned development weeks (see module docstring)."""

    def __init__(self, pack):
        self.pack = pack

    def game_inputs(self, s, w, team):
        return self.pack["games"].get((s, w, team))


class LiveLoader:
    def game_inputs(self, s, w, team):
        raise NotImplementedError("live Phase 1A feature/model regeneration from immutable snapshots is not implemented (freeze blocker)")


def _parse_csv(b):
    return list(csv.DictReader(io.StringIO(b.decode("utf-8"))))


def apply_snapshot_rules(g, horizon, inj_rows, roster_rows, s, w, team, log):
    """Deterministic use of the immutable snapshots: (a) T90 game-day inactive => P(active)=0; (b) a player the T24 report lists as Out cannot be 'active' with high probability."""
    g = {**g, "types": {k: dict(v) for k, v in g["types"].items()}}
    out_ids = {r["gsis_id"] for r in inj_rows if r.get("team") == team and r.get("report_status") in ("Out",) and int(r["week"]) == w and int(r["season"]) == s}
    ina = {r["gsis_id"] for r in roster_rows if r.get("team") == team and r.get("status") == "INA" and int(r["week"]) == w and int(r["season"]) == s} if roster_rows else set()
    key = "pact24" if horizon == "T24" else "pact90"
    for tname, t in g["types"].items():
        p = np.array(t[key], float).copy()
        for j, gid in enumerate(t["ids"]):
            if horizon == "T90" and gid in ina and p[j] > 0.0:
                log.append({"rule": "t90_game_day_inactive", "player": gid, "from": float(p[j])}); p[j] = 0.0
            elif gid in out_ids and p[j] > 0.5:
                log.append({"rule": "reported_out_contradiction", "player": gid, "from": float(p[j])}); p[j] = 0.02
        t[key] = p
    return g


def load_verified_snapshots(sources, kickoff, horizon, root):
    """Bytes + provenance for the required sources; raises SnapshotError on any missing / late / tampered snapshot."""
    return SN.forecast_inputs(list(sources), kickoff, horizon, root)


def calibrate_draws(name, S, cal):
    m = (cal or {}).get(name)
    if not m:
        return S
    if m["method"] == "pit":
        return MT.apply_pit_map(S, {"xs": np.array(m["xs"]), "G": np.array(m["G"])})
    if m["method"] == "scale":
        return MT.apply_scale_map(S, m)
    if m["method"] == "conformal":
        return MT.apply_conformal_map(S, m)
    return S


def build_records(bundle, D, ACT_NAMES, s, w, gsA, res, horizon, kickoff, cutoff, game_id, provenance, snapshot_log):
    """Records for every named player / outcome of one simulated game."""
    from collections import defaultdict
    per = SM.collect(res, gsA, s, w)
    teams = list(res)
    opp_of = {teams[0]: teams[1], teams[1]: teams[0]}
    cut_iso = SN.iso(cutoff)
    recs = []
    pact_by = {}
    role_by = {}
    for tm, g in gsA.items():
        if g is None:
            continue
        pk = "pact24" if horizon == "T24" else "pact90"
        for tname, t in g["types"].items():
            for j, gid in enumerate(t["ids"]):
                pact_by[(s, w, tm, gid)] = max(pact_by.get((s, w, tm, gid), 0.0), float(t[pk][j]))
                if tname in ("carry", "target", "qb_att"):
                    role_by[(s, w, tm, gid)] = max(role_by.get((s, w, tm, gid), 0.0), abs(float(t["P1"][j]) - float(t["P0"][j])))
    opp_map = {n: {k: per[n][1][i] for i, k in enumerate(per[n][0])} for n in ("rush_att", "targets", "pass_att", "def_snaps") if n in per}
    scale0 = {"rush_yds": 5.0, "rec_yds": 5.0, "pass_yds": 20.0, "rush_td": 0.1, "rec_td": 0.1, "pass_td": 0.3, "int": 0.2, "atd": 0.1, "rec": 1.0, "tackles": 1.5, "sacks": 0.1, "def_int": 0.05}
    for outcome, (opp_name, thr) in OUTCOMES.items():
        if outcome not in per:
            continue
        keys, S = per[outcome]
        S = calibrate_draws(outcome, S.astype(np.float64), bundle.calibration)
        sm = MT.summary_row(S)
        q19 = np.quantile(S, MT.GRID19, axis=1)
        p_ev = (S >= 1).mean(1) if thr else None
        opp_S = np.array([opp_map[opp_name][k].mean() if k in opp_map.get(opp_name, {}) else np.nan for k in keys])
        opp_sd = np.array([opp_map[opp_name][k].std() if k in opp_map.get(opp_name, {}) else np.nan for k in keys])
        pact = np.array([pact_by.get(k, np.nan) for k in keys])
        rshift = np.array([role_by.get(k, 0.0) for k in keys])
        U, reasons = MT.predictability(S, np.nan_to_num(pact, nan=1.0), np.nan_to_num(opp_sd / np.maximum(opp_S, 1e-6), nan=0.0), scale0[outcome], role_shift=rshift)
        for i, k in enumerate(keys):
            _, _, tm, gid = k
            rec = {"id": forecast_id(bundle.model_version, game_id, gid, outcome, horizon, cut_iso), "model_version": bundle.model_version, "protocol_version": PROTOCOL_VERSION,
                   "simulation": {"version": SIM_VERSION, "n_draws": bundle.n_draws, "seed": zlib.crc32(repr((bundle.model_version, game_id, horizon)).encode())},
                   "horizon": horizon, "cutoff": cut_iso, "kickoff": SN.iso(kickoff), "season": s, "week": w, "game_id": game_id, "player_id": gid,
                   "player_name": ACT_NAMES.get((s, w, gid), ""), "team": tm, "opponent": opp_of.get(tm, ""), "position": (D.roster.get((s, w, gid)) or D.players.get(gid) or {}).get("pos", ""),
                   "outcome": outcome, "p_active": float(pact[i]) if not np.isnan(pact[i]) else None,
                   "mean": float(sm["mean"][i]), "median": float(sm["median"][i]), "sd": float(sm["sd"][i]),
                   "quantiles": {f"p{int(round(p * 100)):02d}": float(sm[f"p{int(round(p * 100)):02d}"][i]) for p in MT.QS},
                   "quantile_grid": [float(x) for x in q19[:, i]], "event_probability_ge1": None if p_ev is None else float(p_ev[i]),
                   "expected_opportunities": None if np.isnan(opp_S[i]) else float(opp_S[i]),
                   "uncertainty": {"score": float(U[i]), "reasons": reasons[i]},
                   "input_snapshots": provenance, "snapshot_rule_overrides": [x for x in snapshot_log if x.get("player") == gid],
                   "component_versions": bundle.config, "calibration_version": bundle.calibration_version, "calibration_method": (bundle.calibration or {}).get(outcome, {}).get("method")}
            recs.append(rec)
    return recs


def run_game(bundle, loader, D, ACT_NAMES, store, s, w, teamA, teamB, horizon, snapshot_root, run_id, game_id=None):
    """Forecast one game; returns a run-log dict. Never raises for input problems (records the reason); HardError from the store propagates."""
    game_id = game_id or D.game[(s, w, teamA)]["game_id"]
    kickoff = D.game[(s, w, teamA)]["kick"]
    cutoff = SN.forecast_time(kickoff, horizon)
    log = {"game_id": game_id, "horizon": horizon, "teams": [teamA, teamB], "cutoff": SN.iso(cutoff)}
    try:
        srcs, prov = load_verified_snapshots(REQUIRED[horizon] + (("depth_charts",) if s >= 2025 else ()), kickoff, horizon, snapshot_root)
    except SN.SnapshotError as e:
        log.update(status="skipped", reason=f"snapshot: {e}")
        return log
    try:
        inj_rows = _parse_csv(srcs["injuries"]); ros_rows = _parse_csv(srcs["weekly_rosters"]) if "weekly_rosters" in srcs else []
    except Exception as e:  # unparsable snapshot bytes
        log.update(status="skipped", reason=f"snapshot unparsable: {e}")
        return log
    gs, rule_log = {}, []
    for tm in (teamA, teamB):
        try:
            g = loader.game_inputs(s, w, tm)
        except NotImplementedError as e:
            log.update(status="skipped", reason=str(e)); return log
        if g is None:
            log.update(status="skipped", reason=f"no Phase 1A inputs for {tm}"); return log
        gs[tm] = apply_snapshot_rules(g, horizon, inj_rows, ros_rows, s, w, tm, rule_log)
    pack_view = {"games": {(s, w, tm): gs[tm] for tm in gs}, "meta": loader.pack["meta"]}
    seed = zlib.crc32(repr((bundle.model_version, game_id, horizon)).encode())
    res, gsx = SM.run_game(pack_view, teamA, teamB, s, w, bundle.eff, bundle.idx, bundle.defaults, bundle.C, bundle.n_draws, seed, horizon, None, bundle.qb_adjust)
    defaults_used = {tm: r["off"].get("_ev_missing") for tm, r in res.items() if r.get("off") is not None}
    recs = build_records(bundle, D, ACT_NAMES, s, w, gsx, res, horizon, kickoff, cutoff, game_id, prov, rule_log)
    hdr = {"generated_at": datetime.now(UTC).isoformat(), "run_id": run_id, "code_sha": git_sha(), "horizon": horizon, "game_id": game_id}
    r = store.append_batch(f"{s}_wk{w:02d}_{horizon}_{game_id}", hdr, recs)
    log.update(status="ok", n_records=len(recs), snapshot_rules_applied=rule_log, efficiency_defaults_used=defaults_used, **r)
    return log


def read_forecasts(store):
    """All forecast records with their batch header fields (generated_at, run_id, code_sha) joined on."""
    out = []
    for f in store.batch_files():
        h, recs = store.read_batch(f)
        for r in recs:
            out.append({**r, "_generated_at": h.get("generated_at"), "_run_id": h.get("run_id"), "_code_sha": h.get("code_sha")})
    return out
