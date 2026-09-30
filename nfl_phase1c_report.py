"""
NFL_PHASE1C_REPORT  (Phase 1C)

Generates README.md tables and freeze_candidate_validation.md from the Phase 1C result JSON files (no hand-typed numbers in the tables).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
P = REPO / "nfl_models" / "nfl_player_outcome_phase1c"


def J(n):
    f = P / n
    return json.load(open(f)) if f.exists() else None


def fmt(x, d=4):
    return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x))


def table_joint(r):
    L = ["| stat | n | joint T24 CRPS | joint T90 CRPS | independent (same components) CRPS | joint - independent (CRPS gain, p not better) | last-8 empirical CRPS | joint T24 MAE(median) | bias(mean) | cov80 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, e in r["stats"].items():
        j = e["joint_T24"]["combined"]; ind = e["independent_assembly"]
        L.append(f"| {k} | {j['n']} | {fmt(j['crps'])} | {fmt(e['joint_T90']['combined']['crps'])} | {fmt(ind['scores_on_common_rows']['crps'])} | "
                 f"{ind['joint_vs_this']['crps_improvement']:+.4f} (p={ind['joint_vs_this']['p_not_better']}) | {fmt(e['last8_empirical']['combined']['crps'])} | {fmt(j['mae_median'], 3)} | {fmt(j['bias_mean'], 3)} | {fmt(j['cov80'], 3)} |")
    return "\n".join(L)


def table_state(r):
    L = ["| stat | S0 static CRPS | S1 dynamic CRPS | S1 - S0 gain (p not better) |", "|---|---|---|---|"]
    for k, e in r["stats"].items():
        L.append(f"| {k} | {fmt(e['S0']['combined']['crps'])} | {fmt(e['S1']['combined']['crps'])} | {e['S1_vs_S0']['crps_improvement']:+.4f} (p={e['S1_vs_S0']['p_not_better']}) |")
    return "\n".join(L)


def table_cal(r):
    L = ["| stat | uncal cov80(PIT) | cov50(PIT) | CRPS | best candidate | cov80 | cov50 | CRPS change % | passes |", "|---|---|---|---|---|---|---|---|---|"]
    for k, e in r["stats"].items():
        u = e["uncalibrated"]
        for cn, c in e["candidates"].items():
            L.append(f"| {k} | {fmt(u['cov80_pit'], 3)} | {fmt(u['cov50_pit'], 3)} | {fmt(u['crps'])} | {cn} | {fmt(c['cov80_pit'], 3)} | {fmt(c['cov50_pit'], 3)} | {c['crps_change_pct']:+.2f} | {c['passes_rule']} |")
    return "\n".join(L)
