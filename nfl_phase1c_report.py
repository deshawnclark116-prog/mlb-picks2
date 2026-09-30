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


def table_universe(u):
    L = ["| stat | common rows | CRPS accepted-universe pipeline | CRPS depth-universe pipeline | gain (p not better) | MAE accepted | MAE depth | extra rows | extra-row CRPS model vs zero forecast |", "|---|---|---|---|---|---|---|---|---|"]
    for k, e in u["stats"].items():
        x = e.get("extra_rows") or {}
        L.append(f"| {k} | {e['n_common']} | {fmt(e['common_accepted']['crps'])} | {fmt(e['common_depth']['crps'])} | {e['depth_vs_accepted_common']['crps_improvement']:+.4f} (p={e['depth_vs_accepted_common']['p_not_better']}) | "
                 f"{fmt(e['common_accepted']['mae_median'], 3)} | {fmt(e['common_depth']['mae_median'], 3)} | {e['n_extra_players']} | {fmt(x.get('crps_model'))} vs {fmt(x.get('crps_zero_forecast'))} |")
    return "\n".join(L)


def table_conv(c):
    L = ["| N | " + " | ".join(c["stats"]) + " |", "|---|" + "---|" * len(c["stats"])]
    for N in ("200", "1000", "5000", "10000", "25000", "50000", "100000"):
        cells = []
        for s, e in c["stats"].items():
            b = e["by_n"][N]
            cells.append(f"CRPS {b['crps_rel_bias_vs_100k_pct']:+.2f}%, SE(mean) {b['se_mean']:.3g}{'' if b['all_met_amended'] else ' x'}")
        L.append(f"| {N} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def table_curves(r, hz="T24", period="combined", stats=("rush_yds", "rec_yds", "pass_yds")):
    out = []
    for s in stats:
        cur = r["curves"][s][hz][period]
        tols = list(next(iter(cur.values()))["accuracy_curve_median"].keys())
        out += [f"**{s} ({hz}, {period})** - share of player-games with |median forecast - actual| <= tolerance", "", "| subset | n | MAE(med) | " + " | ".join(f"±{t}" for t in tols) + " |", "|---|---|---|" + "---|" * len(tols)]
        for sub, c in cur.items():
            out.append(f"| {sub} | {c['n']} | {c['mae_median']:.2f} | " + " | ".join(f"{c['accuracy_curve_median'][t]:.3f}" for t in tols) + " |")
        out.append("")
    return "\n".join(out)


def table_curves_counts(r, hz="T24", period="combined", stats=("rec", "tackles", "rush_td", "rec_td", "pass_td", "int", "sacks", "def_int")):
    out = []
    for s in stats:
        cur = r["curves"][s].get(hz, {}).get(period)
        if not cur:
            continue
        c = cur["all_eligible"]
        out.append(f"| {s} ({hz}) | {c['n']} | {c['mae_median']:.3f} | " + ", ".join(f"±{k}: {v:.3f}" for k, v in c["accuracy_curve_median"].items()) + " |")
    return "\n".join(["| outcome | n | MAE(median) | exact / within tolerance |", "|---|---|---|---|"] + out)


def table_unc(r):
    L = ["| outcome | Spearman(U, relative error) | decile monotonic violations | Spearman(pred sd, |error|) | violations |", "|---|---|---|---|---|"]
    for k, u in r["uncertainty"].items():
        L.append(f"| {k} | {u['spearman_U_vs_error']} | {u['monotonic_violations']} | {u['spread_skill']['spearman_sd_vs_abs_error']} | {u['spread_skill']['monotonic_violations']} |")
    return "\n".join(L)


def table_v2(v):
    L = ["| target | period | n (played, v2-eligible) | engine cond. median MAE | v2 refit MAE | blend MAE | production v2 MAE (2026 only) | engine vs v2 refit MAE (gain, p) | CRPS engine | CRPS v2 refit+2024 residuals | CRPS gain (p) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, per in v["stats"].items():
        for tag, c in per.items():
            prod = c.get("production_v2_artifact(2026 wk1-3 only)")
            L.append(f"| {name} | {tag} | {c['n']} | {c['engine_conditional_median']['mae']:.2f} | {c['v2_recipe_refit_through_2024']['mae']:.2f} | {c['historical_blend']['mae']:.2f} | "
                     f"{'n/a' if prod is None else format(prod['mae'], '.2f')} | {c['engine_vs_v2_refit_mae']['improvement']:+.3f} (p={c['engine_vs_v2_refit_mae']['p_not_better']}) | "
                     f"{c['crps_engine_conditional']:.3f} | {c['crps_v2_refit_plus_2024_residuals']:.3f} | {c['engine_vs_v2_refit_crps']['improvement']:+.3f} (p={c['engine_vs_v2_refit_crps']['p_not_better']}) |")
    return "\n".join(L)


def table_dep(d):
    L = ["| pair (across dev team-games) | actual | simulated |", "|---|---|---|"]
    for k, v in d["pairs"].items():
        L.append(f"| {k} | {v['actual']:+.3f} | {v['simulated']:+.3f} |")
    return "\n".join(L)


def table_t24t90(r):
    L = ["| outcome | T24 CRPS | T90 CRPS | T90 gain (p not better) |", "|---|---|---|---|"]
    for k, u in r["t24_vs_t90"].items():
        L.append(f"| {k} | {fmt(u['T24']['crps'])} | {fmt(u['T90']['crps'])} | {u['T90_vs_T24']['crps_improvement']:+.4f} (p={u['T90_vs_T24']['p_not_better']}) |")
    return "\n".join(L)
