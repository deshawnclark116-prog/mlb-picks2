#!/usr/bin/env python3
"""
PITCHER_K_CALIBRATION_ABSOLUTE_GATE_A

Same discipline as hits_absolute_significance_gate_a.py, adapted to how the
K market actually works: there's no trained classifier in the live path
(build_strikeout_pick_with_debug -> ksim.simulate directly off real
season/recent K-rates), so "is the model broken" here is fundamentally a
BIAS/calibration question, not a feature-quality question. K_RATE_CALIBRATION
= 1.0504 is a single fixed multiplier, derived once on 2025 dev data and
validated on an early-2026 holdout back on 2026-07-13. A live check on the
current v8.22 window (2026-07-17 to 2026-07-30, n=80 real graded picks)
found a real +0.32 K systematic OVER-projection bias, with OVER picks
hitting 43.3% and UNDER picks hitting 58.0% -- a hardcoded correction
factor that was right three weeks ago silently going stale is the leading
hypothesis. This gate tests that directly and re-derives the factor if
warranted, with the same real-holdout, bootstrap-significance rigor as the
hits investigation -- not another point estimate.

Because a uniform multiplicative calibration constant preserves rank order
by construction, this is NOT an AUC/discrimination problem (that's already
fine at the raw-rate level) -- it's specifically a MEAN-BIAS problem in the
OVER/UNDER decision. So the gate here has two parts:

  BOSS 1 (bias): on a genuinely untouched final certification slice (most
          recent 21 days, never used to derive the candidate factor), the
          95% bootstrap CI of (mean_projected - mean_actual) must contain
          0 -- i.e. NOT be significantly biased in either direction.
  BOSS 2 (does the fix actually help): paired bootstrap significance test
          of |bias| for the newly-derived factor vs the CURRENT 1.0504
          factor, both scored on the SAME cert-slice rows -- the new
          factor's bias must be significantly smaller, not just a smaller
          point estimate.

Strict D-1: every feature for a start uses only that pitcher's OWN prior
starts (battersFaced >= 12, matching pitcher_feature_row()'s exact filter
and recency-decay blend), never anything from the target start itself or
later.

Read-only on the CSV this reads from. Writes only a report to its own
work dir. Does not touch ksim.py or any production file.

Run
---
python -u pitcher_k_calibration_absolute_gate_a.py --source /tmp/pitcher_game_dataset_2026_fresh.csv
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

REPO = Path(__file__).resolve().parent
WORKDIR = REPO / "pitcher_k_calibration_absolute_gate_a_work"

# Exact match to api.py's pitcher_feature_row() constants.
RECENCY_DECAY = 0.15
SEASON_ANCHOR = 0.4
CURRENT_CALIBRATION = 1.0504
MIN_BF = 12
MIN_STARTS = 3
CERT_DAYS = 21
N_BOOTSTRAP = 2000
SEED = 13


def load_starts(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bf = int(float(r.get("batters_faced") or 0))
            if bf < MIN_BF:
                continue
            rows.append({
                "pitcher_id": r["pitcher_id"],
                "game_date": r["game_date"],
                "bf": bf,
                "so": int(float(r.get("strikeouts") or 0)),
            })
    return rows


def build_dataset(rows):
    """Strict D-1: for each start, reconstruct exactly what
    pitcher_feature_row() would have produced as-of that date -- season
    anchor blended with recency-weighted per-start K rate, using only
    that pitcher's strictly-earlier starts this dataset covers."""
    by_pitcher = defaultdict(list)
    for r in rows:
        by_pitcher[r["pitcher_id"]].append(r)

    out = []
    for pid, starts in by_pitcher.items():
        starts.sort(key=lambda r: r["game_date"])
        sos, bfs = [], []
        for s in starts:
            n = len(sos)
            if n >= MIN_STARTS:
                cum_bf = sum(bfs); cum_so = sum(sos)
                season_kbf = cum_so / cum_bf if cum_bf else 0.0
                w = [math.exp(-RECENCY_DECAY * (n - 1 - i)) for i in range(n)]
                rec_kbf = (sum(wi * so for wi, so in zip(w, sos)) /
                           sum(wi * bf for wi, bf in zip(w, bfs))) if sum(w) else season_kbf
                k_per_bf = (1 - SEASON_ANCHOR) * rec_kbf + SEASON_ANCHOR * season_kbf
                out.append({
                    "pitcher_id": pid, "game_date": s["game_date"],
                    "k_per_bf": k_per_bf, "exp_bf": s["bf"],
                    "actual_so": s["so"],
                })
            sos.append(s["so"]); bfs.append(s["bf"])
    return out


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float); labels = np.asarray(labels, dtype=float)
    pos = labels.sum(); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]; i = 0; n = len(s)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def bias_bootstrap_ci(proj, actual, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    proj = np.asarray(proj); actual = np.asarray(actual)
    n = len(proj)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(float(np.mean(proj[idx] - actual[idx])))
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_abs_bias_diff_ci(proj_a, proj_b, actual, n_boot=N_BOOTSTRAP, seed=SEED):
    """95% CI for (|bias_a| - |bias_b|) on paired resamples -- negative and
    significant means arm B has genuinely smaller bias, not just luck."""
    rng = np.random.default_rng(seed)
    pa = np.asarray(proj_a); pb = np.asarray(proj_b); ac = np.asarray(actual)
    n = len(ac)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bias_a = abs(float(np.mean(pa[idx] - ac[idx])))
        bias_b = abs(float(np.mean(pb[idx] - ac[idx])))
        diffs.append(bias_a - bias_b)
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def synthetic_line(prior_so):
    """A fair, no-lookahead proxy for a real sportsbook line: the pitcher's
    own rolling median K's over prior starts, rounded to X.5. Real
    historical FanDuel K lines aren't available for backtesting, so this
    approximates 'a line set close to true recent expectation' without
    using anything from the target start itself."""
    if not prior_so:
        return None
    med = float(np.median(prior_so[-10:]))
    return math.floor(med) + 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--workdir", default=str(WORKDIR))
    args = ap.parse_args()
    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)

    print("PITCHER_K_CALIBRATION_ABSOLUTE_GATE_A\n======================================")
    raw_rows = load_starts(args.source)
    print(f"loaded {len(raw_rows)} qualifying starts (bf>={MIN_BF}) from {args.source}")

    data = build_dataset(raw_rows)
    print(f"strict-D-1 eligible rows (>= {MIN_STARTS} prior starts): {len(data)}")

    dates = sorted({r["game_date"] for r in data})
    last_date = date.fromisoformat(dates[-1])
    cert_cutoff = (last_date - timedelta(days=CERT_DAYS)).isoformat()
    dev = [r for r in data if r["game_date"] < cert_cutoff]
    cert = [r for r in data if r["game_date"] >= cert_cutoff]
    print(f"cert_cutoff: {cert_cutoff}  dev: {len(dev)}  cert (final, untouched): {len(cert)}")

    # ---- Derive a new calibration factor on DEV only (never touches cert) ----
    dev_raw_proj = np.array([r["k_per_bf"] * r["exp_bf"] for r in dev])
    dev_actual = np.array([r["actual_so"] for r in dev], dtype=float)
    new_factor = float(dev_actual.mean() / dev_raw_proj.mean()) if dev_raw_proj.mean() else 1.0
    print(f"\nnewly-derived calibration factor (dev-only, bias-minimizing): {new_factor:.4f}")
    print(f"  (current live factor: {CURRENT_CALIBRATION})")

    # ---- Score three candidates on the untouched CERT slice ----
    cert_raw_proj = np.array([r["k_per_bf"] * r["exp_bf"] for r in cert])
    cert_actual = np.array([r["actual_so"] for r in cert], dtype=float)

    candidates = {
        "no_calibration (1.0)": cert_raw_proj * 1.0,
        f"current_live ({CURRENT_CALIBRATION})": cert_raw_proj * CURRENT_CALIBRATION,
        f"newly_derived ({new_factor:.4f})": cert_raw_proj * new_factor,
    }

    print(f"\n============ CERTIFICATION (n={len(cert)}, dates {cert_cutoff}..{dates[-1]}) ============")
    print(f"  {'candidate':28s} {'mean_bias':>10s} {'95% CI'}")
    results = {}
    for name, proj in candidates.items():
        mean_b, lo, hi = bias_bootstrap_ci(proj, cert_actual)
        results[name] = {"proj": proj, "mean_bias": mean_b, "ci_lower": lo, "ci_upper": hi}
        sig = "SIGNIFICANTLY BIASED" if (lo > 0 or hi < 0) else "not significantly biased"
        print(f"  {name:28s} {mean_b:+10.4f}  [{lo:+.4f}, {hi:+.4f}]  -> {sig}")

    current_proj = results[f"current_live ({CURRENT_CALIBRATION})"]["proj"]
    new_proj = results[f"newly_derived ({new_factor:.4f})"]["proj"]
    diff_mean, diff_lo, diff_hi = paired_abs_bias_diff_ci(current_proj, new_proj, cert_actual)
    print(f"\npaired bootstrap: |bias(current)| - |bias(new)|")
    print(f"  mean={diff_mean:+.4f}  95% CI [{diff_lo:+.4f}, {diff_hi:+.4f}]")

    boss1_pass = not (results[f"newly_derived ({new_factor:.4f})"]["ci_lower"] > 0 or
                       results[f"newly_derived ({new_factor:.4f})"]["ci_upper"] < 0)
    boss2_pass = diff_lo > 0  # current's |bias| significantly bigger than new's
    passed = boss1_pass and boss2_pass

    print(f"\n============ GATE ============")
    print(f"  BOSS 1 (newly-derived factor's bias CI contains 0): {boss1_pass}")
    print(f"  BOSS 2 (newly-derived factor significantly less biased than current live factor): {boss2_pass}")
    print(f"  VERDICT: {'PASS -- recalibrate to ' + f'{new_factor:.4f}' if passed else 'FAIL -- do not change K_RATE_CALIBRATION yet'}")

    # ---- Secondary check: OVER/UNDER decision quality against a fair synthetic line ----
    print(f"\n============ OVER/UNDER DECISION QUALITY (synthetic no-lookahead line) ============")
    # Seed each pitcher's prior-SO history from DEV (real starts before the
    # cert slice even begins), so a pitcher's FIRST cert-slice start still
    # gets a real line instead of being skipped for lack of history.
    seed_hist = defaultdict(list)
    for r in sorted(dev, key=lambda x: x["game_date"]):
        seed_hist[r["pitcher_id"]].append(r["actual_so"])

    for factor_name, factor in [("no_calibration", 1.0), ("current_live", CURRENT_CALIBRATION), ("newly_derived", new_factor)]:
        correct = total = 0
        prior_so_by_pid = defaultdict(list, {k: list(v) for k, v in seed_hist.items()})
        for r in sorted(cert, key=lambda x: x["game_date"]):
            pid = r["pitcher_id"]
            line = synthetic_line(prior_so_by_pid[pid])
            if line is not None:
                proj = r["k_per_bf"] * r["exp_bf"] * factor
                pred_over = proj > line
                actual_over = r["actual_so"] > line
                total += 1
                correct += int(pred_over == actual_over)
            prior_so_by_pid[pid].append(r["actual_so"])
        rate = 100 * correct / total if total else 0
        print(f"  {factor_name:16s} factor={factor:.4f}  decision_accuracy={rate:.1f}%  (n={total})")

    report = {
        "script": "PITCHER_K_CALIBRATION_ABSOLUTE_GATE_A",
        "n_dev": len(dev), "n_cert": len(cert), "cert_cutoff": cert_cutoff,
        "current_calibration": CURRENT_CALIBRATION, "newly_derived_calibration": new_factor,
        "cert_results": {k: {kk: vv for kk, vv in v.items() if kk != "proj"} for k, v in results.items()},
        "paired_diff_current_minus_new": {"mean": diff_mean, "ci_lower": diff_lo, "ci_upper": diff_hi},
        "boss1_pass": boss1_pass, "boss2_pass": boss2_pass, "passed": passed,
    }
    (work / "pitcher_k_calibration_absolute_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {work / 'pitcher_k_calibration_absolute_gate_a_report.json'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
