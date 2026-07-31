#!/usr/bin/env python3
"""
PITCHER_K_REAL_LINEUP_GATE_A

*** RESULT INVALIDATED -- DO NOT TRUST THE NUMBERS THIS SCRIPT PRODUCES ***
lineup_k_dataset_builder_a's input data is contaminated with lookahead.
lineupk.py's general_k_rate_vs_hand / head_to_head_k_rate pass `endDate`
to the MLB Stats API expecting it to bound results to "before this game
only" (the same strict-D-1 pattern proven correct on the pitcher side via
local gameLog date-filtering). Verified directly against the live API:
`endDate` is a silent no-op for stats=statSplits, stats=season, AND
stats=vsPlayer -- three different endDate values (including one dated
before the season started) all returned identical, full-current-season
totals. So every "as-of-date" batter K-rate this script scores is
actually that batter's FULL SEASON rate, including games that hadn't
happened yet as of the historical start being scored. The concordance
jump this produced (0.75 vs pitcher-only's 0.61) is inflated by that
leak and cannot be trusted as evidence the real per-batter lineup signal
helps. This does NOT affect live picks (as_of_date is always "today" in
live serving, so there's no future to leak) -- it only invalidates this
specific historical backtest. A valid version would need each batter's
raw game-by-game log pulled and filtered locally, the same fix already
proven necessary for the pitcher-side feature. Left in the repo as a
disclosed, documented dead end, not a result to build on.

The decisive version of the opponent-context question. pitcher_k_opponent_
context_gate_a tested a cheap team-average proxy and came back inconclusive
(diff CI [-0.0008, +0.0162], touching zero). This tests the REAL signal the
live app actually uses: per-batter K-rate vs. this exact pitcher's
handedness + head-to-head history, for all 9 opposing batters, strict D-1
(lineup_k_dataset_builder_a's output -- 2544 real historical starts, real
network-pulled per-batter data, not team averages).

Same discipline throughout:
  - strict D-1 features (no lookahead) on both sides
  - dev/cert split by date -- blend weight chosen on DEV ONLY, tested on
    an untouched CERT slice
  - concordance (AUC-equivalent, confound-free: rate-vs-rate using actual
    K rate that start, not raw count) as the metric
  - paired bootstrap significance test: same resampled indices scored by
    both arms in each draw

Input: /tmp/lineup_k_reconstruction_2026.jsonl (lineup_k_dataset_builder_a.py output)
"""
import datetime as dt
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent
WORKDIR = REPO / "pitcher_k_real_lineup_gate_a_work"

CERT_DAYS = 21
N_BOOTSTRAP = 2000
SEED = 13
INPUT_JSONL = "/tmp/lineup_k_reconstruction_2026.jsonl"


def load_dataset(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("lineup_avg_kr") is None or d.get("lineup_n_data", 0) < 6:
                continue
            rows.append(d)
    return rows


def concordance(scores, actual, n_pairs=200000, seed=13):
    rng = random.Random(seed)
    n = len(scores)
    conc = disc = tie_s = 0
    for _ in range(n_pairs):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        if actual[i] == actual[j]:
            continue
        if scores[i] == scores[j]:
            tie_s += 1
            continue
        if actual[i] < actual[j]:
            i, j = j, i
        if scores[i] > scores[j]:
            conc += 1
        else:
            disc += 1
    total = conc + disc
    return (conc + 0.5 * tie_s) / (total + tie_s) if (total + tie_s) else float("nan")


def bootstrap_ci_concordance(scores, actual, n_boot=N_BOOTSTRAP, seed=SEED, pair_sub=8000):
    rng = random.Random(seed)
    n = len(scores)
    vals = []
    for b in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        s2 = [scores[i] for i in idx]
        a2 = [actual[i] for i in idx]
        vals.append(concordance(s2, a2, n_pairs=pair_sub, seed=100000 + b))
    vals.sort()
    lo = vals[int(0.025 * n_boot)]
    hi = vals[int(0.975 * n_boot)]
    return sum(vals) / len(vals), lo, hi


def paired_concordance_diff_ci(scores_a, scores_b, actual, n_boot=N_BOOTSTRAP, seed=SEED, pair_sub=8000):
    rng = random.Random(seed)
    n = len(scores_a)
    diffs = []
    for b in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = [scores_a[i] for i in idx]
        sb = [scores_b[i] for i in idx]
        av = [actual[i] for i in idx]
        ca = concordance(sa, av, n_pairs=pair_sub, seed=200000 + b)
        cb = concordance(sb, av, n_pairs=pair_sub, seed=200000 + b)
        diffs.append(cb - ca)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return sum(diffs) / len(diffs), lo, hi


def main():
    ds = load_dataset(INPUT_JSONL)
    print(f"eligible rows (real per-batter lineup data, >=6/9 batters usable): {len(ds)}")

    dates = sorted(r["game_date"] for r in ds)
    last_date = dates[-1]
    cert_cutoff = (dt.date.fromisoformat(last_date) - dt.timedelta(days=CERT_DAYS)).isoformat()

    dev = [r for r in ds if r["game_date"] < cert_cutoff]
    cert = [r for r in ds if r["game_date"] >= cert_cutoff]
    print(f"cert_cutoff: {cert_cutoff}  dev: {len(dev)}  cert (final, untouched): {len(cert)}")

    dev_actual = [r["actual_so"] / r["exp_bf"] for r in dev]
    cert_actual = [r["actual_so"] / r["exp_bf"] for r in cert]

    dev_pitcher = [r["k_per_bf"] for r in dev]
    dev_lineup = [r["lineup_avg_kr"] for r in dev]

    best_w, best_c = None, -1
    grid_results = []
    for w in [i / 20 for i in range(0, 21)]:
        blended = [(1 - w) * p + w * o for p, o in zip(dev_pitcher, dev_lineup)]
        c = concordance(blended, dev_actual, n_pairs=40000, seed=7)
        grid_results.append((w, c))
        if c > best_c:
            best_c, best_w = c, w
    print("\nDEV grid search (weight on real per-batter lineup K-rate):")
    for w, c in grid_results:
        marker = "  <-- best" if w == best_w else ""
        print(f"  w={w:.2f}  concordance={c:.4f}{marker}")
    print(f"\nchosen blend weight (dev-only): w={best_w:.2f}")

    # also score the LIVE weight (0.45 lineup / 0.55 pitcher, from api.py's
    # PITCHER_WEIGHT/LINEUP_WEIGHT) for reference -- what's actually deployed
    live_w = 0.45

    cert_pitcher = [r["k_per_bf"] for r in cert]
    cert_lineup = [r["lineup_avg_kr"] for r in cert]
    cert_blended_dev_w = [(1 - best_w) * p + best_w * o for p, o in zip(cert_pitcher, cert_lineup)]
    cert_blended_live_w = [(1 - live_w) * p + live_w * o for p, o in zip(cert_pitcher, cert_lineup)]

    cert_dates_sorted = sorted(r["game_date"] for r in cert)
    print(f"\nCERTIFICATION (n={len(cert)}, dates {cert_dates_sorted[0]}..{cert_dates_sorted[-1]}):")
    results = {}
    for label, scores in [
        ("pitcher_only", cert_pitcher),
        ("lineup_only", cert_lineup),
        (f"blended(dev_w={best_w:.2f})", cert_blended_dev_w),
        (f"blended(live_w={live_w:.2f})", cert_blended_live_w),
    ]:
        mean_c, lo, hi = bootstrap_ci_concordance(scores, cert_actual)
        results[label] = (mean_c, lo, hi)
        print(f"  {label:24s}  concordance={mean_c:.4f}  CI [{lo:.4f}, {hi:.4f}]")

    mean_diff_dev, lo_diff_dev, hi_diff_dev = paired_concordance_diff_ci(cert_pitcher, cert_blended_dev_w, cert_actual)
    mean_diff_live, lo_diff_live, hi_diff_live = paired_concordance_diff_ci(cert_pitcher, cert_blended_live_w, cert_actual)

    print(f"\npaired bootstrap concordance(dev-weight blend) - concordance(pitcher_only): mean={mean_diff_dev:+.4f}  CI [{lo_diff_dev:+.4f}, {hi_diff_dev:+.4f}]")
    print(f"paired bootstrap concordance(live-weight blend) - concordance(pitcher_only): mean={mean_diff_live:+.4f}  CI [{lo_diff_live:+.4f}, {hi_diff_live:+.4f}]")

    boss_dev = lo_diff_dev > 0
    boss_live = lo_diff_live > 0
    print("\nGATE:")
    print(f"  dev-selected weight significantly beats pitcher-only: {boss_dev}")
    print(f"  live deployed weight significantly beats pitcher-only: {boss_live}")
    passed = boss_dev or boss_live
    print(f"  VERDICT: {'PASS -- real per-batter lineup signal adds proven value' if passed else 'FAIL -- still no proven improvement, even with the real per-batter signal'}")

    WORKDIR.mkdir(exist_ok=True)
    report = {
        "script": "PITCHER_K_REAL_LINEUP_GATE_A",
        "INVALID": True,
        "invalid_reason": ("Input data contaminated with lookahead: lineupk.py's endDate "
                            "parameter is a silent no-op on the MLB Stats API (verified "
                            "directly), so every batter K-rate here is the batter's FULL "
                            "SEASON rate, not their rate as of that historical game. Does "
                            "not affect live picks (as_of_date is always 'today' live), only "
                            "this backtest. Do not treat the concordance/gate results below "
                            "as evidence of anything."),
        "n_dev": len(dev), "n_cert": len(cert), "cert_cutoff": cert_cutoff,
        "dev_grid_search": [{"w": w, "concordance": c} for w, c in grid_results],
        "chosen_dev_weight": best_w, "live_weight": live_w,
        "cert_concordance": {k: {"mean": v[0], "ci_lower": v[1], "ci_upper": v[2]} for k, v in results.items()},
        "paired_diff_dev_weight_minus_pitcher": {"mean": mean_diff_dev, "ci_lower": lo_diff_dev, "ci_upper": hi_diff_dev},
        "paired_diff_live_weight_minus_pitcher": {"mean": mean_diff_live, "ci_lower": lo_diff_live, "ci_upper": hi_diff_live},
        "boss_dev_pass": boss_dev, "boss_live_pass": boss_live, "passed": passed,
    }
    (WORKDIR / "pitcher_k_real_lineup_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {WORKDIR / 'pitcher_k_real_lineup_gate_a_report.json'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
