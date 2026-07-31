#!/usr/bin/env python3
"""
PITCHER_K_OPPONENT_CONTEXT_GATE_A

Question: does adding the OPPONENT TEAM's K-rate as context improve the
pitcher_strikeouts rate projection's discrimination, beyond the pitcher's
own season/recency-blended rate alone?

Motivation: pitcher_k_calibration_absolute_gate_a found the live pitcher-
only rate projection (k_per_bf) has real, significant discrimination at
scale (concordance 0.604, CI [0.591, 0.620] on n=2544 strict-D-1 starts),
but the recency-weighted blend adds nothing over a plain season-to-date
average (0.605). The live app already blends in a real opponent-lineup
signal (lineupk.py's per-batter head-to-head/handedness K rate) for the
Monte Carlo sim, but that per-batter reconstruction is too expensive to
backtest at scale (~9 batters x 2500+ starts = tens of thousands of API
calls). This gate tests the cheaper, still-legitimate team-level proxy:
opponent team's cumulative season K-rate, strict D-1 (only games that
team played BEFORE this pitcher's start date).

Same discipline as the hits investigation and the K calibration gate:
  - strict D-1 features (no lookahead)
  - dev/cert split by date -- the blend weight is chosen on DEV ONLY,
    tested for real on an untouched CERT slice
  - concordance (AUC-equivalent for continuous outcomes) as the metric,
    matching the confound-free (rate-vs-rate, not count-vs-count) test
    already run in the pitcher-only baseline
  - paired bootstrap significance test: is the opponent-aware blend
    significantly better than pitcher-only on CERT, using the SAME
    resampled indices for both arms in each draw

Inputs:
  /tmp/pitcher_game_dataset_2026_fresh.csv  (pitcher_dataset_builder_a.py output)
  /tmp/team_k_gamelogs_2026.json            (fetch_team_k_gamelogs.py output)
"""
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
WORKDIR = REPO / "pitcher_k_opponent_context_gate_a_work"

RECENCY_DECAY = 0.15
SEASON_ANCHOR = 0.4
MIN_BF = 12
MIN_STARTS = 3
MIN_TEAM_GAMES = 10  # need a real sample of the opponent's season before trusting their K rate
CERT_DAYS = 21
N_BOOTSTRAP = 2000
SEED = 13

PITCHER_CSV = "/tmp/pitcher_game_dataset_2026_fresh.csv"
TEAM_GAMELOGS_JSON = "/tmp/team_k_gamelogs_2026.json"


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
                "opponent": r["opponent"],
                "bf": bf,
                "so": int(float(r.get("strikeouts") or 0)),
            })
    return rows


def build_team_asof_index(team_gamelogs):
    """For each team, sorted list of (date, cum_pa_before, cum_so_before,
    n_games_before) -- cumulative counts using STRICTLY earlier games only.
    A final sentinel entry (date='9999-99-99') carries the season-end
    totals, so a lookup for any date after the last tracked game still
    resolves to real cumulative data instead of falling off the end."""
    idx = {}
    for team, games in team_gamelogs.items():
        games = sorted(games, key=lambda g: g["date"])
        entries = []
        cum_pa = cum_so = 0
        for i, g in enumerate(games):
            entries.append((g["date"], cum_pa, cum_so, i))
            cum_pa += g["pa"]
            cum_so += g["so"]
        entries.append(("9999-99-99", cum_pa, cum_so, len(games)))
        idx[team] = entries
    return idx


def team_k_rate_asof(team_idx_entries, game_date):
    """Returns (k_rate, n_games_seen) using only games strictly before
    game_date -- the first entry with date >= game_date holds exactly
    that cumulative total (games strictly before it)."""
    for date, cum_pa, cum_so, n_games in team_idx_entries:
        if date >= game_date:
            return (cum_so / cum_pa if cum_pa else None), n_games
    return None, 0


def build_dataset(rows, team_idx):
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

                team_entries = team_idx.get(s["opponent"])
                opp_kr, opp_n = (None, 0)
                if team_entries:
                    opp_kr, opp_n = team_k_rate_asof(team_entries, s["game_date"])

                if opp_kr is not None and opp_n >= MIN_TEAM_GAMES:
                    out.append({
                        "pitcher_id": pid, "game_date": s["game_date"],
                        "k_per_bf": k_per_bf, "exp_bf": s["bf"],
                        "actual_so": s["so"], "opp_k_rate": opp_kr,
                    })
            sos.append(s["so"]); bfs.append(s["bf"])
    return out


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
    """CI on concordance(b) - concordance(a), same resampled row indices
    scored by both arms in each draw."""
    rng = random.Random(seed)
    n = len(scores_a)
    diffs = []
    for b in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = [scores_a[i] for i in idx]
        sb = [scores_b[i] for i in idx]
        av = [actual[i] for i in idx]
        ca = concordance(sa, av, n_pairs=pair_sub, seed=200000 + b)
        cb = concordance(sb, av, n_pairs=pair_sub, seed=200000 + b)  # same pair draws (seed) -> paired
        diffs.append(cb - ca)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return sum(diffs) / len(diffs), lo, hi


def main():
    rows = load_starts(PITCHER_CSV)
    team_gamelogs = json.load(open(TEAM_GAMELOGS_JSON))
    team_idx = build_team_asof_index(team_gamelogs)

    ds = build_dataset(rows, team_idx)
    print(f"eligible rows (>= {MIN_STARTS} prior starts, opponent team has >= {MIN_TEAM_GAMES} prior games): {len(ds)}")

    dates = sorted(r["game_date"] for r in ds)
    last_date = dates[-1]
    import datetime as dt
    cert_cutoff = (dt.date.fromisoformat(last_date) - dt.timedelta(days=CERT_DAYS)).isoformat()

    dev = [r for r in ds if r["game_date"] < cert_cutoff]
    cert = [r for r in ds if r["game_date"] >= cert_cutoff]
    print(f"cert_cutoff: {cert_cutoff}  dev: {len(dev)}  cert (final, untouched): {len(cert)}")

    # actual K RATE that start (confound-free target, matches pitcher-only test)
    dev_actual = [r["actual_so"] / r["exp_bf"] for r in dev]
    cert_actual = [r["actual_so"] / r["exp_bf"] for r in cert]

    dev_pitcher = [r["k_per_bf"] for r in dev]
    dev_opp = [r["opp_k_rate"] for r in dev]

    # pre-registered grid search for blend weight, DEV ONLY
    best_w, best_c = None, -1
    grid_results = []
    for w in [i / 20 for i in range(0, 21)]:  # 0.00 .. 1.00 step 0.05
        blended = [(1 - w) * p + w * o for p, o in zip(dev_pitcher, dev_opp)]
        c = concordance(blended, dev_actual, n_pairs=40000, seed=7)
        grid_results.append((w, c))
        if c > best_c:
            best_c, best_w = c, w
    print("\nDEV grid search (weight on opponent team K-rate):")
    for w, c in grid_results:
        marker = "  <-- best" if w == best_w else ""
        print(f"  w={w:.2f}  concordance={c:.4f}{marker}")

    print(f"\nchosen blend weight (dev-only): w={best_w:.2f}")

    # CERT (untouched) scoring
    cert_pitcher = [r["k_per_bf"] for r in cert]
    cert_opp = [r["opp_k_rate"] for r in cert]
    cert_blended = [(1 - best_w) * p + best_w * o for p, o in zip(cert_pitcher, cert_opp)]

    cert_dates_sorted = sorted(r["game_date"] for r in cert)
    print(f"\nCERTIFICATION (n={len(cert)}, dates {cert_dates_sorted[0]}..{cert_dates_sorted[-1]}):")
    for label, scores in [("pitcher_only", cert_pitcher), ("opp_team_only", cert_opp), (f"blended(w={best_w:.2f})", cert_blended)]:
        mean_c, lo, hi = bootstrap_ci_concordance(scores, cert_actual)
        print(f"  {label:20s}  concordance={mean_c:.4f}  CI [{lo:.4f}, {hi:.4f}]")

    mean_diff, lo_diff, hi_diff = paired_concordance_diff_ci(cert_pitcher, cert_blended, cert_actual)
    print(f"\npaired bootstrap concordance(blended) - concordance(pitcher_only): mean={mean_diff:+.4f}  CI [{lo_diff:+.4f}, {hi_diff:+.4f}]")

    print("\nGATE:")
    boss1 = lo_diff > 0
    print(f"  BOSS (blended significantly beats pitcher-only, diff CI entirely positive): {boss1}")
    print(f"  VERDICT: {'PASS -- opponent team K-rate context adds real value' if boss1 else 'FAIL -- no proven improvement from opponent team K-rate'}")

    WORKDIR.mkdir(exist_ok=True)
    report = {
        "script": "PITCHER_K_OPPONENT_CONTEXT_GATE_A",
        "n_dev": len(dev), "n_cert": len(cert), "cert_cutoff": cert_cutoff,
        "dev_grid_search": [{"w": w, "concordance": c} for w, c in grid_results],
        "chosen_weight": best_w,
        "cert_concordance": {
            "pitcher_only": cert_pitcher and bootstrap_ci_concordance(cert_pitcher, cert_actual),
            "opp_team_only": bootstrap_ci_concordance(cert_opp, cert_actual),
            "blended": bootstrap_ci_concordance(cert_blended, cert_actual),
        },
        "paired_diff_blended_minus_pitcher": {"mean": mean_diff, "ci_lower": lo_diff, "ci_upper": hi_diff},
        "boss_pass": boss1,
    }
    (WORKDIR / "pitcher_k_opponent_context_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {WORKDIR / 'pitcher_k_opponent_context_gate_a_report.json'}")
    return 0 if boss1 else 2


if __name__ == "__main__":
    raise SystemExit(main())
