#!/usr/bin/env python3
"""
PITCHER_K_SITUATIONAL_THIN_SAMPLE_GATE_A

Backtests the two new governance thresholds added to api.py after a real
user-reported case (Nick Pivetta: 213 career MLB starts, capped to MEDIUM
the same way a true rookie would be, purely because a 60-day IL stint
left him with only a few starts logged this season):

  K_VETERAN_MIN_CAREER_STARTS = 50   -- "established arm" bar
  K_LONG_LAYOFF_MIN_DAYS = 21        -- "real absence, not just a normal
                                        rotation turn" bar

Those two thresholds were shipped as a disclosed, undoubtedly-reasonable
but NOT independently validated heuristic (unlike K_THIN_SAMPLE_MIN_STARTS
itself, which a prior 70k-observation backtest -- pitcher_k_confidence_
cap_gate_a.py -- actually confirmed). This script closes that gap using
the same line-sweep methodology: ksim.simulate() takes no odds/lineup
input at all, so real historical strikeout outcomes at many hypothetical
lines (3.5-8.5, the real sportsbook K-prop range) stand in for real
market history we don't have.

Data: real MLB Stats API gameLogs, 2018-2025, every pitcher with >=3
starts in a season (774 unique pitchers). No historical betting lines
exist for this backtest to use, hence the line sweep. career_starts_asof
and days_since_last_appearance are computed point-in-time from this
same 2018-2025 window walked chronologically per pitcher (so this data
IS the "career" for classification purposes) -- a real, disclosed
limitation: a pitcher who debuted before 2018 has their true lifetime
total left-censored at this window's start, which can only make the
"veteran" bucket smaller/more conservative than reality, never inflate
it. days_since_last_appearance resets every season (splits are fetched
one season at a time, exactly mirroring api.py's own season-scoped
pitcher_feature_row query), so a normal Opening Day start is never
mistaken for a layoff -- consistent with what's actually deployed.

Four groups, all restricted to observations where the raw simulated
confidence is HIGH and n_starts_asof < K_THIN_SAMPLE_MIN_STARTS (8) --
i.e., every observation this script scores is a case the ORIGINAL
K_THIN_SAMPLE_MIN_STARTS rule would have capped to MEDIUM:

  deep_ref   n_starts_asof >= 8, confidence HIGH (the trusted reference
             rate the whole cap exists to protect)
  group_a    thin-sample HIGH, career_starts_asof >= 50, no long layoff
             -- the new "trust it" case
  group_b    thin-sample HIGH, career_starts_asof >= 50, long layoff
             -- the new "still cap it" case
  group_c    thin-sample HIGH, career_starts_asof < 50 (or unknown)
             -- the ORIGINAL case the cap was built for; should keep
             underperforming deep_ref, same as the original finding

Pre-registered pass bar (written before this script has ever been run):
  1. group_a hit rate is NOT confidently worse than deep_ref
     (bootstrap P(group_a worse than deep_ref) < 0.90) -- if it clears
     this, trusting these picks at HIGH is justified.
  2. group_b hit rate IS confidently worse than deep_ref
     (bootstrap P(group_b worse than deep_ref) >= 0.90) -- if it clears
     this, capping these picks is still justified.
  3. group_c hit rate IS confidently worse than deep_ref (sanity check:
     reproduces the original K_THIN_SAMPLE_MIN_STARTS finding on this
     independent dataset).
  4. Each of group_a/group_b/group_c has >= 200 observations (enough N
     to trust the bootstrap, not a handful of lucky/unlucky picks).

If any bar fails, the honest result is "the new thresholds are not
supported by this data" -- report it as such, don't relax the bar to
force a pass.

Run
---
python -u pitcher_k_situational_thin_sample_gate_a.py
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ksim

K_RATE_CALIBRATION = ksim.K_RATE_CALIBRATION


def simulate_fast(k_per_bf, expected_bf, line, start_k_rates=None, sims=10000, rng=None):
    """Vectorized reimplementation of ksim.simulate() -- same model exactly
    (same bf~Normal(expected_bf,2.5) clip [9,30], same time-through-the-
    order decay schedule, same K_RATE_CALIBRATION application, same
    confidence thresholds), just without ksim's pure-Python per-plate-
    appearance loop. That loop is fine for production (one real call per
    pick), but this backtest needs ~170k calls and would otherwise mean
    tens of billions of Python-level random draws -- confirmed to actually
    take many hours (a first, un-vectorized run of this script was killed
    after 69 minutes having barely started the simulation phase).
    Validated against the real ksim.simulate() at realistic K/BF and line
    ranges before use here: mean/side/side_prob/confidence all matched
    within Monte Carlo noise (worst observed diff: mean +/-0.01,
    side_prob +/-0.002, at 300k sims each) -- not assumed equivalent."""
    if rng is None:
        rng = np.random.default_rng()
    k_per_bf_c = k_per_bf * K_RATE_CALIBRATION
    if start_k_rates and len(start_k_rates) >= 4:
        pool = np.array(start_k_rates, dtype=float) * K_RATE_CALIBRATION
        pool = 0.7 * pool + 0.3 * k_per_bf_c
    else:
        pool = np.array([k_per_bf_c], dtype=float)

    pool_idx = rng.integers(0, len(pool), size=sims)
    k_samples = pool[pool_idx]
    bf = np.clip(np.round(rng.normal(expected_bf, 2.5, size=sims)), 9, 30).astype(int)

    positions = np.arange(30)
    pos_matrix = np.broadcast_to(positions, (sims, 30))
    times_through = pos_matrix // 9
    decay = np.where(times_through == 0, 1.0, np.where(times_through == 1, 0.94, 0.85))
    p_k = np.clip(k_samples[:, None] * decay, 0.02, 0.6)
    mask = pos_matrix < bf[:, None]
    draws = rng.random((sims, 30))
    hits = (draws < p_k) & mask
    ks_per_sim = hits.sum(axis=1)

    prob_over = float(np.mean(ks_per_sim > line))
    prob_under = float(np.mean(ks_per_sim < line))
    if prob_over >= prob_under:
        side, side_prob = "OVER", prob_over
    else:
        side, side_prob = "UNDER", prob_under
    if side_prob >= 0.70:
        confidence = "HIGH"
    elif side_prob >= 0.64:
        confidence = "MEDIUM"
    elif side_prob >= 0.59:
        confidence = "LOW"
    else:
        confidence = "NO_BET"
    return {"side": side, "side_prob": round(side_prob, 3), "confidence": confidence}


MLB = "https://statsapi.mlb.com/api/v1"
SEASONS = list(range(2018, 2026))
CACHE_DIR = Path("/data/pitcher_k_situational_gate_a_work")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PID_CACHE = CACHE_DIR / "pids.json"
GAMELOG_CACHE = CACHE_DIR / "gamelogs.json"

RECENCY_DECAY = 0.6
SEASON_ANCHOR = 0.15
K_THIN_SAMPLE_MIN_STARTS = 8
K_VETERAN_MIN_CAREER_STARTS = 50
K_LONG_LAYOFF_MIN_DAYS = 21
LINE_SWEEP = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
SIMS_PER_LINE = 4000  # lighter than production's 10000 -- this script runs
                       # this many thousands of times, production runs it once


def http_get(url, timeout=20, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def discover_pids():
    if PID_CACHE.exists():
        return json.loads(PID_CACHE.read_text())
    pids = set()
    for season in SEASONS:
        url = (f"{MLB}/stats?stats=season&group=pitching&season={season}"
               f"&sportId=1&limit=1000&sortStat=gamesStarted&order=desc")
        d = http_get(url)
        if not d:
            print(f"  WARNING: season leader fetch failed for {season}")
            continue
        splits = d["stats"][0]["splits"]
        for s in splits:
            if (s["stat"].get("gamesStarted") or 0) >= 3:
                pids.add(s["player"]["id"])
    pids = sorted(pids)
    PID_CACHE.write_text(json.dumps(pids))
    return pids


def fetch_one(pid, season):
    url = f"{MLB}/people/{pid}/stats?stats=gameLog&group=pitching&season={season}"
    d = http_get(url)
    if not d:
        return pid, season, []
    try:
        splits = d["stats"][0]["splits"]
    except Exception:
        return pid, season, []
    rows = []
    for sp in splits:
        st = sp.get("stat", {})
        rows.append({
            "date": sp.get("date"),
            "gamesStarted": int(st.get("gamesStarted", 0) or 0),
            "battersFaced": int(st.get("battersFaced", 0) or 0),
            "strikeOuts": int(st.get("strikeOuts", 0) or 0),
        })
    rows.sort(key=lambda r: r["date"] or "")
    return pid, season, rows


def fetch_all_gamelogs(pids):
    if GAMELOG_CACHE.exists():
        cached = json.loads(GAMELOG_CACHE.read_text())
        print(f"  loaded cached gamelogs: {len(cached)} pitchers")
        return cached

    data = {}
    tasks = [(pid, season) for pid in pids for season in SEASONS]
    print(f"  fetching {len(tasks)} (pitcher, season) gameLogs, {len(pids)} pitchers x {len(SEASONS)} seasons ...")
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = [ex.submit(fetch_one, pid, season) for pid, season in tasks]
        for fut in as_completed(futs):
            pid, season, rows = fut.result()
            data.setdefault(str(pid), {})[str(season)] = rows
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{len(tasks)} fetched ({time.time()-t0:.0f}s elapsed)")
    print(f"  fetch complete in {time.time()-t0:.0f}s")
    GAMELOG_CACHE.write_text(json.dumps(data))
    return data


def build_observations(gamelogs):
    """Walks each pitcher's 2018-2025 starts in chronological order,
    replicating pitcher_feature_row's exact within-season feature
    computation (bf>=12 filter, recency-weighted k_per_bf blended with a
    season anchor, n_starts eligibility floor of 3) while separately
    tracking cross-season career_starts_asof and within-season
    days_since_last_appearance -- exactly the two new signals api.py now
    computes live."""
    observations = []
    for pid, by_season in gamelogs.items():
        career_starts_asof = 0
        for season in map(str, SEASONS):
            rows = by_season.get(season) or []
            if not rows:
                continue

            sos, bfs, per_start_krate = [], [], []
            cum_bf = cum_so = 0
            n_starts = 0
            last_appearance_date = None

            for row in rows:
                game_date = row["date"] or ""
                bf = row["battersFaced"]
                so = row["strikeOuts"]
                is_start = row["gamesStarted"] >= 1

                # --- as-of-this-game features (before this game updates state) ---
                if n_starts >= 3:
                    days_since_last = None
                    if last_appearance_date and game_date:
                        try:
                            import datetime as _dt
                            d1 = _dt.date.fromisoformat(last_appearance_date)
                            d2 = _dt.date.fromisoformat(game_date)
                            days_since_last = (d2 - d1).days
                        except Exception:
                            days_since_last = None

                    season_kbf = cum_so / cum_bf if cum_bf else 0
                    n = len(sos)
                    w = [np.exp(-RECENCY_DECAY * (n - 1 - i)) for i in range(n)]
                    rec_kbf = (
                        sum(wi * s for wi, s in zip(w, sos))
                        / sum(wi * b for wi, b in zip(w, bfs))
                    ) if sum(w) else season_kbf
                    k_per_bf = (1 - SEASON_ANCHOR) * rec_kbf + SEASON_ANCHOR * season_kbf
                    avg_bf = sum(bfs[-5:]) / len(bfs[-5:])

                    if k_per_bf > 0 and avg_bf > 0 and is_start:
                        observations.append({
                            "pid": pid, "season": season, "date": game_date,
                            "n_starts_asof": n_starts,
                            "career_starts_asof": career_starts_asof,
                            "days_since_last_appearance": days_since_last,
                            "k_per_bf": k_per_bf, "avg_bf": avg_bf,
                            "per_start_krate": list(per_start_krate[-12:]),
                            "actual_k": so,
                        })

                # --- now fold this game into state ---
                if is_start:
                    career_starts_asof += 1
                if game_date and (last_appearance_date is None or game_date > last_appearance_date):
                    last_appearance_date = game_date
                if bf >= 12:
                    sos.append(so); bfs.append(bf)
                    per_start_krate.append(so / bf if bf else 0)
                    cum_bf += bf; cum_so += so
                    n_starts += 1
    return observations


def grade(obs, rng):
    """Sweeps the real sportsbook K-line range for one real start and
    returns every HIGH-confidence result (side, hit/miss) -- the simulator
    needs no odds/lineup input, so real outcomes at synthetic lines stand
    in for real market history we don't have. Uses simulate_fast(), not
    ksim.simulate() directly -- see that function's docstring."""
    out = []
    for line in LINE_SWEEP:
        sim = simulate_fast(obs["k_per_bf"], obs["avg_bf"], line,
                             start_k_rates=obs["per_start_krate"], sims=SIMS_PER_LINE, rng=rng)
        if sim["confidence"] != "HIGH":
            continue
        actual = obs["actual_k"]
        hit = (actual > line) if sim["side"] == "OVER" else (actual < line)
        out.append(hit)
    return out


def bootstrap_p_worse(hits_a, hits_b, seed=20260907, b=5000):
    """P(arm A's true hit rate is worse than arm B's), via paired
    bootstrap resampling of each arm's own hit list independently."""
    rng = np.random.default_rng(seed)
    a = np.asarray(hits_a, dtype=float)
    b_arr = np.asarray(hits_b, dtype=float)
    worse = 0
    for _ in range(b):
        ra = rng.choice(a, size=len(a), replace=True).mean() if len(a) else 0
        rb = rng.choice(b_arr, size=len(b_arr), replace=True).mean() if len(b_arr) else 0
        if ra < rb:
            worse += 1
    return worse / b


def main():
    print("PITCHER_K_SITUATIONAL_THIN_SAMPLE_GATE_A\n=========================================")
    pids = discover_pids()
    print(f"pitcher universe: {len(pids)} (>=3 starts in some season, 2018-2025)")

    gamelogs = fetch_all_gamelogs(pids)
    print("\nbuilding point-in-time observations ...")
    observations = build_observations(gamelogs)
    print(f"  {len(observations)} eligible pregame observations (n_starts_asof >= 3)")

    deep_hits, group_a_hits, group_b_hits, group_c_hits = [], [], [], []
    n_processed = 0
    t0 = time.time()
    rng = np.random.default_rng(20260907)
    print(f"\nsimulating {len(observations)} observations x {len(LINE_SWEEP)} lines each ...")
    for obs in observations:
        n_processed += 1
        if n_processed % 2000 == 0:
            elapsed = time.time() - t0
            rate_per_s = n_processed / elapsed if elapsed else 0
            eta = (len(observations) - n_processed) / rate_per_s if rate_per_s else 0
            print(f"    simulated {n_processed}/{len(observations)} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        hits = grade(obs, rng)
        if not hits:
            continue

        n_starts = obs["n_starts_asof"]
        career = obs["career_starts_asof"]
        layoff = obs["days_since_last_appearance"]

        if n_starts >= K_THIN_SAMPLE_MIN_STARTS:
            deep_hits.extend(hits)
            continue

        is_veteran = career >= K_VETERAN_MIN_CAREER_STARTS
        is_long_layoff = layoff is not None and layoff >= K_LONG_LAYOFF_MIN_DAYS

        if is_veteran and not is_long_layoff:
            group_a_hits.extend(hits)
        elif is_veteran and is_long_layoff:
            group_b_hits.extend(hits)
        else:
            group_c_hits.extend(hits)

    def rate(hits):
        return (sum(hits) / len(hits)) if hits else float("nan")

    print(f"\n{'group':12s} {'n':>8s} {'hit_rate':>10s}")
    print(f"{'deep_ref':12s} {len(deep_hits):>8d} {rate(deep_hits):>10.4f}")
    print(f"{'group_a':12s} {len(group_a_hits):>8d} {rate(group_a_hits):>10.4f}")
    print(f"{'group_b':12s} {len(group_b_hits):>8d} {rate(group_b_hits):>10.4f}")
    print(f"{'group_c':12s} {len(group_c_hits):>8d} {rate(group_c_hits):>10.4f}")

    p_a_worse = bootstrap_p_worse(group_a_hits, deep_hits) if group_a_hits and deep_hits else float("nan")
    p_b_worse = bootstrap_p_worse(group_b_hits, deep_hits) if group_b_hits and deep_hits else float("nan")
    p_c_worse = bootstrap_p_worse(group_c_hits, deep_hits) if group_c_hits and deep_hits else float("nan")

    print(f"\nbootstrap P(group_a worse than deep_ref) = {p_a_worse:.4f}  (bar: < 0.90 to pass)")
    print(f"bootstrap P(group_b worse than deep_ref) = {p_b_worse:.4f}  (bar: >= 0.90 to pass)")
    print(f"bootstrap P(group_c worse than deep_ref) = {p_c_worse:.4f}  (bar: >= 0.90 to pass, sanity check)")

    c1 = len(group_a_hits) >= 200 and p_a_worse < 0.90
    c2 = len(group_b_hits) >= 200 and p_b_worse >= 0.90
    c3 = len(group_c_hits) >= 200 and p_c_worse >= 0.90
    c4_n = len(group_a_hits) >= 200 and len(group_b_hits) >= 200 and len(group_c_hits) >= 200
    passed = c1 and c2 and c3 and c4_n

    print(f"\nGATE CHECKS:")
    print(f"  1. group_a not confidently worse than deep_ref: {c1}")
    print(f"  2. group_b confidently worse than deep_ref:     {c2}")
    print(f"  3. group_c confidently worse than deep_ref:     {c3}")
    print(f"  4. all groups have >=200 observations:          {c4_n}")
    verdict = "PITCHER_K_SITUATIONAL_THRESHOLDS_PASS" if passed else "PITCHER_K_SITUATIONAL_THRESHOLDS_DO_NOT_CLEAR_GATE"
    print(f"\nVERDICT: {verdict}")

    report = {
        "script": "PITCHER_K_SITUATIONAL_THIN_SAMPLE_GATE_A",
        "seasons": SEASONS,
        "n_pitchers": len(pids),
        "n_observations": len(observations),
        "groups": {
            "deep_ref": {"n": len(deep_hits), "hit_rate": rate(deep_hits)},
            "group_a_veteran_no_layoff": {"n": len(group_a_hits), "hit_rate": rate(group_a_hits)},
            "group_b_veteran_long_layoff": {"n": len(group_b_hits), "hit_rate": rate(group_b_hits)},
            "group_c_non_veteran": {"n": len(group_c_hits), "hit_rate": rate(group_c_hits)},
        },
        "bootstrap_p_worse_than_deep_ref": {
            "group_a": p_a_worse, "group_b": p_b_worse, "group_c": p_c_worse,
        },
        "gate": {"c1_group_a_trusted": c1, "c2_group_b_still_risky": c2,
                 "c3_group_c_sanity_check": c3, "c4_min_n": c4_n},
        "passed": passed, "verdict": verdict,
    }
    (CACHE_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {CACHE_DIR / 'report.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
