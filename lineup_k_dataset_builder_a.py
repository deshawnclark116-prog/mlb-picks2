#!/usr/bin/env python3
"""
LINEUP_K_DATASET_BUILDER_A

Reconstructs the REAL per-batter opposing-lineup K-rate signal (the one
lineupk.py actually blends into the live pitcher_strikeouts projection)
for every strict-D-1-eligible historical pitcher start, so it can be
tested for real significance -- not the cheap team-average proxy used by
pitcher_k_opponent_context_gate_a (which came back inconclusive, CI
[-0.0008, +0.0162], too close to call).

Why this is slow on purpose: for each start, this calls lineupk's real
per-batter K-rate lookups (handedness split + season fallback + head-to-
head vs that exact pitcher) for all ~9 opposing batters, strict D-1
(as_of_date = that start's game_date, no lookahead). That's ~20-27 MLB
Stats API calls per start. At ~2500 eligible starts, that's roughly
50,000-65,000 individual HTTP calls. Paced deliberately slowly (extra
sleep between starts, on top of lineupk's own built-in 0.06s/batter
pacing) to stay well clear of any rate limiting -- this is a one-time
background research pull, not something that needs to be fast.

Game lineups themselves cost ZERO new API calls: pitcher_dataset_builder_a
already pulled and cached every game feed for this window under
./hr_model/cache/game_feed (get_game_feed() reads that cache first).

Checkpointed and resumable: writes one JSON line per completed start to
OUTPUT_JSONL immediately, and skips any (pitcher_id, game_id) pair
already present on restart -- safe to kill and rerun.
"""
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import hr_dataset_builder_a as hb
import lineupk

PITCHER_CSV = "/tmp/pitcher_game_dataset_2026_fresh.csv"
OUTPUT_JSONL = "/tmp/lineup_k_reconstruction_2026.jsonl"

RECENCY_DECAY = 0.15
SEASON_ANCHOR = 0.4
MIN_BF = 12
MIN_STARTS = 3
SEASON = 2026

EXTRA_SLEEP_BETWEEN_STARTS = 2.5  # on top of lineupk's own 0.06s/batter pacing


def load_starts(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bf = int(float(r.get("batters_faced") or 0))
            if bf < MIN_BF:
                continue
            rows.append({
                "game_id": r["game_id"],
                "pitcher_id": r["pitcher_id"],
                "game_date": r["game_date"],
                "team": r["team"],
                "opponent": r["opponent"],
                "side": r["side"],
                "pitcher_hand": r["pitcher_hand"] or "R",
                "bf": bf,
                "so": int(float(r.get("strikeouts") or 0)),
            })
    return rows


def build_eligible_starts(rows):
    """Strict D-1 pitcher-side feature, same as pitcher_k_calibration_absolute_gate_a,
    but keeps game_id/opponent/side/pitcher_hand needed for lineup lookup."""
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
                out.append({**s, "k_per_bf": k_per_bf})
            sos.append(s["so"]); bfs.append(s["bf"])
    return out


def load_done_keys(path):
    done = set()
    p = Path(path)
    if not p.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                done.add((d["pitcher_id"], d["game_id"]))
            except Exception:
                continue
    return done


def main():
    rows = load_starts(PITCHER_CSV)
    eligible = build_eligible_starts(rows)
    eligible.sort(key=lambda r: r["game_date"])
    print(f"eligible strict-D-1 starts: {len(eligible)}")

    done = load_done_keys(OUTPUT_JSONL)
    print(f"already checkpointed: {len(done)}")

    remaining = [r for r in eligible if (r["pitcher_id"], r["game_id"]) not in done]
    print(f"remaining to fetch: {len(remaining)}")
    est_seconds = len(remaining) * (EXTRA_SLEEP_BETWEEN_STARTS + 9 * 0.06 + 2.0)
    print(f"rough estimate: {est_seconds/3600:.1f} hours\n")

    fout = open(OUTPUT_JSONL, "a")
    n_done_this_run = 0
    n_skipped_no_lineup = 0
    t0 = time.time()

    for i, s in enumerate(remaining):
        gid = int(s["game_id"])
        feed = hb.get_game_feed(gid)
        if not feed:
            n_skipped_no_lineup += 1
            continue

        opp_side = "away" if s["side"] == "home" else "home"
        opp_batters = hb.batting_order_ids(feed, opp_side)
        if not opp_batters:
            n_skipped_no_lineup += 1
            continue

        exp_ks, avg_kr, n_data = lineupk.lineup_k_expectation(
            opp_batters,
            s["pitcher_hand"],
            SEASON,
            s["bf"],
            pitcher_id=int(s["pitcher_id"]),
            as_of_date=s["game_date"],
        )

        rec = {
            "pitcher_id": s["pitcher_id"],
            "game_id": s["game_id"],
            "game_date": s["game_date"],
            "k_per_bf": s["k_per_bf"],
            "exp_bf": s["bf"],
            "actual_so": s["so"],
            "lineup_avg_kr": avg_kr,
            "lineup_n_data": n_data,
            "lineup_n_batters": len(opp_batters),
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        n_done_this_run += 1

        if n_done_this_run % 25 == 0:
            elapsed = time.time() - t0
            rate = n_done_this_run / elapsed if elapsed else 0
            eta_h = (len(remaining) - n_done_this_run) / rate / 3600 if rate else float("nan")
            print(f"  [{n_done_this_run}/{len(remaining)}] {s['game_date']} "
                  f"elapsed={elapsed/60:.1f}m eta={eta_h:.1f}h skipped_no_lineup={n_skipped_no_lineup}",
                  flush=True)

        time.sleep(EXTRA_SLEEP_BETWEEN_STARTS)

    fout.close()
    print(f"\ndone. wrote {n_done_this_run} new rows, skipped {n_skipped_no_lineup} (no lineup data).")
    print(f"output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
