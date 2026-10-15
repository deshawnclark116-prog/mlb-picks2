#!/usr/bin/env python3
"""Ahead-of-cutoff NHL V2 research collector -- operations only, NEVER model fitting.

Keeps a checked-out, hash-verified frozen B2 research engine warm across future
NHL cutoffs. Calls the original, unmodified engine exclusively in its legally
open 15-minute pregame window. Every receipt is verified and durably pushed as
soon as captured, and the read-only website shadow is updated immediately.
A process exit without a fresh forecast is NOT counted as successful coverage.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nhl_v2_secondary_cutoff_gate import NHL_SCHEDULE, load_json, upcoming, utc

BRANCH = "feature/nhl-v2-nfl-style-projection-board"
HORIZONS = {1440: "T24H", 90: "T90", 30: "T30"}
CUTOFF_WAKE_SECONDS = 13 * 60
UTC = timezone.utc


def command(argv, cwd, *, capture=False):
    proc = subprocess.run(argv, cwd=str(cwd), text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)
    if proc.returncode:
        raise RuntimeError(f"Command {argv[:3]} failed with exit {proc.returncode}: "
                           f"{(proc.stdout or '')[-1000:]}")
    if capture:
        return proc.stdout
    return ""


def clock():
    return datetime.now(UTC)


def verify_original(original):
    command([sys.executable, "nhl_v2_phase1a_sog_forward.py", "verify"], original)


def original_rows(original):
    p = original / "nhl_models/nhl_player_outcome_v2/phase1a_sog_forward/ledger.jsonl"
    return [json.loads(v) for v in p.read_text().splitlines() if v.strip()]


def git_push(original):
    """Only push allowed model-research RECEIPTS, not locked code or grades."""
    command(["git", "config", "user.name", "github-actions[bot]"], original)
    command(["git", "config", "user.email",
             "41898282+github-actions[bot]@users.noreply.github.com"], original)
    command(["git", "add", "-A",
             "nhl_models/nhl_player_outcome_v2/phase1a_sog_forward"], original)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(original)).returncode == 0:
        return False
    command(["git", "commit", "-m", "Record genuine NHL V2 on-time pregame decision"], original)
    for attempt in range(5):
        if subprocess.run(["git", "push", "origin", "HEAD:" + BRANCH],
                          cwd=str(original)).returncode == 0:
            print("NHL_RESEARCH_DECISION_DURABLY_COMMITTED", flush=True)
            return True
        command(["git", "fetch", "origin", BRANCH], original)
        command(["git", "rebase", "origin/" + BRANCH], original)
        time.sleep(attempt+1)
    raise RuntimeError("Unable to preserve locked original NHL forecast receipt on research branch")


def _render_verified_shadow(original, website):
    """Recreate the whole site artifact from the ORIGINAL frozen, verified model."""
    verify_original(original)
    target = website / "docs/nhl_v2_shadow.json"
    command([sys.executable, "nhl_v2_projection_board_publish.py",
             "--out", str(target.resolve())], original)
    result = json.loads(target.read_text())
    for name, expected in (
        ("verified_original_hash_chain", True),
        ("verified_frozen_code_and_model_lock", True),
        ("actual_player_book_lines_verified", False),
        ("dressed_lineups_verified", False),
        ("not_a_betting_board", True),
        ("historical_corrected_probabilities_were_not_published_pregame", True)
    ):
        if result.get(name) is not expected:
            raise RuntimeError(f"Research shadow violates source contract: {name}")
    result["source_snapshot_commit_sha"] = command(
        ["git", "rev-parse", "HEAD"], original, capture=True).strip()
    result["source_snapshot_repo"] = "deshawnclark116-prog/mlb-picks2"
    result["source_snapshot_branch"] = BRANCH
    result["asof_scope"] = "FROZEN_B2_ORIGINAL_PREGAME_ONLY; not certified for betting"
    target.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False,
                                 allow_nan=False)+"\n")
    return result


def publish(original, website):
    """Race-safe website publication; NEVER rebase stale large JSON over main.

    Two independent on-time workers previously succeeded in appending research
    receipts but collided when rebasing the *same* generated website JSON. The
    failure must not throw away a verified receipt or overwrite another run.
    On retry, cleanly discard ONLY our unpushed site commit, fast-forward to
    latest main and regenerate from the complete verified original ledger.
    There is no force-push, user files are not reverted remotely, and no
    historical forecast is ever fabricated.
    """
    command(["git", "config", "user.name", "github-actions[bot]"], website)
    command(["git", "config", "user.email",
             "41898282+github-actions[bot]@users.noreply.github.com"], website)
    for attempt in range(7):
        if attempt:
            # Another publisher may have changed the exact same large JSON.
            # A textual rebase is unsafe; regenerate against current main.
            command(["git", "fetch", "origin", "main"], website)
            command(["git", "reset", "--hard", "origin/main"], website)
            # The original receipt ledger may itself have advanced on GitHub.
            # Do not overwrite a newer validated model snapshot with a stale
            # local research checkout. Verification follows an FF-only update.
            command(["git", "fetch", "origin", BRANCH], original)
            command(["git", "merge", "--ff-only", "origin/" + BRANCH], original)
            print("NHL_SITE_PUBLISH_REBUILD_FROM_LATEST_REMOTE_LEDGER",
                  attempt+1, flush=True)
        _render_verified_shadow(original, website)
        command(["git", "add", "--", "docs/nhl_v2_shadow.json"], website)
        if subprocess.run(["git", "diff", "--cached", "--quiet"],
                          cwd=str(website)).returncode == 0:
            print("NHL_RESEARCH_PAGE_ALREADY_CURRENT", flush=True)
            return
        command(["git", "commit", "-m",
                 "Publish authentic pregame NHL V2 count forecasts immediately"], website)
        if subprocess.run(["git", "push", "origin", "HEAD:main"],
                          cwd=str(website)).returncode == 0:
            print("NHL_NEW_FORECAST_VISIBLE_ON_RESEARCH_BOARD", flush=True)
            return
        time.sleep(min(attempt+1, 4))
    raise RuntimeError(
        "NHL genuine pregame forecasts saved in immutable research ledger, "
        "but website publication conflicted seven times; never force-push")


def decision_status(rows, candidates):
    """Exact game/horizon/start match; inherited old missed rows never become new."""
    by_key = {}
    for rec in rows:
        if rec.get("record_type") not in (
                "FORECAST", "MISSED_CUTOFF", "MISSED_REVISED_CUTOFF",
                "INVALID_LATE", "SOURCE_FETCH_FAILED_NO_FORECAST",
                "SOURCE_INCOMPLETE_NO_FORECAST"):
            continue
        key = (str(rec["game_id"]), rec["forecast_horizon"], rec["scheduled_start"])
        by_key.setdefault(key, Counter())[rec["record_type"]] += 1
    results = []
    for candidate in candidates:
        key = (candidate["game_id"], HORIZONS[candidate["horizon_minutes"]],
               candidate["start"].replace("+00:00", "Z"))
        # Scheduled starts in the original locked ledger include fractional
        # precision at times: compare instants, not raw format strings.
        found = Counter()
        for (gid, horizon, start), rec in by_key.items():
            if gid == key[0] and horizon == key[1] and utc(start) == utc(candidate["start"]):
                found.update(rec)
        results.append({"game_id":candidate["game_id"],
                        "horizon":key[1], "cutoff":candidate["cutoff"],
                        "forecast_rows":found["FORECAST"],
                        "failure_rows":sum(n for label,n in found.items()
                                           if label != "FORECAST"),
                        "status":"FORECAST" if found["FORECAST"] else
                                 ("MISSED_OR_FAILED" if found else "NO_SAVED_DECISION")})
    return results


def session(original, website, *, duration_minutes=108, now_fn=clock, sleep=time.sleep,
            get_schedule=lambda:load_json(NHL_SCHEDULE), forecast=None, persist=None,
            publish_fn=None, report_path=None):
    """Poll 60 sec and collect every in-window official NHL game until job expires.

    For regression tests, forecast/persist/publish can be injected. Production
    invokes ONLY the immutable engine script; it does not fake inputs.
    """
    start = now_fn()
    finish = start + timedelta(minutes=duration_minutes)
    # Fail BEFORE the 15-minute deadline if original source/model/ledger
    # certification is broken. Prior production worker deferred this expensive
    # check until the cutoff was already imminent.
    if forecast is None:
        verify_original(original)
        print("NHL_FROZEN_MODEL_VERIFIED_BEFORE_LIVE_CUTOFF", flush=True)
    attempted_cutoffs = set()
    observed = []
    failures = []
    last_schedule_error = None
    recent_source_success = False
    while now_fn() < finish:
        now = now_fn()
        try:
            due = upcoming(now, get_schedule(), lead_minutes=14)
            recent_source_success = True
            last_schedule_error = None
        except Exception as e:
            last_schedule_error = str(e)[:180]
            print("NHL_OFFICIAL_SCHEDULE_UNAVAILABLE", last_schedule_error, flush=True)
            sleep(min(60, max(0,(finish-now_fn()).total_seconds())))
            continue
        candidates = [r for r in due if r["seconds_until_cutoff"] <= CUTOFF_WAKE_SECONDS
                      and r["cutoff"] not in attempted_cutoffs]
        cutoffs = sorted({r["cutoff"] for r in candidates})
        for cutoff in cutoffs:
            batch = [r for r in candidates if r["cutoff"]==cutoff]
            attempted_cutoffs.add(cutoff)
            print("NHL_ON_TIME_CUTOFF_ATTEMPT", cutoff, len(batch), flush=True)
            if forecast is None:
                verify_original(original)
                command([sys.executable, "-u", "nhl_v2_phase1a_sog_forward.py", "run"], original)
                verify_original(original)
            else:
                forecast(batch)
            if persist is None:
                git_push(original)
            else:
                persist(batch)
            status = decision_status(original_rows(original), batch)
            observed.extend(status)
            for row in status:
                if row["status"]!="FORECAST":
                    failures.append(row)
            print("NHL_REAL_FORECAST_DECISIONS", json.dumps(status, sort_keys=True), flush=True)
            if any(r["status"]=="FORECAST" for r in status):
                if publish_fn is None:
                    publish(original, website)
                else:
                    publish_fn()
        if not cutoffs:
            sleep(min(60, max(0,(finish-now_fn()).total_seconds())))
        # If model processing takes >= the session lifetime, finish rather than
        # sleeping or inventing late forecasts.
    result = {
        "schema":"NHL_V2_RESIDENT_CUTOFF_DELIVERY_V1",
        "started_at_utc":start.isoformat(),
        "ended_at_utc":now_fn().isoformat(),
        "attempted_game_horizons":len(observed),
        "verified_forecast_decisions":sum(x["status"]=="FORECAST" for x in observed),
        "failed_or_unrecorded_decisions":failures,
        "cutoffs_attempted":len(attempted_cutoffs),
        "last_schedule_error":last_schedule_error,
        "schedule_retrieved_during_session":recent_source_success,
        "no_historical_backfill":True,
        "model_files_modified":False,
        "official_bets_added":0,
    }
    if report_path:
        out = Path(report_path)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NHL_RESIDENT_WORKER_FINAL_RECEIPT",json.dumps(result,sort_keys=True),flush=True)
    if failures:
        raise RuntimeError(f"NHL deadlines reached with no real pregame forecast: {len(failures)}")
    if not recent_source_success:
        raise RuntimeError("NHL schedule was unavailable throughout collector session")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--website-dir",type=Path,required=True)
    p.add_argument("--duration-min",type=int,default=108)
    p.add_argument("--report",type=Path,default=Path("/tmp/nhl_v2_resident_receipt.json"))
    a=p.parse_args()
    if not 1<=a.duration_min<=115:
        p.error("worker lifetime must be within 1-115 minutes")
    session(a.model_dir.resolve(),a.website_dir.resolve(),
            duration_minutes=a.duration_min,report_path=a.report)


if __name__=="__main__":
    main()
