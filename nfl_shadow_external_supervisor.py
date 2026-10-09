"""Opt-in independent NFL deadline supervisor for a continuously running external worker.

This process NEVER reads outcome stats, executes model code, writes forecast state
or modifies cutoff rules. It only verifies upcoming games independently and
dispatches the already-reviewed GitHub collector/publisher ahead of deadlines.
Do not deploy until the repaired collector is on main and the worker is authorized.
"""
import argparse
import base64
import fcntl
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = "deshawnclark116-prog/mlb-picks2"
GOOD = {"DONE", "PARTIAL_V2_MISSING"}
ACTIVE = {"in_progress", "queued", "pending", "waiting", "requested"}
INTERVAL_S = 60
LOOKAHEAD = timedelta(minutes=45)
POSTCUTOFF_GRACE = timedelta(minutes=7)


def utc(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("timezone is required")
        return value.astimezone(timezone.utc)
    obj = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if obj.tzinfo is None:
        raise ValueError("timezone is required")
    return obj.astimezone(timezone.utc)


def decisions(games, ledger, public_games, now, shadow_active=False, publisher_active=False):
    """Pure scheduling policy; inject games, state and time in tests, never in live CLI."""
    from nfl_phase1d_schedule import forecast_cutoff, iso
    now = utc(now)
    needed_shadow, needed_publisher, incidents = [], [], []
    for gid, game in sorted(games.items()):
        kick = utc(game["kick"])
        if kick < now - timedelta(days=1):
            continue
        for hz in ("T24", "T90"):
            cutoff = forecast_cutoff(kick, hz)
            if cutoff > now + LOOKAHEAD or cutoff < now - timedelta(hours=2):
                continue
            key = f"{gid}|{hz}|{iso(cutoff)}"
            state = ledger.get(key, {}).get("state", "MISSING_DISPATCH_KEY")
            if state in GOOD:
                if (gid, hz) not in public_games:
                    if cutoff <= now - POSTCUTOFF_GRACE:
                        incidents.append({"key": key, "kind": "DONE_NOT_PUBLISHED", "state": state})
                    if not publisher_active:
                        needed_publisher.append(key)
            elif cutoff <= now - POSTCUTOFF_GRACE:
                incidents.append({"key": key, "kind": "MISSED_OR_UNFINISHED", "state": state})
            elif now < cutoff and state not in {"FAILED", "MISSED_REAL_CUTOFF"} and not shadow_active:
                needed_shadow.append(key)
    return {"dispatch_shadow": bool(needed_shadow), "dispatch_publisher": bool(needed_publisher),
            "shadow_keys": needed_shadow, "publisher_keys": needed_publisher, "incidents": incidents}


class Github:
    def __init__(self, token, repo=REPO):
        if not token:
            raise RuntimeError("NFL_SUPERVISOR_GITHUB_TOKEN is required")
        self.token, self.repo = token, repo

    def request(self, method, path, payload=None):
        url = "https://api.github.com/repos/" + self.repo + "/" + path
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, method=method,
                                     headers={"Authorization": "Bearer " + self.token,
                                              "Accept": "application/vnd.github+json",
                                              "X-GitHub-Api-Version": "2022-11-28",
                                              "User-Agent": "nfl-independent-deadline-supervisor",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=18) as response:
                raw = response.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"GITHUB_API_{exc.code}: {method} {path}") from exc

    def content(self, path, ref):
        name = urllib.parse.quote(path, safe="/")
        obj = self.request("GET", f"contents/{name}?ref={urllib.parse.quote(ref)}")
        if obj.get("encoding") != "base64":
            raise RuntimeError("GITHUB_CONTENT_UNAVAILABLE: expected base64 document")
        return base64.b64decode(obj["content"])

    def runs(self, name):
        data = self.request("GET", f"actions/workflows/{name}/runs?per_page=40")
        return data.get("workflow_runs", [])

    def dispatch(self, name, inputs=None):
        return self.request("POST", f"actions/workflows/{name}/dispatches",
                            {"ref": "main", "inputs": inputs or {}})

    def alert(self, message, issue=71):
        return self.request("POST", f"issues/{issue}/comments", {"body": message})


def read_ledger(raw):
    last = {}
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        last[row["key"]] = row
    return last


def read_public(raw):
    doc = json.loads(raw)
    fields = doc.get("row_fields") or []
    if "game_id" not in fields or "id" not in fields:
        raise RuntimeError("PUBLICATION_SCHEMA_INVALID")
    result = set()
    for hz in ("T24", "T90"):
        for row in doc.get("forecasts", {}).get(hz, []):
            if len(row) != len(fields):
                raise RuntimeError("PUBLICATION_SCHEMA_INVALID")
            result.add((row[fields.index("game_id")], hz))
    return result


def api_active(runs):
    return any(run.get("status") in ACTIVE and run.get("head_branch") == "main"
               for run in runs)


def load_log(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {"dispatch": {}, "alerts": {}}


def save_log(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    os.replace(temp, path)


def recent(record, now, mins):
    if not record:
        return False
    return now - utc(record) < timedelta(minutes=mins)


def one_tick(gh, store, now=None, fetch_schedule=None):
    """Decide from an independent live schedule, remote ledger and public result."""
    from nfl_phase1d_schedule import parse_schedule
    import nfl_shadow_schedule as schedule
    now = utc(now or datetime.now(timezone.utc))
    fetch_schedule = fetch_schedule or schedule.fetch
    raw, _ = fetch_schedule()
    games = parse_schedule(schedule.sanitize(raw, now)[0])
    led = read_ledger(gh.content("dispatch_ledger.jsonl", "nfl-shadow-state"))
    public = read_public(gh.content("docs/nfl_phase1_shadow.json", "main"))
    shadow = gh.runs("nfl_phase1e_shadow.yml")
    pubruns = gh.runs("nfl_new_engine_publish.yml")
    choice = decisions(games, led, public, now, api_active(shadow), api_active(pubruns))
    log = load_log(store)
    if choice["dispatch_shadow"]:
        key = "|".join(choice["shadow_keys"])
        if not recent(log["dispatch"].get("shadow:" + key), now, 12):
            gh.dispatch("nfl_phase1e_shadow.yml", {"mode": "run", "duration_min": "90"})
            log["dispatch"]["shadow:" + key] = now.isoformat()
            choice["shadow_dispatched"] = True
    if choice["dispatch_publisher"]:
        key = "|".join(choice["publisher_keys"])
        if not recent(log["dispatch"].get("publisher:" + key), now, 8):
            gh.dispatch("nfl_new_engine_publish.yml")
            log["dispatch"]["publisher:" + key] = now.isoformat()
            choice["publisher_dispatched"] = True
    for issue in choice["incidents"]:
        fingerprint = issue["kind"] + "|" + issue["key"] + "|" + issue["state"]
        if recent(log["alerts"].get(fingerprint), now, 180):
            continue
        message = ("AUTOMATED NFL DELIVERY FAILURE (external deadline supervisor)\n\n"
                   f"- Decision: \`{issue['kind']}\`\n- Key: \`{issue['key']}\`\n"
                   f"- State: \`{issue['state']}\`\n- Observed UTC: \`{now.isoformat()}\`\n\n"
                   "No forecast was backfilled. Review the dispatch ledger, run logs and public receipt audit.")
        gh.alert(message)
        log["alerts"][fingerprint] = now.isoformat()
    save_log(store, log)
    return choice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="Run one real-time audit (default: continuous independent worker)")
    ap.add_argument("--interval-sec", type=int, default=INTERVAL_S)
    ap.add_argument("--state-file", default=os.environ.get("NFL_SUPERVISOR_STATE_FILE", "/var/data/nfl-supervisor-state.json"))
    args = ap.parse_args()
    if args.interval_sec < 30:
        raise SystemExit("minimum poll interval is 30 seconds")
    gh = Github(os.environ.get("NFL_SUPERVISOR_GITHUB_TOKEN"))
    path = Path(args.state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "a+") as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise SystemExit("SUPERVISOR_ALREADY_RUNNING: only one worker may own this state") from e
        while True:
            try:
                decision = one_tick(gh, path)
                print(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "decision": decision},
                                 sort_keys=True), flush=True)
            except Exception as exc:
                print(json.dumps({"status": "SUPERVISOR_POLL_FAILED",
                                  "at": datetime.now(timezone.utc).isoformat(),
                                  "error": str(exc)}), file=sys.stderr, flush=True)
                if args.once:
                    raise
            if args.once:
                break
            time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
